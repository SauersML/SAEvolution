"""
This script serves as the main entry point and conductor for the simulation.
It loads parameters from the configuration file, establishes the initial
population of agents, controls the primary loop iterating through generations,
sequences the execution of game rounds and the subsequent evolutionary updates
for each generation, and oversees overall simulation logging and state saving
for restartability and dashboard data.
"""

import yaml
import os
import logging
import datetime
import random
import json
import uuid
import argparse
import shutil # For copying config
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from manager import Agent, initialize_population, calculate_fitness, evolve_population
from engine import run_game_round

load_dotenv()

SIMULATION_STATE_BASE_DIR_CONFIG_KEY = 'state_saving.directory' # Key in config.yaml
LOG_DIR_CONFIG_KEY = 'logging.log_directory'

def load_config(config_path='config.yaml'):
    if not os.path.exists(config_path):
        # If running from a checkpoint, config might be loaded from snapshot
        logging.warning(f"Primary configuration file {config_path} not found. Relying on checkpoint if resuming.")
        return None # Allow to proceed if resuming, load_checkpoint will handle config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
             raise ValueError(f"Configuration file {config_path} is empty or invalid.")
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        raise ValueError(f"Error parsing configuration file {config_path}: {e}") from e
    except Exception as e:
        logging.error(f"An unexpected error occurred loading configuration {config_path}: {e}")
        raise

def setup_logging(log_level_str: str, log_dir_base: str, simulation_run_id: str, resume_mode: bool = False):
    """
    Sets up logging for the simulation.
    If resuming, appends to the existing log file for that simulation_run_id.
    """
    os.makedirs(log_dir_base, exist_ok=True)
    log_filename = Path(log_dir_base) / f"sim_log_{simulation_run_id}.log"

    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_str.upper(), logging.INFO)

    # Remove existing handlers if any, to prevent duplicate logging on re-runs in same interpreter session (e.g. testing)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, mode='a' if resume_mode else 'w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configured. Log file: {log_filename}. Resume mode: {resume_mode}")


def save_generation_checkpoint(
    simulation_run_id: str,
    config_snapshot: dict,
    generation_number: int,
    population: list[Agent],
    game_details_for_generation: list[dict],
    rng_state: any,
    generation_summary_metrics: dict,
    state_base_dir: str
):
    """Saves the complete state of a generation for restart and dashboarding."""
    run_dir = Path(state_base_dir) / simulation_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot only once per run (or if it changes, which it shouldn't mid-run)
    config_snapshot_path = run_dir / "config_snapshot.json"
    if not config_snapshot_path.exists():
        try:
            with open(config_snapshot_path, 'w') as f:
                json.dump(config_snapshot, f, indent=4)
            logging.info(f"Saved config snapshot to {config_snapshot_path}")
        except Exception as e:
            logging.error(f"Failed to save config snapshot: {e}")

    # Generation state file
    gen_state_filename = run_dir / f"generation_{generation_number:04d}.json"
    population_data_for_save = []
    for agent in population:
        agent_dict = agent.to_dict()
        # Ensure fitness is captured if available (calculate_fitness should ideally store it on agent or pass it here)
        if hasattr(agent, 'current_fitness_score'): 
            agent_dict['fitness_score'] = agent.current_fitness_score
        elif 'fitness_scores_map' in generation_summary_metrics: # Fallback if not directly on agent
            agent_dict['fitness_score'] = generation_summary_metrics.get('fitness_scores_map', {}).get(agent.agent_id, None)
        population_data_for_save.append(agent_dict)

    state_data = {
        'simulation_run_id': simulation_run_id,
        'generation_number': generation_number,
        'timestamp_completed': datetime.datetime.now().isoformat(),
        'rng_state': rng_state,
        'population_state': population_data_for_save,
        'generation_summary_metrics': generation_summary_metrics,
    }
    try:
        with open(gen_state_filename, 'w') as f:
            json.dump(state_data, f, indent=4)
        logging.info(f"Saved generation {generation_number} state to {gen_state_filename}")
    except Exception as e:
        logging.error(f"Failed to save generation {generation_number} state: {e}")

    # Games data file (JSON Lines)
    games_data_filename = run_dir / f"games_generation_{generation_number:04d}.jsonl"
    try:
        with open(games_data_filename, 'w') as f: # Overwrite if exists for this generation
            for game_detail in game_details_for_generation:
                f.write(json.dumps(game_detail) + '\n')
        logging.info(f"Saved {len(game_details_for_generation)} game details for generation {generation_number} to {games_data_filename}")
    except Exception as e:
        logging.error(f"Failed to save game details for generation {generation_number}: {e}")

    # Update latest generation tracker
    latest_gen_tracker_path = run_dir / "_latest_generation_number.txt"
    try:
        with open(latest_gen_tracker_path, 'w') as f:
            f.write(str(generation_number))
    except Exception as e:
        logging.error(f"Failed to update latest generation tracker: {e}")


def load_checkpoint(simulation_run_id_to_resume: str, state_base_dir: str) -> tuple | None:
    """Loads the latest checkpoint for a given simulation_run_id."""
    run_dir = Path(state_base_dir) / simulation_run_id_to_resume
    if not run_dir.is_dir():
        logging.error(f"Resume failed: Simulation run directory not found: {run_dir}")
        return None

    latest_gen_tracker_path = run_dir / "_latest_generation_number.txt"
    config_snapshot_path = run_dir / "config_snapshot.json"

    if not latest_gen_tracker_path.exists():
        logging.error(f"Resume failed: Latest generation tracker not found in {run_dir}")
        return None
    if not config_snapshot_path.exists():
        logging.error(f"Resume failed: Config snapshot not found in {run_dir}")
        return None

    try:
        with open(latest_gen_tracker_path, 'r') as f:
            last_saved_generation = int(f.read().strip())
    except Exception as e:
        logging.error(f"Resume failed: Could not read latest generation number: {e}")
        return None

    gen_state_filename = run_dir / f"generation_{last_saved_generation:04d}.json"
    if not gen_state_filename.exists():
        logging.error(f"Resume failed: Generation state file not found: {gen_state_filename}")
        return None

    try:
        with open(gen_state_filename, 'r') as f:
            state_data = json.load(f)
        with open(config_snapshot_path, 'r') as f:
            loaded_config = json.load(f)
    except Exception as e:
        logging.error(f"Resume failed: Could not load state or config files: {e}")
        return None

    # The rng_state was saved as a list [version, state_list, gaussian_state],
    # where state_list itself was originally a tuple but became a list during JSON serialization.
    # We need to convert the outer list to a tuple, and also the inner state_list back to a tuple.
    loaded_rng_state_as_list = state_data['rng_state']
    if isinstance(loaded_rng_state_as_list, list) and len(loaded_rng_state_as_list) == 3 and isinstance(loaded_rng_state_as_list[1], list):
        version = loaded_rng_state_as_list[0]
        internal_mt_state_tuple = tuple(loaded_rng_state_as_list[1]) # Convert inner list to tuple
        gaussian_state = loaded_rng_state_as_list[2]
        rng_state_to_set = (version, internal_mt_state_tuple, gaussian_state)
        random.setstate(rng_state_to_set)
    else:
        logging.error(f"Resume failed: RNG state from checkpoint is malformed. Expected list of 3 elements with 2nd element being a list. Got: {loaded_rng_state_as_list}")
    
    resumed_population = [Agent.from_dict(agent_data) for agent_data in state_data['population_state']]
    
    start_generation = last_saved_generation + 1
    
    logging.info(f"Successfully loaded checkpoint for run '{simulation_run_id_to_resume}' from generation {last_saved_generation}.")
    return resumed_population, start_generation, loaded_config, simulation_run_id_to_resume


def find_latest_run_id(state_base_dir: str) -> str | None:
    """Finds the most recent simulation_run_id to resume from."""
    p_state_base_dir = Path(state_base_dir)
    if not p_state_base_dir.is_dir():
        return None

    latest_run_id = None
    latest_mod_time = 0

    for run_dir in p_state_base_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "_latest_generation_number.txt").exists():
            try:
                # Use modification time of the tracker file as a proxy for recency
                mod_time = (run_dir / "_latest_generation_number.txt").stat().st_mtime
                if mod_time > latest_mod_time:
                    latest_mod_time = mod_time
                    latest_run_id = run_dir.name
            except Exception:
                continue # Ignore directories that cause errors
    
    if latest_run_id:
        logging.info(f"Found latest run ID to resume: {latest_run_id}")
    else:
        logging.info("No resumable runs found in state directory.")
    return latest_run_id


async def run_simulation(args):
    # --- Initial Config Loading (might be overridden by checkpoint) ---
    primary_config = load_config(args.config_file)
    if primary_config is None and not args.resume_run_id and not args.resume_latest:
        # Use a basic logger for this pre-setup error message
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.error(f"Error: Main config file '{args.config_file}' not found and not resuming. Exiting.")
        return

    # --- Determine Run Mode and Load State if Resuming ---
    current_population: list[Agent] = []
    start_generation = 1
    simulation_run_id: str = ""
    config: dict = {} # This will hold the definitive config for the run

    # Determine base directories from primary_config if available, else use defaults
    # These might be updated if resuming and config snapshot has different paths
    state_base_dir = (primary_config or {}).get('state_saving', {}).get('directory', 'simulation_state')
    log_dir_base = (primary_config or {}).get('logging', {}).get('log_directory', 'logs')


    resuming = False
    if args.resume_run_id:
        # Initial basic logging if setup_logging hasn't run yet
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Attempting to resume run ID: {args.resume_run_id}")
        checkpoint_data = load_checkpoint(args.resume_run_id, state_base_dir)
        if checkpoint_data:
            current_population, start_generation, config, simulation_run_id = checkpoint_data
            resuming = True
            # Update state_base_dir and log_dir_base from loaded config if they exist and differ
            state_base_dir = config.get('state_saving', {}).get('directory', state_base_dir)
            log_dir_base = config.get('logging', {}).get('log_directory', log_dir_base)
        else:
            logging.error(f"Failed to load checkpoint for run ID '{args.resume_run_id}'. Exiting.")
            return
    elif args.resume_latest:
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Attempting to resume the latest run.")
        run_id_to_resume = find_latest_run_id(state_base_dir)
        if run_id_to_resume:
            checkpoint_data = load_checkpoint(run_id_to_resume, state_base_dir)
            if checkpoint_data:
                current_population, start_generation, config, simulation_run_id = checkpoint_data
                resuming = True
                state_base_dir = config.get('state_saving', {}).get('directory', state_base_dir)
                log_dir_base = config.get('logging', {}).get('log_directory', log_dir_base)
            else:
                logging.error(f"Failed to load latest checkpoint for run ID '{run_id_to_resume}'. Exiting.")
                return
        else:
            logging.info("No previous run found to resume with --resume-latest. Starting a new simulation.")
            # Fall through to new simulation logic

    if not resuming: # New simulation
        if primary_config is None: # Should have been caught, but defensive
            # This case is tricky as logging might not be set up.
            print("CRITICAL: Cannot start a new simulation without a valid primary config file. Exiting.")
            return
        config = primary_config
        simulation_run_id = f"sim_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        log_level = config.get('logging', {}).get('log_level', 'INFO')
        setup_logging(log_level, log_dir_base, simulation_run_id, resume_mode=False) # log_dir_base from primary
        logging.info(f"Starting new simulation run: {simulation_run_id}")
        logging.info(f"Configuration loaded: {config}")

        try:
            current_population = initialize_population(config)
        except Exception as e:
            logging.critical(f"Failed to initialize population: {e}", exc_info=True)
            return
        logging.info(f"Initial population of {len(current_population)} agents created.")
        
        run_dir = Path(state_base_dir) / simulation_run_id # state_base_dir from primary
        run_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = run_dir / "config_snapshot.json"
        try:
            with open(config_snapshot_path, 'w') as f:
                json.dump(config, f, indent=4)
            logging.info(f"Saved initial config snapshot to {config_snapshot_path}")
        except Exception as e:
            logging.error(f"Failed to save initial config snapshot: {e}")

    else: # Resuming simulation
        log_level = config.get('logging', {}).get('log_level', 'INFO') # Config is from checkpoint
        setup_logging(log_level, log_dir_base, simulation_run_id, resume_mode=True) # log_dir_base from checkpoint or primary
        logging.info(f"Resuming simulation run: {simulation_run_id} from generation {start_generation}")
        logging.info(f"Resumed with configuration: {config}")


    # --- Simulation Parameters from definitive config ---
    sim_config_params = config.get('simulation', {})
    num_generations = sim_config_params.get('num_generations', 10)

    state_saving_config = config.get('state_saving', {})
    save_state_enabled = state_saving_config.get('enabled', True)
    save_state_interval = state_saving_config.get('interval', 1)
    # state_base_dir is already determined

    last_successfully_saved_generation = start_generation -1 

    # --- Main Simulation Loop ---
    for gen_num in range(start_generation, num_generations + 1):
        logging.info(f"--- Starting Generation {gen_num}/{num_generations} for run {simulation_run_id} ---")
        generation_game_details = [] 

    try:
        # Call the asynchronous run_game_round function
        population_after_round, generation_game_details = await run_game_round(
            current_population, config, simulation_run_id, gen_num
        )

        if not isinstance(population_after_round, list) or len(population_after_round) != len(current_population):
                 logging.error(f"Population state invalid after game round in Generation {gen_num}. Expected {len(current_population)} agents, got {len(population_after_round)}. Aborting.")
                 break
            logging.info(f"Completed game round for Generation {gen_num}. {len(generation_game_details)} games played.")

            fitness_scores_list = calculate_fitness(population_after_round)
            fitness_scores_map = {}
            if len(population_after_round) == len(fitness_scores_list):
                for agent, score in zip(population_after_round, fitness_scores_list):
                    if hasattr(agent, 'agent_id'): # Ensure agent has an ID for the map
                         agent.current_fitness_score = score # Store on agent instance for easier access
                         fitness_scores_map[agent.agent_id] = score
                    else:
                        logging.warning("Agent found without agent_id during fitness score mapping.")
            else:
                logging.error(f"Mismatch between population size ({len(population_after_round)}) and fitness scores ({len(fitness_scores_list)}) in Gen {gen_num}. Cannot map fitness scores.")
            
            if not isinstance(fitness_scores_list, list) or len(fitness_scores_list) != len(population_after_round):
                 logging.error(f"Fitness scores invalid after calculation in Generation {gen_num}. Expected {len(population_after_round)} scores, got {len(fitness_scores_list)}. Aborting.")
                 break
            logging.info(f"Calculated fitness for Generation {gen_num}.")
            
            # Pass generation_game_details to evolve_population
            next_population = evolve_population(
                population_after_round, 
                fitness_scores_list, 
                config,
                generation_game_details # Pass the list of game detail dicts
            )
            if not isinstance(next_population, list) or len(next_population) != len(population_after_round): # Should maintain pop size
                 logging.error(f"Population state invalid after evolution in Generation {gen_num}. Expected {len(population_after_round)} agents, got {len(next_population)}. Aborting.")
                 break
            logging.info(f"Population evolved for Generation {gen_num}.")

            current_population = next_population
            last_successfully_saved_generation = gen_num # Mark this generation as completed

            # --- State Saving / Checkpointing ---
            if save_state_enabled and (gen_num % save_state_interval == 0 or gen_num == num_generations):
                generation_summary_metrics = {
                    "avg_fitness": (sum(fitness_scores_list) / len(fitness_scores_list)) if fitness_scores_list else 0,
                    "max_fitness": max(fitness_scores_list) if fitness_scores_list else 0,
                    "min_fitness": min(fitness_scores_list) if fitness_scores_list else 0,
                    "avg_wealth": (sum(a.wealth for a in current_population) / len(current_population)) if current_population else 0,
                    "total_games_played_in_generation": len(generation_game_details),
                    "fitness_scores_map": fitness_scores_map 
                }
                save_generation_checkpoint(
                    simulation_run_id=simulation_run_id,
                    config_snapshot=config,
                    generation_number=gen_num,
                    population=current_population,
                    game_details_for_generation=generation_game_details,
                    rng_state=list(random.getstate()),
                    generation_summary_metrics=generation_summary_metrics,
                    state_base_dir=state_base_dir # Use the determined state_base_dir
                )

        except KeyboardInterrupt:
             logging.warning(f"Simulation run {simulation_run_id} interrupted by user during Generation {gen_num}.")
             logging.info(f"Last successfully completed and potentially saved generation was: {last_successfully_saved_generation if last_successfully_saved_generation >= (start_generation if resuming else 1) else 'None this session'}")
             break
        except Exception as e:
            logging.critical(f"Critical error during Generation {gen_num} of run {simulation_run_id}: {e}", exc_info=True)
            break
        finally:
            # This block will execute whether the try block completed normally or due to an exception/break.
            # We can log the effective end generation here.
            # 'gen_num' will hold the value of the generation that was being processed or just finished.
            # If the loop didn't even start (e.g. error in init), gen_num might not be defined.
            current_gen_for_log = locals().get('gen_num', start_generation if resuming else 0)


    # Final log message outside the loop
    final_gen_message_val = current_gen_for_log if 'current_gen_for_log' in locals() else (start_generation -1 if resuming else 0)
    logging.info(f"Simulation run {simulation_run_id} finished or stopped. Last processed generation was {final_gen_message_val}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agent Evolution Simulation.")
    parser.add_argument(
        '--config-file',
        type=str,
        default='config.yaml',
        help='Path to the YAML configuration file.'
    )
    parser.add_argument(
        '--resume-run-id',
        type=str,
        default=None,
        help='Specific simulation_run_id to resume from. Looks in state_saving.directory.'
    )
    parser.add_argument(
        '--resume-latest',
        action='store_true',
        help='Resume the most recently modified simulation run found in state_saving.directory.'
    )
    
    cli_args = parser.parse_args()

    if cli_args.resume_run_id and cli_args.resume_latest:
        # Basic print as logging might not be set up
        print("Error: Cannot use both --resume-run-id and --resume-latest. Choose one.")
    else:
        # Run the main asynchronous simulation function using asyncio.run()
        asyncio.run(run_simulation(cli_args))
