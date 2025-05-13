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
from pathlib import Path
from dotenv import load_dotenv
from manager import Agent, initialize_population, calculate_fitness, evolve_population
from engine import run_game_round # Assuming run_game_round will return (updated_population, game_details_list)

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
        if not hasattr(agent, 'current_fitness_score'): # Placeholder for actual attribute name
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

    random.setstate(tuple(state_data['rng_state'])) # RNG state is often a tuple
    
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


def run_simulation(args):
    # --- Initial Config Loading (might be overridden by checkpoint) ---
    primary_config = load_config(args.config_file)
    if primary_config is None and not args.resume_run_id and not args.resume_latest:
        print(f"Error: Main config file '{args.config_file}' not found and not resuming. Exiting.")
        return

    # --- Determine Run Mode and Load State if Resuming ---
    current_population: list[Agent] = []
    start_generation = 1
    simulation_run_id: str = ""
    config: dict = {} # This will hold the definitive config for the run

    state_base_dir_from_config = (primary_config or {}).get('state_saving', {}).get('directory', 'simulation_state')
    log_dir_base_from_config = (primary_config or {}).get('logging', {}).get('log_directory', 'logs')


    resuming = False
    if args.resume_run_id:
        checkpoint_data = load_checkpoint(args.resume_run_id, state_base_dir_from_config)
        if checkpoint_data:
            current_population, start_generation, config, simulation_run_id = checkpoint_data
            resuming = True
        else:
            logging.error(f"Failed to load checkpoint for run ID '{args.resume_run_id}'. Exiting.")
            return
    elif args.resume_latest:
        run_id_to_resume = find_latest_run_id(state_base_dir_from_config)
        if run_id_to_resume:
            checkpoint_data = load_checkpoint(run_id_to_resume, state_base_dir_from_config)
            if checkpoint_data:
                current_population, start_generation, config, simulation_run_id = checkpoint_data
                resuming = True
            else:
                logging.error(f"Failed to load latest checkpoint for run ID '{run_id_to_resume}'. Exiting.")
                return
        else:
            logging.info("No previous run found to resume with --resume-latest. Starting a new simulation.")
            # Fall through to new simulation logic

    if not resuming: # New simulation
        if primary_config is None: # Should have been caught, but defensive
            print("Cannot start a new simulation without a valid config file. Exiting.")
            return
        config = primary_config
        simulation_run_id = f"sim_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        log_level = config.get('logging', {}).get('log_level', 'INFO')
        setup_logging(log_level, log_dir_base_from_config, simulation_run_id, resume_mode=False)
        logging.info(f"Starting new simulation run: {simulation_run_id}")
        logging.info(f"Configuration loaded: {config}")

        try:
            current_population = initialize_population(config)
        except Exception as e:
            logging.critical(f"Failed to initialize population: {e}", exc_info=True)
            return
        logging.info(f"Initial population of {len(current_population)} agents created.")
        
        # Save initial config snapshot for this new run
        run_dir = Path(state_base_dir_from_config) / simulation_run_id
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
        # log_dir_base_from_config may not match what's in loaded config if primary_config was None
        # Safest to use the dir from the loaded config if available, else the one from primary_config
        loaded_log_dir_base = config.get('logging', {}).get('log_directory', log_dir_base_from_config)
        setup_logging(log_level, loaded_log_dir_base, simulation_run_id, resume_mode=True)
        logging.info(f"Resuming simulation run: {simulation_run_id} from generation {start_generation}")
        logging.info(f"Resumed with configuration: {config}")


    # --- Simulation Parameters from definitive config ---
    sim_config_params = config.get('simulation', {})
    num_generations = sim_config_params.get('num_generations', 10)
    # population_size is mainly for initial setup, subsequent gens maintain size via evolution logic

    state_saving_config = config.get('state_saving', {})
    save_state_enabled = state_saving_config.get('enabled', True) # Default to True for dashboarding
    save_state_interval = state_saving_config.get('interval', 1)  # Save every generation for dashboard
    # state_base_dir is already determined (state_base_dir_from_config)

    # --- Main Simulation Loop ---
    for gen_num in range(start_generation, num_generations + 1):
        logging.info(f"--- Starting Generation {gen_num}/{num_generations} for run {simulation_run_id} ---")
        generation_game_details = [] # To store details of all games in this generation

        try:
            # run_game_round now needs to return (updated_population, list_of_game_detail_dicts)
            population_after_round, generation_game_details = run_game_round(current_population, config)
            
            if not isinstance(population_after_round, list) or len(population_after_round) != len(current_population):
                 logging.error(f"Population state invalid after game round in Generation {gen_num}. Aborting.")
                 break
            logging.info(f"Completed game round for Generation {gen_num}. {len(generation_game_details)} games played.")

            fitness_scores_list = calculate_fitness(population_after_round)
            # Store fitness on agents or create a map for saving
            fitness_scores_map = {}
            for agent, score in zip(population_after_round, fitness_scores_list):
                agent.current_fitness_score = score # Assuming Agent class can store this
                fitness_scores_map[agent.agent_id] = score
            
            if not isinstance(fitness_scores_list, list) or len(fitness_scores_list) != len(population_after_round):
                 logging.error(f"Fitness scores invalid after calculation in Generation {gen_num}. Aborting.")
                 break
            logging.info(f"Calculated fitness for Generation {gen_num}.")
            
            next_population = evolve_population(population_after_round, fitness_scores_list, config)
            if not isinstance(next_population, list) or len(next_population) != len(population_after_round):
                 logging.error(f"Population state invalid after evolution in Generation {gen_num}. Aborting.")
                 break
            logging.info(f"Population evolved for Generation {gen_num}.")

            current_population = next_population

            # --- State Saving / Checkpointing ---
            if save_state_enabled and (gen_num % save_state_interval == 0 or gen_num == num_generations):
                generation_summary_metrics = {
                    "avg_fitness": sum(fitness_scores_list) / len(fitness_scores_list) if fitness_scores_list else 0,
                    "max_fitness": max(fitness_scores_list) if fitness_scores_list else 0,
                    "min_fitness": min(fitness_scores_list) if fitness_scores_list else 0,
                    "avg_wealth": sum(a.wealth for a in current_population) / len(current_population) if current_population else 0,
                    "total_games_played_in_generation": len(generation_game_details),
                    "fitness_scores_map": fitness_scores_map # For dashboard to easily get per-agent fitness
                }
                save_generation_checkpoint(
                    simulation_run_id=simulation_run_id,
                    config_snapshot=config, # The definitive config for this run
                    generation_number=gen_num,
                    population=current_population,
                    game_details_for_generation=generation_game_details,
                    rng_state=list(random.getstate()), # Convert tuple to list for JSON
                    generation_summary_metrics=generation_summary_metrics,
                    state_base_dir=state_base_dir_from_config
                )

        except KeyboardInterrupt:
             logging.warning(f"Simulation run {simulation_run_id} interrupted by user during Generation {gen_num}.")
             logging.info("Attempting to save state for the last *completed* generation if applicable...")
             # The save_generation_checkpoint is already at the end of a successful generation loop.
             # If interrupt happens before that, it relies on the *previous* generation's save.
             # If it happens *after* save but *before* next loop, current gen is saved.
             # This behavior is generally fine.
             break
        except Exception as e:
            logging.critical(f"Critical error during Generation {gen_num} of run {simulation_run_id}: {e}", exc_info=True)
            break

    logging.info(f"Simulation run {simulation_run_id} finished or stopped at generation {gen_num if 'gen_num' in locals() else start_generation-1}.")


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
        print("Error: Cannot use both --resume-run-id and --resume-latest. Choose one.")
    else:
        run_simulation(cli_args)
