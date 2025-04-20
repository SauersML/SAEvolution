"""
This script serves as the main entry point and conductor for the simulation. 
It loads parameters from the configuration file, establishes the initial 
population of agents, controls the primary loop iterating through generations, 
sequences the execution of game rounds and the subsequent evolutionary updates 
for each generation, and oversees overall simulation logging and state saving.
"""

import yaml
import os
import logging
import datetime
import random
import json
from dotenv import load_dotenv
from manager import Agent, initialize_population, calculate_fitness, evolve_population
from engine import run_game_round

load_dotenv()

def load_config(config_path='config.yaml'):
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
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

def setup_logging(log_level_str, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_str.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configured.")
    logging.info(f"Log file path: {log_filename}")

def save_state(population, generation, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state_filename = os.path.join(save_dir, f"state_generation_{generation}.json")
    state_data = {
        'generation': generation,
        'population': [agent.to_dict() for agent in population] 
    }
    try:
        with open(state_filename, 'w') as f:
            json.dump(state_data, f, indent=4)
        logging.info(f"Saved simulation state to {state_filename}")
    except Exception as e:
        logging.error(f"Failed to save state at generation {generation} to {state_filename}: {e}")

def run_simulation():
    try:
        config = load_config()
    except Exception:
        print("Failed to load configuration. Exiting.") 
        return 

    log_dir = config.get('logging', {}).get('log_directory', 'logs')
    log_level = config.get('logging', {}).get('log_level', 'INFO')
    setup_logging(log_level, log_dir)
    
    logging.info("Starting simulation driver...")
    logging.info(f"Configuration loaded: {config}")

    sim_config = config.get('simulation', {})
    num_generations = sim_config.get('num_generations', 10)
    population_size = sim_config.get('population_size', 10)
    
    state_saving_config = config.get('state_saving', {})
    save_state_enabled = state_saving_config.get('enabled', False)
    save_state_interval = state_saving_config.get('interval', 10)
    save_state_dir = state_saving_config.get('directory', 'simulation_state')

    try:
        current_population = initialize_population(config)
    except Exception as e:
        logging.critical(f"Failed to initialize population: {e}", exc_info=True)
        return 

    if len(current_population) != population_size:
        logging.warning(f"Initialized population size {len(current_population)} differs from configured size {population_size}.")

    logging.info(f"Initial population of {len(current_population)} agents created.")

    for generation in range(1, num_generations + 1):
        logging.info(f"Starting Generation {generation}/{num_generations}")

        try:
            population_after_round = run_game_round(current_population, config)
            
            if not isinstance(population_after_round, list) or len(population_after_round) != len(current_population):
                 logging.error(f"Population state invalid after game round in Generation {generation}. Expected list of size {len(current_population)}, got {type(population_after_round)} of size {len(population_after_round) if isinstance(population_after_round, list) else 'N/A'}. Aborting.")
                 break 
            logging.info(f"Completed game round for Generation {generation}.")

            fitness_scores = calculate_fitness(population_after_round)
            if not isinstance(fitness_scores, list) or len(fitness_scores) != len(population_after_round):
                 logging.error(f"Fitness scores invalid after calculation in Generation {generation}. Expected list of size {len(population_after_round)}, got {type(fitness_scores)} of size {len(fitness_scores) if isinstance(fitness_scores, list) else 'N/A'}. Aborting.")
                 break
            logging.info(f"Calculated fitness for Generation {generation}.")
            
            next_population = evolve_population(population_after_round, fitness_scores, config)
            if not isinstance(next_population, list) or len(next_population) != population_size:
                 logging.error(f"Population state invalid after evolution in Generation {generation}. Expected list of size {population_size}, got {type(next_population)} of size {len(next_population) if isinstance(next_population, list) else 'N/A'}. Aborting.")
                 break 
            logging.info(f"Population evolved for Generation {generation}.")

            current_population = next_population

            if save_state_enabled and (generation % save_state_interval == 0 or generation == num_generations):
                save_state(current_population, generation, save_state_dir)

        except KeyboardInterrupt:
             logging.warning(f"Simulation interrupted by user during Generation {generation}. Exiting.")
             break
        except Exception as e:
            logging.critical(f"Critical error during Generation {generation}: {e}", exc_info=True)
            break 

    logging.info("Simulation run finished.")

if __name__ == "__main__":
    run_simulation()
