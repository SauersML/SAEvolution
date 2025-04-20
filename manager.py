"""
This module defines the data structure for an individual agent, encompassing 
its behavioral genome settings and accumulated wealth. It manages the group 
of agents and implements the core evolutionary mechanics: calculating relative 
success scores from wealth, determining which agents reproduce based on these 
scores, and algorithmically modifying the genomes of offspring by automatically 
adjusting features based on a contrastive analysis of the parent's performance 
in recent successful versus unsuccessful interactions.
"""

import random
import logging
import uuid
import math
import copy
from interface import perform_contrastive_analysis

class Agent:
    def __init__(self, agent_id, model_id, initial_genome=None, initial_wealth=30.0):
        self.agent_id = agent_id
        self.model_id = model_id
        if initial_genome is None:
            self.genome = {}
        else:
            self.genome = initial_genome
        self.wealth = float(initial_wealth)
        self.round_history = []

    def reset_round_state(self, initial_wealth=30.0):
        self.wealth = float(initial_wealth)
        self.round_history = []

    def add_game_result(self, result_data):
        if isinstance(result_data, dict):
            self.round_history.append(result_data)
        else:
            logging.warning(f"Agent {self.agent_id}: Attempted to add invalid game result data type: {type(result_data)}")

    def get_transcripts_by_outcome(self):
        winning_transcripts = []
        losing_transcripts = []
        for result in self.round_history:
            outcome = result.get('outcome')
            transcript = result.get('transcript')
            if isinstance(transcript, list):
                if outcome == 'win':
                    winning_transcripts.append(transcript)
                elif outcome == 'loss':
                    losing_transcripts.append(transcript)
        return winning_transcripts, losing_transcripts

    def to_dict(self):
        return {
            'agent_id': self.agent_id,
            'model_id': self.model_id,
            'genome': self.genome,
            'wealth': self.wealth,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError("Data provided to Agent.from_dict must be a dictionary.")

        agent_id = data.get('agent_id', str(uuid.uuid4()))
        model_id = data.get('model_id')
        if not model_id:
            raise ValueError("model_id is required when creating Agent from dict.")

        genome = data.get('genome', {})
        if not isinstance(genome, dict):
            logging.warning(f"Agent {agent_id}: Invalid genome type in from_dict data. Defaulting to empty genome.")
            genome = {}

        wealth = data.get('wealth', 30.0)
        try:
            wealth = float(wealth)
        except (ValueError, TypeError):
            logging.warning(f"Agent {agent_id}: Invalid wealth type in from_dict data. Defaulting to 30.0.")
            wealth = 30.0

        agent = cls(
            agent_id=agent_id,
            model_id=model_id,
            initial_genome=genome,
            initial_wealth=wealth
        )
        return agent

def initialize_population(config):
    population = []
    if not isinstance(config, dict):
        logging.error("Invalid configuration provided to initialize_population.")
        raise TypeError("Configuration must be a dictionary.")

    sim_config = config.get('simulation', {})
    agent_config = config.get('agent', {})

    pop_size = sim_config.get('population_size', 10)
    model_id = agent_config.get('model_id')
    initial_wealth = agent_config.get('initial_wealth', 30.0)

    if not model_id:
        logging.error("Agent model_id not specified in configuration.")
        raise ValueError("Agent model_id must be specified in configuration.")

    initial_genome_config = agent_config.get('initial_genome', {})
    if not isinstance(initial_genome_config, dict):
        logging.warning("Initial genome configuration is not a dictionary. Defaulting to empty genome for all agents.")
        initial_genome_config = {}

    for _ in range(pop_size):
        agent_id = str(uuid.uuid4())
        agent_genome = copy.deepcopy(initial_genome_config)
        try:
            agent = Agent(agent_id, model_id, agent_genome, initial_wealth)
            population.append(agent)
            logging.debug(f"Initialized Agent {agent_id} with model {model_id}")
        except Exception as e:
            logging.error(f"Failed to initialize agent {agent_id}: {e}", exc_info=True)
            raise RuntimeError(f"Agent initialization failed for agent {agent_id}") from e

    return population

def calculate_fitness(population):
    if not isinstance(population, list):
        logging.error("Invalid population data type provided to calculate_fitness.")
        return []

    if not population:
        return []

    num_agents = len(population)
    valid_agents = [agent for agent in population if hasattr(agent, 'wealth') and isinstance(agent.wealth, (int, float))]
    if len(valid_agents) != num_agents:
        logging.warning("Some agents in population lack valid wealth attribute.")
    
    total_wealth = sum(agent.wealth for agent in valid_agents)
    fitness_scores = []

    if total_wealth <= 0 or num_agents == 0:
        logging.warning("Total population wealth is zero or negative, or population is empty. Assigning equal fitness.")
        equal_fitness = 1.0 / num_agents if num_agents > 0 else 0
        fitness_scores = [equal_fitness] * num_agents
    else:
        for agent in population:
            agent_wealth = getattr(agent, 'wealth', 0.0)
            if not isinstance(agent_wealth, (int, float)):
                agent_wealth = 0.0
            fitness = max(0.0, agent_wealth / total_wealth)
            fitness_scores.append(fitness)

    sum_fitness = sum(fitness_scores)
    if math.isclose(sum_fitness, 0.0) and num_agents > 0:
        equal_fitness = 1.0 / num_agents
        normalized_scores = [equal_fitness] * num_agents
        logging.debug("Calculated fitness resulted in zero sum, assigning equal normalized scores.")
    elif sum_fitness > 0:
        try:
            normalized_scores = [score / sum_fitness for score in fitness_scores]
        except ZeroDivisionError:
            logging.error("Zero division error during fitness normalization.")
            equal_fitness = 1.0 / num_agents if num_agents > 0 else 0
            normalized_scores = [equal_fitness] * num_agents
    else:
        normalized_scores = [0.0] * num_agents

    if normalized_scores and not math.isclose(sum(normalized_scores), 1.0, rel_tol=1e-9):
        logging.warning(f"Normalized fitness scores do not sum precisely to 1.0 (Sum: {sum(normalized_scores)}). Check calculation.")

    return normalized_scores

def select_parents(population, fitness_scores, num_offspring):
    if not population or len(population) != len(fitness_scores):
        logging.error("Population and fitness scores mismatch or empty population in select_parents.")
        return []

    population_to_select_from = population
    weights_to_use = fitness_scores

    sum_weights = sum(weights_to_use)

    if math.isclose(sum_weights, 0.0, rel_tol=1e-9) or sum_weights < 0:
        logging.warning("Total fitness is zero or negative for selection. Using uniform random selection.")
        weights_to_use = None

    if num_offspring <= 0:
        return []

    try:
        selected_parents = random.choices(
            population=population_to_select_from,
            weights=weights_to_use,
            k=num_offspring
        )
        return selected_parents
    except ValueError as e:
        logging.error(f"Error during parent selection with random.choices: {e}. Returning empty list.")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during parent selection: {e}", exc_info=True)
        return []

def apply_algorithmic_genome_update(parent_genome, winning_features, losing_features, config):
    if not isinstance(config, dict):
         logging.error("Invalid configuration provided to apply_algorithmic_genome_update.")
         return parent_genome # Return original genome on config error

    evo_config = config.get('evolution', {})
    learning_rate = evo_config.get('learning_rate', 0.05)
    num_winning_to_use = evo_config.get('num_winning_features', 3)
    num_losing_to_use = evo_config.get('num_losing_features', 3)
    min_activation = evo_config.get('activation_min', -1.0)
    max_activation = evo_config.get('activation_max', 1.0)
    target_pos = evo_config.get('target_positive', 1.0)
    target_neg = evo_config.get('target_negative', -0.5)
    update_method = evo_config.get('update_method', 'increment')

    offspring_genome = copy.deepcopy(parent_genome)
    updated_winning_keys = set()

    if isinstance(winning_features, list):
        for i, feature in enumerate(winning_features):
            if i >= num_winning_to_use:
                break
            if hasattr(feature, 'uuid'):
                feature_key = str(feature.uuid)
                current_value = offspring_genome.get(feature_key, 0.0)

                if update_method == 'target':
                    new_value = current_value + learning_rate * (target_pos - current_value)
                else:
                    new_value = current_value + learning_rate

                new_value = max(min_activation, min(max_activation, float(new_value)))
                offspring_genome[feature_key] = new_value
                updated_winning_keys.add(feature_key)
                logging.debug(f"Genome update: Increasing winning feature {feature_key} to {new_value:.4f}")
            else:
                 logging.warning("Winning feature object lacks 'uuid' attribute during genome update.")

    if isinstance(losing_features, list):
        for i, feature in enumerate(losing_features):
            if i >= num_losing_to_use:
                break
            if hasattr(feature, 'uuid'):
                feature_key = str(feature.uuid)

                if feature_key in updated_winning_keys:
                    logging.debug(f"Skipping update for losing feature {feature_key} as it was already updated as winning.")
                    continue

                current_value = offspring_genome.get(feature_key, 0.0)

                if update_method == 'target':
                    new_value = current_value + learning_rate * (target_neg - current_value)
                else:
                    new_value = current_value - learning_rate

                new_value = max(min_activation, min(max_activation, float(new_value)))
                offspring_genome[feature_key] = new_value
                logging.debug(f"Genome update: Decreasing losing feature {feature_key} to {new_value:.4f}")
            else:
                 logging.warning("Losing feature object lacks 'uuid' attribute during genome update.")

    return offspring_genome

def evolve_population(population, fitness_scores, config):
    if not population:
        logging.warning("Cannot evolve empty population.")
        return []
        
    new_population = []
    population_size = len(population)

    if not isinstance(config, dict):
         logging.error("Invalid configuration provided to evolve_population.")
         return population # Return original population on config error

    evo_config = config.get('evolution', {})
    contrast_top_k = evo_config.get('contrast_top_k', 10)

    agent_config = config.get('agent', {})
    agent_model_id = agent_config.get('model_id')
    initial_wealth = agent_config.get('initial_wealth', 30.0)

    if not agent_model_id:
         logging.error("Agent model ID missing in config during evolution.")
         raise ValueError("Agent model_id must be specified in configuration for evolution.")

    selected_parents = select_parents(population, fitness_scores, population_size)
    if not selected_parents:
         logging.error("Parent selection failed. Returning original population.")
         return population

    num_offspring_created = 0
    for parent_agent in selected_parents:
        winning_transcripts, losing_transcripts = parent_agent.get_transcripts_by_outcome()
        
        offspring_genome = parent_agent.genome

        if winning_transcripts and losing_transcripts:
            try:
                winning_features, losing_features = perform_contrastive_analysis(
                    losing_transcripts,
                    winning_transcripts,
                    parent_agent.model_id,
                    contrast_top_k,
                    config
                )

                if winning_features is not None and losing_features is not None:
                    offspring_genome = apply_algorithmic_genome_update(
                        parent_agent.genome,
                        winning_features,
                        losing_features,
                        config
                    )
                else:
                    logging.info(f"Contrastive analysis returned None results for parent {parent_agent.agent_id}. Skipping genome update.")

            except ImportError:
                logging.error("Could not import perform_contrastive_analysis from interface module.")
                raise
            except Exception as e:
                logging.error(f"Error during contrastive analysis or genome update for parent {parent_agent.agent_id}: {e}", exc_info=True)

        else:
            logging.info(f"Parent {parent_agent.agent_id} lacked sufficient distinct outcome transcripts ({len(winning_transcripts)} win, {len(losing_transcripts)} loss) for contrast. Inheriting genome without update.")

        offspring_agent_id = str(uuid.uuid4())
        try:
            offspring = Agent(
                agent_id=offspring_agent_id,
                model_id=agent_model_id,
                initial_genome=offspring_genome,
                initial_wealth=initial_wealth
            )
            offspring.reset_round_state(initial_wealth)
            new_population.append(offspring)
            num_offspring_created += 1
        except Exception as e:
             logging.error(f"Failed to create offspring agent {offspring_agent_id} from parent {parent_agent.agent_id}: {e}", exc_info=True)
             logging.warning(f"Skipping offspring creation for parent {parent_agent.agent_id}")

    if num_offspring_created != population_size:
         logging.error(f"Evolution resulted in population size {num_offspring_created}, expected {population_size}. Returning partially created population.")

    return new_population
