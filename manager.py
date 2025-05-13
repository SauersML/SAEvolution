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
from interface import perform_contrastive_analysis # Assuming this is correctly implemented in interface.py

class Agent:
    """
    Represents an individual agent in the simulation.
    """
    def __init__(self, agent_id: str, model_id: str, initial_genome: dict | None = None, initial_wealth: float = 30.0):
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string.")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string.")
        if initial_genome is not None and not isinstance(initial_genome, dict):
            raise TypeError("initial_genome must be a dictionary or None.")
        if not isinstance(initial_wealth, (int, float)):
            raise TypeError("initial_wealth must be a float or int.")

        self.agent_id: str = agent_id
        self.model_id: str = model_id
        self.genome: dict = copy.deepcopy(initial_genome) if initial_genome is not None else {}
        self.wealth: float = float(initial_wealth)
        self.round_history: list[dict] = [] # List of game result dictionaries

    def reset_round_state(self, initial_wealth: float = 30.0) -> None:
        """
        Resets the agent's wealth and round history for a new round.
        """
        if not isinstance(initial_wealth, (int, float)):
            logging.warning(f"Agent {self.agent_id}: Invalid initial_wealth type for reset. Defaulting to 30.0.")
            self.wealth = 30.0
        else:
            self.wealth = float(initial_wealth)
        self.round_history = []
        logging.debug(f"Agent {self.agent_id} round state reset. Wealth: {self.wealth:.2f}")

    def add_game_result(self, result_data: dict) -> None:
        """
        Adds a game result to the agent's round history.
        """
        if not isinstance(result_data, dict):
            logging.warning(f"Agent {self.agent_id}: Attempted to add invalid game result data (type: {type(result_data)}). Expected dict.")
            return
        self.round_history.append(result_data)

    def get_transcripts_by_outcome(self) -> tuple[list[list[dict]], list[list[dict]]]:
        """
        Separates transcripts from the round history into winning and losing lists.
        Each transcript is a list of message dictionaries (e.g., [{"role": "Role A", "content": "..."}]).
        Returns:
            A tuple (winning_transcripts_list, losing_transcripts_list).
        """
        winning_transcripts_list: list[list[dict]] = []
        losing_transcripts_list: list[list[dict]] = []

        for result in self.round_history:
            outcome = result.get('outcome') # Should be 'win', 'loss', 'tie', 'error'
            transcript = result.get('transcript') # Should be a list of dicts

            if isinstance(transcript, list) and transcript: # transcript is a non-empty list
                if outcome == 'win':
                    winning_transcripts_list.append(transcript)
                elif outcome == 'loss':
                    losing_transcripts_list.append(transcript)
            elif transcript is not None: # Log if transcript exists but is not a valid list
                logging.warning(f"Agent {self.agent_id}: Game result has invalid transcript type ({type(transcript)}) or empty transcript for outcome '{outcome}'.")

        return winning_transcripts_list, losing_transcripts_list

    def to_dict(self) -> dict:
        """
        Serializes the agent's state to a dictionary.
        """
        return {
            'agent_id': self.agent_id,
            'model_id': self.model_id,
            'genome': copy.deepcopy(self.genome), # a copy is returned
            'wealth': self.wealth,
            # 'round_history': self.round_history
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Agent':
        """
        Creates an Agent instance from a dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError("Data for Agent.from_dict must be a dictionary.")

        agent_id = data.get('agent_id')
        model_id = data.get('model_id')
        initial_genome = data.get('genome')
        initial_wealth = data.get('wealth')

        if not agent_id or not isinstance(agent_id, str):
            logging.warning("Missing or invalid 'agent_id' in agent data, generating a new UUID.")
            agent_id = str(uuid.uuid4())
        if not model_id or not isinstance(model_id, str):
            raise ValueError("Missing or invalid 'model_id' (string) in agent data.")
        
        if initial_genome is None: # Explicitly allow None to mean empty genome
            initial_genome = {}
        elif not isinstance(initial_genome, dict):
            logging.warning(f"Agent {agent_id}: Invalid 'genome' type in data (expected dict, got {type(initial_genome)}). Defaulting to empty genome.")
            initial_genome = {}
        
        if initial_wealth is None or not isinstance(initial_wealth, (int, float)):
            logging.warning(f"Agent {agent_id}: Missing or invalid 'wealth' in data (expected float/int, got {type(initial_wealth)}). Defaulting to 30.0.")
            initial_wealth = 30.0

        return cls(
            agent_id=agent_id,
            model_id=model_id,
            initial_genome=initial_genome,
            initial_wealth=float(initial_wealth)
        )

def initialize_population(config: dict) -> list[Agent]:
    """
    Initializes a population of agents based on the simulation configuration.
    """
    if not isinstance(config, dict):
        logging.error("Invalid configuration provided to initialize_population (must be dict).")
        raise TypeError("Configuration must be a dictionary.")

    sim_config = config.get('simulation', {})
    agent_config = config.get('agent', {})

    pop_size = sim_config.get('population_size')
    model_id = agent_config.get('model_id')
    initial_wealth = agent_config.get('initial_wealth', 30.0) # Default if not specified

    if not isinstance(pop_size, int) or pop_size <= 0:
        raise ValueError("simulation.population_size must be a positive integer.")
    if not model_id or not isinstance(model_id, str):
        raise ValueError("agent.model_id must be a non-empty string.")
    if not isinstance(initial_wealth, (int, float)):
        logging.warning(f"agent.initial_wealth is not a number ({initial_wealth}). Defaulting to 30.0.")
        initial_wealth = 30.0
    
    initial_genome_template = agent_config.get('initial_genome', {})
    if not isinstance(initial_genome_template, dict):
        logging.warning(f"agent.initial_genome is not a dictionary ({initial_genome_template}). Defaulting to empty genome for all agents.")
        initial_genome_template = {}

    population: list[Agent] = []
    for i in range(pop_size):
        agent_id = str(uuid.uuid4())
        # Each agent gets a deep copy of the initial genome template
        agent_genome = copy.deepcopy(initial_genome_template) 
        try:
            agent = Agent(agent_id, model_id, agent_genome, float(initial_wealth))
            population.append(agent)
            logging.debug(f"Initialized Agent {agent_id} (model: {model_id}, wealth: {initial_wealth:.2f}, genome keys: {list(agent_genome.keys())})")
        except Exception as e:
            logging.critical(f"Failed to initialize agent {i+1}/{pop_size} (ID: {agent_id}): {e}", exc_info=True)
            # Decide on error handling: re-raise, or skip agent, or use defaults?
            raise RuntimeError(f"Agent initialization failed for agent {agent_id}") from e
    
    logging.info(f"Initialized population of {len(population)} agents.")
    return population

def calculate_fitness(population: list[Agent]) -> list[float]:
    """
    Calculates fitness scores for each agent in the population, typically based on wealth.
    Fitness scores are normalized to sum to 1.0.
    """
    if not isinstance(population, list):
        logging.error("Invalid population (must be list) provided to calculate_fitness.")
        return []
    if not all(isinstance(agent, Agent) for agent in population):
        logging.error("Population list contains non-Agent elements in calculate_fitness.")
        return [] # Or handle by filtering

    if not population:
        logging.warning("calculate_fitness called with empty population.")
        return []

    num_agents = len(population)
    agent_wealths = []
    for agent in population:
        if hasattr(agent, 'wealth') and isinstance(agent.wealth, (int, float)):
            agent_wealths.append(max(0.0, agent.wealth)) # Fitness cannot be negative
        else:
            logging.warning(f"Agent {getattr(agent, 'agent_id', 'Unknown')} missing valid wealth attribute. Assigning 0 wealth for fitness.")
            agent_wealths.append(0.0)
    
    total_wealth = sum(agent_wealths)
    fitness_scores: list[float] = []

    if total_wealth <= 0: # Handles all-zero or negative total wealth
        logging.warning("Total population wealth is zero or negative. Assigning equal (potentially zero) fitness.")
        # If num_agents > 0, assign 1/N, otherwise fitness is 0 per agent (empty list handled).
        equal_fitness = (1.0 / num_agents) if num_agents > 0 else 0.0
        fitness_scores = [equal_fitness] * num_agents
    else:
        for wealth in agent_wealths:
            fitness_scores.append(wealth / total_wealth)

    # scores sum to 1.0 (or close enough) if there was positive total wealth
    # If all fitness_scores are 0 (e.g., total_wealth was 0), sum_fitness will be 0.
    # In such a case, normalized_scores will also be all 0 or 1/N if we re-normalized.
    # The current logic already handles this by assigning 1/N above if total_wealth <=0.
    sum_fitness = sum(fitness_scores)
    if not math.isclose(sum_fitness, 1.0, rel_tol=1e-9) and sum_fitness > 1e-9: # Check only if sum isn't effectively zero
        logging.warning(f"Raw fitness scores sum to {sum_fitness}, not 1.0. This might indicate an issue if total_wealth was positive.")
        # If scores are all zero, they will remain so.
    
    logging.debug(f"Calculated fitness scores: {['{:.4f}'.format(s) for s in fitness_scores]}")
    return fitness_scores


def select_parents(population: list[Agent], fitness_scores: list[float], num_offspring: int) -> list[Agent]:
    """
    Selects parent agents from the population based on their fitness scores.
    Uses roulette wheel selection (random.choices).
    """
    if not isinstance(population, list) or not isinstance(fitness_scores, list) or not isinstance(num_offspring, int):
        logging.error("Invalid arguments to select_parents.")
        return []
    if len(population) != len(fitness_scores):
        logging.error("Population size and fitness_scores length mismatch in select_parents.")
        return []
    if not population:
        logging.warning("select_parents called with empty population.")
        return []
    if num_offspring <= 0:
        logging.warning("select_parents called with num_offspring <= 0.")
        return []

    # random.choices requires positive weights if any are provided.
    # If all fitness_scores are zero (or negative, though fitness is floored at 0),
    # random.choices will raise a ValueError if weights sum to zero.
    # In such a case, fall back to uniform random selection without weights.
    
    valid_weights_present = any(f > 1e-9 for f in fitness_scores) # Check if any fitness is positive
    
    if not valid_weights_present:
        logging.warning("All fitness scores are zero or non-positive. Performing uniform random parent selection.")
        selected_parents = random.choices(population, k=num_offspring)
    else:
        try:
            selected_parents = random.choices(
                population=population,
                weights=fitness_scores, # These are already normalized
                k=num_offspring
            )
        except ValueError as e:
            logging.error(f"ValueError during parent selection (likely due to weights): {e}. Falling back to uniform selection.")
            selected_parents = random.choices(population, k=num_offspring)
        except Exception as e: # Catch any other unexpected errors
            logging.error(f"Unexpected error during parent selection: {e}", exc_info=True)
            return [] # Return empty if selection fails critically

    logging.info(f"Selected {len(selected_parents)} parents for reproduction.")
    return selected_parents

def apply_algorithmic_genome_update(
    parent_genome: dict,
    features_correlated_with_winning: list, # List of Feature objects from Goodfire
    features_correlated_with_losing: list,  # List of Feature objects from Goodfire
    config: dict
) -> dict:
    """
    Applies algorithmic updates to a parent's genome based on features
    correlated with winning and losing.
    """
    if not isinstance(parent_genome, dict):
        logging.error("apply_algorithmic_genome_update: parent_genome must be a dict.")
        return {}
    if not isinstance(features_correlated_with_winning, list) or \
       not isinstance(features_correlated_with_losing, list):
        logging.error("apply_algorithmic_genome_update: Winning/losing features must be lists.")
        return copy.deepcopy(parent_genome) # Return original on error
    if not isinstance(config, dict):
         logging.error("Invalid configuration provided to apply_algorithmic_genome_update.")
         return copy.deepcopy(parent_genome)

    evo_config = config.get('evolution', {})
    learning_rate = float(evo_config.get('learning_rate', 0.05))
    num_winning_to_use = int(evo_config.get('num_winning_features', 3))
    num_losing_to_use = int(evo_config.get('num_losing_features', 3))
    min_activation = float(evo_config.get('activation_min', -1.0))
    max_activation = float(evo_config.get('activation_max', 1.0))
    target_pos = float(evo_config.get('target_positive', 1.0))
    target_neg = float(evo_config.get('target_negative', -0.5))
    update_method = evo_config.get('update_method', 'increment') # 'increment' or 'target'

    offspring_genome = copy.deepcopy(parent_genome)
    updated_feature_keys_in_this_step = set() # To avoid conflicting updates if a feature is in both lists

    # Process features correlated with winning (reinforce them)
    if features_correlated_with_winning:
        for i, feature_obj in enumerate(features_correlated_with_winning):
            if i >= num_winning_to_use:
                break
            if hasattr(feature_obj, 'uuid') and isinstance(feature_obj.uuid, (str, uuid.UUID)): # Goodfire Feature objects have UUIDs
                feature_key = str(feature_obj.uuid)
                current_value = float(offspring_genome.get(feature_key, 0.0)) # Default to 0.0 if feature not in genome

                if update_method == 'target':
                    new_value = current_value + learning_rate * (target_pos - current_value)
                elif update_method == 'increment':
                    new_value = current_value + learning_rate
                else:
                    logging.warning(f"Unknown update_method '{update_method}'. Defaulting to 'increment'.")
                    new_value = current_value + learning_rate
                
                new_value = max(min_activation, min(max_activation, new_value)) # Clamp
                offspring_genome[feature_key] = new_value
                updated_feature_keys_in_this_step.add(feature_key)
                logging.debug(f"Genome update (winning correlation): Feature {feature_key} from {current_value:.4f} to {new_value:.4f}")
            else:
                 logging.warning(f"Winning feature object (index {i}) lacks 'uuid' attribute or uuid is not string/UUID. Skipping update for this feature.")

    # Process features correlated with losing (suppress them)
    if features_correlated_with_losing:
        for i, feature_obj in enumerate(features_correlated_with_losing):
            if i >= num_losing_to_use:
                break
            if hasattr(feature_obj, 'uuid') and isinstance(feature_obj.uuid, (str, uuid.UUID)):
                feature_key = str(feature_obj.uuid)

                if feature_key in updated_feature_keys_in_this_step:
                    logging.debug(f"Skipping update for losing-correlated feature {feature_key} as it was already updated (likely as winning-correlated).")
                    continue

                current_value = float(offspring_genome.get(feature_key, 0.0))

                if update_method == 'target':
                    new_value = current_value + learning_rate * (target_neg - current_value)
                elif update_method == 'increment': # 'increment' implies decrement for losing features
                    new_value = current_value - learning_rate
                else:
                    logging.warning(f"Unknown update_method '{update_method}'. Defaulting to 'increment' (decrement for losing).")
                    new_value = current_value - learning_rate

                new_value = max(min_activation, min(max_activation, new_value)) # Clamp
                offspring_genome[feature_key] = new_value
                # updated_feature_keys_in_this_step.add(feature_key) # Not strictly necessary to add here again
                logging.debug(f"Genome update (losing correlation): Feature {feature_key} from {current_value:.4f} to {new_value:.4f}")
            else:
                 logging.warning(f"Losing feature object (index {i}) lacks 'uuid' attribute or uuid is not string/UUID. Skipping update for this feature.")
    
    return offspring_genome

def evolve_population(population: list[Agent], fitness_scores: list[float], config: dict) -> list[Agent]:
    """
    Evolves the population to create a new generation of agents.
    This involves parent selection and genome modification for offspring.
    """
    if not isinstance(population, list) or not isinstance(fitness_scores, list) or not isinstance(config, dict):
        logging.error("evolve_population: Invalid arguments.")
        return population # Return original if critical args are wrong
    if not population:
        logging.warning("evolve_population: Cannot evolve an empty population.")
        return []
    if len(population) != len(fitness_scores):
        logging.error("evolve_population: Population size and fitness_scores length mismatch.")
        return population # Return original

    population_size = len(population) # Maintain constant population size
    new_population: list[Agent] = []

    evo_config = config.get('evolution', {})
    contrast_top_k = int(evo_config.get('contrast_top_k', 10))

    agent_config = config.get('agent', {})
    agent_model_id = agent_config.get('model_id')
    initial_wealth = float(agent_config.get('initial_wealth', 30.0))

    if not agent_model_id:
         logging.critical("Agent model_id missing in config during evolution. Cannot create offspring.")
         raise ValueError("Agent model_id must be specified in configuration for evolution.")

    # Select parents for the new generation
    selected_parents = select_parents(population, fitness_scores, population_size)
    if not selected_parents or len(selected_parents) != population_size :
         logging.error(f"Parent selection failed or returned incorrect number of parents ({len(selected_parents)} instead of {population_size}). Aborting evolution for this generation.")
         # Fallback: return a population of clones of the original population or just the original one
         # For safety, let's return clones of the original to avoid modifying it in place if it was accidentally passed.
         cloned_population = [Agent.from_dict(p.to_dict()) for p in population]
         for agent in cloned_population: agent.reset_round_state(initial_wealth) # Reset for next round
         return cloned_population


    num_offspring_created = 0
    for parent_agent in selected_parents:
        winning_transcripts, losing_transcripts = parent_agent.get_transcripts_by_outcome()
        
        logging.debug(f"Parent {parent_agent.agent_id} has {len(winning_transcripts)} winning and {len(losing_transcripts)} losing transcripts for contrastive analysis.")
        
        # Start with a copy of the parent's genome
        offspring_genome = copy.deepcopy(parent_agent.genome)

        # Perform contrastive analysis only if there's data for both outcomes
        if winning_transcripts and losing_transcripts:
            try:
                # CRITICAL FIX: Correctly interpret the output of perform_contrastive_analysis
                # perform_contrastive_analysis(dataset_1_to_avoid, dataset_2_to_encourage, ...)
                # returns (features_explaining_dataset_1, features_explaining_dataset_2)
                
                features_from_losing_games, features_from_winning_games = perform_contrastive_analysis(
                    dataset_1=losing_transcripts,    # Behavior to avoid (associated with loss)
                    dataset_2=winning_transcripts,  # Behavior to encourage (associated with win)
                    agent_variant_or_model_id=parent_agent, # Pass parent agent to use its genome/model
                    top_k=contrast_top_k,
                    config=config
                )

                if features_from_winning_games is not None and features_from_losing_games is not None:
                    logging.debug(f"Contrastive analysis for parent {parent_agent.agent_id}: "
                                  f"{len(features_from_winning_games)} win-correlated features, "
                                  f"{len(features_from_losing_games)} lose-correlated features.")
                    offspring_genome = apply_algorithmic_genome_update(
                        parent_genome=offspring_genome, # Can be parent_agent.genome or current offspring_genome
                        features_correlated_with_winning=features_from_winning_games, # Correctly passed
                        features_correlated_with_losing=features_from_losing_games,   # Correctly passed
                        config=config
                    )
                else:
                    logging.info(f"Contrastive analysis returned None for one or both feature sets for parent {parent_agent.agent_id}. Offspring inherits genome without update from contrast.")
            except ImportError: # Should not happen if imports are correct
                logging.critical("ImportError: perform_contrastive_analysis not found. Check interface.py.", exc_info=True)
                raise # This is a structural problem
            except Exception as e:
                logging.error(f"Error during contrastive analysis or genome update for offspring of parent {parent_agent.agent_id}: {e}", exc_info=True)
                # Offspring will inherit the (potentially partially modified) genome as is.
        else:
            logging.info(f"Parent {parent_agent.agent_id} lacked sufficient distinct outcome transcripts "
                         f"({len(winning_transcripts)} win, {len(losing_transcripts)} loss) for contrastive analysis. "
                         "Offspring inherits genome without update from contrast.")

        # Create the new offspring agent
        offspring_agent_id = str(uuid.uuid4())
        try:
            offspring = Agent(
                agent_id=offspring_agent_id,
                model_id=agent_model_id, # All offspring use the configured base model ID
                initial_genome=offspring_genome,
                initial_wealth=initial_wealth # All offspring start with the standard initial wealth
            )
            # offspring.reset_round_state(initial_wealth) # Agent.__init__ already sets wealth, history. Redundant here.
            new_population.append(offspring)
            num_offspring_created += 1
            logging.debug(f"Created offspring {offspring_agent_id} from parent {parent_agent.agent_id}.")
        except Exception as e:
             logging.error(f"Failed to create offspring agent {offspring_agent_id} from parent {parent_agent.agent_id}: {e}", exc_info=True)

    if num_offspring_created != population_size:
         # This could happen if Agent creation fails above, or parent selection returned fewer.
         logging.warning(f"Evolution resulted in population size {num_offspring_created}, but expected {population_size}. "
                       "This might be due to errors in offspring creation or parent selection issues.")
         # If strictly maintaining population size is critical, might need to add random new agents or re-clone.

    logging.info(f"Evolution complete. New population size: {len(new_population)}.")
    return new_population
