"""
This module defines the data structure for an individual agent, encompassing
its behavioral genome settings and accumulated wealth. It manages the group
of agents and implements the core evolutionary mechanics: calculating relative
success scores from wealth, determining which agents reproduce based on these
scores, and algorithmically modifying the genomes of offspring by automatically
adjusting features based on inspecting active features from the parent's
performance in the most recent game.
"""

import random
import logging
import uuid
import math
import copy
import re
import goodfire # For goodfire.Variant and goodfire.Feature
# ContextInspector will be referred to by string literal in type hints
# as it's not directly importable from 'goodfire' or 'goodfire.features'
# in the current SDK
from interface import get_goodfire_async_client, get_goodfire_client 

class Agent:
    """
    Represents an individual agent in the simulation.
    """
    def __init__(self,
                 agent_id: str,
                 model_id: str,
                 initial_genome: dict | None = None,
                 initial_wealth: float = 30.0,
                 parent_id: str | None = None,
                 evolutionary_input_positive_features: list[str] | None = None,
                 evolutionary_input_negative_features: list[str] | None = None
                 ):
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string.")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string.")
        if initial_genome is not None and not isinstance(initial_genome, dict):
            raise TypeError("initial_genome must be a dictionary or None.")
        if not isinstance(initial_wealth, (int, float)):
            raise TypeError("initial_wealth must be a float or int.")
        if parent_id is not None and not isinstance(parent_id, str):
            raise TypeError("parent_id must be a string or None.")
        if evolutionary_input_positive_features is not None and not isinstance(evolutionary_input_positive_features, list):
            raise TypeError("evolutionary_input_positive_features must be a list or None.")
        if evolutionary_input_negative_features is not None and not isinstance(evolutionary_input_negative_features, list):
            raise TypeError("evolutionary_input_negative_features must be a list or None.")

        self.agent_id: str = agent_id
        self.model_id: str = model_id
        # Genome structure: dict[str_uuid, {'activation': float, 'label': str, 'index_in_sae': int | None}]
        # 'index_in_sae' is crucial for creating goodfire.Feature objects if needed for variant.set
        # It might be None if the feature was added to genome before this field was tracked or if lookup failed.
        self.genome: dict = copy.deepcopy(initial_genome) if initial_genome is not None else {}
        self.wealth: float = float(initial_wealth)
        self.round_history: list[dict] = [] # List of game result dictionaries for the current round

        self.parent_id: str | None = parent_id
        self.evolutionary_input_positive_features: list[str] = evolutionary_input_positive_features if evolutionary_input_positive_features is not None else []
        self.evolutionary_input_negative_features: list[str] = evolutionary_input_negative_features if evolutionary_input_negative_features is not None else []

        self.current_fitness_score: float | None = None

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

    def get_genome_for_goodfire_variant(self) -> dict[goodfire.Feature, float]:
        """
        Converts the agent's internal genome to the format expected by goodfire.Variant().set(),
        which is dict[goodfire.Feature, float].
        Requires 'index_in_sae' to be present and valid in genome entries for them to be included.
        Skips features where 'index_in_sae' is missing or invalid.
        """
        genome_for_api = {}
        for uuid_str, data in self.genome.items():
            if isinstance(data, dict) and \
               'activation' in data and \
               'label' in data and \
               data.get('index_in_sae') is not None:
                try:
                    feature_uuid_obj = uuid.UUID(uuid_str) if isinstance(uuid_str, str) else uuid_str

                    feature_obj = goodfire.Feature(
                        uuid=feature_uuid_obj, # Pass UUID object
                        label=str(data['label']),
                        index_in_sae=int(data['index_in_sae'])
                    )
                    genome_for_api[feature_obj] = float(data['activation'])
                except (ValueError, TypeError) as e: # Catches errors from int(), float(), UUID()
                    logging.warning(f"Agent {self.agent_id}: Could not create Feature object or convert values for genome entry UUID '{uuid_str}' (label: '{data.get('label')}', index: '{data.get('index_in_sae')}'). Error: {e}. Skipping this feature for Variant.")
            else:
                logging.debug(f"Agent {self.agent_id}: Genome entry for UUID '{uuid_str}' is missing 'activation', 'label', or a valid 'index_in_sae'. Full data: {data}. Skipping for Variant.")
        return genome_for_api


    def to_dict(self) -> dict:
        """
        Serializes the agent's state to a dictionary.
        """
        return {
            'agent_id': self.agent_id,
            'model_id': self.model_id,
            'genome': copy.deepcopy(self.genome),
            'wealth': self.wealth,
            'parent_id': self.parent_id,
            'evolutionary_input_positive_features': copy.deepcopy(self.evolutionary_input_positive_features),
            'evolutionary_input_negative_features': copy.deepcopy(self.evolutionary_input_negative_features),
            'current_fitness_score': self.current_fitness_score
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
        initial_genome_data = data.get('genome', {}) # Default to empty dict
        initial_wealth = data.get('wealth')
        parent_id = data.get('parent_id')
        evo_pos_features = data.get('evolutionary_input_positive_features')
        evo_neg_features = data.get('evolutionary_input_negative_features')
        current_fitness_score = data.get('current_fitness_score')

        if not agent_id or not isinstance(agent_id, str):
            logging.warning("Missing or invalid 'agent_id' in agent data, generating a new UUID.")
            agent_id = str(uuid.uuid4())
        if not model_id or not isinstance(model_id, str):
            raise ValueError("Missing or invalid 'model_id' (string) in agent data.")

        parsed_genome = {}
        if not isinstance(initial_genome_data, dict):
            logging.warning(f"Agent {agent_id}: Invalid 'genome' type in data (expected dict, got {type(initial_genome_data)}). Defaulting to empty genome.")
        else:
            for f_uuid, f_data in initial_genome_data.items():
                if isinstance(f_data, dict) and 'activation' in f_data and 'label' in f_data:
                    try:
                        parsed_genome[str(f_uuid)] = {
                            'activation': float(f_data['activation']),
                            'label': str(f_data['label']),
                            'index_in_sae': int(f_data['index_in_sae']) if f_data.get('index_in_sae') is not None else None
                        }
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Agent {agent_id}: Error parsing genome feature {f_uuid} data '{f_data}': {e}. Skipping.")
                elif isinstance(f_data, (int, float)): # Handle old format: direct activation value
                    logging.debug(f"Agent {agent_id}: Genome feature {f_uuid} in old format. Converting. Label will be generic, index_in_sae will be None.")
                    parsed_genome[str(f_uuid)] = {
                        'activation': float(f_data),
                        'label': f"Feature {str(f_uuid)[:8]} (label N/A)",
                        'index_in_sae': None
                    }
                else:
                    logging.warning(f"Agent {agent_id}: Malformed genome entry for {f_uuid}: {f_data}. Skipping.")

        if initial_wealth is None or not isinstance(initial_wealth, (int, float)):
            logging.warning(f"Agent {agent_id}: Missing or invalid 'wealth' in data (expected float/int, got {type(initial_wealth)}). Defaulting to 30.0.")
            initial_wealth = 30.0

        agent = cls(
            agent_id=agent_id,
            model_id=model_id,
            initial_genome=parsed_genome, # Pass the processed genome
            initial_wealth=float(initial_wealth),
            parent_id=parent_id,
            evolutionary_input_positive_features=evo_pos_features,
            evolutionary_input_negative_features=evo_neg_features
        )
        if current_fitness_score is not None:
            try:
                agent.current_fitness_score = float(current_fitness_score)
            except (ValueError, TypeError):
                logging.warning(f"Agent {agent_id}: Invalid current_fitness_score value '{current_fitness_score}'. Setting to None.")
                agent.current_fitness_score = None
        return agent

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
    initial_wealth = agent_config.get('initial_wealth', 30.0)

    if not isinstance(pop_size, int) or pop_size <= 0:
        raise ValueError("simulation.population_size must be a positive integer.")
    if not model_id or not isinstance(model_id, str):
        raise ValueError("agent.model_id must be a non-empty string.")
    if not isinstance(initial_wealth, (int, float)):
        logging.warning(f"agent.initial_wealth is not a number ({initial_wealth}). Defaulting to 30.0.")
        initial_wealth = 30.0

    initial_genome_template_data = agent_config.get('initial_genome', {})
    parsed_initial_genome_template = {}
    if not isinstance(initial_genome_template_data, dict):
        logging.warning(f"agent.initial_genome is not a dictionary ({initial_genome_template_data}). Defaulting to empty genome template.")
    else:
        for f_uuid, f_data in initial_genome_template_data.items():
            if isinstance(f_data, dict) and 'activation' in f_data and 'label' in f_data:
                try:
                    parsed_initial_genome_template[str(f_uuid)] = {
                        'activation': float(f_data['activation']),
                        'label': str(f_data['label']),
                        'index_in_sae': int(f_data['index_in_sae']) if f_data.get('index_in_sae') is not None else None
                    }
                except (ValueError, TypeError):
                     logging.warning(f"Skipping malformed initial_genome template item (dict type): {f_uuid} -> {f_data}")
            elif isinstance(f_data, (int, float)): # Old format template item
                parsed_initial_genome_template[str(f_uuid)] = {
                    'activation': float(f_data),
                    'label': f"Initial Feature {str(f_uuid)[:8]}", # Generic label for old format
                    'index_in_sae': None # Cannot know index for arbitrary initial genome feature
                }
            else:
                 logging.warning(f"Skipping malformed initial_genome template item (unknown type): {f_uuid} -> {f_data}")

    population: list[Agent] = []
    for i in range(pop_size):
        agent_id = str(uuid.uuid4())
        # Each agent gets a deep copy of the parsed template
        agent_genome = copy.deepcopy(parsed_initial_genome_template)
        try:
            agent = Agent(
                agent_id=agent_id,
                model_id=model_id,
                initial_genome=agent_genome,
                initial_wealth=float(initial_wealth),
                parent_id=None,
                evolutionary_input_positive_features=[],
                evolutionary_input_negative_features=[]
            )
            population.append(agent)
            logging.debug(f"Initialized Agent {agent_id} (model: {model_id}, wealth: {initial_wealth:.2f}, genome keys: {list(agent_genome.keys())})")
        except Exception as e:
            logging.critical(f"Failed to initialize agent {i+1}/{pop_size} (ID: {agent_id}): {e}", exc_info=True)
            raise RuntimeError(f"Agent initialization failed for agent {agent_id}") from e

    logging.info(f"Initialized population of {len(population)} agents.")
    return population

def calculate_fitness(population: list[Agent]) -> list[float]:
    """
    Calculates fitness scores for each agent in the population, typically based on wealth.
    Fitness scores are normalized to sum to 1.0.
    Also stores the fitness score on each agent instance.
    """
    if not isinstance(population, list):
        logging.error("Invalid population (must be list) provided to calculate_fitness.")
        return []
    if not all(isinstance(agent, Agent) for agent in population):
        logging.error("Population list contains non-Agent elements in calculate_fitness.")
        return []

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

    if total_wealth <= 1e-9: # Use a small epsilon for float comparison
        logging.warning("Total population wealth is zero or effectively zero. Assigning equal fitness.")
        equal_fitness = (1.0 / num_agents) if num_agents > 0 else 0.0
        fitness_scores = [equal_fitness] * num_agents
    else:
        for wealth in agent_wealths:
            fitness_scores.append(wealth / total_wealth)

    for i, agent in enumerate(population):
        agent.current_fitness_score = fitness_scores[i]

    sum_fitness = sum(fitness_scores)
    if not math.isclose(sum_fitness, 1.0, rel_tol=1e-9) and sum_fitness > 1e-9 :
        logging.warning(f"Raw fitness scores sum to {sum_fitness:.6f}, not 1.0. This might indicate an issue if total_wealth was positive.")

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

    valid_weights_present = any(f > 1e-9 for f in fitness_scores)

    if not valid_weights_present:
        logging.warning("All fitness scores are zero or effectively zero. Performing uniform random parent selection.")
        selected_parents = random.choices(population, k=num_offspring)
    else:
        # weights are non-negative for random.choices
        sanitized_weights = [max(0.0, w) for w in fitness_scores]
        if sum(sanitized_weights) < 1e-9: # Still all zero after sanitizing
            logging.warning("All sanitized fitness scores are zero. Uniform random parent selection fallback.")
            selected_parents = random.choices(population, k=num_offspring)
        else:
            try:
                selected_parents = random.choices(
                    population=population,
                    weights=sanitized_weights,
                    k=num_offspring
                )
            except ValueError as e:
                logging.error(f"ValueError during parent selection (likely due to weights sum to zero): {e}. Falling back to uniform selection.")
                selected_parents = random.choices(population, k=num_offspring)
            except Exception as e:
                logging.error(f"Unexpected error during parent selection: {e}", exc_info=True)
                return []

    logging.info(f"Selected {len(selected_parents)} parents for reproduction.")
    return selected_parents

def apply_algorithmic_genome_update(
    current_genome_state: dict,
    features_to_reinforce: list[goodfire.Feature],
    features_to_suppress: list[goodfire.Feature],
    config: dict
) -> dict:
    """
    Applies algorithmic updates to a genome based on features to reinforce (from wins)
    or suppress (from losses).
    Genome structure: dict[str_uuid, {'activation': float, 'label': str, 'index_in_sae': int | None}]
    Input features are goodfire.Feature objects, which contain uuid, label, and index_in_sae.
    """
    if not isinstance(current_genome_state, dict):
        logging.error("apply_algorithmic_genome_update: current_genome_state must be a dict.")
        return {}
    if not isinstance(features_to_reinforce, list) or not isinstance(features_to_suppress, list):
        logging.error("apply_algorithmic_genome_update: features_to_reinforce/suppress must be lists.")
        return copy.deepcopy(current_genome_state)
    if not isinstance(config, dict):
         logging.error("Invalid configuration provided to apply_algorithmic_genome_update.")
         return copy.deepcopy(current_genome_state)

    evo_config = config.get('evolution', {})
    learning_rate = float(evo_config.get('learning_rate', 0.1)) # Updated default per config
    num_winning_to_use = int(evo_config.get('num_winning_features', 3))
    num_losing_to_use = int(evo_config.get('num_losing_features', 3))
    min_activation = float(evo_config.get('activation_min', -5.0))
    max_activation = float(evo_config.get('activation_max', 5.0))
    target_pos = float(evo_config.get('target_positive', 1.0))
    target_neg = float(evo_config.get('target_negative', -0.1)) # Updated per config
    update_method = evo_config.get('update_method', 'increment')

    offspring_genome = copy.deepcopy(current_genome_state)

    # Process features to reinforce (from winning game)
    if features_to_reinforce:
        for i, feature_obj in enumerate(features_to_reinforce):
            if i >= num_winning_to_use:
                break
            if not (feature_obj and hasattr(feature_obj, 'uuid') and hasattr(feature_obj, 'label') and hasattr(feature_obj, 'index_in_sae')):
                logging.warning(f"Reinforce feature object (index {i}) invalid or lacks uuid/label/index_in_sae. Obj: {feature_obj}")
                continue

            feature_key = str(feature_obj.uuid)
            feature_label = str(feature_obj.label)
            feature_index_in_sae = int(feature_obj.index_in_sae)

            current_genome_entry = offspring_genome.get(feature_key, {'activation': 0.0, 'label': feature_label, 'index_in_sae': feature_index_in_sae})
            current_activation = float(current_genome_entry.get('activation', 0.0))

            if update_method == 'target':
                new_activation = current_activation + learning_rate * (target_pos - current_activation)
            elif update_method == 'increment':
                new_activation = current_activation + learning_rate
            else:
                logging.warning(f"Unknown update_method '{update_method}' in config. Defaulting to 'increment'.")
                new_activation = current_activation + learning_rate

            new_activation = max(min_activation, min(max_activation, new_activation))
            offspring_genome[feature_key] = {'activation': new_activation, 'label': feature_label, 'index_in_sae': feature_index_in_sae}
            logging.debug(f"Genome reinforce: Feature {feature_key} ('{feature_label}') from {current_activation:.4f} to {new_activation:.4f}")

    # Process features to suppress (from losing game)
    if features_to_suppress:
        for i, feature_obj in enumerate(features_to_suppress):
            if i >= num_losing_to_use:
                break
            if not (feature_obj and hasattr(feature_obj, 'uuid') and hasattr(feature_obj, 'label') and hasattr(feature_obj, 'index_in_sae')):
                logging.warning(f"Suppress feature object (index {i}) invalid or lacks uuid/label/index_in_sae. Obj: {feature_obj}")
                continue

            feature_key = str(feature_obj.uuid)
            feature_label = str(feature_obj.label)
            feature_index_in_sae = int(feature_obj.index_in_sae)

            current_genome_entry = offspring_genome.get(feature_key, {'activation': 0.0, 'label': feature_label, 'index_in_sae': feature_index_in_sae})
            current_activation = float(current_genome_entry.get('activation', 0.0))

            if update_method == 'target':
                new_activation = current_activation + learning_rate * (target_neg - current_activation)
            elif update_method == 'increment':
                new_activation = current_activation - learning_rate
            else:
                logging.warning(f"Unknown update_method '{update_method}' in config. Defaulting to 'increment' (decrement for losing).")
                new_activation = current_activation - learning_rate

            new_activation = max(min_activation, min(max_activation, new_activation))
            offspring_genome[feature_key] = {'activation': new_activation, 'label': feature_label, 'index_in_sae': feature_index_in_sae}
            logging.debug(f"Genome suppress: Feature {feature_key} ('{feature_label}') from {current_activation:.4f} to {new_activation:.4f}")

    return offspring_genome


async def _inspect_parent_features_task(
    client: goodfire.AsyncClient, 
    parent_agent: Agent, 
    final_messages_for_api_inspect: list[dict], 
    inspect_aggregate_by_val: str, 
    config: dict # Pass config for logging/potential future use
) -> "ContextInspector" | Exception | None:
    """
    Helper coroutine to run feature inspection for a single parent agent.
    Returns the ContextInspector object on success, an Exception on failure, or None if no inspection is done.
    """
    if not final_messages_for_api_inspect:
        logging.debug(f"Parent {parent_agent.agent_id}: No messages provided for inspection. Skipping task.")
        return None 
    
    parent_variant = goodfire.Variant(parent_agent.model_id)
    parent_genome_for_api = parent_agent.get_genome_for_goodfire_variant()
    if parent_genome_for_api:
        try:
            parent_variant.set(parent_genome_for_api)
        except Exception as e_set_variant:
            logging.error(f"Parent {parent_agent.agent_id}: Error setting variant for inspection task: {e_set_variant}", exc_info=True)
            return e_set_variant # Return exception if variant setting fails

    try:
        logging.debug(f"Parent {parent_agent.agent_id}: Starting inspection task with {len(final_messages_for_api_inspect)} messages.")
        context_inspector = await client.features.inspect(
            messages=final_messages_for_api_inspect,
            model=parent_variant,
            features=None, 
            aggregate_by=inspect_aggregate_by_val
        )
        logging.debug(f"Parent {parent_agent.agent_id}: Inspection task completed successfully.")
        return context_inspector
    except Exception as e:
        logging.error(f"Parent {parent_agent.agent_id}: Error during inspection task: {e}", exc_info=True)
        return e # Return the exception to be handled by the caller

async def evolve_population(
    population: list[Agent], 
    fitness_scores: list[float], 
    config: dict,
    all_games_this_generation: list[dict],
    global_feature_cache: dict # Added parameter for the global feature metadata cache
) -> list[Agent]:
    """
    Evolves the population to create a new generation of agents.
    This involves parent selection, concurrent feature inspection for eligible parents,
    and genome modification for offspring based on inspection results.
    The global_feature_cache is populated with metadata from successful inspections.
    """
    if not isinstance(population, list) or \
       not isinstance(fitness_scores, list) or \
       not isinstance(config, dict) or \
       not isinstance(all_games_this_generation, list) or \
       not isinstance(global_feature_cache, dict): # Check cache type
        logging.error("evolve_population: Invalid arguments (population, fitness_scores, config, all_games_this_generation, or global_feature_cache).")
        return copy.deepcopy(population) if isinstance(population, list) else []

    population_size = len(population)
    
    evo_config = config.get('evolution', {})
    inspect_top_k_val = int(evo_config.get('inspect_top_k', 10))
    inspect_aggregate_by_val = evo_config.get('inspect_aggregate_by', 'max')

    agent_config = config.get('agent', {})
    agent_model_id = agent_config.get('model_id')
    initial_wealth = float(agent_config.get('initial_wealth', 30.0))

    if not agent_model_id:
         logging.critical("Agent model_id missing in config during evolution. Cannot create offspring.")
         raise ValueError("Agent model_id must be specified in configuration for evolution.")

    selected_parents = select_parents(population, fitness_scores, population_size)
    if not selected_parents or len(selected_parents) != population_size :
         logging.error(f"Parent selection failed or returned incorrect number of parents ({len(selected_parents)} instead of {population_size}). Returning clones of original population.")
         cloned_population = [Agent.from_dict(p.to_dict()) for p in population]
         for agent_instance in cloned_population: agent_instance.reset_round_state(initial_wealth)
         return cloned_population

    try:
        client = get_goodfire_async_client(config)
    except Exception as e:
        logging.critical(f"Failed to initialize Goodfire client in evolve_population: {e}. Cannot proceed with feature inspection based evolution.", exc_info=True)
        cloned_population = [Agent.from_dict(p.to_dict()) for p in selected_parents]
        for agent_instance in cloned_population: agent_instance.reset_round_state(initial_wealth)
        logging.warning("Falling back to cloning parents due to Goodfire client initialization failure.")
        return cloned_population

    # Prepare inspection tasks
    inspection_tasks_with_context = []
    for parent_agent in selected_parents:
        final_messages_for_api_inspect_for_current_parent: list[dict] = []
        parent_game_id_for_current_parent: str | None = None # For logging
        outcome_for_current_parent: str | None = None # For decision making

        if not parent_agent.round_history:
            logging.info(f"Parent {parent_agent.agent_id} has no game history this round. No inspection task created.")
        else:
            parent_game_summary = parent_agent.round_history[0]
            parent_game_id_for_current_parent = parent_game_summary.get('game_id')
            outcome_for_current_parent = parent_game_summary.get('outcome')
            
            full_game_details_for_parent = None
            if parent_game_id_for_current_parent:
                full_game_details_for_parent = next((g for g in all_games_this_generation if g.get("game_id") == parent_game_id_for_current_parent), None)

            transcript_for_inspection = full_game_details_for_parent.get('transcript') if full_game_details_for_parent else None

            if transcript_for_inspection and isinstance(transcript_for_inspection, list) and outcome_for_current_parent in ['win', 'loss']:
                message_content_for_inspect = None
                role_for_inspect_api = "assistant"
                parent_agent_role_in_this_game = None
                player_A_id_from_game = full_game_details_for_parent.get('player_A_id')
                player_B_id_from_game = full_game_details_for_parent.get('player_B_id')
                player_A_game_role_from_game = full_game_details_for_parent.get('player_A_game_role')
                player_B_game_role_from_game = full_game_details_for_parent.get('player_B_game_role')

                if player_A_id_from_game == parent_agent.agent_id: parent_agent_role_in_this_game = player_A_game_role_from_game
                elif player_B_id_from_game == parent_agent.agent_id: parent_agent_role_in_this_game = player_B_game_role_from_game
                
                if not parent_agent_role_in_this_game:
                    logging.warning(f"Parent {parent_agent.agent_id} (Game: {parent_game_id_for_current_parent}) - Could not determine parent's game role. Skipping targeted message identification for inspection.")
                
                target_message_id_from_adj = None
                if outcome_for_current_parent == 'win' and parent_agent_role_in_this_game:
                    target_message_id_from_adj = full_game_details_for_parent.get('adjudication_win_message_id')
                elif outcome_for_current_parent == 'loss' and parent_agent_role_in_this_game:
                    target_message_id_from_adj = full_game_details_for_parent.get('adjudication_lose_message_id')

                if target_message_id_from_adj and target_message_id_from_adj.lower() not in ["n/a", "tie", "none", ""]:
                    match_id = re.match(r"([AB])([0-9]+)", target_message_id_from_adj.upper())
                    if match_id:
                        role_prefix_adj, turn_num_1_based_adj_str = match_id.group(1), match_id.group(2)
                        turn_num_1_based_adj = int(turn_num_1_based_adj_str)
                        expected_speaker_game_role_adj = player_A_game_role_from_game if role_prefix_adj == "A" else player_B_game_role_from_game
                        if expected_speaker_game_role_adj == parent_agent_role_in_this_game:
                            current_role_turn_count = 0
                            for turn_data in transcript_for_inspection:
                                if turn_data.get("role") == parent_agent_role_in_this_game:
                                    current_role_turn_count += 1
                                    if current_role_turn_count == turn_num_1_based_adj:
                                        message_content_for_inspect = turn_data.get("content")
                                        break
                        else: logging.warning(f"Parent {parent_agent.agent_id}: Adjudicator message ID '{target_message_id_from_adj}' refers to opponent's message. Using fallback.")
                    else: logging.warning(f"Parent {parent_agent.agent_id}: Could not parse adjudicator message ID '{target_message_id_from_adj}'. Using fallback.")
                elif target_message_id_from_adj: logging.info(f"Parent {parent_agent.agent_id}: Adjudicator message ID was '{target_message_id_from_adj}'. Using fallback.")

                if not message_content_for_inspect and outcome_for_current_parent in ['win', 'loss']: # Fallback
                    for turn_data in reversed(transcript_for_inspection):
                        if turn_data.get("agent_id") == parent_agent.agent_id:
                            message_content_for_inspect = turn_data.get("content")
                            break
                
                if message_content_for_inspect:
                    final_messages_for_api_inspect_for_current_parent = [{"role": role_for_inspect_api, "content": message_content_for_inspect}]
                else:
                    logging.warning(f"Parent {parent_agent.agent_id} (Game: {parent_game_id_for_current_parent}): No suitable message found for inspection.")
            else: # No transcript, or not win/loss
                 log_msg_reason = "no valid transcript" if not transcript_for_inspection else f"outcome '{outcome_for_current_parent}' not win/loss"
                 logging.info(f"Parent {parent_agent.agent_id} (Game: {parent_game_id_for_current_parent}): No inspection task due to {log_msg_reason}.")
        
        task_coro = None
        if final_messages_for_api_inspect_for_current_parent:
            task_coro = _inspect_parent_features_task(client, parent_agent, final_messages_for_api_inspect_for_current_parent, inspect_aggregate_by_val, config)
        
        inspection_tasks_with_context.append({
            "task_coro": task_coro,
            "parent": parent_agent,
            "outcome": outcome_for_current_parent, # Store outcome for later use
            "game_id": parent_game_id_for_current_parent # For logging
        })

    # Execute inspection tasks concurrently
    coroutines_to_await = [item["task_coro"] for item in inspection_tasks_with_context if item["task_coro"] is not None]
    gathered_results = []
    if coroutines_to_await:
        logging.info(f"Starting concurrent inspection for {len(coroutines_to_await)} parents.")
        gathered_results = await asyncio.gather(*coroutines_to_await, return_exceptions=True)
        logging.info(f"Finished concurrent inspection. Received {len(gathered_results)} results/exceptions.")
    
    # Process results and create offspring
    new_population: list[Agent] = []
    result_idx = 0 # To map results from gathered_results back to tasks that were actually run

    for item_context in inspection_tasks_with_context:
        parent_agent = item_context["parent"]
        outcome = item_context["outcome"] # This is the outcome of the parent's game
        game_id_for_log = item_context["game_id"] # For logging purposes

        offspring_genome = copy.deepcopy(parent_agent.genome)
        positive_feature_uuids_for_record: list[str] = []
        negative_feature_uuids_for_record: list[str] = []
        features_to_reinforce_objs: list[goodfire.Feature] = []
        features_to_suppress_objs: list[goodfire.Feature] = []
        
        context_inspector_result = None
        if item_context["task_coro"] is not None: # A task was scheduled for this parent
            if result_idx < len(gathered_results):
                current_result_from_gather = gathered_results[result_idx]
                result_idx += 1

                # Duck-typing: Check if the result has the 'top' method,
                # which is characteristic of a ContextInspector object.
                # This is used because ContextInspector itself cannot be reliably imported for isinstance checks.
                if hasattr(current_result_from_gather, 'top') and callable(getattr(current_result_from_gather, 'top')):
                    context_inspector_result = current_result_from_gather
                    # Populate global_feature_cache with any newly fetched feature metadata
                    if hasattr(context_inspector_result, '_features') and isinstance(context_inspector_result._features, dict):
                        logging.debug(f"Parent {parent_agent.agent_id} (Game: {game_id_for_log}): ContextInspector has {len(context_inspector_result._features)} features in its internal cache.")
                        for f_uuid_str, gf_feature_obj in context_inspector_result._features.items():
                            if f_uuid_str not in global_feature_cache:
                                global_feature_cache[f_uuid_str] = gf_feature_obj
                                logging.debug(f"Parent {parent_agent.agent_id} (Game: {game_id_for_log}): Added feature {f_uuid_str} ('{gf_feature_obj.label}') to global cache.")
                            # else:
                                # logging.debug(f"Parent {parent_agent.agent_id} (Game: {game_id_for_log}): Feature {f_uuid_str} already in global cache.")
                elif isinstance(current_result_from_gather, Exception):
                    logging.error(f"Inspection task for parent {parent_agent.agent_id} (Game: {game_id_for_log}) resulted in an exception: {current_result_from_gather}", exc_info=current_result_from_gather)
                else: # Should not happen if _inspect_parent_features_task returns ContextInspector, Exception, or None
                    logging.error(f"Unexpected result type '{type(current_result_from_gather)}' from inspection task for parent {parent_agent.agent_id} (Game: {game_id_for_log}).")
            else:
                logging.error(f"Logic error: Mismatch between tasks scheduled and results received for parent {parent_agent.agent_id}. Expected result_idx < {len(gathered_results)} but got {result_idx}.")


        if context_inspector_result:
            logging.debug(f"Parent {parent_agent.agent_id} (Game: {game_id_for_log}): Processing successful inspection result.")
            top_feature_activations = context_inspector_result.top(k=inspect_top_k_val)
            activated_features_in_game = [fa.feature for fa in top_feature_activations if hasattr(fa, 'feature') and isinstance(fa.feature, goodfire.Feature)]
            
            num_usable_features = len(activated_features_in_game)
            logging.info(f"Parent {parent_agent.agent_id} (Outcome: {outcome}, Game: {game_id_for_log}): Inspection yielded {num_usable_features} usable features from top {inspect_top_k_val}.")

            if outcome == 'win':
                features_to_reinforce_objs = activated_features_in_game
                positive_feature_uuids_for_record = [str(f.uuid) for f in features_to_reinforce_objs if f and hasattr(f, 'uuid')]
            elif outcome == 'loss':
                features_to_suppress_objs = activated_features_in_game
                negative_feature_uuids_for_record = [str(f.uuid) for f in features_to_suppress_objs if f and hasattr(f, 'uuid')]
        else:
            # This covers cases where task_coro was None, or the task failed, or returned None
            logging.info(f"Parent {parent_agent.agent_id} (Game: {game_id_for_log}): No inspection result to process. Offspring inherits genome without inspection-based update.")

        # Apply genome updates regardless of whether inspection happened (it might be empty lists)
        if features_to_reinforce_objs or features_to_suppress_objs:
            offspring_genome = apply_algorithmic_genome_update(
                current_genome_state=offspring_genome,
                features_to_reinforce=features_to_reinforce_objs,
                features_to_suppress=features_to_suppress_objs,
                config=config
            )
        
        # Create offspring
        offspring_agent_id = str(uuid.uuid4())
        try:
            offspring = Agent(
                agent_id=offspring_agent_id,
                model_id=agent_model_id,
                initial_genome=offspring_genome,
                initial_wealth=initial_wealth,
                parent_id=parent_agent.agent_id,
                evolutionary_input_positive_features=positive_feature_uuids_for_record,
                evolutionary_input_negative_features=negative_feature_uuids_for_record
            )
            new_population.append(offspring)
        except Exception as e_create_offspring:
            logging.error(f"Failed to create offspring for parent {parent_agent.agent_id} (ID: {offspring_agent_id}): {e_create_offspring}", exc_info=True)
            # Fallback: add a clone of the parent to maintain population size
            logging.warning(f"Adding a clone of parent {parent_agent.agent_id} due to offspring creation error.")
            cloned_parent = Agent.from_dict(parent_agent.to_dict()) # Create a new instance
            cloned_parent.agent_id = offspring_agent_id # Give it the new ID
            cloned_parent.parent_id = parent_agent.agent_id # Set parent ID
            cloned_parent.reset_round_state(initial_wealth) # Reset its state for the new generation
            new_population.append(cloned_parent)

    # population size is maintained
    if len(new_population) != population_size:
         logging.warning(f"Evolution resulted in population size {len(new_population)}, but expected {population_size}. Adjusting population size.")
         while len(new_population) < population_size and selected_parents: # selected_parents should not be empty if we reached here
             parent_to_clone = random.choice(selected_parents)
             cloned_offspring_id = str(uuid.uuid4())
             clone = Agent(
                 agent_id=cloned_offspring_id, model_id=parent_to_clone.model_id,
                 initial_genome=copy.deepcopy(parent_to_clone.genome), initial_wealth=initial_wealth,
                 parent_id=parent_to_clone.agent_id
             )
             new_population.append(clone)
             logging.debug(f"Added clone of {parent_to_clone.agent_id} (as {cloned_offspring_id}) to meet population size.")
         if len(new_population) > population_size:
             new_population = new_population[:population_size]
             logging.debug("Truncated new population to meet target size.")

    logging.info(f"Evolution complete using concurrent inspection. New population size: {len(new_population)}.")
    return new_population
