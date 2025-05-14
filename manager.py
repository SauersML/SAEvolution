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
import goodfire # For goodfire.Variant and goodfire.Feature
from interface import get_goodfire_client # To get the API client instance

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


def evolve_population(
    population: list[Agent], 
    fitness_scores: list[float], 
    config: dict,
    all_games_this_generation: list[dict] # New parameter
) -> list[Agent]:
    """
    Evolves the population to create a new generation of agents.
    This involves parent selection and genome modification for offspring using feature inspection.
    Args:
        population: The current population of agents.
        fitness_scores: Fitness scores for the current population.
        config: The simulation configuration dictionary.
        all_games_this_generation: A list of all game detail dictionaries from the current generation.
    """
    if not isinstance(population, list) or \
       not isinstance(fitness_scores, list) or \
       not isinstance(config, dict) or \
       not isinstance(all_games_this_generation, list):
        logging.error("evolve_population: Invalid arguments.")
        return copy.deepcopy(population) if isinstance(population, list) else []

    population_size = len(population)
    new_population: list[Agent] = []

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
         for agent_instance in cloned_population: agent_instance.reset_round_state(initial_wealth) # Renamed agent to agent_instance
         return cloned_population

    try:
        client = get_goodfire_client(config)
    except Exception as e:
        logging.critical(f"Failed to initialize Goodfire client in evolve_population: {e}. Cannot proceed with feature inspection based evolution.", exc_info=True)
        # Fallback: Create offspring by cloning parents without genome modification
        cloned_population = [Agent.from_dict(p.to_dict()) for p in selected_parents] # Use selected_parents
        for agent_instance in cloned_population: agent_instance.reset_round_state(initial_wealth)
        logging.warning("Falling back to cloning parents due to Goodfire client initialization failure.")
        return cloned_population


    for parent_agent in selected_parents:
        offspring_genome = copy.deepcopy(parent_agent.genome)
        positive_feature_uuids_for_record: list[str] = []
        negative_feature_uuids_for_record: list[str] = []
        features_to_reinforce_objs: list[goodfire.Feature] = []
        features_to_suppress_objs: list[goodfire.Feature] = []

        if not parent_agent.round_history:
            logging.info(f"Parent {parent_agent.agent_id} has no game history this round. Offspring inherits genome without inspection update.")
        else:
            # Get the summary of the game played by the parent (we'll use its game_id)
            # The first game in round_history is the one to inspect for evolution for now.
            # THIS REQUIRES THERE TO BE ONE GAME PER GENERATION
            parent_game_summary = parent_agent.round_history[0]
            parent_game_id = parent_game_summary.get('game_id')
            outcome = parent_game_summary.get('outcome')
            
            full_game_details_for_parent = None
            if parent_game_id:
                full_game_details_for_parent = next((g for g in all_games_this_generation if g.get("game_id") == parent_game_id), None)

            if not full_game_details_for_parent:
                logging.warning(f"Parent {parent_agent.agent_id} played game {parent_game_id}, but full details not found in all_games_this_generation. Skipping inspection.")
                transcript_for_inspection = None
            else:
                transcript_for_inspection = full_game_details_for_parent.get('transcript') # This is List[Dict[str, str]] like ChatMessage

            if transcript_for_inspection and isinstance(transcript_for_inspection, list) and outcome in ['win', 'loss']:
                logging.info(f"Parent {parent_agent.agent_id} (outcome: {outcome}) - Preparing to inspect game {parent_game_id} with {len(transcript_for_inspection)} transcript entries.")
                try:
                    parent_variant = goodfire.Variant(parent_agent.model_id)
                    parent_genome_for_api_variant_set = parent_agent.get_genome_for_goodfire_variant()
                    if parent_genome_for_api_variant_set:
                        parent_variant.set(parent_genome_for_api_variant_set)

                    logging.debug(f"Inspecting game transcript for parent {parent_agent.agent_id} (outcome: {outcome}) using aggregate_by='{inspect_aggregate_by_val}'")

                    context_inspector = client.features.inspect(
                        messages=transcript, # transcript should be List[ChatMessage]
                        model=parent_variant,
                        features=None,
                        aggregate_by=inspect_aggregate_by_val
                    )

                    top_feature_activations = context_inspector.top(k=inspect_top_k_val)
                    activated_features_in_game = [fa.feature for fa in top_feature_activations if hasattr(fa, 'feature')]

                    logging.info(f"Parent {parent_agent.agent_id} (outcome: {outcome}): Inspected game, found {len(activated_features_in_game)} top active features.")

                    if outcome == 'win':
                        features_to_reinforce_objs = activated_features_in_game
                        positive_feature_uuids_for_record = [str(f.uuid) for f in features_to_reinforce_objs if f and hasattr(f, 'uuid')]
                    elif outcome == 'loss':
                        features_to_suppress_objs = activated_features_in_game
                        negative_feature_uuids_for_record = [str(f.uuid) for f in features_to_suppress_objs if f and hasattr(f, 'uuid')]

                except Exception as e:
                    logging.error(f"Error during feature inspection for parent {parent_agent.agent_id}: {e}", exc_info=True)
            else:
                log_msg_detail = "no transcript" if not transcript else f"outcome '{outcome}' not win/loss"
                logging.info(f"Parent {parent_agent.agent_id} had {log_msg_detail}. Offspring inherits genome without inspection update.")

        if features_to_reinforce_objs or features_to_suppress_objs:
            offspring_genome = apply_algorithmic_genome_update(
                current_genome_state=offspring_genome,
                features_to_reinforce=features_to_reinforce_objs,
                features_to_suppress=features_to_suppress_objs,
                config=config
            )
        else:
            logging.debug(f"No features identified from inspection for parent {parent_agent.agent_id} to apply updates, or outcome was not win/loss/tie. Offspring inherits genome.")


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
            logging.debug(f"Created offspring {offspring_agent_id} from parent {parent_agent.agent_id}.")
        except Exception as e:
             logging.error(f"Failed to create offspring agent {offspring_agent_id} from parent {parent_agent.agent_id}: {e}", exc_info=True)

    if len(new_population) != population_size:
         logging.warning(f"Evolution resulted in population size {len(new_population)}, but expected {population_size}. "
                       "This might be due to errors in offspring creation during the loop.")

    logging.info(f"Evolution complete. New population size: {len(new_population)}.")
    return new_population
