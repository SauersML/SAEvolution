"""
This module governs the procedure for a single two-player game interaction.
It assigns roles, directs the creation of the game's scenario description,
manages the turn-based conversational exchange between the participants,
initiates the evaluation of the interaction's outcome based on scenario rules,
and calculates the resulting wealth adjustment between the players according to
bets placed and the final verdict.
It also collects detailed information about each game for logging and dashboarding.
"""

import random
import logging
import copy
import datetime
import uuid # For unique game IDs
from manager import Agent
from interface import generate_scenario, generate_agent_response, adjudicate_interaction

def _apply_wealth_change(winner: Agent, loser: Agent, winner_bet: float, loser_bet: float, max_loss_multiplier: float, max_gain_ratio: float) -> tuple[float, float]:
    """
    Applies wealth changes to winner and loser based on bets and game parameters.
    Returns: A tuple (actual_gain_for_winner, actual_loss_for_loser).
    """
    if not all(isinstance(agent, Agent) for agent in [winner, loser]):
        logging.error("_apply_wealth_change: Invalid Agent objects provided.")
        return 0.0, 0.0
    if not all(isinstance(val, (int, float)) for val in [winner_bet, loser_bet, max_loss_multiplier, max_gain_ratio]):
         logging.error("_apply_wealth_change: Invalid numerical arguments provided.")
         return 0.0, 0.0
    if max_loss_multiplier < 0:
         logging.error(f"_apply_wealth_change: Invalid max_loss_multiplier ({max_loss_multiplier}). Must be non-negative.")
         return 0.0, 0.0
    if max_gain_ratio < 1.0:
         logging.error(f"_apply_wealth_change: Invalid max_gain_ratio ({max_gain_ratio}). Must be >= 1.0.")
         return 0.0, 0.0

    potential_loss_for_loser = loser_bet * max_loss_multiplier
    actual_loss_for_loser = min(loser.wealth, potential_loss_for_loser)
    actual_loss_for_loser = max(0.0, actual_loss_for_loser)

    gain_from_loser = actual_loss_for_loser
    gain_capped_by_loser_original_bet = min(gain_from_loser, loser_bet)
    max_gain_amount_for_winner = winner.wealth * (max_gain_ratio - 1.0)
    if max_gain_amount_for_winner < 0:
        max_gain_amount_for_winner = 0.0

    actual_gain_for_winner = min(gain_capped_by_loser_original_bet, max_gain_amount_for_winner)
    actual_gain_for_winner = max(0.0, actual_gain_for_winner)

    try:
        winner.wealth += actual_gain_for_winner
        loser.wealth -= actual_loss_for_loser
    except AttributeError:
        logging.error(f"AttributeError updating wealth for {getattr(winner, 'agent_id', 'Unknown Winner')} or {getattr(loser, 'agent_id', 'Unknown Loser')}.")
        return 0.0, 0.0

    logging.debug(f"Wealth change: Winner {winner.agent_id} (bet {winner_bet:.2f}) gains {actual_gain_for_winner:.2f}. Loser {loser.agent_id} (bet {loser_bet:.2f}) loses {actual_loss_for_loser:.2f}.")
    return actual_gain_for_winner, actual_loss_for_loser

def _determine_bets(agent1: Agent, agent2: Agent, config: dict) -> tuple[float, float]:
    """
    Determines the bet amounts for agent1 and agent2 based on the configuration.
    """
    if not all(isinstance(agent, Agent) for agent in [agent1, agent2]):
        logging.error("_determine_bets: Invalid Agent objects provided.")
        return 0.0, 0.0
    if not isinstance(config, dict):
        logging.error("_determine_bets: Invalid config provided.")
        return 0.0, 0.0

    betting_config = config.get('game', {}).get('betting', {})
    strategy = betting_config.get('strategy', 'fixed')
    fixed_amount = float(betting_config.get('fixed_amount', 5.0))
    min_bet = float(betting_config.get('min_bet', 1.0))
    max_bet_ratio = float(betting_config.get('max_bet_ratio', 0.5))

    bet1_val, bet2_val = 0.0, 0.0

    if strategy == 'fixed':
        bet1_val = fixed_amount
        bet2_val = fixed_amount
    else:
        logging.warning(f"Unsupported betting strategy '{strategy}' in config. Defaulting to fixed amount: {fixed_amount}.")
        bet1_val = fixed_amount
        bet2_val = fixed_amount

    bet1_val = min(bet1_val, agent1.wealth * max_bet_ratio)
    bet2_val = min(bet2_val, agent2.wealth * max_bet_ratio)
    bet1_val = max(min_bet, bet1_val)
    bet1_val = min(agent1.wealth, bet1_val)
    bet2_val = max(min_bet, bet2_val)
    bet2_val = min(agent2.wealth, bet2_val)
    bet1_val = max(0.0, bet1_val)
    bet2_val = max(0.0, bet2_val)

    logging.debug(f"Agent {agent1.agent_id} (wealth {agent1.wealth:.2f}) bets {bet1_val:.2f}. Agent {agent2.agent_id} (wealth {agent2.wealth:.2f}) bets {bet2_val:.2f}.")
    return bet1_val, bet2_val

def _play_single_game(agent1: Agent, agent2: Agent, config: dict, run_id: str, generation_number: int, game_index_in_round: int) -> dict:
    """
    Manages a single game interaction between two agents.
    Updates agent wealth and internal history.
    Returns a dictionary with comprehensive details of the game for external logging/dashboarding.
    """
    if not all(isinstance(agent, Agent) for agent in [agent1, agent2]):
        logging.error("Invalid Agent objects provided to _play_single_game.")
        return {} # Return empty dict on error
    if not isinstance(config, dict):
        logging.error("Invalid config provided to _play_single_game.")
        return {}

    game_id = f"{run_id}_gen{generation_number}_game{game_index_in_round}_{uuid.uuid4().hex[:8]}"
    timestamp_start = datetime.datetime.now().isoformat()

    game_details_dict = {
        "game_id": game_id,
        "run_id": run_id,
        "generation_number": generation_number,
        "timestamp_start": timestamp_start,
        "timestamp_end": None, # Will be set later
        "proposer_agent_id": None,
        "opponent_in_proposal_agent_id": None,
        "player_A_id": agent1.agent_id,
        "player_B_id": agent2.agent_id,
        "player_A_game_role": None,
        "player_B_game_role": None,
        "scenario_text": "Error: Scenario not processed.",
        "scenario_raw_llm_output": None,
        "scenario_generation_successful": False,
        "transcript": [], # Will be populated
        "adjudication_prompt": None,
        "adjudication_raw_llm_output": None, # If interface provides it
        "adjudication_result": "Error: Adjudication not run.",
        "betting_details": None,
        "wealth_changes": None,
        "final_player_A_wealth": agent1.wealth, # Initial wealth for now
        "final_player_B_wealth": agent2.wealth, # Initial wealth for now
        "defaulted_to_tie_reason": None
    }

    game_config = config.get('game', {})
    interaction_turns = game_config.get('interaction_turns_per_agent', 3)
    betting_conf = config.get('game', {}).get('betting', {})
    max_loss_mult = float(betting_conf.get('max_loss_multiplier', 1.0))
    max_gain_ratio_val = float(betting_conf.get('max_gain_ratio', 2.0))

    proposer_obj, opponent_obj = random.sample([agent1, agent2], 2)
    game_details_dict["proposer_agent_id"] = proposer_obj.agent_id
    game_details_dict["opponent_in_proposal_agent_id"] = opponent_obj.agent_id

    logging.info(f"Game {game_id}: Proposer: {proposer_obj.agent_id}, Opponent: {opponent_obj.agent_id}. (Full pair: {agent1.agent_id} vs {agent2.agent_id})")

    # For Agent's internal history (simplified compared to full game_details_dict)
    hist_entry_for_agent1 = {
        'game_id': game_id,
        'opponent_id': agent2.agent_id,
        'role_in_proposal': 'proposer' if agent1 == proposer_obj else 'opponent',
        'role_in_game': None,
        'scenario_text': game_details_dict["scenario_text"], # Will be updated
        'transcript_snippet': None, # Could be a snippet or reference
        'bets': None,
        'outcome': 'error',
        'wealth_change': 0.0
    }

    actual_winner_agent: Agent | None = None
    actual_loser_agent: Agent | None = None
    adjudication_final_text = "Error" # Default adjudication text

    try:
        # 1. Scenario Generation
        scenario_info = generate_scenario(proposer_obj, config) # generate_scenario should return a dict

        if isinstance(scenario_info, dict) and scenario_info.get('scenario_text') and scenario_info.get('role_assignment'):
            game_details_dict["scenario_text"] = scenario_info['scenario_text']
            game_details_dict["scenario_raw_llm_output"] = scenario_info.get('raw_output') # Assuming interface.py might add this
            game_details_dict["scenario_generation_successful"] = True
            hist_entry_for_agent1['scenario_text'] = scenario_info['scenario_text'] # Update for agent history too
            logging.debug(f"Game {game_id}: Scenario generated by {proposer_obj.agent_id}: '{scenario_info['scenario_text'][:100]}...'")

            # Assign roles for the game interaction based on scenario_info
            role_map = scenario_info['role_assignment'] # Should contain {proposer_id: 'Role A/B'}
            proposer_game_role = role_map[proposer_obj.agent_id]
            opponent_game_role = 'Role B' if proposer_game_role == 'Role A' else 'Role A'
            
            # Store game roles for agent1 and agent2
            if agent1 == proposer_obj:
                game_details_dict["player_A_game_role"] = proposer_game_role
                game_details_dict["player_B_game_role"] = opponent_game_role
            else: # agent1 == opponent_obj
                game_details_dict["player_A_game_role"] = opponent_game_role
                game_details_dict["player_B_game_role"] = proposer_game_role
            
            hist_entry_for_agent1['role_in_game'] = game_details_dict["player_A_game_role"]
            logging.info(f"Game {game_id} roles: {agent1.agent_id} is {game_details_dict['player_A_game_role']}, {agent2.agent_id} is {game_details_dict['player_B_game_role']}")

            # 2. Interaction Phase
            interaction_players = [agent1, agent2]
            random.shuffle(interaction_players) # Randomize who speaks first in the dialogue
            
            current_transcript = []
            for turn_idx in range(interaction_turns * 2):
                current_turn_agent = interaction_players[turn_idx % 2]
                current_turn_agent_game_role = game_details_dict["player_A_game_role"] if current_turn_agent == agent1 else game_details_dict["player_B_game_role"]
                
                logging.debug(f"Game {game_id} Turn {turn_idx + 1}: Agent {current_turn_agent.agent_id} ({current_turn_agent_game_role}) responding.")
                # scenario_data_dict for generate_agent_response should be the `scenario_info` from generate_scenario
                response_txt = generate_agent_response(current_turn_agent, scenario_info, current_transcript, current_turn_agent_game_role, config)

                turn_entry = {"role": current_turn_agent_game_role, "agent_id": current_turn_agent.agent_id, "content": "[Agent failed to provide a response]"}
                if response_txt and isinstance(response_txt, str) and response_txt.strip():
                    turn_entry["content"] = response_txt
                    logging.debug(f"Game {game_id}: Agent {current_turn_agent.agent_id} response: '{response_txt[:100]}...'")
                else:
                    logging.warning(f"Game {game_id}: Agent {current_turn_agent.agent_id} failed to generate a valid/non-empty response.")
                current_transcript.append(turn_entry)
            
            game_details_dict["transcript"] = current_transcript
            hist_entry_for_agent1['transcript_snippet'] = f"{len(current_transcript)} turns" # Or first/last turn snippet

            # 3. Adjudication
            # adjudicate_interaction should accept the scenario_info dict and the transcript
            adjudication_final_text = adjudicate_interaction(scenario_info, current_transcript, config)
            game_details_dict["adjudication_result"] = adjudication_final_text
            # game_details_dict["adjudication_prompt"] = ... # if interface.py can provide this
            # game_details_dict["adjudication_raw_llm_output"] = ... # if interface.py can provide this

            if adjudication_final_text in ['Role A Wins', 'Role B Wins', 'Tie']:
                if adjudication_final_text == 'Role A Wins':
                    actual_winner_agent = agent1 if game_details_dict["player_A_game_role"] == 'Role A' else agent2
                    actual_loser_agent = agent2 if game_details_dict["player_A_game_role"] == 'Role A' else agent1
                elif adjudication_final_text == 'Role B Wins':
                    actual_winner_agent = agent1 if game_details_dict["player_A_game_role"] == 'Role B' else agent2
                    actual_loser_agent = agent2 if game_details_dict["player_A_game_role"] == 'Role B' else agent1
                logging.info(f"Game {game_id}: Adjudication result: {adjudication_final_text}.")
            else:
                logging.warning(f"Game {game_id}: Invalid or non-standard adjudication result: '{adjudication_final_text}'. Game defaulted to Tie.")
                game_details_dict["defaulted_to_tie_reason"] = f"Adjudication error: {adjudication_final_text}"
                # actual_winner_agent and actual_loser_agent remain None -> Tie

        else: # Scenario generation failed
            game_details_dict["scenario_text"] = "Scenario generation failed or returned invalid data."
            game_details_dict["scenario_generation_successful"] = False
            logging.warning(f"Game {game_id}: Scenario generation failed. Proposer ({proposer_obj.agent_id}) loses by default.")
            actual_winner_agent, actual_loser_agent = opponent_obj, proposer_obj
            game_details_dict["defaulted_to_tie_reason"] = "Scenario generation failure (proposer loss)" # Or specific outcome
            game_details_dict["adjudication_result"] = "Proposer Loss (Scenario Fail)" # More specific than "Error"


        # 4. Determine Bets
        bet_agent1, bet_agent2 = _determine_bets(agent1, agent2, config)
        game_details_dict["betting_details"] = {"player_A_bet": bet_agent1, "player_B_bet": bet_agent2}
        hist_entry_for_agent1['bets'] = {'agent1_bet': bet_agent1, 'agent2_bet': bet_agent2}

        # 5. Apply Wealth Change and Finalize Game Details
        wealth_A_change, wealth_B_change = 0.0, 0.0
        if actual_winner_agent and actual_loser_agent:
            winner_bet = bet_agent1 if actual_winner_agent == agent1 else bet_agent2
            loser_bet = bet_agent2 if actual_winner_agent == agent1 else bet_agent1 # This logic seems off, should be loser's bet
            
            # Corrected bet passing:
            gain, loss = _apply_wealth_change(actual_winner_agent, actual_loser_agent, 
                                              bet_agent1 if actual_winner_agent == agent1 else bet_agent2, # winner's bet amount
                                              bet_agent1 if actual_loser_agent == agent1 else bet_agent2,   # loser's bet amount
                                              max_loss_mult, max_gain_ratio_val)

            if actual_winner_agent == agent1:
                wealth_A_change = gain
                wealth_B_change = -loss
                hist_entry_for_agent1['outcome'] = 'win'
            else: # actual_winner_agent == agent2
                wealth_A_change = -loss
                wealth_B_change = gain
                hist_entry_for_agent1['outcome'] = 'loss'
            hist_entry_for_agent1['wealth_change'] = wealth_A_change
        else: # Tie
            hist_entry_for_agent1['outcome'] = 'tie'
            hist_entry_for_agent1['wealth_change'] = 0.0
            # adjudication_result might already be 'Tie' or an error message
            if not game_details_dict["defaulted_to_tie_reason"]:
                game_details_dict["defaulted_to_tie_reason"] = "Adjudicated as Tie"
            if adjudication_final_text != "Tie": # If it wasn't explicitly a Tie from adjudicator
                game_details_dict["adjudication_result"] = "Tie (Defaulted)"


        game_details_dict["wealth_changes"] = {"player_A_wealth_change": wealth_A_change, "player_B_wealth_change": wealth_B_change}
        logging.info(f"Game {game_id}: Outcome for {agent1.agent_id}: {hist_entry_for_agent1['outcome']}, wealth change: {wealth_A_change:.2f}. For {agent2.agent_id} wealth change: {wealth_B_change:.2f}")

    except Exception as e:
        logging.critical(f"Game {game_id}: Critical unhandled error during single game logic: {e}", exc_info=True)
        hist_entry_for_agent1['outcome'] = 'error'
        hist_entry_for_agent1['wealth_change'] = 0.0
        game_details_dict["adjudication_result"] = "Critical Game Error"
        game_details_dict["defaulted_to_tie_reason"] = f"Critical error: {str(e)[:100]}"
    finally:
        # 6. Record Game History for agents (internal)
        try:
            agent1.add_game_result(copy.deepcopy(hist_entry_for_agent1))
            hist_entry_for_agent2 = copy.deepcopy(hist_entry_for_agent1)
            hist_entry_for_agent2['opponent_id'] = agent1.agent_id
            hist_entry_for_agent2['role_in_proposal'] = 'proposer' if agent2 == proposer_obj else 'opponent'
            hist_entry_for_agent2['role_in_game'] = game_details_dict["player_B_game_role"]
            if hist_entry_for_agent1['outcome'] == 'win': hist_entry_for_agent2['outcome'] = 'loss'
            elif hist_entry_for_agent1['outcome'] == 'loss': hist_entry_for_agent2['outcome'] = 'win'
            else: hist_entry_for_agent2['outcome'] = hist_entry_for_agent1['outcome']
            hist_entry_for_agent2['wealth_change'] = -hist_entry_for_agent1['wealth_change']
            agent2.add_game_result(hist_entry_for_agent2)
        except Exception as e:
             logging.error(f"Game {game_id}: Unexpected error saving game history for agents: {e}", exc_info=True)

        game_details_dict["timestamp_end"] = datetime.datetime.now().isoformat()
        game_details_dict["final_player_A_wealth"] = agent1.wealth
        game_details_dict["final_player_B_wealth"] = agent2.wealth
        
    return game_details_dict


def run_game_round(population: list[Agent], config: dict, run_id: str, generation_number: int) -> tuple[list[Agent], list[dict]]:
    """
    Manages a round of games for the entire population.
    Agents are paired, play games, and their states (wealth, history) are updated.
    Returns the updated population and a list of detailed game dictionaries.
    """
    if not isinstance(population, list):
        logging.error("run_game_round: Population must be a list.")
        return [], []
    if not all(isinstance(agent, Agent) for agent in population):
        logging.error("run_game_round: Population list contains non-Agent elements.")
        return population, []
    if not population:
        logging.warning("run_game_round: Called with empty population.")
        return [], []
    if not isinstance(config, dict):
        logging.error("run_game_round: Config must be a dictionary.")
        return population, []

    games_per_agent_target = int(config.get('simulation', {}).get('games_per_agent_in_round', 3))
    pairing_strategy = config.get('round', {}).get('pairing_strategy', 'random_shuffle')
    num_agents = len(population)
    initial_wealth_val = float(config.get('agent', {}).get('initial_wealth', 30.0))

    all_games_details_this_round: list[dict] = []

    # Reset agent states for the round
    for current_agent in population:
        try:
            current_agent.reset_round_state(initial_wealth_val)
        except AttributeError:
             logging.critical(f"Agent {getattr(current_agent, 'agent_id', 'Unknown')} missing reset_round_state method.")
             raise

    if num_agents < 2:
        logging.warning("Not enough agents (<2) for pairing. Skipping game round.")
        return population, [] # Return empty list for game details

    if pairing_strategy == 'random_shuffle':
        games_played_this_round_count = {agent.agent_id: 0 for agent in population}
        total_participations_needed = num_agents * games_per_agent_target
        if total_participations_needed % 2 != 0:
            logging.warning(f"Target total participations ({total_participations_needed}) is odd. May lead to unequal game counts.")
        
        num_total_games_to_schedule = total_participations_needed // 2
        max_iters = num_total_games_to_schedule * 3 + num_agents
        if num_agents <= 3: max_iters = games_per_agent_target * num_agents * 2

        actual_games_played_count = 0
        for iter_num in range(max_iters):
            if all(count >= games_per_agent_target for count in games_played_this_round_count.values()):
                logging.info(f"All agents appear to have met target of {games_per_agent_target} games by iteration {iter_num + 1}.")
                break

            shuffled_indices = list(range(num_agents))
            random.shuffle(shuffled_indices)
            was_paired_in_pass = [False] * num_agents

            for i in range(0, num_agents - (num_agents % 2), 2):
                idx1, idx2 = shuffled_indices[i], shuffled_indices[i+1]
                if was_paired_in_pass[idx1] or was_paired_in_pass[idx2]:
                    continue

                agent_A_obj, agent_B_obj = population[idx1], population[idx2]

                if games_played_this_round_count.get(agent_A_obj.agent_id, 0) < games_per_agent_target and \
                   games_played_this_round_count.get(agent_B_obj.agent_id, 0) < games_per_agent_target:
                    
                    game_detail_dict = _play_single_game(agent_A_obj, agent_B_obj, config, run_id, generation_number, actual_games_played_count + 1)
                    if game_detail_dict: # Only append if a valid detail dict was returned
                        all_games_details_this_round.append(game_detail_dict)
                    
                    games_played_this_round_count[agent_A_obj.agent_id] += 1
                    games_played_this_round_count[agent_B_obj.agent_id] += 1
                    was_paired_in_pass[idx1] = True
                    was_paired_in_pass[idx2] = True
                    actual_games_played_count += 1
            
            if iter_num == max_iters - 1:
                 logging.warning(f"Reached max pairing iterations ({max_iters}). Game counts per agent may not be perfectly uniform at {games_per_agent_target}.")

        logging.info(f"Game round finished. Total distinct games played: {actual_games_played_count}.")
        for p_agent in population:
            final_count = games_played_this_round_count.get(p_agent.agent_id, 0)
            logging.info(f"Agent {p_agent.agent_id} played {final_count} games this round (target: {games_per_agent_target}).")
        
        agents_below_target = [aid for aid, count in games_played_this_round_count.items() if count < games_per_agent_target]
        if agents_below_target:
             logging.warning(f"{len(agents_below_target)} agent(s) did not meet {games_per_agent_target} games: {agents_below_target}")

    else:
        logging.error(f"Unsupported pairing strategy: '{pairing_strategy}'")
        raise NotImplementedError(f"Pairing strategy '{pairing_strategy}' is not implemented.")

    return population, all_games_details_this_round
