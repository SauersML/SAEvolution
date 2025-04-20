"""
This module governs the procedure for a single two-player game interaction. 
It assigns roles, directs the creation of the game's scenario description, 
manages the turn-based conversational exchange between the participants, 
initiates the evaluation of the interaction's outcome based on scenario rules, 
and calculates the resulting wealth adjustment between the players according to 
bets placed and the final verdict.
"""

import random
import logging
import math
import copy
from manager import Agent
from interface import generate_scenario, generate_agent_response, adjudicate_interaction, perform_contrastive_analysis

def _apply_wealth_change(winner, loser, winner_bet, loser_bet, max_loss_multiplier, max_gain_ratio):
    if not all(isinstance(agent, Agent) for agent in [winner, loser]):
        logging.error("_apply_wealth_change: Invalid Agent objects provided.")
        return 0.0, 0.0
    if not all(isinstance(val, (int, float)) for val in [winner_bet, loser_bet, max_loss_multiplier, max_gain_ratio]):
         logging.error("_apply_wealth_change: Invalid numerical arguments provided.")
         return 0.0, 0.0
    if max_loss_multiplier < 0 or max_gain_ratio < 1.0:
         logging.error(f"_apply_wealth_change: Invalid multipliers/ratios provided (max_loss={max_loss_multiplier}, max_gain={max_gain_ratio}).")
         return 0.0, 0.0

    potential_loss = loser_bet * max_loss_multiplier
    actual_loss = min(loser.wealth, potential_loss)
    actual_loss = max(0.0, actual_loss) 

    potential_gain_from_loser = min(actual_loss, loser_bet)

    max_gain_limit = winner.wealth * (max_gain_ratio - 1.0)
    potential_gain = min(potential_gain_from_loser, max_gain_limit)

    actual_gain = max(0.0, potential_gain)

    try:
        winner.wealth += actual_gain
        loser.wealth -= actual_loss
    except AttributeError:
        logging.error(f"AttributeError updating wealth for {winner.agent_id} or {loser.agent_id}.")
        return 0.0, 0.0 

    logging.debug(f"Wealth change: Winner {winner.agent_id} gains {actual_gain:.2f} (bet {winner_bet:.2f}). Loser {loser.agent_id} loses {actual_loss:.2f} (bet {loser_bet:.2f}).")

    return actual_gain, actual_loss

def _determine_bets(agent1, agent2, config):
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
    max_bet_ratio = float(betting_config.get('max_bet_ratio', 1.0))

    bet1 = fixed_amount
    bet2 = fixed_amount

    bet1 = min(bet1, agent1.wealth * max_bet_ratio)
    bet2 = min(bet2, agent2.wealth * max_bet_ratio)
    
    bet1 = max(min_bet, min(agent1.wealth, bet1))
    bet2 = max(min_bet, min(agent2.wealth, bet2))
    
    bet1 = max(0.0, bet1)
    bet2 = max(0.0, bet2)

    logging.debug(f"Agent {agent1.agent_id} bets {bet1:.2f}. Agent {agent2.agent_id} bets {bet2:.2f}.")
    return bet1, bet2

def _play_single_game(agent1: Agent, agent2: Agent, config: dict):
    if not all(isinstance(agent, Agent) for agent in [agent1, agent2]):
        logging.error("Invalid Agent objects provided to _play_single_game.")
        return
    if not isinstance(config, dict):
        logging.error("Invalid config provided to _play_single_game.")
        return

    game_config = config.get('game', {})
    interaction_turns = game_config.get('interaction_turns_per_agent', 3)
    betting_config = game_config.get('betting', {})
    max_loss_mult = float(betting_config.get('max_loss_multiplier', 1.0))
    max_gain_ratio = float(betting_config.get('max_gain_ratio', 2.0))

    try:
        scenario_proposer, opponent = random.sample([agent1, agent2], 2)
    except ValueError:
         logging.error("Could not sample agents for roles.")
         return

    logging.info(f"Starting game between {agent1.agent_id} and {agent2.agent_id}. Proposer: {scenario_proposer.agent_id}")

    winner, loser = None, None
    game_outcome = 'tie'
    scenario_data = None
    transcript = []
    game_history_entry = {
        'opponent_id': opponent.agent_id if scenario_proposer == agent1 else scenario_proposer.agent_id,
        'role': 'proposer' if scenario_proposer == agent1 else 'opponent',
        'scenario': None,
        'transcript': transcript,
        'bets': None,
        'outcome': 'error', 
        'wealth_change': 0.0
    }
    bet1, bet2 = 0.0, 0.0 

    try:
        scenario_data = generate_scenario(scenario_proposer, config)
        if not isinstance(scenario_data, dict) or 'scenario_text' not in scenario_data or 'role_assignment' not in scenario_data:
            logging.warning(f"Scenario generation failed or returned invalid data for proposer {scenario_proposer.agent_id}. Proposer loses.")
            winner, loser = opponent, scenario_proposer
            game_outcome = 'loss' if scenario_proposer == agent1 else 'win'
            bet1, bet2 = _determine_bets(agent1, agent2, config)
            game_history_entry['bets'] = {'agent1_bet': bet1, 'agent2_bet': bet2}
            winner_bet = bet2 if winner == agent2 else bet1 
            loser_bet = bet1 if loser == agent1 else bet2 
            gain, loss = _apply_wealth_change(winner, loser, winner_bet, loser_bet, max_loss_mult, max_gain_ratio)
            game_history_entry['outcome'] = game_outcome
            game_history_entry['wealth_change'] = -loss if loser == agent1 else gain 

        else:
            logging.debug(f"Scenario generated by {scenario_proposer.agent_id}")
            game_history_entry['scenario'] = scenario_data.get('scenario_text', 'Error retrieving text')

            agent1_role = scenario_data['role_assignment'].get(agent1.agent_id)
            agent2_role = scenario_data['role_assignment'].get(agent2.agent_id)
            
            if not agent1_role or not agent2_role:
                 logging.error(f"Failed to assign roles correctly for game involving {agent1.agent_id} and {agent2.agent_id}. Defaulting to Tie.")
                 game_outcome = 'tie'
            else:
                current_player_index = random.choice([0, 1])
                players = [agent1, agent2]
                roles = [agent1_role, agent2_role]

                for turn in range(interaction_turns * 2):
                    current_agent = players[current_player_index]
                    current_role = roles[current_player_index]

                    logging.debug(f"Turn {turn+1}: Agent {current_agent.agent_id} ({current_role})")

                    response_text = generate_agent_response(current_agent, scenario_data, transcript, current_role, config)

                    if response_text is None or not isinstance(response_text, str):
                        logging.warning(f"Agent {current_agent.agent_id} failed to generate a valid response. Turn skipped.")
                    else:
                        transcript.append({"role": current_role, "content": response_text})
                        logging.debug(f"Agent {current_agent.agent_id} response captured.")

                    current_player_index = 1 - current_player_index

                adjudication_result = adjudicate_interaction(scenario_data, transcript, config)

                if isinstance(adjudication_result, str) and adjudication_result in ['Role A Wins', 'Role B Wins', 'Tie']:
                    role_map = scenario_data.get('role_assignment', {})
                    role_a_agent_id = next((aid for aid, r in role_map.items() if r == 'Role A'), None)
                    role_b_agent_id = next((aid for aid, r in role_map.items() if r == 'Role B'), None)

                    if adjudication_result == 'Role A Wins' and role_a_agent_id:
                        winner = agent1 if agent1.agent_id == role_a_agent_id else agent2
                        loser = agent2 if agent1.agent_id == role_a_agent_id else agent1
                        game_outcome = 'win' if winner == agent1 else 'loss'
                    elif adjudication_result == 'Role B Wins' and role_b_agent_id:
                        winner = agent1 if agent1.agent_id == role_b_agent_id else agent2
                        loser = agent2 if agent1.agent_id == role_b_agent_id else agent1
                        game_outcome = 'win' if winner == agent1 else 'loss'
                    else: 
                        winner, loser = None, None
                        game_outcome = 'tie'

                    logging.info(f"Adjudication result: {adjudication_result}. Game outcome for agent {agent1.agent_id}: {game_outcome}")

                else:
                    logging.warning(f"Invalid or non-standard adjudication result received: '{adjudication_result}'. Game defaulted to Tie.")
                    winner, loser = None, None
                    game_outcome = 'tie'
            
            if game_history_entry['outcome'] != 'error' and game_history_entry.get('bets') is None:
                 bet1, bet2 = _determine_bets(agent1, agent2, config)
                 game_history_entry['bets'] = {'agent1_bet': bet1, 'agent2_bet': bet2}


            if winner and loser:
                winner_bet = bet1 if winner == agent1 else bet2
                loser_bet = bet2 if loser == agent1 else bet1
                gain, loss = _apply_wealth_change(winner, loser, winner_bet, loser_bet, max_loss_mult, max_gain_ratio)
                game_history_entry['outcome'] = game_outcome
                game_history_entry['wealth_change'] = gain if winner == agent1 else -loss
            else:
                 game_history_entry['outcome'] = 'tie' 
                 game_history_entry['wealth_change'] = 0.0
                 if game_outcome != 'tie': 
                     logging.warning(f"Game outcome was '{game_outcome}' but no winner/loser identified. Setting wealth change to 0.")

    except ImportError as e:
         logging.critical(f"ImportError in game engine: {e}. Check interface functions.", exc_info=True)
         game_history_entry['outcome'] = 'error'
         game_history_entry['wealth_change'] = 0.0
         raise 
    except Exception as e:
        logging.error(f"Unhandled error during single game between {agent1.agent_id} and {agent2.agent_id}: {e}", exc_info=True)
        game_history_entry['outcome'] = 'error'
        game_history_entry['wealth_change'] = 0.0

    finally:
        try:
            agent1.add_game_result(game_history_entry)
            agent2_history_entry = copy.deepcopy(game_history_entry)
            agent2_history_entry['opponent_id'] = agent1.agent_id
            agent2_history_entry['role'] = 'opponent' if game_history_entry['role'] == 'proposer' else 'proposer'
            if game_history_entry['outcome'] == 'win': agent2_history_entry['outcome'] = 'loss'
            elif game_history_entry['outcome'] == 'loss': agent2_history_entry['outcome'] = 'win'
            elif game_history_entry['outcome'] == 'tie': agent2_history_entry['outcome'] = 'tie'
            else: agent2_history_entry['outcome'] = 'error' 
            agent2_history_entry['wealth_change'] = -game_history_entry['wealth_change']
            agent2.add_game_result(agent2_history_entry)
        except AttributeError:
             logging.error(f"Failed to add game result to agent history for {agent1.agent_id} or {agent2.agent_id}.")
        except Exception as e:
             logging.error(f"Unexpected error saving game history: {e}", exc_info=True)


def run_game_round(population: list[Agent], config: dict) -> list[Agent]:
    if not isinstance(population, list):
        logging.error("run_game_round received non-list population.")
        return [] 
    if not population:
        logging.warning("run_game_round called with empty population.")
        return []
    if not isinstance(config, dict):
        logging.error("run_game_round received non-dict config.")
        return population 

    round_config = config.get('round', {})
    games_per_agent = round_config.get('games_per_agent', 3)
    pairing_strategy = round_config.get('pairing_strategy', 'random_shuffle')

    num_agents = len(population)
    agent_indices = list(range(num_agents))
    games_played = {agent.agent_id: 0 for agent in population}

    agent_config = config.get('agent', {})
    initial_wealth = float(agent_config.get('initial_wealth', 30.0))
    for agent in population:
        try:
            agent.reset_round_state(initial_wealth)
        except AttributeError:
             logging.error(f"Agent {getattr(agent, 'agent_id', 'Unknown')} missing reset_round_state method.")
             raise TypeError(f"Agent object {getattr(agent, 'agent_id', 'Unknown')} does not support round reset.")


    round_games_count = 0
    expected_total_games = (num_agents * games_per_agent) 

    if pairing_strategy == 'random_shuffle':
        if num_agents < 2:
            logging.warning("Not enough agents for pairing. Skipping round.")
            return population

        matchups_needed = (expected_total_games + 1) // 2 
        max_iterations = matchups_needed * 3 

        for i in range(max_iterations):
            random.shuffle(agent_indices)
            logging.debug(f"Round pairing iteration {i+1}")
            iteration_played_count = 0

            for j in range(0, num_agents - 1, 2):
                idx1, idx2 = agent_indices[j], agent_indices[j+1]
                
                if idx1 >= num_agents or idx2 >= num_agents:
                     logging.error(f"Invalid agent index generated during pairing: {idx1}, {idx2}")
                     continue

                agent1 = population[idx1]
                agent2 = population[idx2]

                if games_played.get(agent1.agent_id, 0) < games_per_agent and games_played.get(agent2.agent_id, 0) < games_per_agent:
                    logging.info(f"Running game {round_games_count + 1}: {agent1.agent_id} vs {agent2.agent_id}")
                    _play_single_game(agent1, agent2, config)
                    games_played[agent1.agent_id] = games_played.get(agent1.agent_id, 0) + 1
                    games_played[agent2.agent_id] = games_played.get(agent2.agent_id, 0) + 1
                    round_games_count += 1
                    iteration_played_count += 1

            all_agents_met_target = all(count >= games_per_agent for count in games_played.values())
            if all_agents_met_target:
                logging.info(f"All agents appear to have met target of {games_per_agent} games. Ending round.")
                break
                
            if i == max_iterations - 1:
                 logging.warning(f"Reached max pairing iterations ({max_iterations}). Round may not have completed perfectly for all agents.")

        logging.info(f"Round finished. Total games played: {round_games_count}.")
        final_game_counts = [games_played.get(agent.agent_id, 0) for agent in population]
        logging.info(f"Final games per agent counts: {final_game_counts}")
        if not all(count >= games_per_agent for count in final_game_counts):
             logging.warning(f"Target games per agent ({games_per_agent}) may not have been met for all agents due to pairing variation.")

    else:
        logging.error(f"Unsupported pairing strategy specified in config: {pairing_strategy}")
        raise NotImplementedError(f"Pairing strategy '{pairing_strategy}' is not implemented.")

    return population
