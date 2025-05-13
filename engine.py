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
import copy # math module was imported but not used.
from manager import Agent
from interface import generate_scenario, generate_agent_response, adjudicate_interaction

def _apply_wealth_change(winner: Agent, loser: Agent, winner_bet: float, loser_bet: float, max_loss_multiplier: float, max_gain_ratio: float) -> tuple[float, float]:
    """
    Applies wealth changes to winner and loser based on bets and game parameters.

    Args:
        winner: The winning Agent object.
        loser: The losing Agent object.
        winner_bet: The amount bet by the winner (used for logging, not directly for gain calculation cap here).
        loser_bet: The amount bet by the loser (used for gain calculation cap for winner).
        max_loss_multiplier: Multiplier for the loser's bet to determine max potential loss.
        max_gain_ratio: Max ratio of winner's initial wealth they can have after winning (e.g., 2.0 for doubling).
                       A ratio of 2.0 means the winner's wealth can at most double (gain = current wealth).

    Returns:
        A tuple (actual_gain_for_winner, actual_loss_for_loser).
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
    if max_gain_ratio < 1.0: # Winning implies gain, so ratio must allow for at least maintaining wealth.
         logging.error(f"_apply_wealth_change: Invalid max_gain_ratio ({max_gain_ratio}). Must be >= 1.0.")
         return 0.0, 0.0

    # Calculate loser's actual loss
    # Loser cannot lose more than their current wealth or their bet * multiplier.
    potential_loss_for_loser = loser_bet * max_loss_multiplier
    actual_loss_for_loser = min(loser.wealth, potential_loss_for_loser)
    actual_loss_for_loser = max(0.0, actual_loss_for_loser) # loss is not negative (e.g. if wealth was 0)

    # Calculate winner's actual gain
    # 1. Gain is fundamentally limited by what the loser actually lost.
    gain_from_loser = actual_loss_for_loser
    
    # 2. As per paper: "winner can only gain up to the loser's bet" (original bet, not multiplied loss).
    gain_capped_by_loser_original_bet = min(gain_from_loser, loser_bet)

    # 3. As per paper: "winner cannot more than double their own money".
    # If max_gain_ratio is 2.0, max_gain_amount is winner.wealth * (2.0 - 1.0) = winner.wealth.
    # New wealth will be winner.wealth + winner.wealth = 2 * winner.wealth.
    max_gain_amount_for_winner = winner.wealth * (max_gain_ratio - 1.0)
    if max_gain_amount_for_winner < 0: # Could happen if winner.wealth is negative (though unlikely with current rules) or max_gain_ratio < 1
        max_gain_amount_for_winner = 0.0

    actual_gain_for_winner = min(gain_capped_by_loser_original_bet, max_gain_amount_for_winner)
    actual_gain_for_winner = max(0.0, actual_gain_for_winner) # gain is not negative

    try:
        winner.wealth += actual_gain_for_winner
        loser.wealth -= actual_loss_for_loser
    except AttributeError: # Should not happen if type checks pass
        logging.error(f"AttributeError updating wealth for {getattr(winner, 'agent_id', 'Unknown Winner')} or {getattr(loser, 'agent_id', 'Unknown Loser')}.")
        return 0.0, 0.0 # Return zero change if update fails

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
    # Example for a future strategy:
    # elif strategy == 'percentage_wealth':
    #     percentage = betting_config.get('percentage_amount', 0.1) # e.g., 10%
    #     bet1_val = agent1.wealth * percentage
    #     bet2_val = agent2.wealth * percentage
    else:
        logging.warning(f"Unsupported betting strategy '{strategy}' in config. Defaulting to fixed amount: {fixed_amount}.")
        bet1_val = fixed_amount
        bet2_val = fixed_amount

    # Apply cap: bet cannot exceed max_bet_ratio of current wealth
    bet1_val = min(bet1_val, agent1.wealth * max_bet_ratio)
    bet2_val = min(bet2_val, agent2.wealth * max_bet_ratio)
    
    # Apply floor: bet must be at least min_bet (but not more than current wealth)
    bet1_val = max(min_bet, bet1_val)
    bet1_val = min(agent1.wealth, bet1_val) # Cannot bet more than available wealth

    bet2_val = max(min_bet, bet2_val)
    bet2_val = min(agent2.wealth, bet2_val) # Cannot bet more than available wealth
    
    # Final check: bets are non-negative
    bet1_val = max(0.0, bet1_val)
    bet2_val = max(0.0, bet2_val)

    logging.debug(f"Agent {agent1.agent_id} (wealth {agent1.wealth:.2f}) bets {bet1_val:.2f}. Agent {agent2.agent_id} (wealth {agent2.wealth:.2f}) bets {bet2_val:.2f}.")
    return bet1_val, bet2_val

def _play_single_game(agent1: Agent, agent2: Agent, config: dict) -> None:
    """
    Manages a single game interaction between two agents (agent1 and agent2).
    Updates agent wealth and game history for both participating agents.
    """
    if not all(isinstance(agent, Agent) for agent in [agent1, agent2]):
        logging.error("Invalid Agent objects provided to _play_single_game.")
        return
    if not isinstance(config, dict):
        logging.error("Invalid config provided to _play_single_game.")
        return

    game_config = config.get('game', {})
    interaction_turns = game_config.get('interaction_turns_per_agent', 3) # Total turns = this * 2
    
    betting_conf = config.get('game', {}).get('betting', {})
    max_loss_mult = float(betting_conf.get('max_loss_multiplier', 1.0))
    max_gain_ratio_val = float(betting_conf.get('max_gain_ratio', 2.0))

    # Randomly assign one agent as the scenario proposer
    # The other agent is the opponent in the proposal phase
    # These are agent1 and agent2, just in a specific order for this game step
    proposer_obj, opponent_obj = random.sample([agent1, agent2], 2)

    logging.info(f"Starting game. Proposer: {proposer_obj.agent_id}, Opponent: {opponent_obj.agent_id}. (Full pair: {agent1.agent_id} vs {agent2.agent_id})")

    scenario_data_dict = None
    transcript_list = []
    bet_agent1, bet_agent2 = 0.0, 0.0 # Bets for the specific agents agent1 and agent2
    
    # This history entry is built from the perspective of agent1
    # It will be deepcopied and adapted for agent2 later.
    hist_entry_for_agent1 = {
        'opponent_id': agent2.agent_id, # agent1's opponent is agent2
        'role_in_proposal': 'proposer' if agent1 == proposer_obj else 'opponent',
        'role_in_game': None, # 'Role A' or 'Role B' for agent1
        'scenario_text': "Scenario not generated or error.", # Default
        'transcript': transcript_list, # Appended to directly
        'bets': None, # Dict: {'agent1_bet': bet_agent1, 'agent2_bet': bet_agent2}
        'outcome': 'error', # For agent1
        'wealth_change': 0.0  # For agent1
    }

    actual_winner_agent: Agent | None = None
    actual_loser_agent: Agent | None = None

    try:
        # 1. Scenario Generation by proposer_obj
        scenario_data_dict = generate_scenario(proposer_obj, config)

        if not isinstance(scenario_data_dict, dict) or \
           'scenario_text' not in scenario_data_dict or \
           not isinstance(scenario_data_dict.get('scenario_text'), str) or \
           'role_assignment' not in scenario_data_dict or \
           not isinstance(scenario_data_dict.get('role_assignment'), dict) or \
           proposer_obj.agent_id not in scenario_data_dict['role_assignment']:
            
            logging.warning(f"Scenario generation failed or returned invalid data for proposer {proposer_obj.agent_id}. Proposer ({proposer_obj.agent_id}) loses by default.")
            actual_winner_agent, actual_loser_agent = opponent_obj, proposer_obj
            hist_entry_for_agent1['scenario_text'] = "Scenario generation failed or invalid."
        else:
            # Scenario generation successful
            hist_entry_for_agent1['scenario_text'] = scenario_data_dict['scenario_text']
            logging.debug(f"Scenario generated by {proposer_obj.agent_id}: '{scenario_data_dict['scenario_text'][:100]}...'")

            # *** CRITICAL FIX FOR ROLE ASSIGNMENT ***
            # scenario_data_dict['role_assignment'] initially only has proposer's role.
            proposer_assigned_game_role = scenario_data_dict['role_assignment'][proposer_obj.agent_id]
            opponent_assigned_game_role = 'Role B' if proposer_assigned_game_role == 'Role A' else 'Role A'
            scenario_data_dict['role_assignment'][opponent_obj.agent_id] = opponent_assigned_game_role
            # Now role_assignment contains entries for both proposer_obj and opponent_obj

            # Determine game roles for agent1 and agent2 (the specific pair passed to this function)
            agent1_game_role = scenario_data_dict['role_assignment'].get(agent1.agent_id)
            agent2_game_role = scenario_data_dict['role_assignment'].get(agent2.agent_id)
            hist_entry_for_agent1['role_in_game'] = agent1_game_role

            if not agent1_game_role or not agent2_game_role:
                 logging.error(f"Failed to assign game roles correctly for game involving {agent1.agent_id} and {agent2.agent_id} despite scenario success. Defaulting to Tie.")
                 # Outcome remains 'error' or becomes 'tie' based on later logic for wealth_change
            else:
                logging.info(f"Game roles: {agent1.agent_id} is {agent1_game_role}, {agent2.agent_id} is {agent2_game_role}")

                # 2. Interaction Phase
                # Determine who starts the interaction: random choice for now
                interaction_players = [agent1, agent2] # Use the specific agents for this game
                random.shuffle(interaction_players) 
                
                for turn_idx in range(interaction_turns * 2): # Each agent gets `interaction_turns`
                    current_turn_agent = interaction_players[turn_idx % 2]
                    current_turn_agent_game_role = scenario_data_dict['role_assignment'][current_turn_agent.agent_id]

                    logging.debug(f"Turn {turn_idx + 1}: Agent {current_turn_agent.agent_id} ({current_turn_agent_game_role}) is responding.")
                    response_txt = generate_agent_response(current_turn_agent, scenario_data_dict, transcript_list, current_turn_agent_game_role, config)

                    if response_txt is None or not isinstance(response_txt, str) or not response_txt.strip():
                        logging.warning(f"Agent {current_turn_agent.agent_id} failed to generate a valid/non-empty response. Turn content will be marked.")
                        transcript_list.append({"role": current_turn_agent_game_role, "content": "[Agent failed to provide a response]"})
                    else:
                        transcript_list.append({"role": current_turn_agent_game_role, "content": response_txt})
                        logging.debug(f"Agent {current_turn_agent.agent_id} response captured: '{response_txt[:100]}...'")
                
                # 3. Adjudication
                adjudication_txt = adjudicate_interaction(scenario_data_dict, transcript_list, config)
                
                if isinstance(adjudication_txt, str) and adjudication_txt in ['Role A Wins', 'Role B Wins', 'Tie']:
                    role_map = scenario_data_dict['role_assignment']
                    
                    # Find which Agent object was Role A and Role B
                    agent_is_role_A = None
                    agent_is_role_B = None
                    for agent_id_in_map, game_r in role_map.items():
                        current_agent_obj = agent1 if agent1.agent_id == agent_id_in_map else agent2
                        if game_r == 'Role A': agent_is_role_A = current_agent_obj
                        elif game_r == 'Role B': agent_is_role_B = current_agent_obj
                    
                    if adjudication_txt == 'Role A Wins' and agent_is_role_A:
                        actual_winner_agent, actual_loser_agent = agent_is_role_A, agent_is_role_B
                    elif adjudication_txt == 'Role B Wins' and agent_is_role_B:
                        actual_winner_agent, actual_loser_agent = agent_is_role_B, agent_is_role_A
                    # If 'Tie', actual_winner_agent and actual_loser_agent remain None
                    logging.info(f"Adjudication result: {adjudication_txt}.")
                else: 
                    logging.warning(f"Invalid or non-standard adjudication result: '{adjudication_txt}'. Game defaulted to Tie.")
                    # actual_winner_agent and actual_loser_agent remain None, leading to a tie.

        # This block executes if scenario generation failed OR if it succeeded.
        # 4. Determine Bets (if not already set due to early exit)
        # Bets are determined regardless of scenario success for record keeping.
        bet_agent1, bet_agent2 = _determine_bets(agent1, agent2, config)
        hist_entry_for_agent1['bets'] = {'agent1_bet': bet_agent1, 'agent2_bet': bet_agent2}

        # 5. Determine Final Outcome for agent1 and Apply Wealth Change
        if actual_winner_agent and actual_loser_agent: # A clear winner and loser
            hist_entry_for_agent1['outcome'] = 'win' if agent1 == actual_winner_agent else 'loss'
            
            winner_bet_val = bet_agent1 if actual_winner_agent == agent1 else bet_agent2
            loser_bet_val = bet_agent1 if actual_loser_agent == agent1 else bet_agent2

            gain, loss = _apply_wealth_change(actual_winner_agent, actual_loser_agent, winner_bet_val, loser_bet_val, max_loss_mult, max_gain_ratio_val)
            hist_entry_for_agent1['wealth_change'] = gain if agent1 == actual_winner_agent else -loss
        else: # Tie (or error that resolved to tie)
            hist_entry_for_agent1['outcome'] = 'tie'
            hist_entry_for_agent1['wealth_change'] = 0.0
        
        logging.info(f"Game outcome for {agent1.agent_id}: {hist_entry_for_agent1['outcome']}, wealth change: {hist_entry_for_agent1['wealth_change']:.2f}")

    except Exception as e: # Catch-all for unexpected issues during game play
        logging.critical(f"Critical unhandled error during single game logic between {agent1.agent_id} and {agent2.agent_id}: {e}", exc_info=True)
        hist_entry_for_agent1['outcome'] = 'error' # Mark as error for agent1
        hist_entry_for_agent1['wealth_change'] = 0.0
    finally:
        # 6. Record Game History for both agents
        try:
            agent1.add_game_result(copy.deepcopy(hist_entry_for_agent1)) # hist_entry_for_agent1 is already from agent1's perspective

            hist_entry_for_agent2 = copy.deepcopy(hist_entry_for_agent1)
            hist_entry_for_agent2['opponent_id'] = agent1.agent_id
            hist_entry_for_agent2['role_in_proposal'] = 'proposer' if agent2 == proposer_obj else 'opponent'
            if scenario_data_dict and 'role_assignment' in scenario_data_dict: # scenario_data_dict is available
                hist_entry_for_agent2['role_in_game'] = scenario_data_dict['role_assignment'].get(agent2.agent_id)
            else:
                hist_entry_for_agent2['role_in_game'] = None # Or 'unknown' if scenario failed early

            if hist_entry_for_agent1['outcome'] == 'win': hist_entry_for_agent2['outcome'] = 'loss'
            elif hist_entry_for_agent1['outcome'] == 'loss': hist_entry_for_agent2['outcome'] = 'win'
            else: hist_entry_for_agent2['outcome'] = hist_entry_for_agent1['outcome'] # tie or error
            
            hist_entry_for_agent2['wealth_change'] = -hist_entry_for_agent1['wealth_change']
            # Bets dict {'agent1_bet': ..., 'agent2_bet': ...} remains consistent, referring to agent1 and agent2 objects.
            
            agent2.add_game_result(hist_entry_for_agent2)
        except AttributeError as e: # If Agent objects are malformed
             logging.error(f"AttributeError adding game result to agent history ({agent1.agent_id} or {agent2.agent_id}): {e}")
        except Exception as e: # Other errors during history saving
             logging.error(f"Unexpected error saving game history for agents {agent1.agent_id}, {agent2.agent_id}: {e}", exc_info=True)

def run_game_round(population: list[Agent], config: dict) -> list[Agent]:
    """
    Manages a round of games for the entire population.
    Agents are paired, play games, and their states (wealth, history) are updated.
    """
    if not isinstance(population, list):
        logging.error("run_game_round: Population must be a list. Returning empty list.")
        return [] 
    if not all(isinstance(agent, Agent) for agent in population):
        logging.error("run_game_round: Population list contains non-Agent elements. Behavior undefined. Returning original list.")
        return population 
    if not population:
        logging.warning("run_game_round: Called with empty population. Returning empty list.")
        return []
    if not isinstance(config, dict):
        logging.error("run_game_round: Config must be a dictionary. Returning original population.")
        return population 

    # Use a more specific config key if available, or a sensible default.
    # Paper states "exactly three games". Let's make this configurable via 'simulation.games_per_agent_in_round'.
    games_per_agent_target = int(config.get('simulation', {}).get('games_per_agent_in_round', 3))
    
    # Pairing strategy: currently only 'random_shuffle' is directly handled below.
    # config.yaml does not specify 'round:pairing_strategy', so this defaults to 'random_shuffle'.
    pairing_strategy = config.get('round', {}).get('pairing_strategy', 'random_shuffle')

    num_agents = len(population)
    
    agent_conf = config.get('agent', {})
    initial_wealth_val = float(agent_conf.get('initial_wealth', 30.0))

    # Reset agent states for the round
    for current_agent in population:
        try:
            current_agent.reset_round_state(initial_wealth_val)
        except AttributeError:
             logging.critical(f"Agent {getattr(current_agent, 'agent_id', 'Unknown')} missing reset_round_state method. Agent class implementation error.")
             raise # This is a critical error indicating wrong Agent structure.

    if num_agents < 2:
        logging.warning("Not enough agents (<2) for pairing. Skipping game round.")
        return population

    if pairing_strategy == 'random_shuffle':
        # This strategy attempts to each agent plays `games_per_agent_target` games.
        # Exactness can vary, especially with odd numbers or few iterations.
        games_played_this_round = {agent.agent_id: 0 for agent in population}
        
        total_participations_needed = num_agents * games_per_agent_target
        if total_participations_needed % 2 != 0:
            logging.warning(f"Target total participations ({total_participations_needed}) is odd. This may lead to unequal game counts.")
        
        # Number of unique game instances to be played.
        num_total_games_to_schedule = total_participations_needed // 2

        # Heuristic for max iterations to find pairings.
        # Can be tuned based on population size and games_per_agent_target.
        max_iters = num_total_games_to_schedule * 3 + num_agents # enough chances
        if num_agents <= 3: max_iters = games_per_agent_target * num_agents * 2


        actual_games_played_in_round = 0
        for iter_num in range(max_iters):
            # Check if all agents have met their target number of games
            if all(count >= games_per_agent_target for count in games_played_this_round.values()):
                logging.info(f"All agents appear to have met target of {games_per_agent_target} games by iteration {iter_num + 1}. Ending pairing phase.")
                break

            shuffled_indices = list(range(num_agents))
            random.shuffle(shuffled_indices)
            
            # Track who has been paired in this specific pass to avoid re-pairing immediately
            was_paired_in_pass = [False] * num_agents

            for i in range(0, num_agents - (num_agents % 2), 2): # Iterate through pairs
                idx1, idx2 = shuffled_indices[i], shuffled_indices[i+1]

                # Skip if either agent in this potential pair was already paired in this pass
                if was_paired_in_pass[idx1] or was_paired_in_pass[idx2]:
                    continue 

                agent_A, agent_B = population[idx1], population[idx2]

                # Check if both agents still need to play more games this round
                if games_played_this_round.get(agent_A.agent_id, 0) < games_per_agent_target and \
                   games_played_this_round.get(agent_B.agent_id, 0) < games_per_agent_target:
                    
                    logging.info(f"Pairing Iteration {iter_num + 1}, Game {actual_games_played_in_round + 1}: {agent_A.agent_id} vs {agent_B.agent_id}")
                    _play_single_game(agent_A, agent_B, config)
                    
                    games_played_this_round[agent_A.agent_id] += 1
                    games_played_this_round[agent_B.agent_id] += 1
                    was_paired_in_pass[idx1] = True
                    was_paired_in_pass[idx2] = True
                    actual_games_played_in_round += 1
            
            if iter_num == max_iters - 1:
                 logging.warning(f"Reached max pairing iterations ({max_iters}). Game counts per agent may not be perfectly uniform at the target of {games_per_agent_target}.")

        logging.info(f"Game round finished. Total distinct games played: {actual_games_played_in_round}.")
        for p_agent in population:
            final_count = games_played_this_round.get(p_agent.agent_id, 0)
            logging.info(f"Agent {p_agent.agent_id} played {final_count} games this round (target: {games_per_agent_target}).")
        
        # Final check and warning if targets weren't met for some.
        agents_below_target = [aid for aid, count in games_played_this_round.items() if count < games_per_agent_target]
        if agents_below_target:
             logging.warning(f"{len(agents_below_target)} agent(s) did not meet the target of {games_per_agent_target} games: {agents_below_target}")

    else:
        logging.error(f"Unsupported pairing strategy specified in config: '{pairing_strategy}'")
        raise NotImplementedError(f"Pairing strategy '{pairing_strategy}' is not implemented.")

    return population
