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
from pathlib import Path # For path manipulation
import json # For writing JSON
import asyncio
from typing import Coroutine, Any
from manager import Agent
from interface import generate_scenario, generate_agent_response, adjudicate_interaction

def _apply_wealth_change(winner: Agent, loser: Agent, winner_bet: float, loser_bet: float, max_loss_multiplier: float, max_gain_ratio: float) -> tuple[float, float]:
    """
    Applies wealth changes to winner and loser based on bets and game parameters.
    Returns: A tuple (actual_gain_for_winner, actual_loss_for_loser).
    """
    if not all(isinstance(agent, Agent) for agent in [winner, loser]):
        logging.error("_apply_wealth_change: Invalid Agent objects provided (winner or loser is not an Agent instance).")
        return 0.0, 0.0
    if not all(isinstance(val, (int, float)) for val in [winner_bet, loser_bet, max_loss_multiplier, max_gain_ratio]):
        logging.error(f"_apply_wealth_change: Invalid numerical arguments provided. Types: wb={type(winner_bet)}, lb={type(loser_bet)}, mlm={type(max_loss_multiplier)}, mgr={type(max_gain_ratio)}")
        return 0.0, 0.0
    if max_loss_multiplier < 0:
        logging.error(f"_apply_wealth_change: Invalid max_loss_multiplier ({max_loss_multiplier}). Must be non-negative.")
        return 0.0, 0.0
    if max_gain_ratio < 1.0: # Max gain ratio implies multiplier, e.g., 2.0 means up to 100% gain of own wealth
        logging.error(f"_apply_wealth_change: Invalid max_gain_ratio ({max_gain_ratio}). Must be >= 1.0.")
        return 0.0, 0.0

    # Calculate actual loss for the loser, capped by their current wealth and the bet amount
    potential_loss_for_loser = loser_bet * max_loss_multiplier
    actual_loss_for_loser = min(loser.wealth, potential_loss_for_loser)
    actual_loss_for_loser = max(0.0, actual_loss_for_loser) # loss is not negative

    # Calculate actual gain for the winner
    # Winner can gain at most what the loser actually lost from their bet,
    # AND winner's gain is capped by a ratio of their own wealth (max_gain_ratio).
    # The amount transferred is the smaller of what loser lost (from their bet) and what winner is allowed to gain.
    
    # Max winner can take from loser is what loser actually lost (capped by loser's wealth)
    # AND this amount cannot exceed what the loser bet (ensures winner doesn't get more than loser risked * multiplier if loser had less wealth)
    gain_from_loser_bet = min(actual_loss_for_loser, loser_bet)

    # Max winner can gain based on their own wealth
    # Example: if max_gain_ratio is 2.0, winner can gain up to 1.0 * their current wealth.
    max_potential_gain_for_winner = winner.wealth * (max_gain_ratio - 1.0)
    if max_potential_gain_for_winner < 0: # Should not happen if max_gain_ratio >= 1.0
        max_potential_gain_for_winner = 0.0

    actual_gain_for_winner = min(gain_from_loser_bet, max_potential_gain_for_winner)
    actual_gain_for_winner = max(0.0, actual_gain_for_winner) # gain is not negative

    try:
        winner.wealth += actual_gain_for_winner
        loser.wealth -= actual_loss_for_loser
    except AttributeError as e:
        logging.error(f"AttributeError updating wealth for {getattr(winner, 'agent_id', 'Unknown Winner')} or {getattr(loser, 'agent_id', 'Unknown Loser')}: {e}")
        return 0.0, 0.0 # No wealth change if attribute error

    logging.debug(
        f"Wealth change: Winner {getattr(winner, 'agent_id', 'N/A')} (bet {winner_bet:.2f}) gains {actual_gain_for_winner:.2f} (new wealth: {winner.wealth:.2f}). "
        f"Loser {getattr(loser, 'agent_id', 'N/A')} (bet {loser_bet:.2f}) loses {actual_loss_for_loser:.2f} (new wealth: {loser.wealth:.2f})."
    )
    return actual_gain_for_winner, actual_loss_for_loser

def _determine_bets(agent1: Agent, agent2: Agent, config: dict) -> tuple[float, float]:
    """
    Determines the bet amounts for agent1 and agent2 based on the configuration.
    Ensures bets are non-negative and do not exceed agent's wealth or max_bet_ratio.
    """
    if not all(isinstance(agent, Agent) for agent in [agent1, agent2]):
        logging.error("_determine_bets: Invalid Agent objects provided.")
        return 0.0, 0.0
    if not isinstance(config, dict):
        logging.error("_determine_bets: Invalid config provided (must be a dictionary).")
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
    else: # Add other strategies here if needed
        logging.warning(f"Unsupported betting strategy '{strategy}' in config. Defaulting to fixed amount: {fixed_amount}.")
        bet1_val = fixed_amount
        bet2_val = fixed_amount

    # Apply constraints: bet <= wealth * max_bet_ratio, bet >= min_bet, bet <= wealth, bet >= 0
    bet1_val = min(bet1_val, agent1.wealth * max_bet_ratio)
    bet1_val = max(min_bet, bet1_val)
    bet1_val = min(agent1.wealth, bet1_val) # Cannot bet more than current wealth
    bet1_val = max(0.0, bet1_val)           # Bet cannot be negative

    bet2_val = min(bet2_val, agent2.wealth * max_bet_ratio)
    bet2_val = max(min_bet, bet2_val)
    bet2_val = min(agent2.wealth, bet2_val) # Cannot bet more than current wealth
    bet2_val = max(0.0, bet2_val)           # Bet cannot be negative

    logging.debug(f"Agent {agent1.agent_id} (wealth {agent1.wealth:.2f}) bets {bet1_val:.2f}. Agent {agent2.agent_id} (wealth {agent2.wealth:.2f}) bets {bet2_val:.2f}.")
    return bet1_val, bet2_val


async def _play_single_game(agent1: Agent, agent2: Agent, config: dict, run_id: str, generation_number: int, game_index_in_round: int) -> dict:
    """
    Manages a single game interaction between two agents.
    Updates agent wealth and internal history.
    Returns a dictionary with comprehensive details of the game for external logging/dashboarding.
    """
    if not all(isinstance(agent, Agent) for agent in [agent1, agent2]):
        logging.error("Invalid Agent objects provided to _play_single_game. Both must be Agent instances.")
        return {"game_id": f"error_invalid_agents_{uuid.uuid4().hex[:4]}", "adjudication_result": "Critical Game Error: Invalid Agents", "defaulted_to_tie_reason": "Invalid agents"}
    if not isinstance(config, dict):
        logging.error("Invalid config provided to _play_single_game (must be a dictionary).")
        return {"game_id": f"error_invalid_config_{uuid.uuid4().hex[:4]}", "adjudication_result": "Critical Game Error: Invalid Config", "defaulted_to_tie_reason": "Invalid config"}

    game_id = f"{run_id}_gen{generation_number}_game{game_index_in_round}_{uuid.uuid4().hex[:8]}"
    timestamp_start = datetime.datetime.now().isoformat()

    game_details_dict = {
        "game_id": game_id,
        "run_id": run_id,
        "generation_number": generation_number,
        "timestamp_start": timestamp_start,
        "timestamp_end": None,
        "proposer_agent_id": None,
        "opponent_in_proposal_agent_id": None,
        "player_A_id": agent1.agent_id,
        "player_B_id": agent2.agent_id,
        "player_A_game_role": None,
        "player_B_game_role": None,
        "scenario_text": "Error: Scenario not processed due to internal error.",
        "scenario_generation_prompt": None,
        "scenario_raw_llm_output": None,
        "scenario_generation_successful": False,
        "transcript": [],
        "adjudication_prompt_scratchpad": None,
        "adjudication_raw_llm_output_scratchpad": None,
        "adjudication_scratchpad": None,
        "adjudication_prompt_outcome_ids": None,
        "adjudication_raw_llm_output_outcome_ids": None,
        "adjudication_result": "Error: Adjudication not run or failed.",
        "adjudication_win_message_id": None,
        "adjudication_lose_message_id": None,
        "betting_details": None,
        "wealth_changes": None,
        "final_player_A_wealth": agent1.wealth,
        "final_player_B_wealth": agent2.wealth,
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

    logging.info(f"Game {game_id}: Starting. Proposer: {proposer_obj.agent_id}, Opponent: {opponent_obj.agent_id}. (Full pair: {agent1.agent_id} vs {agent2.agent_id})")

    hist_entry_for_agent1 = {
        'game_id': game_id,
        'opponent_id': agent2.agent_id,
        'role_in_proposal': 'proposer' if agent1 == proposer_obj else 'opponent',
        'role_in_game': None,
        'scenario_text': game_details_dict["scenario_text"],
        'transcript_snippet': None,
        'bets': None,
        'outcome': 'error',
        'wealth_change': 0.0
    }

    actual_winner_agent: Agent | None = None
    actual_loser_agent: Agent | None = None
    adjudication_final_text = "Error: Adjudication not reached or failed"
    bet_agent1, bet_agent2 = 0.0, 0.0 # Initialize bets here

    try:
        logging.debug(f"Game {game_id}: Requesting scenario generation by {proposer_obj.agent_id}.")
        scenario_info, scenario_gen_prompt_text = await generate_scenario(proposer_obj, config) # Assuming generate_scenario is async
        game_details_dict["scenario_generation_prompt"] = scenario_gen_prompt_text

        scenario_processing_succeeded = False
        if isinstance(scenario_info, dict):
            scenario_text_val = scenario_info.get('scenario_text')
            role_assignment_val = scenario_info.get('role_assignment')
            is_scenario_text_valid = isinstance(scenario_text_val, str) and bool(scenario_text_val.strip())
            is_role_assignment_valid = isinstance(role_assignment_val, dict) and bool(role_assignment_val)
            logging.debug(f"Game {game_id}: Post-generate_scenario check: scenario_info is_dict={True}, "
                          f"scenario_text_valid='{is_scenario_text_valid}', scenario_text_content_preview='{str(scenario_text_val)[:70] if scenario_text_val else 'N/A'}...', "
                          f"role_assignment_valid='{is_role_assignment_valid}', role_assignment_content='{role_assignment_val}'.")
            scenario_processing_succeeded = is_scenario_text_valid and is_role_assignment_valid
        else:
            logging.debug(f"Game {game_id}: Post-generate_scenario check: scenario_info type is {type(scenario_info)}, not a dict.")

        if scenario_processing_succeeded:
            logging.info(f"Game {game_id}: Scenario successfully processed by engine after generation.")
            game_details_dict["scenario_text"] = scenario_info['scenario_text']
            game_details_dict["scenario_raw_llm_output"] = scenario_info.get('raw_output')
            game_details_dict["scenario_generation_successful"] = True
            hist_entry_for_agent1['scenario_text'] = scenario_info['scenario_text']

            role_map = scenario_info['role_assignment']
            proposer_actual_game_role = role_map[proposer_obj.agent_id]
            opponent_actual_game_role = 'Role B' if proposer_actual_game_role == 'Role A' else 'Role A'

            if agent1 == proposer_obj:
                game_details_dict["player_A_game_role"] = proposer_actual_game_role
                game_details_dict["player_B_game_role"] = opponent_actual_game_role
            else:
                game_details_dict["player_A_game_role"] = opponent_actual_game_role
                game_details_dict["player_B_game_role"] = proposer_actual_game_role
            
            hist_entry_for_agent1['role_in_game'] = game_details_dict["player_A_game_role"]
            logging.info(f"Game {game_id}: Roles assigned - Player A ({agent1.agent_id}) is {game_details_dict['player_A_game_role']}, Player B ({agent2.agent_id}) is {game_details_dict['player_B_game_role']}.")

            current_transcript = []
            interaction_players_ordered = [agent1, agent2]
            random.shuffle(interaction_players_ordered)

            for turn_idx in range(interaction_turns * 2):
                current_turn_agent = interaction_players_ordered[turn_idx % 2]
                current_turn_agent_game_role = game_details_dict["player_A_game_role"] if current_turn_agent == agent1 else game_details_dict["player_B_game_role"]
                
                logging.debug(f"Game {game_id} Turn {turn_idx + 1}: Agent {current_turn_agent.agent_id} ({current_turn_agent_game_role}) responding.")
        
                response_text, agent_action_llm_prompt = await generate_agent_response(
                    current_turn_agent, scenario_info, current_transcript,
                    current_turn_agent_game_role, config
                )
        
                turn_entry = {
                    "role": current_turn_agent_game_role,
                    "agent_id": current_turn_agent.agent_id,
                    "content": "[Agent failed to provide a response or response was empty/invalid]",
                    "raw_llm_prompt": agent_action_llm_prompt
                }
                if response_text is not None and isinstance(response_text, str) and response_text.strip():
                    turn_entry["content"] = response_text
                    logging.debug(f"Game {game_id}: Agent {current_turn_agent.agent_id} response: '{response_text[:100]}...'")
                else:
                    logging.warning(f"Game {game_id}: Agent {current_turn_agent.agent_id} provided no valid/non-empty response for turn {turn_idx + 1}.")
                current_transcript.append(turn_entry)
            
            game_details_dict["transcript"] = current_transcript
            hist_entry_for_agent1['transcript_snippet'] = f"{len(current_transcript)} turns, first: {current_transcript[0]['content'][:30] if current_transcript else 'N/A'}..."

            logging.debug(f"Game {game_id}: Preparing transcript with IDs for adjudication.")
            player_A_role_label = game_details_dict.get("player_A_game_role")
            player_B_role_label = game_details_dict.get("player_B_game_role")
            
            labeled_transcript_lines = []
            turn_counters = {player_A_role_label: 0, player_B_role_label: 0}

            for turn_data in current_transcript:
                role_in_turn = turn_data.get("role")
                content = turn_data.get("content", "[empty_content]")
                
                id_prefix = "U"
                if role_in_turn == player_A_role_label:
                    id_prefix = "A"
                elif role_in_turn == player_B_role_label:
                    id_prefix = "B"
                
                if role_in_turn in turn_counters: # Ensure role_in_turn is a valid key
                    turn_counters[role_in_turn] += 1
                    message_id_for_adj = f"{id_prefix}{turn_counters[role_in_turn]}"
                else: # Fallback if role_in_turn is unexpected (e.g., None)
                    turn_counters.setdefault("UnknownRole", 0)
                    turn_counters["UnknownRole"] +=1
                    message_id_for_adj = f"U{turn_counters['UnknownRole']}"

                labeled_transcript_lines.append(f"{message_id_for_adj} ({role_in_turn}): {content}")
            
            labeled_transcript_str_for_adj = "\n".join(labeled_transcript_lines)
            logging.debug(f"Game {game_id}: Labeled transcript for adjudicator:\n{labeled_transcript_str_for_adj}")

            logging.debug(f"Game {game_id}: Requesting adjudication (two calls: scratchpad, then outcome/IDs).")
            (final_parsed_outcome, win_msg_id, lose_msg_id, scratchpad_content,
             prompt_for_scratchpad, raw_output_scratchpad,
             prompt_for_outcome_ids, raw_output_outcome_ids) = await adjudicate_interaction(
                scenario_info, labeled_transcript_str_for_adj, config
            )
                  
            game_details_dict["adjudication_result"] = final_parsed_outcome 
            game_details_dict["adjudication_win_message_id"] = win_msg_id
            game_details_dict["adjudication_lose_message_id"] = lose_msg_id
            game_details_dict["adjudication_scratchpad"] = scratchpad_content
            game_details_dict["adjudication_prompt_scratchpad"] = prompt_for_scratchpad
            game_details_dict["adjudication_raw_llm_output_scratchpad"] = raw_output_scratchpad
            game_details_dict["adjudication_prompt_outcome_ids"] = prompt_for_outcome_ids
            game_details_dict["adjudication_raw_llm_output_outcome_ids"] = raw_output_outcome_ids

            adjudication_final_text = final_parsed_outcome

            if adjudication_final_text == 'Role A Wins':
                actual_winner_agent = agent1 if game_details_dict["player_A_game_role"] == 'Role A' else agent2
                actual_loser_agent = agent2 if game_details_dict["player_A_game_role"] == 'Role A' else agent1
            elif adjudication_final_text == 'Role B Wins':
                actual_winner_agent = agent1 if game_details_dict["player_A_game_role"] == 'Role B' else agent2
                actual_loser_agent = agent2 if game_details_dict["player_A_game_role"] == 'Role B' else agent1
            elif adjudication_final_text == 'Tie':
                pass
            else:
                logging.warning(f"Game {game_id}: Adjudication result was '{adjudication_final_text}'. Game defaulted to Tie.")
                game_details_dict["defaulted_to_tie_reason"] = f"Adjudication error/non-standard: {adjudication_final_text}"
        else:
            game_details_dict["scenario_text"] = "Scenario generation failed or returned invalid/incomplete data."
            game_details_dict["scenario_generation_successful"] = False
            logging.warning(f"Game {game_id}: Scenario generation determined as FAILED by engine. Proposer ({proposer_obj.agent_id}) loses by default.")
            actual_winner_agent, actual_loser_agent = opponent_obj, proposer_obj
            game_details_dict["defaulted_to_tie_reason"] = "Scenario generation failure (proposer loss by default)"
            game_details_dict["adjudication_result"] = "Proposer Loss (Scenario Gen Fail)"

        bet_agent1, bet_agent2 = _determine_bets(agent1, agent2, config)
        game_details_dict["betting_details"] = {"player_A_bet": bet_agent1, "player_B_bet": bet_agent2}
        hist_entry_for_agent1['bets'] = {'agent1_bet': bet_agent1, 'agent2_bet': bet_agent2}

        wealth_A_change, wealth_B_change = 0.0, 0.0
        if actual_winner_agent and actual_loser_agent:
            winner_bet_val = bet_agent1 if actual_winner_agent == agent1 else bet_agent2
            loser_bet_val = bet_agent2 if actual_winner_agent == agent1 else bet_agent1

            gain, loss = _apply_wealth_change(
                actual_winner_agent, actual_loser_agent,
                winner_bet_val, loser_bet_val,
                max_loss_mult, max_gain_ratio_val
            )

            if actual_winner_agent == agent1:
                wealth_A_change = gain
                wealth_B_change = -loss
                hist_entry_for_agent1['outcome'] = 'win'
            else:
                wealth_A_change = -loss
                wealth_B_change = gain 
                hist_entry_for_agent1['outcome'] = 'loss'
            hist_entry_for_agent1['wealth_change'] = wealth_A_change
            logging.info(f"Game {game_id}: Adjudicated Outcome: {game_details_dict['adjudication_result']}. Winner: {actual_winner_agent.agent_id}, Loser: {actual_loser_agent.agent_id}.")
        else:
            hist_entry_for_agent1['outcome'] = 'tie'
            hist_entry_for_agent1['wealth_change'] = 0.0
            if not game_details_dict["defaulted_to_tie_reason"]:
                game_details_dict["defaulted_to_tie_reason"] = "Adjudicated as Tie or game ended inconclusively"
            if game_details_dict["adjudication_result"] not in ["Tie", "Role A Wins", "Role B Wins", "Proposer Loss (Scenario Gen Fail)"]:
                game_details_dict["adjudication_result"] = "Tie (Defaulted)"
            logging.info(f"Game {game_id}: Game outcome is a Tie. Reason: {game_details_dict['defaulted_to_tie_reason']}.")

        game_details_dict["wealth_changes"] = {"player_A_wealth_change": wealth_A_change, "player_B_wealth_change": wealth_B_change}
        logging.info(f"Game {game_id}: Final wealth changes - Agent1 ({agent1.agent_id}): {wealth_A_change:+.2f}, Agent2 ({agent2.agent_id}): {wealth_B_change:+.2f}")

    except Exception as e:
        logging.critical(f"Game {game_id}: CRITICAL UNHANDLED ERROR during game: {e}", exc_info=True)
        hist_entry_for_agent1['outcome'] = 'error_critical'
        hist_entry_for_agent1['wealth_change'] = 0.0
        game_details_dict["adjudication_result"] = "Critical Game Error"
        game_details_dict["defaulted_to_tie_reason"] = f"Critical error during game: {str(e)[:150]}"
        game_details_dict["scenario_text"] = game_details_dict.get("scenario_text", "Error due to critical failure in game.")
        game_details_dict.setdefault("scenario_generation_prompt", "Unavailable due to critical error")
        game_details_dict.setdefault("adjudication_prompt_scratchpad", "Unavailable due to critical error")
        game_details_dict.setdefault("adjudication_prompt_outcome_ids", "Unavailable due to critical error")

    finally:
        try:
            agent1.add_game_result(copy.deepcopy(hist_entry_for_agent1))

            hist_entry_for_agent2 = copy.deepcopy(hist_entry_for_agent1)
            hist_entry_for_agent2['opponent_id'] = agent1.agent_id
            hist_entry_for_agent2['role_in_proposal'] = 'proposer' if agent2 == proposer_obj else 'opponent'
            hist_entry_for_agent2['role_in_game'] = game_details_dict.get("player_B_game_role")
            
            if hist_entry_for_agent1['outcome'] == 'win':
                hist_entry_for_agent2['outcome'] = 'loss'
            elif hist_entry_for_agent1['outcome'] == 'loss':
                hist_entry_for_agent2['outcome'] = 'win'
            else:
                hist_entry_for_agent2['outcome'] = hist_entry_for_agent1['outcome']
            
            hist_entry_for_agent2['wealth_change'] = game_details_dict.get("wealth_changes", {}).get("player_B_wealth_change", 0.0)
            # Bets for agent2: their bet is what agent1 saw as agent2's bet, and opponent's bet is what agent1 bet
            hist_entry_for_agent2['bets'] = {'agent1_bet': bet_agent2, 'agent2_bet': bet_agent1} # Corrected perspective for agent2


            agent2.add_game_result(hist_entry_for_agent2)

        except Exception as e_hist:
            logging.error(f"Game {game_id}: Unexpected error saving game history to agents: {e_hist}", exc_info=True)

        game_details_dict["timestamp_end"] = datetime.datetime.now().isoformat()
        game_details_dict["final_player_A_wealth"] = agent1.wealth
        game_details_dict["final_player_B_wealth"] = agent2.wealth
        logging.info(f"Game {game_id}: Finished. Final wealth A:{agent1.wealth:.2f}, B:{agent2.wealth:.2f}. Adjudication: {game_details_dict['adjudication_result']}")
        
        try:
            state_base_dir_from_config = config.get('state_saving', {}).get('directory', 'simulation_state')
            if not isinstance(run_id, str) or not run_id:
                logging.error(f"Game {game_id}: Invalid or missing run_id ('{run_id}') for live data persistence. Skipping.")
            else:
                run_dir_path_in_engine = Path(state_base_dir_from_config) / run_id
                run_dir_path_in_engine.mkdir(parents=True, exist_ok=True)

                games_data_filename_live = run_dir_path_in_engine / f"games_generation_{generation_number:04d}.jsonl"
                try:
                    with open(games_data_filename_live, 'a') as f:
                        f.write(json.dumps(game_details_dict) + '\n')
                    logging.debug(f"Game {game_id}: Appended live game detail to {games_data_filename_live}")
                except Exception as e_append:
                    logging.error(f"Game {game_id}: Failed to append live game detail to {games_data_filename_live}: {e_append}")
        except Exception as e_live_persist_outer:
            logging.error(f"Game {game_id}: Outer error during live data persistence: {e_live_persist_outer}", exc_info=True)
        
    return game_details_dict


async def run_game_round(population: list[Agent], config: dict, run_id: str, generation_number: int) -> tuple[list[Agent], list[dict]]:
    """
    Manages a round of games for the entire population asynchronously.
    Agents are paired, play games concurrently within pairing passes, and their states (wealth, history) are updated.
    Returns the updated population and a list of detailed game dictionaries.
    """
    if not isinstance(population, list):
        logging.error("run_game_round: Population argument must be a list.")
        return [], [] 
    if not all(isinstance(agent, Agent) for agent in population):
        logging.error("run_game_round: All elements in population list must be Agent instances.")
        return population, [] 
    if not population:
        logging.warning("run_game_round: Called with an empty population. No games will be played.")
        return [], []
    if not isinstance(config, dict):
        logging.error("run_game_round: Config argument must be a dictionary.")
        return population, [] 

    sim_settings = config.get('simulation', {})
    games_per_agent_target = int(sim_settings.get('games_per_agent_target', config.get('game',{}).get('interaction_turns_per_agent',0)))   
    pairing_strategy = config.get('round', {}).get('pairing_strategy', 'random_shuffle') 
    num_agents = len(population)
    initial_wealth_val = float(config.get('agent', {}).get('initial_wealth', 30.0))
    all_games_details_this_round: list[dict] = []

    for ag_idx, current_agent in enumerate(population):
        try:
            current_agent.reset_round_state(initial_wealth_val)
        except AttributeError:
            logging.critical(f"Agent {getattr(current_agent, 'agent_id', f'Unknown_at_index_{ag_idx}')} missing reset_round_state method. Cannot proceed with game round.")
            raise 

    if num_agents < 2:
        logging.warning(f"Not enough agents ({num_agents}) for pairing. At least 2 are required. Skipping game round.")
        return population, []

    if pairing_strategy == 'random_shuffle':
        games_played_this_round_count = {agent.agent_id: 0 for agent in population}
        total_participations_needed = num_agents * games_per_agent_target
        
        if total_participations_needed % 2 != 0:
            logging.warning(
                f"Target total participations ({total_participations_needed}) is odd. "
                f"Game counts per agent might vary slightly around the target of {games_per_agent_target}."
            )
        
        num_total_games_to_schedule = total_participations_needed // 2
        max_iters_pairing = num_total_games_to_schedule * 5 + num_agents 
        if num_agents <= 3: 
            max_iters_pairing = games_per_agent_target * num_agents * 2 

        actual_games_played_this_round_idx = 0 
        for iter_num in range(max_iters_pairing):
            tasks_for_this_pass: list[Coroutine[Any, Any, dict]] = [] 
            if all(count >= games_per_agent_target for count in games_played_this_round_count.values()):
                logging.info(f"All agents have met or exceeded target of {games_per_agent_target} games by pairing iteration {iter_num + 1}.")
                break

            shuffled_indices = list(range(num_agents))
            random.shuffle(shuffled_indices)
            
            was_paired_in_this_shuffle_pass = [False] * num_agents

            for i in range(0, num_agents - (num_agents % 2), 2): 
                idx1, idx2 = shuffled_indices[i], shuffled_indices[i+1]
                
                if was_paired_in_this_shuffle_pass[idx1] or was_paired_in_this_shuffle_pass[idx2]:
                    continue

                agent_A_obj, agent_B_obj = population[idx1], population[idx2]

                if games_played_this_round_count.get(agent_A_obj.agent_id, 0) < games_per_agent_target and \
                   games_played_this_round_count.get(agent_B_obj.agent_id, 0) < games_per_agent_target:
                    
                    actual_games_played_this_round_idx += 1
                    # _play_single_game is  async, so it returns a coroutine object directly
                    game_task = _play_single_game(
                        agent_A_obj, agent_B_obj, config,
                        run_id, generation_number, actual_games_played_this_round_idx
                    )
                    tasks_for_this_pass.append(game_task) 
                    
                    was_paired_in_this_shuffle_pass[idx1] = True
                    was_paired_in_this_shuffle_pass[idx2] = True
            
            if tasks_for_this_pass:
                logging.info(f"Game round (iter {iter_num+1}): Launching {len(tasks_for_this_pass)} games concurrently for this pass.")
                game_results_this_pass = await asyncio.gather(*tasks_for_this_pass, return_exceptions=True)
                
                for result_or_exc in game_results_this_pass:
                    if isinstance(result_or_exc, Exception):
                        logging.error(f"Game round (iter {iter_num+1}): A game task failed with an exception: {result_or_exc}", exc_info=result_or_exc)
                    else:
                        game_detail_dict = result_or_exc 
                        if game_detail_dict and isinstance(game_detail_dict, dict) and game_detail_dict.get("game_id"):
                            all_games_details_this_round.append(game_detail_dict)
                            player_a_id = game_detail_dict.get("player_A_id")
                            player_b_id = game_detail_dict.get("player_B_id")
                            if player_a_id and player_a_id in games_played_this_round_count:
                                games_played_this_round_count[player_a_id] += 1
                            if player_b_id and player_b_id in games_played_this_round_count:
                                games_played_this_round_count[player_b_id] += 1
                        else:
                            logging.error(f"Game round (iter {iter_num+1}): A game task returned an invalid result or no game_id: {game_detail_dict}")
            
            if iter_num == max_iters_pairing - 1: 
                logging.warning(
                    f"Reached max pairing iterations ({max_iters_pairing}). "
                    f"Game counts per agent may not be perfectly uniform at {games_per_agent_target}."
                )

        logging.info(f"Game round completed. Total distinct games attempted/logged: {len(all_games_details_this_round)}.")
        for p_agent in population:
            final_count = games_played_this_round_count.get(p_agent.agent_id, 0)
            logging.info(f"Agent {p_agent.agent_id} played {final_count} games this round (target: {games_per_agent_target}).")
        
        agents_below_target = [aid for aid, count in games_played_this_round_count.items() if count < games_per_agent_target]
        if agents_below_target:
            logging.warning(f"{len(agents_below_target)} agent(s) did not meet the target of {games_per_agent_target} games: {agents_below_target}")

    else: 
        logging.error(f"Unsupported pairing strategy: '{pairing_strategy}'. Only 'random_shuffle' is implemented.")
        raise NotImplementedError(f"Pairing strategy '{pairing_strategy}' is not implemented.")

    return population, all_games_details_this_round
