import streamlit as st
import pandas as pd
import json
from pathlib import Path
import altair as alt
import re
import numpy as np
from collections import defaultdict, Counter


class UIElementKeyGenerator:
    """
    Generates unique and stable keys for Streamlit UI elements.
    """
    @staticmethod
    def create(element_type: str, base_name: str) -> str:
        """
        Creates a key string based on element type and a base name.
        Example: create("slider", "my_slider") -> "uielement_slider_my_slider"
        This provides a stable key for elements.
        """
        return f"uielement_{element_type}_{base_name}"

# --- Configuration ---
DEFAULT_STATE_BASE_DIR = "simulation_state"
CONFIG_FILENAME = "config_snapshot.json"
LATEST_GEN_TRACKER_FILENAME = "_latest_generation_number.txt"

# ---  Utility Functions ---

def get_agent_display_name(agent_id: str) -> str:
    return f"Agent {agent_id[:8]}" if isinstance(agent_id, str) and agent_id else "N/A"

def get_feature_display_name(feature_uuid: str, label: str | None) -> str:
    if label and label.strip():
        return f"{label} ({feature_uuid[:8]}...)"
    return f"Feature {feature_uuid[:8]}..."

# --- Data Loading and Pre-computation ---

@st.cache_data(ttl=300) # Increased TTL for expensive precomputation
def load_full_run_data(run_id: str, state_base_dir: str):
    """
    Loads all necessary data for a simulation run and performs pre-computation.
    Returns a dictionary containing:
        - config_snapshot: dict
        - latest_generation_number: int
        - all_generation_data: dict[int, dict] (gen_num -> gen_data)
        - all_agents_by_id: dict[str, list[dict]] (agent_id -> list of agent states across gens)
        - lineages: dict[str, list[str]] (progenitor_id -> list of all descendant_ids including self)
        - progenitor_map: dict[str, str] (agent_id -> progenitor_id)
        - all_active_features: dict[str_uuid, str_label] (features with non-zero activation)
        - generation_summary_df: pd.DataFrame (for overview curves)
        - interesting_events: list[dict]
    """
    if not run_id:
        return None

    base_path = Path(state_base_dir) / run_id
    if not base_path.is_dir():
        st.error(f"Run directory not found: {base_path}")
        return None

    # Load config
    config_path = base_path / CONFIG_FILENAME
    config_snapshot = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_snapshot = json.load(f)
    else:
        st.warning(f"Config snapshot not found for run {run_id}")
        return None # Config is essential













    # Load latest fully checkpointed generation number from the tracker file
    tracker_path = base_path / LATEST_GEN_TRACKER_FILENAME
    last_fully_checkpointed_gen_num = 0 # Default if tracker is missing or invalid
    if tracker_path.exists():
        with open(tracker_path, 'r') as f:
            content = f.read().strip()
            if content:
                try:
                    last_fully_checkpointed_gen_num = int(content)
                except ValueError:
                    st.error(f"Invalid content in latest generation tracker for {run_id}. Could not parse generation number. Will attempt to load data assuming generation 0 was last checkpoint.")
                    # last_fully_checkpointed_gen_num remains 0. This allows the dashboard to still try loading games for gen 1.
    
    # This variable will store the highest generation number for which any data (full state or just games) is found.
    # It's used to inform the UI about the extent of available data.
    max_gen_for_dashboard_display = last_fully_checkpointed_gen_num

    # Initialize data structures
    all_generation_data = {} # Stores content of generation_XXXX.json (full state)
    all_agents_by_id = defaultdict(list)
    all_active_features = {} # Stores uuid -> label for features encountered
    generation_summary_list = [] # List to build the generation_summary_df
    interesting_events = [] # List for various event log entries

    # Define numeric columns for generation_summary_df conversion upfront
    numeric_cols_for_summary_df = ["avg_fitness", "min_fitness", "max_fitness", "avg_wealth", "unique_genomes_approx", "total_games_played_in_generation"]

    # --- Phase 1: Load data from fully checkpointed generations (generation_XXXX.json files) ---
    # This includes population states, detailed agent genomes, and generation summary metrics.
    # Events derived from this full state data are also processed here.
    for gen_num in range(1, last_fully_checkpointed_gen_num + 1):
        gen_file_path = base_path / f"generation_{gen_num:04d}.json"
        if gen_file_path.exists():
            gen_data = None
            try:
                with open(gen_file_path, 'r') as f:
                    gen_data = json.load(f)
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON from {gen_file_path}: {e}")
                continue # Skip this problematic generation file

            if gen_data is None: continue # Should not happen if open was successful and no JSONDecodeError

            all_generation_data[gen_num] = gen_data # Store the full state for this generation

            # Event: Generation Completed
            interesting_events.append({
                "generation": gen_num,
                "timestamp": gen_data.get("timestamp_completed"),
                "type": "Generation Completed",
                "description": f"Generation {gen_num} processing finished."
            })

            # Process population state for agent tracking and feature collection
            population_state = gen_data.get("population_state", [])
            for agent_dict in population_state:
                agent_dict['generation_number'] = gen_num # Augment agent data with its generation
                all_agents_by_id[agent_dict['agent_id']].append(agent_dict)
                
                genome = agent_dict.get("genome", {})
                for feature_uuid, feature_data in genome.items():
                    if isinstance(feature_data, dict): # Modern genome format
                        activation = feature_data.get('activation', 0.0)
                        label = feature_data.get('label', f"Label for {feature_uuid[:8]}...")
                    else: # Older genome format (direct activation value)
                        activation = feature_data
                        label = f"Unknown Label ({feature_uuid[:8]})"

                    if abs(activation) > 1e-9: # If feature has non-zero activation
                        if feature_uuid not in all_active_features:
                            all_active_features[feature_uuid] = label
                            # Event: Feature Appeared
                            interesting_events.append({
                                "generation": gen_num,
                                "timestamp": gen_data.get("timestamp_completed"),
                                "type": "Feature Appeared",
                                "description": f"Feature '{get_feature_display_name(feature_uuid, label)}' first appeared with non-zero activation (Agent: {get_agent_display_name(agent_dict['agent_id'])})."
                            })
            
            # Prepare data for Generation Summary DataFrame (for overview curves)
            summary_metrics = gen_data.get("generation_summary_metrics", {})
            summary_entry = {
                "generation_number": gen_num,
                "timestamp_completed": gen_data.get("timestamp_completed"),
                "avg_fitness": summary_metrics.get("avg_fitness"),
                "min_fitness": summary_metrics.get("min_fitness"),
                "max_fitness": summary_metrics.get("max_fitness"),
                "avg_wealth": summary_metrics.get("avg_wealth"),
                "total_games_played_in_generation": summary_metrics.get("total_games_played_in_generation")
            }
            # Calculate unique genomes for summary
            current_gen_population_for_summary = gen_data.get("population_state", [])
            if current_gen_population_for_summary:
                genome_fingerprints = []
                for agent_s in current_gen_population_for_summary:
                    g = agent_s.get("genome", {})
                    fp_items = []
                    for uuid_key, data_val in g.items():
                        act = data_val.get('activation') if isinstance(data_val, dict) else data_val
                        fp_items.append((uuid_key, round(act, 5))) # Round to avoid float precision issues
                    genome_fingerprints.append(tuple(sorted(fp_items)))
                summary_entry["unique_genomes_approx"] = len(set(genome_fingerprints))
            else:
                summary_entry["unique_genomes_approx"] = 0
            generation_summary_list.append(summary_entry)

    # Build the Generation Summary DataFrame from fully checkpointed generations
    generation_summary_df = pd.DataFrame(generation_summary_list)
    for col in numeric_cols_for_summary_df:
        if col in generation_summary_df.columns:
            generation_summary_df[col] = pd.to_numeric(generation_summary_df[col], errors='coerce')

    # --- Reconstruct Lineages (based on data from fully checkpointed generations) ---
    lineages = {} 
    progenitor_map = {} 
    # First pass: identify progenitors (Gen 1 agents)
    if 1 in all_generation_data: # Check if gen 1 data was loaded (i.e., last_fully_checkpointed_gen_num >= 1)
        for agent_dict in all_generation_data[1].get("population_state", []):
            agent_id = agent_dict['agent_id']
            lineages[agent_id] = [agent_id] # Progenitor is part of its own lineage
            progenitor_map[agent_id] = agent_id
            # Event: Lineage Founded
            interesting_events.append({
                "generation": 1, "timestamp": all_generation_data[1].get("timestamp_completed"),
                "type": "Lineage Founded",
                "description": f"Progenitor {get_agent_display_name(agent_id)} founded a new lineage."
            })

    # Subsequent passes: build out lineages up to the last fully checkpointed generation
    for gen_num in range(1, last_fully_checkpointed_gen_num + 1):
        if gen_num not in all_generation_data: continue # Should not happen if loop is correct
        for agent_dict in all_generation_data[gen_num].get("population_state", []):
            agent_id = agent_dict['agent_id']
            parent_id = agent_dict.get('parent_id') 
            if parent_id and parent_id in progenitor_map:
                progenitor_id_of_parent = progenitor_map[parent_id]
                if agent_id not in lineages[progenitor_id_of_parent]: 
                    lineages[progenitor_id_of_parent].append(agent_id)
                progenitor_map[agent_id] = progenitor_id_of_parent
            elif parent_id and parent_id not in progenitor_map:
                # Handle cases where an agent's parent isn't in the progenitor map (e.g., orphan or data anomaly)
                if agent_id not in lineages: 
                    lineages[agent_id] = [agent_id]
                    progenitor_map[agent_id] = agent_id
                    st.sidebar.warning(f"Agent {agent_id} (Gen {gen_num}) has parent {parent_id} not in progenitor_map. Treating as new lineage root for now.")

    # --- Process Threshold-based and Evolutionary Events (based on fully checkpointed data) ---
    LINEAGE_MEMBER_COUNT_THRESHOLDS = [5, 10, 25, 50, 100] 
    FEATURE_AVG_ACTIVATION_THRESHOLD = 0.5
    lineage_thresholds_met = defaultdict(lambda: defaultdict(bool)) 
    feature_activation_thresholds_met = defaultdict(bool) 

    # Iterate through fully checkpointed generations for these events
    for gen_num in range(1, last_fully_checkpointed_gen_num + 1):
        if gen_num not in all_generation_data: continue
        
        current_gen_data_for_events = all_generation_data[gen_num]
        current_gen_population_for_events = current_gen_data_for_events.get("population_state", [])
        current_gen_timestamp_for_events = current_gen_data_for_events.get("timestamp_completed")

        # Event: Lineage reached K members
        lineage_counts_this_gen_for_events = Counter()
        for agent_dict_event_lm in current_gen_population_for_events:
            progenitor_id_event_lm = progenitor_map.get(agent_dict_event_lm['agent_id'])
            if progenitor_id_event_lm:
                lineage_counts_this_gen_for_events[progenitor_id_event_lm] += 1
        for prog_id_event_lm, count_event_lm in lineage_counts_this_gen_for_events.items():
            for threshold_event_lm in LINEAGE_MEMBER_COUNT_THRESHOLDS:
                if count_event_lm >= threshold_event_lm and not lineage_thresholds_met[prog_id_event_lm][threshold_event_lm]:
                    interesting_events.append({
                        "generation": gen_num, "timestamp": current_gen_timestamp_for_events,
                        "type": "Lineage Milestone",
                        "description": f"Lineage of {get_agent_display_name(prog_id_event_lm)} reached {count_event_lm} members (threshold: {threshold_event_lm})."
                    })
                    lineage_thresholds_met[prog_id_event_lm][threshold_event_lm] = True

        # Event: Feature average activation exceeded threshold
        feature_activations_this_gen_for_events = defaultdict(list)
        for agent_dict_event_fm in current_gen_population_for_events:
            genome_event_fm = agent_dict_event_fm.get("genome", {})
            for f_uuid_event_fm, f_data_event_fm in genome_event_fm.items():
                act_val_event_fm = f_data_event_fm.get('activation') if isinstance(f_data_event_fm, dict) else f_data_event_fm
                feature_activations_this_gen_for_events[f_uuid_event_fm].append(act_val_event_fm)
        for f_uuid_event_fm, activations_event_fm in feature_activations_this_gen_for_events.items():
            if activations_event_fm: # Check if list is not empty
                avg_activation_event_fm = np.mean(activations_event_fm)
                if abs(avg_activation_event_fm) >= FEATURE_AVG_ACTIVATION_THRESHOLD and not feature_activation_thresholds_met[f_uuid_event_fm]:
                    feature_label_event_fm = all_active_features.get(f_uuid_event_fm, f"UUID {f_uuid_event_fm[:8]}")
                    interesting_events.append({
                        "generation": gen_num, "timestamp": current_gen_timestamp_for_events,
                        "type": "Feature Milestone",
                        "description": f"Feature '{get_feature_display_name(f_uuid_event_fm, feature_label_event_fm)}' average activation reached {avg_activation_event_fm:.3f} (threshold: {FEATURE_AVG_ACTIVATION_THRESHOLD})."
                    })
                    feature_activation_thresholds_met[f_uuid_event_fm] = True
        
        # Evolutionary Feature Update Events (from offspring in this generation)
        # These depend on 'evolutionary_input_positive_features' stored in the agent's state from generation_XXXX.json
        for agent_dict_evo_event in current_gen_population_for_events:
            agent_id_evo_event = agent_dict_evo_event.get('agent_id')
            pos_features_evo_event = agent_dict_evo_event.get('evolutionary_input_positive_features', [])
            neg_features_evo_event = agent_dict_evo_event.get('evolutionary_input_negative_features', [])

            if pos_features_evo_event:
                top_feature_uuid_evo_pos = pos_features_evo_event[0] 
                feature_label_evo_pos = all_active_features.get(top_feature_uuid_evo_pos, f"UUID {top_feature_uuid_evo_pos[:8]}")
                interesting_events.append({
                    "generation": gen_num, "timestamp": current_gen_timestamp_for_events,
                    "type": "Feature Reinforced",
                    "description": f"Agent {get_agent_display_name(agent_id_evo_event)}: Feature '{get_feature_display_name(top_feature_uuid_evo_pos, feature_label_evo_pos)}' reinforced (parent won)."
                })
            if neg_features_evo_event:
                top_feature_uuid_evo_neg = neg_features_evo_event[0] 
                feature_label_evo_neg = all_active_features.get(top_feature_uuid_evo_neg, f"UUID {top_feature_uuid_evo_neg[:8]}")
                interesting_events.append({
                    "generation": gen_num, "timestamp": current_gen_timestamp_for_events,
                    "type": "Feature Suppressed",
                    "description": f"Agent {get_agent_display_name(agent_id_evo_event)}: Feature '{get_feature_display_name(top_feature_uuid_evo_neg, feature_label_evo_neg)}' suppressed (parent lost)."
                })

    # --- Phase 2: Load game data (games_generation_XXXX.jsonl) and "Game Outcome" events ---
    # First, load games for all fully checkpointed generations
    for gen_num_games in range(1, last_fully_checkpointed_gen_num + 1):
        games_this_gen = load_games_for_generation_cached(run_id, gen_num_games, state_base_dir)
        for game in games_this_gen:
            game_id_short = game.get("game_id", "UnknownGame")[:15]
            pA_id = game.get("player_A_id")
            pB_id = game.get("player_B_id")
            pA_role = game.get("player_A_game_role", "Role A")
            pB_role = game.get("player_B_game_role", "Role B")
            adj_result = game.get("adjudication_result", "Unknown")
            game_timestamp = game.get("timestamp_end", game.get("timestamp_start"))
            outcome_desc = "Unknown Outcome"
            if adj_result == "Role A Wins": outcome_desc = f"{pA_role} ({get_agent_display_name(pA_id)}) Wins vs {pB_role} ({get_agent_display_name(pB_id)})"
            elif adj_result == "Role B Wins": outcome_desc = f"{pB_role} ({get_agent_display_name(pB_id)}) Wins vs {pA_role} ({get_agent_display_name(pA_id)})"
            elif "Tie" in adj_result: outcome_desc = f"Tie between {pA_role} ({get_agent_display_name(pA_id)}) and {pB_role} ({get_agent_display_name(pB_id)})"
            else: outcome_desc = f"Result: {adj_result} for {pA_role} ({get_agent_display_name(pA_id)}) vs {pB_role} ({get_agent_display_name(pB_id)})"
            interesting_events.append({
                "generation": gen_num_games, "timestamp": game_timestamp,
                "type": "Game Outcome", "description": f"Game {game_id_short}...: {outcome_desc}"
            })

    # Attempt to load game data for the *next* (potentially in-progress) generation
    in_progress_gen_candidate = last_fully_checkpointed_gen_num + 1
    in_progress_games_file_path = base_path / f"games_generation_{in_progress_gen_candidate:04d}.jsonl"
    
    if in_progress_games_file_path.exists():
        games_in_progress_gen = load_games_for_generation_cached(run_id, in_progress_gen_candidate, state_base_dir)
        if games_in_progress_gen: # Check if the file actually contained games
            for game_ip in games_in_progress_gen: # Use game_ip to avoid variable name collision
                game_id_short_ip = game_ip.get("game_id", "UnknownGame")[:15]
                pA_id_ip = game_ip.get("player_A_id")
                pB_id_ip = game_ip.get("player_B_id")
                pA_role_ip = game_ip.get("player_A_game_role", "Role A")
                pB_role_ip = game_ip.get("player_B_game_role", "Role B")
                adj_result_ip = game_ip.get("adjudication_result", "Unknown")
                game_timestamp_ip = game_ip.get("timestamp_end", game_ip.get("timestamp_start"))
                outcome_desc_ip = "Unknown Outcome"
                if adj_result_ip == "Role A Wins": outcome_desc_ip = f"{pA_role_ip} ({get_agent_display_name(pA_id_ip)}) Wins vs {pB_role_ip} ({get_agent_display_name(pB_id_ip)})"
                elif adj_result_ip == "Role B Wins": outcome_desc_ip = f"{pB_role_ip} ({get_agent_display_name(pB_id_ip)}) Wins vs {pA_role_ip} ({get_agent_display_name(pA_id_ip)})"
                elif "Tie" in adj_result_ip: outcome_desc_ip = f"Tie between {pA_role_ip} ({get_agent_display_name(pA_id_ip)}) and {pB_role_ip} ({get_agent_display_name(pB_id_ip)})"
                else: outcome_desc_ip = f"Result: {adj_result_ip} for {pA_role_ip} ({get_agent_display_name(pA_id_ip)}) vs {pB_role_ip} ({get_agent_display_name(pB_id_ip)})"
                interesting_events.append({
                    "generation": in_progress_gen_candidate, "timestamp": game_timestamp_ip,
                    "type": "Game Outcome", "description": f"Game {game_id_short_ip}...: {outcome_desc_ip}"
                })
            
            max_gen_for_dashboard_display = in_progress_gen_candidate # Update if in-progress games found
            st.sidebar.info(f"Showing games from in-progress generation {in_progress_gen_candidate}.")
            
            # Add a synthetic entry to generation_summary_list for the in-progress generation
            # This helps charts extend to this generation, even if most metrics are NaN.
            # Check if an entry for this generation doesn't already exist from a full load (unlikely but safe).
            if not any(s["generation_number"] == in_progress_gen_candidate for s in generation_summary_list):
                generation_summary_list.append({
                    "generation_number": in_progress_gen_candidate,
                    "timestamp_completed": None, # This generation is not fully completed
                    "total_games_played_in_generation": len(games_in_progress_gen)
                    # Other metrics like avg_fitness, avg_wealth, etc., will be missing/NaN
                })
            # Rebuild DataFrame to include the in-progress generation's game count
            generation_summary_df = pd.DataFrame(generation_summary_list)
            for col in numeric_cols_for_summary_df: # Re-apply numeric conversion
                if col in generation_summary_df.columns:
                    generation_summary_df[col] = pd.to_numeric(generation_summary_df[col], errors='coerce')

    # Sort all collected interesting events by generation then timestamp
    interesting_events.sort(key=lambda x: (x["generation"], x.get("timestamp", "")))

    # If no data at all was found (e.g., empty state directory or brand new run with no files yet)
    if max_gen_for_dashboard_display == 0 and not (base_path / f"games_generation_{1:04d}.jsonl").exists():
        st.warning(f"No generation data (neither full checkpoints nor game files) found for run '{run_id}'. The simulation may not have started or no data has been saved yet.")
        # Return a structure that allows the dashboard to load without breaking,
        # but indicates no data.
        return {
            "config_snapshot": config_snapshot, # config_snapshot is loaded earlier, should be present
            "latest_generation_number": 0,
            "max_fully_checkpointed_generation": 0, # Max gen with full state (generation_XXXX.json)
            "all_generation_data": {},
            "all_agents_by_id": {},
            "lineages": {},
            "progenitor_map": {},
            "all_active_features": {},
            "generation_summary_df": pd.DataFrame(),
            "interesting_events": []
        }

    return {
        "config_snapshot": config_snapshot,
        "latest_generation_number": max_gen_for_dashboard_display, # This is the highest gen with *any* data (full state or just games)
        "max_fully_checkpointed_generation": last_fully_checkpointed_gen_num, # This is the highest gen with full state (generation_XXXX.json)
        "all_generation_data": all_generation_data, # Contains full state up to last_fully_checkpointed_gen_num
        "all_agents_by_id": dict(all_agents_by_id), 
        "lineages": lineages,
        "progenitor_map": progenitor_map,
        "all_active_features": all_active_features,
        "generation_summary_df": generation_summary_df, # May include a row for in-progress gen
        "interesting_events": interesting_events
    }

@st.cache_data(ttl=60)
def load_games_for_generation_cached(run_id: str, generation_number: int, state_base_dir: str) -> list[dict]:
    if not run_id or generation_number is None: return []
    games_file_path = Path(state_base_dir) / run_id / f"games_generation_{generation_number:04d}.jsonl"
    games_data = []
    if games_file_path.exists():
        try:
            with open(games_file_path, 'r') as f:
                for line in f:
                    try:
                        games_data.append(json.loads(line))
                    except json.JSONDecodeError as jde:
                        st.warning(f"Skipping invalid JSON line in {games_file_path}: {jde}")
        except Exception as e:
            st.error(f"Error loading games for generation {generation_number} of run {run_id}: {e}")
    return games_data

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="SAEvolution Dashboard v2")
st.title("🧬 Agent Evolution Dashboard v2 🔬")

# --- Session State Initialization ---
# (Handles selections persisting across interactions and reruns)
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📈 Overview & Curves"
if 'selected_run_id' not in st.session_state:
    st.session_state.selected_run_id = None
# For cross-tab navigation
if 'nav_to_agent_id' not in st.session_state:
    st.session_state.nav_to_agent_id = None
if 'nav_to_generation' not in st.session_state:
    st.session_state.nav_to_generation = None
if 'nav_to_feature_uuid' not in st.session_state:
    st.session_state.nav_to_feature_uuid = None
if 'nav_to_lineage_progenitor' not in st.session_state:
    st.session_state.nav_to_lineage_progenitor = None


# --- Sidebar ---
st.sidebar.header("Simulation Controls")
state_dir_input = st.sidebar.text_input("State Directory Path", value=DEFAULT_STATE_BASE_DIR, help="Path to the base directory where simulation states are saved.")

# available_runs will now be a list of dictionaries: [{'id': str, 'timestamp': float}]
available_runs_data = []
if Path(state_dir_input).is_dir():
    for item in Path(state_dir_input).iterdir():
        if item.is_dir():
            tracker_file = item / LATEST_GEN_TRACKER_FILENAME
            if tracker_file.exists():
                try:
                    # Get modification timestamp of the tracker file
                    mod_timestamp = tracker_file.stat().st_mtime
                    available_runs_data.append({'id': item.name, 'timestamp': mod_timestamp})
                except Exception as e:
                    st.sidebar.warning(f"Could not get timestamp for run {item.name}: {e}")
    # Sort by timestamp, most recent first
    available_runs_data = sorted(available_runs_data, key=lambda x: x['timestamp'], reverse=True)

# Extract just the IDs for the selectbox options, maintaining the sorted order
available_run_ids = [run_info['id'] for run_info in available_runs_data]

if not available_run_ids:
    st.sidebar.warning("No valid simulation runs found in the specified directory.")
    st.stop()

# Refined logic for handling session state for selected_run_id
if 'selected_run_id' not in st.session_state or st.session_state.selected_run_id is None:
    # If no run ID is in session state, or it's explicitly None, default to the latest available
    st.session_state.selected_run_id = available_run_ids[0] if available_run_ids else None
elif st.session_state.selected_run_id not in available_run_ids:
    # If the run ID in session state is no longer in the list of available runs, default to the latest
    st.sidebar.info(f"Previously selected run '{st.session_state.selected_run_id}' no longer found or invalid. Selecting the latest available run.")
    st.session_state.selected_run_id = available_run_ids[0] if available_run_ids else None
# Otherwise, st.session_state.selected_run_id is a valid, existing run, and we keep it.


# Determine index for the selectbox
selectbox_index = 0
if st.session_state.selected_run_id and available_run_ids:
    try:
        selectbox_index = available_run_ids.index(st.session_state.selected_run_id)
    except ValueError:
        # This case means selected_run_id (from session) is not in the current available_run_ids list.
        # The logic above should have reset it, but as a fallback:
        st.sidebar.warning(f"Fallback: Run ID '{st.session_state.selected_run_id}' from session not in current list. Defaulting to latest.")
        st.session_state.selected_run_id = available_run_ids[0] if available_run_ids else None
        selectbox_index = 0 # selectbox_index remains 0 if selected_run_id becomes None here
elif not available_run_ids: # Should have been caught by st.stop() but defensive
    st.session_state.selected_run_id = None

selected_run_id_from_ui = st.sidebar.selectbox(
    "Select Simulation Run ID",
    options=available_run_ids, # Use the list of IDs
    index=selectbox_index,
    help="Choose a simulation run to inspect."
)

# Update session state if selection changes
if selected_run_id_from_ui != st.session_state.selected_run_id:
    st.session_state.selected_run_id = selected_run_id_from_ui
    st.cache_data.clear() # Clear cache when run changes
    # Reset navigation states
    st.session_state.nav_to_agent_id = None
    st.session_state.nav_to_generation = None
    st.session_state.nav_to_feature_uuid = None
    st.session_state.nav_to_lineage_progenitor = None
    st.rerun()


if st.sidebar.button("🔄 Refresh Data & Clear Cache", help="Reload all data from the selected run and clear all caches."):
    st.cache_data.clear()
    st.session_state.nav_to_agent_id = None # Reset nav states on full refresh
    st.session_state.nav_to_generation = None
    st.session_state.nav_to_feature_uuid = None
    st.session_state.nav_to_lineage_progenitor = None
    st.rerun()

if not st.session_state.selected_run_id:
    st.info("Please select a Simulation Run ID from the sidebar.")
    st.stop()

# --- Load Full Run Data (Cached) ---
run_data = load_full_run_data(st.session_state.selected_run_id, state_dir_input)

if not run_data:
    st.error(f"Failed to load data for run {st.session_state.selected_run_id}. Please check logs and data integrity.")
    st.stop()

# Unpack run_data for easier access
config_snapshot = run_data["config_snapshot"]
latest_gen_num = run_data["latest_generation_number"] # Max gen with *any* data (full state or just games)
max_fully_checkpointed_generation = run_data["max_fully_checkpointed_generation"] # Max gen with full state (generation_XXXX.json)
all_gen_data = run_data["all_generation_data"] # Contains full state up to max_fully_checkpointed_generation
all_agents_by_id = run_data["all_agents_by_id"]
lineages = run_data["lineages"]
progenitor_map = run_data["progenitor_map"]
all_active_features = run_data["all_active_features"] # dict: uuid -> label
generation_summary_df = run_data["generation_summary_df"]
interesting_events = run_data["interesting_events"]

# --- Tab Navigation Setup ---
TABS = {
    "📈 Overview": "overview_tab",
    "👑 Lineage Explorer": "lineage_tab",
    "🧬 Feature Explorer": "feature_tab",
    "🔬 Generation Detail": "generation_tab",
    "👤 Agent Detail": "agent_tab",
    "📜 Game Viewer": "game_tab",
    "🗓️ Event Log": "event_tab",
}
tab_names = list(TABS.keys())

# Handle navigation requests
if st.session_state.nav_to_lineage_progenitor is not None and st.session_state.active_tab != "👑 Lineage Explorer":
    st.session_state.active_tab = "👑 Lineage Explorer"
elif st.session_state.nav_to_feature_uuid is not None and st.session_state.active_tab != "🧬 Feature Explorer":
    st.session_state.active_tab = "🧬 Feature Explorer"
elif st.session_state.nav_to_agent_id is not None and st.session_state.active_tab != "👤 Agent Detail":
    st.session_state.active_tab = "👤 Agent Detail"


# Create tabs
try:
    default_tab_index = tab_names.index(st.session_state.active_tab)
except ValueError:
    default_tab_index = 0
    st.session_state.active_tab = tab_names[0]

# Render tabs using st.radio for a different style, or st.tabs
# For simplicity, using st.tabs; st.radio can be used for more custom tab-like navigation
# active_tab_key = st.sidebar.radio("Navigation", tab_names, index=default_tab_index, key="main_nav")
# st.session_state.active_tab = active_tab_key
# active_tab_container_map = {name: st.container() for name in tab_names} # If using radio

# Using st.tabs
tab_ui_objects = st.tabs(tab_names)
tab_container_map = {name: tab_ui_objects[i] for i, name in enumerate(tab_names)}


# --- Tab Implementations ---

# Helper for navigation buttons
def create_nav_button(label, target_tab, nav_state_key, nav_state_value, params=None, key_suffix=None):
    """
    Creates a Streamlit button that, when clicked, navigates to a specified tab and
    sets associated navigation state in st.session_state.

    Args:
        label (str): The text to display on the button.
        target_tab (str): The key of the tab (from TABS) to navigate to.
        nav_state_key (str): The session state key to update with nav_state_value (e.g., 'nav_to_agent_id').
        nav_state_value (any): The value to set for nav_state_key (e.g., an agent ID).
        params (dict, optional): Additional key-value pairs to set in st.session_state. Defaults to None.
        key_suffix (str, optional): A suffix to append to the button's generated key for uniqueness
                                    if multiple buttons might otherwise have the same base key. Defaults to None.
    """
    # components of the key are strings for reliable key generation.
    str_nav_state_key = str(nav_state_key)
    str_nav_state_value = str(nav_state_value)
    str_target_tab = str(target_tab)

    base_key = f"nav_btn_{str_nav_state_key}_{str_nav_state_value}_{str_target_tab}"
    
    final_key = base_key
    if key_suffix and isinstance(key_suffix, str) and key_suffix.strip(): # Append suffix if it's a non-empty string
        final_key = f"{base_key}_{key_suffix.strip()}"
        
    if st.button(label, key=final_key):
        st.session_state.active_tab = target_tab
        st.session_state[nav_state_key] = nav_state_value # Store the original (potentially non-string) nav_state_value
        if params: # For additional context like generation number or other parameters
            for k, v in params.items():
                st.session_state[k] = v
        st.rerun()

# --- 📈 Overview Tab ---
with tab_container_map["📈 Overview"]:
    st.header(f"Run Overview: {st.session_state.selected_run_id}")
    col_overview1, col_overview2 = st.columns(2)
    with col_overview1:
        if config_snapshot:
            with st.expander("Simulation Configuration Snapshot", expanded=False):
                st.json(config_snapshot)
    with col_overview2:
        st.metric("Generations Completed", latest_gen_num if latest_gen_num is not None else "N/A")

    if not generation_summary_df.empty:
        st.subheader("Performance Curves Over Generations")
        plot_cols = st.columns(2)
        with plot_cols[0]:
            st.markdown("##### Fitness Metrics (Avg, Min, Max)")
            fitness_metrics = ["avg_fitness", "min_fitness", "max_fitness"]
            avail_fit_metrics = [m for m in fitness_metrics if m in generation_summary_df.columns and generation_summary_df[m].notna().any()]
            if avail_fit_metrics:
                df_melt = generation_summary_df[["generation_number"] + avail_fit_metrics].melt(
                    "generation_number", var_name="Metric", value_name="Fitness"
                ).dropna()
                if not df_melt.empty:
                    chart = alt.Chart(df_melt).mark_line(point=True).encode(
                        x=alt.X("generation_number:Q", title="Generation"),
                        y=alt.Y("Fitness:Q", scale=alt.Scale(zero=False)),
                        color="Metric:N",
                        tooltip=["generation_number", "Metric", alt.Tooltip("Fitness:Q", format=".3f")]
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
        with plot_cols[1]:
            st.markdown("##### Average Wealth")
            if "avg_wealth" in generation_summary_df.columns and generation_summary_df["avg_wealth"].notna().any():
                df_plot = generation_summary_df[["generation_number", "avg_wealth"]].dropna()
                if not df_plot.empty:
                    chart = alt.Chart(df_plot).mark_line(point=True).encode(
                        x=alt.X("generation_number:Q", title="Generation"),
                        y=alt.Y("avg_wealth:Q", title="Average Wealth", scale=alt.Scale(zero=False)),
                        tooltip=["generation_number", alt.Tooltip("avg_wealth:Q", format=".2f")]
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

        plot_cols2 = st.columns(2)
        with plot_cols2[0]:
            st.markdown("##### Games Played per Generation")
            if "total_games_played_in_generation" in generation_summary_df.columns and generation_summary_df["total_games_played_in_generation"].notna().any():
                df_plot = generation_summary_df[["generation_number", "total_games_played_in_generation"]].dropna()
                if not df_plot.empty:
                    chart = alt.Chart(df_plot).mark_bar().encode(
                        x=alt.X("generation_number:Q", title="Generation", bin=alt.Bin(maxbins=max(10, latest_gen_num // 5 if latest_gen_num else 10))),
                        y=alt.Y("total_games_played_in_generation:Q", title="Games Played"),
                        tooltip=["generation_number", "total_games_played_in_generation"]
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
        with plot_cols2[1]:
            st.markdown("##### Unique Genomes (Approx.)")
            if "unique_genomes_approx" in generation_summary_df.columns and generation_summary_df["unique_genomes_approx"].notna().any():
                df_plot = generation_summary_df[["generation_number", "unique_genomes_approx"]].dropna()
                if not df_plot.empty:
                    chart = alt.Chart(df_plot).mark_line(point=True).encode(
                        x=alt.X("generation_number:Q", title="Generation"),
                        y=alt.Y("unique_genomes_approx:Q", title="Unique Genomes (Approx.)"),
                        tooltip=["generation_number", "unique_genomes_approx"]
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No generation summary data available for plotting curves.")

# --- 👑 Lineage Explorer Tab ---
with tab_container_map["👑 Lineage Explorer"]:
    st.header("👑 Lineage Explorer")
    
    progenitor_ids = sorted([pid for pid in lineages.keys() if lineages[pid]]) # Progenitors with descendants
    
    # Handle navigation or direct selection
    default_progenitor_idx = 0
    if st.session_state.nav_to_lineage_progenitor and st.session_state.nav_to_lineage_progenitor in progenitor_ids:
        default_progenitor_idx = progenitor_ids.index(st.session_state.nav_to_lineage_progenitor)
        st.session_state.nav_to_lineage_progenitor = None # Consume nav state

    selected_progenitor_id = st.selectbox(
        "Select Progenitor Agent ID",
        options=progenitor_ids,
        format_func=get_agent_display_name,
        index=default_progenitor_idx,
        key="lineage_progenitor_select"
    )

    if selected_progenitor_id:
        lineage_member_ids = lineages.get(selected_progenitor_id, [])
        st.write(f"**Progenitor:** {get_agent_display_name(selected_progenitor_id)}")
        st.write(f"**Total Descendants (incl. self) by latest gen:** {len(lineage_member_ids)}")

        # 1. Lineage Fitness Over Time
        lineage_fitness_data = []
        for gen_num in range(1, latest_gen_num + 1):
            gen_pop_fitness_scores = []
            lineage_members_in_gen_fitness = []
            
            current_gen_agents = all_gen_data.get(gen_num, {}).get("population_state", [])
            fitness_map_this_gen = all_gen_data.get(gen_num, {}).get("generation_summary_metrics", {}).get("fitness_scores_map", {})

            for agent_data in current_gen_agents:
                agent_id = agent_data['agent_id']
                fitness = agent_data.get("fitness_score", fitness_map_this_gen.get(agent_id))
                if fitness is not None:
                    gen_pop_fitness_scores.append(fitness)
                    if agent_id in lineage_member_ids:
                        lineage_members_in_gen_fitness.append(fitness)
            
            if lineage_members_in_gen_fitness:
                lineage_fitness_data.append({
                    "generation": gen_num,
                    "metric": "Lineage Avg Fitness",
                    "value": np.mean(lineage_members_in_gen_fitness)
                })
            if gen_pop_fitness_scores: # Population average for comparison
                 lineage_fitness_data.append({
                    "generation": gen_num,
                    "metric": "Population Avg Fitness",
                    "value": np.mean(gen_pop_fitness_scores)
                })


        if lineage_fitness_data:
            df_fitness_lineage = pd.DataFrame(lineage_fitness_data)
            fitness_chart = alt.Chart(df_fitness_lineage).mark_line(point=True).encode(
                x=alt.X("generation:Q", title="Generation"),
                y=alt.Y("value:Q", title="Fitness", scale=alt.Scale(zero=False)),
                color="metric:N",
                tooltip=["generation", "metric", alt.Tooltip("value:Q", format=".3f")]
            ).properties(title="Lineage Fitness vs. Population Average").interactive()
            st.altair_chart(fitness_chart, use_container_width=True)

        # 2. Lineage Proportion Over Time (Stacked Area for Top N Lineages)
        st.subheader("Lineage Proportions Over Time (Top 5 Lineages + Other)")
        lineage_proportion_data = []
        all_progenitors = sorted(lineages.keys()) # All progenitors, not just those with descendants
        
        # Identify top 5 lineages by number of members in the final generation
        final_gen_lineage_counts = Counter()
        if latest_gen_num in all_gen_data:
            for agent_data in all_gen_data[latest_gen_num].get("population_state", []):
                prog_id = progenitor_map.get(agent_data['agent_id'])
                if prog_id:
                    final_gen_lineage_counts[prog_id] += 1
        
        top_5_progenitors = [p[0] for p in final_gen_lineage_counts.most_common(5)]

        for gen_num in range(1, latest_gen_num + 1):
            total_pop_in_gen = len(all_gen_data.get(gen_num, {}).get("population_state", []))
            if total_pop_in_gen == 0: continue

            lineage_counts_in_gen = Counter()
            for agent_data in all_gen_data[gen_num].get("population_state", []):
                prog_id = progenitor_map.get(agent_data['agent_id'])
                if prog_id:
                    lineage_counts_in_gen[prog_id] += 1
            
            other_proportion = 0.0
            for prog_id in all_progenitors:
                proportion = lineage_counts_in_gen[prog_id] / total_pop_in_gen if total_pop_in_gen > 0 else 0
                if prog_id in top_5_progenitors:
                    lineage_proportion_data.append({
                        "generation": gen_num,
                        "lineage": get_agent_display_name(prog_id),
                        "proportion": proportion
                    })
                else:
                    other_proportion += proportion
            
            if other_proportion > 0:
                 lineage_proportion_data.append({
                    "generation": gen_num,
                    "lineage": "Other Lineages",
                    "proportion": other_proportion
                })


        if lineage_proportion_data:
            df_proportions = pd.DataFrame(lineage_proportion_data)
            proportion_chart = alt.Chart(df_proportions).mark_area().encode(
                x=alt.X("generation:Q", title="Generation"),
                y=alt.Y("proportion:Q", stack="normalize", axis=alt.Axis(format='%'), title="Population Proportion"),
                color=alt.Color("lineage:N", title="Lineage Progenitor"),
                tooltip=["generation", "lineage", alt.Tooltip("proportion:Q", format=".2%")]
            ).properties(title="Lineage Proportions Over Time").interactive()
            st.altair_chart(proportion_chart, use_container_width=True)

        # 3. Genome Evolution Along Most Successful Descendant Path (for selected lineage)
        st.subheader(f"Genome Evolution for Most Successful Descendant of {get_agent_display_name(selected_progenitor_id)}")
        # Find most successful descendant in the last generation
        most_successful_desc_id = None
        max_fitness_in_lineage_last_gen = -1

        if latest_gen_num in all_gen_data:
            fitness_map_last_gen = all_gen_data[latest_gen_num].get("generation_summary_metrics", {}).get("fitness_scores_map", {})
            for agent_data in all_gen_data[latest_gen_num].get("population_state", []):
                if agent_data['agent_id'] in lineage_member_ids:
                    fitness = agent_data.get("fitness_score", fitness_map_last_gen.get(agent_data['agent_id']))
                    if fitness is not None and fitness > max_fitness_in_lineage_last_gen:
                        max_fitness_in_lineage_last_gen = fitness
                        most_successful_desc_id = agent_data['agent_id']
        
        if most_successful_desc_id:
            st.write(f"Tracing lineage to: {get_agent_display_name(most_successful_desc_id)} (Fitness: {max_fitness_in_lineage_last_gen:.3f})")
            path = []
            curr_agent_id_in_path = most_successful_desc_id
            while curr_agent_id_in_path:
                # Find the agent's data (need its generation to find its specific record)
                # This assumes agent_id is unique across generations.
                # We need to find the specific instance of this agent_id.
                # all_agents_by_id stores a list of states for each agent_id.
                # We need the LATEST state for that agent ID if it appears multiple times (it shouldn't in this context of tracing back)
                
                # Find the generation of curr_agent_id_in_path
                agent_instance = None
                for gen_idx_path in reversed(range(1, latest_gen_num + 1)): # Search backwards
                    gen_agents_path = all_gen_data.get(gen_idx_path, {}).get("population_state", [])
                    found = next((a for a in gen_agents_path if a['agent_id'] == curr_agent_id_in_path), None)
                    if found:
                        agent_instance = found
                        break
                
                if agent_instance:
                    path.append(agent_instance)
                    if agent_instance['agent_id'] == selected_progenitor_id: # Reached progenitor
                        break
                    curr_agent_id_in_path = agent_instance.get('parent_id')
                else: # Should not happen if data is consistent
                    st.warning(f"Could not find agent data for {curr_agent_id_in_path} while tracing lineage path.")
                    break
            path.reverse() # Progenitor first

            for i, agent_on_path in enumerate(path):
                with st.expander(f"Gen {agent_on_path['generation_number']}: {get_agent_display_name(agent_on_path['agent_id'])} (Fitness: {agent_on_path.get('fitness_score', 'N/A'):.3f})", expanded=(i == len(path) -1)):
                    st.write(f"**Wealth:** {agent_on_path.get('wealth', 'N/A'):.2f}")
                    
                    genome_data_current = agent_on_path.get("genome", {})
                    parent_agent_on_path = path[i-1] if i > 0 else None
                    genome_data_parent = parent_agent_on_path.get("genome", {}) if parent_agent_on_path else {}

                    if genome_data_current:
                        st.markdown("**Genome Changes from Parent:**")
                        diff_data = []
                        all_feature_uuids_in_comparison = set(genome_data_current.keys()) | set(genome_data_parent.keys())
                        
                        activation_change_threshold = 0.1 # Define a threshold for "significant change"

                        for f_uuid in sorted(list(all_feature_uuids_in_comparison)):
                            current_feature_info = genome_data_current.get(f_uuid)
                            parent_feature_info = genome_data_parent.get(f_uuid)

                            current_act = 0.0
                            current_label = all_active_features.get(f_uuid, f"Feature {f_uuid[:8]}...") # Default label
                            if current_feature_info:
                                current_act = current_feature_info.get('activation') if isinstance(current_feature_info, dict) else current_feature_info
                                if isinstance(current_feature_info, dict) and 'label' in current_feature_info:
                                    current_label = current_feature_info['label']
                            
                            parent_act = 0.0
                            if parent_feature_info:
                                parent_act = parent_feature_info.get('activation') if isinstance(parent_feature_info, dict) else parent_feature_info

                            status = ""
                            delta = current_act - parent_act

                            if f_uuid in genome_data_current and f_uuid not in genome_data_parent:
                                status = "New"
                            elif f_uuid not in genome_data_current and f_uuid in genome_data_parent:
                                status = "Removed"
                                # For removed features, display parent's activation as 'from', current is implicitly 0
                                current_act = 0 # Explicitly for display consistency if needed for "To"
                            elif abs(delta) >= activation_change_threshold:
                                status = "Changed"
                            elif current_act != 0 : # Present, but not significantly changed, and not zero
                                status = "Maintained" 
                            
                            if status: # Only show features that are new, removed, changed, or actively maintained
                                diff_data.append({
                                    "Feature": get_feature_display_name(f_uuid, current_label),
                                    "Status": status,
                                    "Activation (Parent)": parent_act if parent_agent_on_path else "N/A (Progenitor)",
                                    "Activation (Current)": current_act,
                                    "Change": delta if status in ["Changed", "New"] else "N/A" # New features have delta from 0
                                })
                        
                        if diff_data:
                            df_diff = pd.DataFrame(diff_data)
                            st.dataframe(df_diff, height=250, use_container_width=True,
                                         column_config={
                                             "Activation (Parent)": st.column_config.NumberColumn(format="%.3f"),
                                             "Activation (Current)": st.column_config.NumberColumn(format="%.3f"),
                                             "Change": st.column_config.NumberColumn(format="%+.3f")
                                         })
                        else:
                            st.caption("No significant genome changes or no parent for comparison (progenitor). Full genome below:")
                            # Fallback to showing full genome if no diff data or progenitor
                            genome_list_for_display = []
                            for f_uuid, f_data in genome_data_current.items():
                                act_val = f_data.get('activation') if isinstance(f_data, dict) else f_data
                                label_val = (f_data.get('label') if isinstance(f_data, dict) 
                                             else all_active_features.get(f_uuid, f"Feature {f_uuid[:8]}..."))
                                genome_list_for_display.append({"Feature": get_feature_display_name(f_uuid, label_val) , "Activation": act_val})
                            st.dataframe(pd.DataFrame(genome_list_for_display).sort_values(by="Activation", key=abs, ascending=False), height=200)

                    # Display evolutionary inputs for this agent (if not Gen 1)
                    if agent_on_path['generation_number'] > 1:
                        pos_inputs = agent_on_path.get('evolutionary_input_positive_features', [])
                        neg_inputs = agent_on_path.get('evolutionary_input_negative_features', [])
                        if pos_inputs:
                            st.markdown("**Features Reinforced (from Parent's Success):**")
                            st.markdown(", ".join([get_feature_display_name(fuuid, all_active_features.get(fuuid)) for fuuid in pos_inputs]))
                        if neg_inputs:
                            st.markdown("**Features Suppressed (from Parent's Failure):**")
                            st.markdown(", ".join([get_feature_display_name(fuuid, all_active_features.get(fuuid)) for fuuid in neg_inputs]))
        else:
            st.info("Could not determine a most successful descendant for genome evolution display.")

# --- 🧬 Feature Explorer Tab ---
with tab_container_map["🧬 Feature Explorer"]:
    st.header("🧬 Feature Explorer")
    
    if not all_active_features:
        st.info("No features with non-zero activations found in this simulation run.")
    else:
        feature_options = {uuid: label for uuid, label in sorted(all_active_features.items(), key=lambda item: item[1])}

        # Handle navigation
        default_feature_uuid = list(feature_options.keys())[0]
        if st.session_state.nav_to_feature_uuid and st.session_state.nav_to_feature_uuid in feature_options:
            default_feature_uuid = st.session_state.nav_to_feature_uuid
            st.session_state.nav_to_feature_uuid = None # Consume nav state

        selected_feature_uuid = st.selectbox(
            "Select Feature to Explore",
            options=list(feature_options.keys()),
            format_func=lambda uuid: get_feature_display_name(uuid, feature_options.get(uuid)),
            index=list(feature_options.keys()).index(default_feature_uuid),
            key="feature_explorer_select"
        )

        if selected_feature_uuid:
            st.subheader(f"Details for: {get_feature_display_name(selected_feature_uuid, feature_options.get(selected_feature_uuid))}")

            # 1. Cross-Generation Activation Trend
            feature_trend_data = []
            for gen_num in range(1, latest_gen_num + 1):
                activations_in_gen = []
                num_with_feature = 0
                total_agents_in_gen = 0
                if gen_num in all_gen_data:
                    pop_state = all_gen_data[gen_num].get("population_state", [])
                    total_agents_in_gen = len(pop_state)
                    for agent_data in pop_state:
                        genome = agent_data.get("genome", {})
                        feature_info = genome.get(selected_feature_uuid)
                        if feature_info: # Feature is present
                            num_with_feature +=1
                            act_val = feature_info.get('activation') if isinstance(feature_info, dict) else feature_info
                            activations_in_gen.append(act_val)
                
                if activations_in_gen: # Only add data if feature was active in that gen
                    feature_trend_data.append({
                        "generation": gen_num,
                        "metric": "Avg Activation",
                        "value": np.mean(activations_in_gen)
                    })
                    feature_trend_data.append({
                        "generation": gen_num,
                        "metric": "Min Activation",
                        "value": np.min(activations_in_gen)
                    })
                    feature_trend_data.append({
                        "generation": gen_num,
                        "metric": "Max Activation",
                        "value": np.max(activations_in_gen)
                    })
                if total_agents_in_gen > 0:
                    feature_trend_data.append({
                        "generation": gen_num,
                        "metric": "Prevalence (%)",
                        "value": (num_with_feature / total_agents_in_gen) * 100 if total_agents_in_gen > 0 else 0
                    })


            if feature_trend_data:
                df_trend = pd.DataFrame(feature_trend_data)
                
                base_chart = alt.Chart(df_trend).encode(x='generation:Q')
                
                activation_lines = base_chart.transform_filter(
                    alt.datum.metric != "Prevalence (%)"
                ).mark_line(point=True).encode(
                    y=alt.Y('value:Q', title='Activation', scale=alt.Scale(zero=False)),
                    color='metric:N',
                    tooltip=['generation', 'metric', alt.Tooltip('value:Q', format=".3f")]
                )
                
                prevalence_line = base_chart.transform_filter(
                    alt.datum.metric == "Prevalence (%)"
                ).mark_line(point=True, strokeDash=[5,5], color='gray').encode(
                    y=alt.Y('value:Q', title='Prevalence (%)', axis=alt.Axis(orient='right')),
                     tooltip=['generation', 'metric', alt.Tooltip('value:Q', format=".1f")]
                )
                
                layered_chart = alt.layer(activation_lines, prevalence_line).resolve_scale(
                    y='independent'
                ).properties(title=f"Feature Activation & Prevalence Over Generations").interactive()
                st.altair_chart(layered_chart, use_container_width=True)

            # 2. Correlation with Success (selected generation)
            gen_for_corr_feature = st.slider("Select Generation for Correlation Plot", 1, latest_gen_num, latest_gen_num, key="feature_corr_gen_slider")
            correlation_data = []
            if gen_for_corr_feature in all_gen_data:
                pop_state_corr = all_gen_data[gen_for_corr_feature].get("population_state", [])
                fitness_map_corr = all_gen_data[gen_for_corr_feature].get("generation_summary_metrics", {}).get("fitness_scores_map", {})

                for agent_data in pop_state_corr:
                    genome = agent_data.get("genome", {})
                    feature_info = genome.get(selected_feature_uuid)
                    if feature_info:
                        act_val = feature_info.get('activation') if isinstance(feature_info, dict) else feature_info
                        fitness = agent_data.get("fitness_score", fitness_map_corr.get(agent_data['agent_id']))
                        if fitness is not None:
                            correlation_data.append({
                                "activation": act_val,
                                "fitness": fitness,
                                "agent_id_short": get_agent_display_name(agent_data['agent_id'])
                            })
            
            if correlation_data:
                df_corr = pd.DataFrame(correlation_data)
                corr_chart = alt.Chart(df_corr).mark_circle(size=60).encode(
                    x=alt.X('activation:Q', title=f'Activation of {feature_options.get(selected_feature_uuid, "Feature")}'),
                    y=alt.Y('fitness:Q', title='Agent Fitness'),
                    tooltip=['agent_id_short', alt.Tooltip('activation:Q', format=".3f"), alt.Tooltip('fitness:Q', format=".3f")]
                ).properties(title=f"Feature Activation vs. Fitness (Gen {gen_for_corr_feature})").interactive()
                st.altair_chart(corr_chart, use_container_width=True)
                # Calculate and display Pearson correlation
                if len(df_corr['activation']) > 1 and len(df_corr['fitness']) > 1: # Need at least 2 points
                     pearson_r = df_corr['activation'].corr(df_corr['fitness'])
                     st.metric(label="Pearson Correlation (Activation vs. Fitness)", value=f"{pearson_r:.3f}")
            else:
                st.caption(f"No agents with this feature found in Generation {gen_for_corr_feature} for correlation plot.")

            # 3. Agent List by Feature (Table - for selected generation)
            st.markdown("---")
            st.subheader(f"Agents with Feature '{get_feature_display_name(selected_feature_uuid, feature_options.get(selected_feature_uuid))}' in a Selected Generation")
            
            # Slider for selecting generation for the agent list table
            if latest_gen_num == 1:
                gen_for_agent_list_feature = 1
                st.markdown("Displaying agent list for Generation 1 (only generation available).")
            else:
                gen_for_agent_list_feature = st.slider(
                    "Select Generation for Agent List", 
                    1, 
                    latest_gen_num, 
                    latest_gen_num, # Default to latest gen
                    key="feature_agent_list_gen_slider"
                )

            agents_with_feature_data = []
            if gen_for_agent_list_feature in all_gen_data:
                pop_state_agent_list = all_gen_data[gen_for_agent_list_feature].get("population_state", [])
                fitness_map_agent_list = all_gen_data[gen_for_agent_list_feature].get("generation_summary_metrics", {}).get("fitness_scores_map", {})

                for agent_data in pop_state_agent_list:
                    genome = agent_data.get("genome", {})
                    feature_info = genome.get(selected_feature_uuid)
                    if feature_info: # Agent has the feature
                        act_val = feature_info.get('activation') if isinstance(feature_info, dict) else feature_info
                        fitness = agent_data.get("fitness_score", fitness_map_agent_list.get(agent_data['agent_id']))
                        wealth = agent_data.get("wealth")
                        
                        agents_with_feature_data.append({
                            "Agent ID": get_agent_display_name(agent_data['agent_id']),
                            "Activation": act_val,
                            "Fitness": fitness,
                            "Wealth": wealth,
                            "_raw_agent_id": agent_data['agent_id'] # For potential navigation
                        })
            
            if agents_with_feature_data:
                df_agents_with_feature = pd.DataFrame(agents_with_feature_data)
                # Sort by activation value by default (absolute, descending)
                df_agents_with_feature = df_agents_with_feature.sort_values(by="Activation", key=lambda x: abs(x), ascending=False)
                st.dataframe(df_agents_with_feature[["Agent ID", "Activation", "Fitness", "Wealth"]], 
                             use_container_width=True,
                             column_config={
                                 "Activation": st.column_config.NumberColumn(format="%.3f"),
                                 "Fitness": st.column_config.NumberColumn(format="%.3f"),
                                 "Wealth": st.column_config.NumberColumn(format="%.2f"),
                             })
                # Potential for adding navigation buttons per row if desired, using _raw_agent_id
            else:
                st.caption(f"No agents found with this feature in Generation {gen_for_agent_list_feature}.")


# --- 🔬 Generation Detail Tab ---
with tab_container_map["🔬 Generation Detail"]:
    st.header("🔬 Generation Detail")




    latest_gen = run_data.get("latest_generation_number", 0)
    all_generations_data = run_data.get("all_generation_data", {})
    progenitor_map = run_data.get("progenitor_map", {}) # Added from context
    all_active_features = run_data.get("all_active_features", {}) # Added from context

    if latest_gen == 0:
        st.info("No generation data available to display details.") # Modified st.info text slightly
    else:
        # Generation selection slider
        # Default to the generation context from navigation, or latest generation
        default_gen_val_slider = st.session_state.get("nav_target_generation", latest_gen)
        if not (isinstance(default_gen_val_slider, int) and 1 <= default_gen_val_slider <= latest_gen):
            default_gen_val_slider = latest_gen
        
        selected_gen_num_for_detail = default_gen_val_slider
        if latest_gen > 1:
            selected_gen_num_for_detail = st.slider(
                "Select Generation to View Details",
                min_value=1, max_value=latest_gen, value=default_gen_val_slider,
                key=UIElementKeyGenerator.create("slider", "gen_detail_generation_select")
            )
        else:
            st.markdown("Displaying details for Generation 1 (the only generation available).")
        
        # Persist this selection for potential cross-tab navigation context
        st.session_state.nav_target_generation = selected_gen_num_for_detail

        gen_data_for_display = all_generations_data.get(selected_gen_num_for_detail)
        if not gen_data_for_display:
            st.error(f"Data for Generation {selected_gen_num_for_detail} could not be loaded or is missing.")
        else:
            st.subheader(f"Summary for Generation {selected_gen_num_for_detail}")
            summary_metrics_data = gen_data_for_display.get("generation_summary_metrics", {})
            
            metric_cols = st.columns(4)
            metric_cols[0].metric("Avg Fitness", f"{summary_metrics_data.get('avg_fitness', 0.0):.3f}")
            metric_cols[1].metric("Max Fitness", f"{summary_metrics_data.get('max_fitness', 0.0):.3f}")
            metric_cols[2].metric("Avg Wealth", f"{summary_metrics_data.get('avg_wealth', 0.0):.2f}")
            metric_cols[3].metric("Games Played", summary_metrics_data.get('total_games_played_in_generation', "N/A"))

            st.markdown("##### Population Snapshot")
            population_state_list = gen_data_for_display.get("population_state", [])
            if population_state_list:
                agents_table_data = []
                fitness_map_for_gen_detail = summary_metrics_data.get("fitness_scores_map", {})
                for agent_rec in population_state_list:
                    agent_id_val = agent_rec.get("agent_id")
                    fitness_val = agent_rec.get("fitness_score", fitness_map_for_gen_detail.get(agent_id_val))
                    genome_data_val = agent_rec.get("genome", {})
                    
                    agents_table_data.append({
                        "Agent ID (Short)": get_agent_display_name(agent_id_val),
                        "Parent ID (Short)": get_agent_display_name(agent_rec.get('parent_id')) if agent_rec.get('parent_id') else "N/A (Gen 1)",
                        "Progenitor (Short)": get_agent_display_name(progenitor_map.get(agent_id_val)),
                        "Wealth": float(agent_rec.get("wealth", 0.0)),
                        "Fitness": float(fitness_val) if fitness_val is not None else None,
                        "Genome Size": len(genome_data_val),
                        "_raw_agent_id": agent_id_val # For potential navigation
                    })
                
                df_agents_snapshot = pd.DataFrame(agents_table_data)
                st.dataframe(
                    df_agents_snapshot[["Agent ID (Short)", "Parent ID (Short)", "Progenitor (Short)", "Wealth", "Fitness", "Genome Size"]], 
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Wealth": st.column_config.NumberColumn(format="%.2f"),
                        "Fitness": st.column_config.NumberColumn(format="%.3f")
                    }
                )
            else:
                st.caption("No population data available for this generation.")

            # Population Genome Heatmap (Top N features by absolute activation magnitude)
            st.markdown("##### Population Genome Heatmap (Top Features by Max Absolute Activation)")
            if population_state_list:
                feature_max_abs_activations_map = defaultdict(float)
                for agent_rec in population_state_list:
                    genome_data = agent_rec.get("genome", {})
                    for f_uuid_val, f_data_val in genome_data.items():
                        act_val = f_data_val.get('activation',0.0) if isinstance(f_data_val, dict) else float(f_data_val or 0.0)
                        if abs(act_val) > feature_max_abs_activations_map[f_uuid_val]:
                            feature_max_abs_activations_map[f_uuid_val] = abs(act_val)
                
                # Select top N features based on their maximum absolute activation in this generation
                num_features_for_heatmap = 5 
                sorted_features_by_max_abs = sorted(
                    feature_max_abs_activations_map.items(), key=lambda item: item[1], reverse=True
                )
                top_n_feature_uuids_heatmap = [item[0] for item in sorted_features_by_max_abs[:num_features_for_heatmap]]

                if top_n_feature_uuids_heatmap:
                    heatmap_plot_data = []
                    for agent_rec in population_state_list:
                        agent_id_short_name = get_agent_display_name(agent_rec.get('agent_id'))
                        genome_data = agent_rec.get("genome", {})
                        for f_uuid_heatmap in top_n_feature_uuids_heatmap:
                            feature_raw_val = genome_data.get(f_uuid_heatmap)
                            activation_for_heatmap = 0.0
                            if feature_raw_val is not None:
                                activation_for_heatmap = feature_raw_val.get('activation',0.0) if isinstance(feature_raw_val, dict) else float(feature_raw_val or 0.0)
                            
                            heatmap_plot_data.append({
                                "agent_label": agent_id_short_name,
                                "feature_display_label": get_feature_display_name(f_uuid_heatmap, all_active_features.get(f_uuid_heatmap)),
                                "activation_level": activation_for_heatmap
                            })
                    
                    if heatmap_plot_data:
                        df_heatmap_plot = pd.DataFrame(heatmap_plot_data)
                        genome_heatmap_chart = alt.Chart(df_heatmap_plot).mark_rect().encode(
                            x=alt.X('feature_display_label:N', title='Feature', sort=None), 
                            y=alt.Y('agent_label:N', title='Agent', sort=None),
                            color=alt.Color('activation_level:Q', 
                                            scale=alt.Scale(scheme='redblue', domain=[-1, 1], clamp=True), 
                                            legend=alt.Legend(title="Activation Level")),
                            tooltip=['agent_label', 'feature_display_label', alt.Tooltip('activation_level:Q', format=".3f")]
                        ).properties(title=f"Top {num_features_for_heatmap} Feature Activations (Generation {selected_gen_num_for_detail})").interactive()
                        st.altair_chart(genome_heatmap_chart, use_container_width=True)
                    else:
                        st.caption("Not enough data to render genome heatmap.")
                else:
                    st.caption("No features found with significant activations for heatmap display.")
            else:
                st.caption("No population data to generate genome heatmap.")

# --- 👤 Agent Detail Tab ---
with tab_container_map["👤 Agent Detail"]:
    st.header("👤 Agent Detail Viewer")

    # Generation selection for agent context.
    # If only one generation exists, it must be 1. Otherwise, allow selection via slider.
    if latest_gen_num == 1:
        selected_gen_for_agent = 1
        st.markdown("Agent context from Generation 1 (only generation available).")
        # Check if navigation state is consistent.
        if st.session_state.nav_to_generation is not None and st.session_state.nav_to_generation != 1:
            st.warning(
                f"Navigation previously targeted generation {st.session_state.nav_to_generation} for agent context, "
                f"but only generation 1 is currently available. Using generation 1."
            )
    else: # latest_gen_num > 1
        # Determine the default value for the slider.
        # Default to the latest generation, unless a specific generation is navigated to.
        default_agent_gen_value = latest_gen_num
        if st.session_state.nav_to_generation is not None and \
           1 <= st.session_state.nav_to_generation <= latest_gen_num:
            default_agent_gen_value = st.session_state.nav_to_generation
        
        selected_gen_for_agent = st.slider(
            "Select Generation for Agent Context", 
            min_value=1, 
            max_value=latest_gen_num, 
            value=default_agent_gen_value, 
            key="agent_detail_gen_slider"
        )
    
    # Update session state with the selected generation, useful for navigation persistence.
    st.session_state.nav_to_generation = selected_gen_for_agent

    if selected_gen_for_agent and selected_gen_for_agent in all_gen_data:
        agents_in_selected_gen = all_gen_data[selected_gen_for_agent].get("population_state", [])
        agent_options = {a['agent_id']: get_agent_display_name(a['agent_id']) for a in agents_in_selected_gen}

        if not agent_options:
            st.info(f"No agents found in Generation {selected_gen_for_agent}.")
        else:
            default_agent_id = list(agent_options.keys())[0]
            if st.session_state.nav_to_agent_id and st.session_state.nav_to_agent_id in agent_options:
                default_agent_id = st.session_state.nav_to_agent_id
                # Consume nav_to_agent_id after use if this is the target generation
                if selected_gen_for_agent == st.session_state.get('nav_to_generation_for_agent', selected_gen_for_agent):
                     st.session_state.nav_to_agent_id = None 
            
            selected_agent_id = st.selectbox(
                "Select Agent ID",
                options=list(agent_options.keys()),
                format_func=lambda aid: agent_options[aid],
                index = list(agent_options.keys()).index(default_agent_id) if default_agent_id in agent_options else 0,
                key="agent_detail_agent_select"
            )

            if selected_agent_id:
                agent_data = next((a for a in agents_in_selected_gen if a['agent_id'] == selected_agent_id), None)
                if agent_data:
                    st.subheader(f"Agent: {get_agent_display_name(selected_agent_id)} (Generation {selected_gen_for_agent})")
                    
                    parent_id = agent_data.get('parent_id')
                    progenitor_id_agent = progenitor_map.get(selected_agent_id)
                    
                    cols_agent_info = st.columns(3)
                    cols_agent_info[0].metric("Wealth", f"{agent_data.get('wealth', 0):.2f}")
                    fitness_map_agent = all_gen_data[selected_gen_for_agent].get("generation_summary_metrics", {}).get("fitness_scores_map", {})
                    fitness_agent = agent_data.get("fitness_score", fitness_map_agent.get(selected_agent_id))
                    cols_agent_info[1].metric("Fitness", f"{fitness_agent:.3f}" if fitness_agent is not None else "N/A")
                    cols_agent_info[2].metric("Model ID", agent_data.get("model_id", "N/A"))

                    if parent_id:
                        create_nav_button(f"View Parent: {get_agent_display_name(parent_id)}", "👤 Agent Detail", "nav_to_agent_id", parent_id, params={'nav_to_generation': selected_gen_for_agent-1 if selected_gen_for_agent > 1 else 1})
                    if progenitor_id_agent:
                         create_nav_button(f"Explore Lineage of {get_agent_display_name(progenitor_id_agent)}", "👑 Lineage Explorer", "nav_to_lineage_progenitor", progenitor_id_agent)

                    st.markdown("##### Genome Activations")
                    genome = agent_data.get("genome", {})
                    if genome:
                        genome_list = []
                        for f_uuid, f_data in genome.items():
                            act_val = f_data.get('activation') if isinstance(f_data, dict) else f_data
                            label_val = f_data.get('label') if isinstance(f_data, dict) else all_active_features.get(f_uuid, "Unknown")
                            genome_list.append({"Feature Label": label_val, "Activation": act_val, "UUID": f_uuid})
                        
                        df_genome = pd.DataFrame(genome_list).sort_values(by="Activation", key=abs, ascending=False)
                        
                        col_gt, col_gc = st.columns([0.6, 0.4])
                        with col_gt:
                            # Display genome with clickable feature labels
                            st.markdown("##### Genome Details (Click Feature Label to Explore)")
                            for _, row in df_genome.iterrows():
                                feature_uuid_for_nav = row["UUID"]
                                feature_label_for_nav = row["Feature Label"]
                                activation_val = row["Activation"]
                                
                                # Use columns for better layout of button and activation
                                cols_genome_row = st.columns([0.7, 0.3])
                                with cols_genome_row[0]:
                                    if st.button(f"{feature_label_for_nav}", key=f"nav_genome_feature_{selected_agent_id}_{feature_uuid_for_nav}", help=f"Explore feature {feature_label_for_nav}"):
                                        st.session_state.active_tab = "🧬 Feature Explorer"
                                        st.session_state.nav_to_feature_uuid = feature_uuid_for_nav
                                        st.rerun()
                                with cols_genome_row[1]:
                                    st.markdown(f"`{activation_val:.3f}`")
                            st.caption("Sorted by absolute activation value.")


                        with col_gc:
                            # Chart top/bottom N by absolute activation
                            n_to_show_genome_chart = min(10, len(df_genome))
                            chart_df_genome = df_genome.nlargest(n_to_show_genome_chart, 'Activation', keep='all').append(
                                df_genome.nsmallest(n_to_show_genome_chart, 'Activation', keep='all')
                            ).drop_duplicates().reset_index(drop=True)
                            
                            if not chart_df_genome.empty:
                                genome_bar_chart = alt.Chart(chart_df_genome).mark_bar().encode(
                                    x='Activation:Q',
                                    y=alt.Y('Feature Label:N', sort='-x', title="Feature"),
                                    tooltip=['Feature Label:N', alt.Tooltip('Activation:Q', format=".3f"), 'UUID:N'],
                                    color=alt.condition(alt.datum.Activation > 0, alt.value('steelblue'), alt.value('orange'))
                                ).properties(title="Strongest Genome Activations").interactive()
                                st.altair_chart(genome_bar_chart, use_container_width=True)

                        # Display Evolutionary Inputs
                        if selected_gen_for_agent > 1: # Only for agents not in Gen 1
                            pos_inputs = agent_data.get('evolutionary_input_positive_features', [])
                            neg_inputs = agent_data.get('evolutionary_input_negative_features', [])
                            if pos_inputs or neg_inputs:
                                st.markdown("---")
                                st.markdown("##### Evolutionary Inputs (From Parent's Last Round)")
                            if pos_inputs:
                                st.markdown("**Features Reinforced (Parent Won With These):**")
                                for i, f_uuid_pos in enumerate(pos_inputs): # Use enumerate to get an index
                                    create_nav_button(
                                        get_feature_display_name(f_uuid_pos, all_active_features.get(f_uuid_pos)), 
                                        "🧬 Feature Explorer", 
                                        "nav_to_feature_uuid", 
                                        f_uuid_pos,
                                        key_suffix=f"pos_input_feature_{i}" # Indexed suffix for uniqueness
                                    )
                            if neg_inputs:
                                st.markdown("**Features Suppressed (Parent Lost With These):**")
                                for i, f_uuid_neg in enumerate(neg_inputs): # Use enumerate to get an index
                                    create_nav_button(
                                        get_feature_display_name(f_uuid_neg, all_active_features.get(f_uuid_neg)), 
                                        "🧬 Feature Explorer", 
                                        "nav_to_feature_uuid", 
                                        f_uuid_neg,
                                        key_suffix=f"neg_input_feature_{i}" # Indexed suffix for uniqueness
                                    )
                                st.markdown("---")


                    # Games played by this agent in this generation
                    st.markdown(f"##### Games Played by {get_agent_display_name(selected_agent_id)} (Gen {selected_gen_for_agent})")
                    games_this_gen = load_games_for_generation_cached(st.session_state.selected_run_id, selected_gen_for_agent, state_dir_input)
                    agent_games = [g for g in games_this_gen if selected_agent_id in [g.get("player_A_id"), g.get("player_B_id")]]
                    
                    if agent_games:
                        for game in agent_games:
                            is_player_A = game.get("player_A_id") == selected_agent_id
                            opponent_id = game.get("player_B_id") if is_player_A else game.get("player_A_id")
                            agent_role = game.get("player_A_game_role") if is_player_A else game.get("player_B_game_role")
                            wealth_change = game.get("wealth_changes", {}).get("player_A_wealth_change" if is_player_A else "player_B_wealth_change", "N/A")
                            outcome = game.get("adjudication_result", "Unknown")
                            
                            agent_game_outcome_display = "N/A"
                            if outcome not in ["Unknown", "Error", "Critical Game Error", "error"] and agent_role:
                                if "Tie" in outcome: agent_game_outcome_display = "Tie"
                                elif (outcome == "Role A Wins" and agent_role == "Role A") or \
                                     (outcome == "Role B Wins" and agent_role == "Role B"):
                                    agent_game_outcome_display = "Win"
                                else: agent_game_outcome_display = "Loss"

                            game_id_short = game.get("game_id", "N/A_Game")[:15]
                            with st.expander(f"Game vs {get_agent_display_name(opponent_id)} ({game_id_short}...) - Outcome: {agent_game_outcome_display}, ΔWealth: {wealth_change}"):
                                st.json(game, expanded=False) # Simple display, could be more formatted
                                create_nav_button(f"View Full Game {game_id_short}", "📜 Game Viewer", "nav_to_agent_id", game.get("game_id"), # using agent_id key for game_id
                                                  params={'nav_to_generation': selected_gen_for_agent})


# --- 📜 Game Viewer Tab ---
with tab_container_map["📜 Game Viewer"]:
    st.header("📜 Game Viewer")
    # Similar logic to original, but using cached game data
    # Nav state: st.session_state.nav_to_agent_id will hold game_id here
    # Nav state: st.session_state.nav_to_generation will hold generation_number
    
    # Determine the generation for game viewing.
    # If only one generation exists, it must be 1. Otherwise, allow selection via slider.
    if latest_gen_num == 1:
        selected_gen_for_games = 1
        st.markdown("Viewing games from Generation 1 (only generation available).")
        # Check if navigation state is consistent.
        if st.session_state.nav_to_generation is not None and st.session_state.nav_to_generation != 1:
            st.warning(
                f"Navigation previously targeted generation {st.session_state.nav_to_generation} for game viewing, "
                f"but only generation 1 is currently available. Using generation 1."
            )
    else: # latest_gen_num > 1
        # Determine the default value for the slider.
        # Default to the latest generation, unless a specific generation is navigated to.
        default_game_gen_value = latest_gen_num
        if st.session_state.nav_to_generation is not None and \
           1 <= st.session_state.nav_to_generation <= latest_gen_num:
            default_game_gen_value = st.session_state.nav_to_generation
        
        selected_gen_for_games = st.slider(
            "Select Generation for Games", 
            min_value=1, 
            max_value=latest_gen_num, 
            value=default_game_gen_value, 
            key="game_view_gen_slider"
        )
    
    st.session_state.nav_to_generation = selected_gen_for_games

    if selected_gen_for_games:
        games_in_gen = load_games_for_generation_cached(st.session_state.selected_run_id, selected_gen_for_games, state_dir_input)
        if not games_in_gen:
            st.info(f"No game records found for Generation {selected_gen_for_games}.")
        else:
            game_options = {g.get("game_id", f"UnknownGame_{i}"): f"{g.get('game_id', f'UnknownGame_{i}')[:15]}... (Players: {get_agent_display_name(g.get('player_A_id'))} vs {get_agent_display_name(g.get('player_B_id'))})" for i, g in enumerate(games_in_gen)}
            
            default_game_id_gv = list(game_options.keys())[0]
            if st.session_state.nav_to_agent_id and st.session_state.nav_to_agent_id in game_options: # nav_to_agent_id stores game_id for this tab
                default_game_id_gv = st.session_state.nav_to_agent_id
                # Consume nav states if this is the target generation
                if selected_gen_for_games == st.session_state.get('nav_to_generation_for_game', selected_gen_for_games):
                    st.session_state.nav_to_agent_id = None
                    st.session_state.nav_to_generation = None


            selected_game_id_viewer = st.selectbox(
                "Select Game ID",
                options=list(game_options.keys()),
                format_func=lambda gid: game_options[gid],
                index=list(game_options.keys()).index(default_game_id_gv) if default_game_id_gv in game_options else 0,
                key="game_viewer_game_select"
            )

            if selected_game_id_viewer:
                game_to_display = next((g for g in games_in_gen if g.get("game_id") == selected_game_id_viewer), None)
                if game_to_display:
                    # Display logic similar to original, but more structured
                    pA_id = game_to_display.get('player_A_id')
                    pB_id = game_to_display.get('player_B_id')
                    pA_role = game_to_display.get('player_A_game_role', 'Role A')
                    pB_role = game_to_display.get('player_B_game_role', 'Role B')

                    st.subheader(f"Game: {selected_game_id_viewer}")
                    game_info_cols = st.columns(2)
                    with game_info_cols[0]:
                        st.markdown(f"**Player A ({pA_role}):**")
                        create_nav_button(
                            get_agent_display_name(pA_id), 
                            "👤 Agent Detail", 
                            "nav_to_agent_id", 
                            pA_id, 
                            params={'nav_to_generation': selected_gen_for_games},
                            key_suffix="playerA_game_viewer_link" # Unique suffix for Player A button
                        )
                    with game_info_cols[1]:
                        st.markdown(f"**Player B ({pB_role}):**")
                        create_nav_button(
                            get_agent_display_name(pB_id), 
                            "👤 Agent Detail", 
                            "nav_to_agent_id", 
                            pB_id, 
                            params={'nav_to_generation': selected_gen_for_games},
                            key_suffix="playerB_game_viewer_link" # Unique suffix for Player B button
                        )
                    
                    st.markdown(f"**Scenario Proposer:** {get_agent_display_name(game_to_display.get('proposer_agent_id'))}")

                    with st.expander("📜 Scenario Details", expanded=True):
                        st.text_area("Scenario Text", game_to_display.get("scenario_text", "N/A"), height=200, disabled=True)
                        # Could add raw scenario LLM output here if desired

                    st.markdown("##### 💬 Conversation Transcript")
                    transcript = game_to_display.get("transcript", [])
                    if transcript:
                        for turn in transcript:
                            speaker_role = turn.get('role', 'Unknown')
                            speaker_agent_id = turn.get('agent_id')
                            avatar = "🧑‍💻" if speaker_role == pA_role else "🤖" if speaker_role == pB_role else "👤"
                            with st.chat_message(name=speaker_role, avatar=avatar):
                                st.write(f"**{speaker_role} ({get_agent_display_name(speaker_agent_id)})**: ")
                                st.markdown(turn.get('content', ''))
                    else: st.info("No transcript.")

                    st.markdown("##### ⚖️ Adjudication & Outcome")
                    adj_result = game_to_display.get("adjudication_result", "N/A")
                    # Adjudication prompts
                    adj_prompt_scratchpad = game_to_display.get("adjudication_prompt_scratchpad")
                    adj_prompt_outcome_ids = game_to_display.get("adjudication_prompt_outcome_ids")
                    # Adjudication raw outputs
                    adj_raw_llm_output_scratchpad = game_to_display.get("adjudication_raw_llm_output_scratchpad")
                    adj_raw_llm_output_outcome_ids = game_to_display.get("adjudication_raw_llm_output_outcome_ids")
                    # Parsed scratchpad content
                    adj_scratchpad_content = game_to_display.get("adjudication_scratchpad")
                    
                    wealth_ch = game_to_display.get("wealth_changes", {})
                    final_a_wealth = game_to_display.get("final_player_A_wealth")
                    final_b_wealth = game_to_display.get("final_player_B_wealth")
                    default_tie_reason = game_to_display.get("defaulted_to_tie_reason")
                    betting_details = game_to_display.get("betting_details")

                    st.metric("Adjudication Result", str(adj_result)) # it's a string

                    if default_tie_reason:
                        st.caption(f"Note on Outcome: {default_tie_reason}")
                    
                    col_wealth_outcome1, col_wealth_outcome2 = st.columns(2)
                    with col_wealth_outcome1:
                        player_A_wealth_change_val = wealth_ch.get('player_A_wealth_change', 0.0)
                        st.write(f"Player A ({pA_role}) Wealth Change: {player_A_wealth_change_val:+.2f}")
                        if final_a_wealth is not None:
                            st.write(f"Player A ({pA_role}) Final Wealth: {float(final_a_wealth):.2f}")
                        else:
                            st.write(f"Player A ({pA_role}) Final Wealth: N/A")
                    
                    with col_wealth_outcome2:
                        player_B_wealth_change_val = wealth_ch.get('player_B_wealth_change', 0.0)
                        st.write(f"Player B ({pB_role}) Wealth Change: {player_B_wealth_change_val:+.2f}")
                        if final_b_wealth is not None:
                            st.write(f"Player B ({pB_role}) Final Wealth: {float(final_b_wealth):.2f}")
                        else:
                            st.write(f"Player B ({pB_role}) Final Wealth: N/A")

                    if betting_details:
                        with st.expander("Betting Details"):
                            st.json(betting_details)

                    if adj_scratchpad_content:
                        with st.expander("Adjudicator Scratchpad Content (Parsed from 1st Call)", expanded=False):
                            st.text_area("Parsed Scratchpad", value=adj_scratchpad_content, height=150, disabled=True, key=f"adj_scratchpad_content_{selected_game_id_viewer}")

                    if adj_prompt_scratchpad:
                        with st.expander("Adjudication Prompt (Scratchpad Call - 1st Call)", expanded=False):
                            st.text_area("Prompt for Scratchpad Generation", value=adj_prompt_scratchpad, height=200, disabled=True, key=f"adj_prompt_scratchpad_{selected_game_id_viewer}")
                    if adj_raw_llm_output_scratchpad:
                        with st.expander("Raw LLM Output (Scratchpad Call - 1st Call)", expanded=False):
                             st.text_area("Raw LLM Output for Scratchpad Generation", value=adj_raw_llm_output_scratchpad, height=100, disabled=True, key=f"adj_raw_scratchpad_{selected_game_id_viewer}")
                    
                    if adj_prompt_outcome_ids:
                        with st.expander("Adjudication Prompt (Outcome/IDs Call - 2nd Call)", expanded=False):
                            st.text_area("Prompt for Outcome/IDs Determination", value=adj_prompt_outcome_ids, height=200, disabled=True, key=f"adj_prompt_outcome_ids_{selected_game_id_viewer}")
                    if adj_raw_llm_output_outcome_ids: 
                        with st.expander("Raw LLM Output (Outcome/IDs Call - 2nd Call)", expanded=False):
                             st.text_area("Raw LLM Output for Outcome/IDs Determination", value=adj_raw_llm_output_outcome_ids, height=100, disabled=True, key=f"adj_raw_outcome_ids_{selected_game_id_viewer}")

# --- 🗓️ Event Log Tab ---
with tab_container_map["🗓️ Event Log"]:
    st.header("🗓️ Event Log (Normal Milestones)")
    if interesting_events:
        df_events = pd.DataFrame(interesting_events)
        st.dataframe(df_events[["generation", "type", "description", "timestamp"]], use_container_width=True, hide_index=True)
    else:
        st.info("No interesting events logged for this run yet.")
