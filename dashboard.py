import streamlit as st
import pandas as pd
import json
from pathlib import Path
import altair as alt # For plotting
import os # For listing directories
import re # For sorting generation files naturally

# --- Configuration ---
DEFAULT_STATE_BASE_DIR = "simulation_state"
CONFIG_FILENAME = "config_snapshot.json"
LATEST_GEN_TRACKER_FILENAME = "_latest_generation_number.txt"
GENERATION_FILE_PATTERN = r"generation_(\d+)\.json"
GAMES_FILE_PATTERN = r"games_generation_(\d+)\.jsonl"

# --- Utility Functions ---

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Key for natural sorting (e.g., gen_2 before gen_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

@st.cache_data(ttl=60) # Cache for 60 seconds to allow for semi-live updates
def get_available_simulation_runs(state_base_dir: str) -> list[str]:
    """Scans the state directory for available simulation run IDs."""
    base_path = Path(state_base_dir)
    if not base_path.is_dir():
        st.sidebar.error(f"State directory not found: {state_base_dir}")
        return []
    
    run_ids = []
    for item in base_path.iterdir():
        if item.is_dir() and (item / LATEST_GEN_TRACKER_FILENAME).exists():
            run_ids.append(item.name)
    return sorted(run_ids, reverse=True) # Show newest first

@st.cache_data(ttl=60)
def load_config_snapshot(run_id: str, state_base_dir: str) -> dict | None:
    """Loads the config snapshot for a given run."""
    if not run_id: return None
    config_path = Path(state_base_dir) / run_id / CONFIG_FILENAME
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading config for {run_id}: {e}")
            return None
    return None

@st.cache_data(ttl=60)
def get_latest_generation_number(run_id: str, state_base_dir: str) -> int | None:
    """Reads the latest successfully saved generation number for a run."""
    if not run_id: return None
    tracker_path = Path(state_base_dir) / run_id / LATEST_GEN_TRACKER_FILENAME
    if tracker_path.exists():
        try:
            with open(tracker_path, 'r') as f:
                return int(f.read().strip())
        except Exception as e:
            st.error(f"Error reading latest generation tracker for {run_id}: {e}")
            return None
    return None

@st.cache_data(ttl=60)
def load_generation_data(run_id: str, generation_number: int, state_base_dir: str) -> dict | None:
    """Loads the state data for a specific generation of a run."""
    if not run_id: return None
    gen_file_path = Path(state_base_dir) / run_id / f"generation_{generation_number:04d}.json"
    if gen_file_path.exists():
        try:
            with open(gen_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading data for generation {generation_number} of run {run_id}: {e}")
            return None
    return None

@st.cache_data(ttl=60)
def load_all_generation_summary_data(run_id: str, state_base_dir: str) -> pd.DataFrame:
    """Loads summary metrics from all available generation files for a run."""
    if not run_id: return pd.DataFrame()
    
    latest_gen = get_latest_generation_number(run_id, state_base_dir)
    if latest_gen is None:
        return pd.DataFrame()

    all_summaries = []
    for gen_num in range(1, latest_gen + 1): # Assuming generations start from 1
        gen_data = load_generation_data(run_id, gen_num, state_base_dir)
        if gen_data and "generation_summary_metrics" in gen_data:
            summary = gen_data["generation_summary_metrics"]
            summary["generation_number"] = gen_data["generation_number"]
            summary["timestamp_completed"] = gen_data.get("timestamp_completed")
            all_summaries.append(summary)
    
    return pd.DataFrame(all_summaries)

@st.cache_data(ttl=60)
def load_games_for_generation(run_id: str, generation_number: int, state_base_dir: str) -> list[dict]:
    """Loads all game details for a specific generation from its .jsonl file."""
    if not run_id: return []
    games_file_path = Path(state_base_dir) / run_id / f"games_generation_{generation_number:04d}.jsonl"
    games_data = []
    if games_file_path.exists():
        try:
            with open(games_file_path, 'r') as f:
                for line in f:
                    games_data.append(json.loads(line))
        except Exception as e:
            st.error(f"Error loading games for generation {generation_number} of run {run_id}: {e}")
    return games_data


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="SAEvolution Dashboard")
st.title("🚀 Socio-Cultural Agent Evolution Dashboard")

# --- Sidebar for Global Controls ---
st.sidebar.header("Simulation Controls")
state_dir = st.sidebar.text_input("State Directory Path", value=DEFAULT_STATE_BASE_DIR)

available_runs = get_available_simulation_runs(state_dir)

if not available_runs:
    st.sidebar.warning("No simulation runs found in the specified state directory.")
    st.info("No simulation data to display. Please the simulation has run and saved state, and the 'State Directory Path' is correct.")
    st.stop()

selected_run_id = st.sidebar.selectbox("Select Simulation Run ID", options=available_runs)

if st.sidebar.button("Refresh Data"):
    # Clear all memoized functions to reload data
    st.cache_data.clear()
    st.rerun()

if not selected_run_id:
    st.info("Please select a Simulation Run ID from the sidebar.")
    st.stop()

# --- Main Content Area ---

# Load core data for the selected run
config_data = load_config_snapshot(selected_run_id, state_dir)
latest_gen_num_for_run = get_latest_generation_number(selected_run_id, state_dir)
all_gen_summary_df = load_all_generation_summary_data(selected_run_id, state_dir)


tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview & Curves", 
    "🔬 Generation Explorer", 
    "👤 Agent Detail", 
    "📜 Game Viewer"
])

# --- Tab 1: Overview & Curves ---
with tab1:
    st.header(f"Run Overview: {selected_run_id}")
    if config_data:
        with st.expander("Simulation Configuration"):
            st.json(config_data, expanded=False)
    else:
        st.warning("Configuration snapshot not found for this run.")

    if latest_gen_num_for_run is not None:
        st.metric("Generations Completed", latest_gen_num_for_run)
    else:
        st.metric("Generations Completed", "N/A (No data)")

    if not all_gen_summary_df.empty:
        st.subheader("Performance Curves")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Fitness Over Generations")
            fitness_df = all_gen_summary_df[["generation_number", "avg_fitness", "min_fitness", "max_fitness"]].melt(
                id_vars=["generation_number"], var_name="Metric", value_name="Fitness"
            )
            fitness_chart = alt.Chart(fitness_df).mark_line(point=True).encode(
                x=alt.X('generation_number:Q', title='Generation'),
                y=alt.Y('Fitness:Q', title='Fitness Value'),
                color='Metric:N',
                tooltip=['generation_number', 'Metric', 'Fitness']
            ).interactive()
            st.altair_chart(fitness_chart, use_container_width=True)

        with col2:
            st.markdown("##### Wealth Over Generations")
            # Assuming 'avg_wealth' is in summary, add min/max if available or calculated
            if "avg_wealth" in all_gen_summary_df.columns:
                wealth_df = all_gen_summary_df[["generation_number", "avg_wealth"]].copy()
                wealth_df.rename(columns={"avg_wealth": "Average Wealth"}, inplace=True)
                
                wealth_chart = alt.Chart(wealth_df).mark_line(point=True).encode(
                    x=alt.X('generation_number:Q', title='Generation'),
                    y=alt.Y('Average Wealth:Q', title='Wealth Value'),
                    tooltip=['generation_number', 'Average Wealth']
                ).interactive()
                st.altair_chart(wealth_chart, use_container_width=True)
            else:
                st.caption("Average wealth data not found in generation summaries.")
        
        st.markdown("##### Game Statistics Over Generations")
        if "total_games_played_in_generation" in all_gen_summary_df.columns:
            games_chart = alt.Chart(all_gen_summary_df).mark_bar().encode(
                x=alt.X('generation_number:Q', title='Generation', bin=alt.Bin(maxbins=max(1,latest_gen_num_for_run or 1))), # at least 1 bin
                y=alt.Y('total_games_played_in_generation:Q', title='Games Played'),
                tooltip=['generation_number', 'total_games_played_in_generation']
            ).interactive()
            st.altair_chart(games_chart, use_container_width=True)
        else:
            st.caption("Total games played data not found.")

    else:
        st.info("No generation summary data found to plot curves for this run.")


# --- Tab 2: Generation Explorer ---
with tab2:
    st.header("Generation Explorer")
    if latest_gen_num_for_run is None:
        st.warning("No generation data available for this run.")
    else:
        gen_numbers = list(range(1, latest_gen_num_for_run + 1))
        selected_gen_num = st.selectbox(
            "Select Generation", 
            options=gen_numbers, 
            format_func=lambda x: f"Generation {x}",
            index=len(gen_numbers)-1 # Default to latest
        )

        if selected_gen_num:
            gen_data = load_generation_data(selected_run_id, selected_gen_num, state_dir)
            if gen_data:
                st.subheader(f"Details for Generation {selected_gen_num}")
                
                summary_metrics = gen_data.get("generation_summary_metrics", {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Fitness", f"{summary_metrics.get('avg_fitness', 0):.3f}")
                col2.metric("Max Fitness", f"{summary_metrics.get('max_fitness', 0):.3f}")
                col3.metric("Avg Wealth", f"{summary_metrics.get('avg_wealth', 0):.2f}")
                col4.metric("Games Played", summary_metrics.get('total_games_played_in_generation', 'N/A'))

                st.markdown("##### Agents in this Generation")
                population_state = gen_data.get("population_state", [])
                if population_state:
                    agents_df_data = []
                    for agent_s in population_state:
                        agents_df_data.append({
                            "Agent ID": agent_s.get("agent_id"),
                            "Model ID": agent_s.get("model_id"),
                            "Wealth": agent_s.get("wealth"),
                            "Fitness": agent_s.get("fitness_score", summary_metrics.get("fitness_scores_map", {}).get(agent_s.get("agent_id"))), # Try to get fitness
                            "Genome Size": len(agent_s.get("genome", {})),
                        })
                    agents_df = pd.DataFrame(agents_df_data)
                    st.dataframe(agents_df, use_container_width=True)
                else:
                    st.info("No agent population data found for this generation.")
            else:
                st.error(f"Could not load data for Generation {selected_gen_num}.")


# --- Tab 3: Agent Detail ---
with tab3:
    st.header("Agent Detail Viewer")
    if latest_gen_num_for_run is None:
        st.warning("No generation data available to select agents from.")
    else:
        gen_numbers_for_agent = list(range(1, latest_gen_num_for_run + 1))
        selected_gen_for_agent = st.selectbox(
            "Select Generation (for Agent)", 
            options=gen_numbers_for_agent, 
            format_func=lambda x: f"Generation {x}",
            index=len(gen_numbers_for_agent)-1, # Default to latest
            key="agent_detail_gen_select"
        )

        if selected_gen_for_agent:
            gen_data_for_agent = load_generation_data(selected_run_id, selected_gen_for_agent, state_dir)
            if gen_data_for_agent and "population_state" in gen_data_for_agent:
                agent_ids = [a["agent_id"] for a in gen_data_for_agent["population_state"]]
                if not agent_ids:
                    st.info("No agents found in the selected generation.")
                else:
                    selected_agent_id = st.selectbox(
                        "Select Agent ID", 
                        options=agent_ids,
                        key="agent_detail_agent_select"
                    )
                    
                    if selected_agent_id:
                        agent_data = next((a for a in gen_data_for_agent["population_state"] if a["agent_id"] == selected_agent_id), None)
                        if agent_data:
                            st.subheader(f"Agent: {selected_agent_id} (Generation {selected_gen_for_agent})")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Wealth", f"{agent_data.get('wealth', 0):.2f}")
                            fitness_val = agent_data.get("fitness_score")
                            if fitness_val is None: # Try to get from gen summary if not directly on agent
                                fitness_val = gen_data_for_agent.get("generation_summary_metrics", {}).get("fitness_scores_map", {}).get(selected_agent_id)

                            col2.metric("Fitness", f"{fitness_val:.3f}" if fitness_val is not None else "N/A")
                            col3.metric("Model ID", agent_data.get("model_id", "N/A"))

                            st.markdown("##### Genome Activations")
                            genome = agent_data.get("genome", {})
                            if genome:
                                genome_df = pd.DataFrame(list(genome.items()), columns=['Feature UUID', 'Activation'])
                                st.dataframe(genome_df, height=200, use_container_width=True)
                            else:
                                st.info("No genome data for this agent.")

                            st.markdown("##### Games Played by this Agent (in this generation)")
                            games_this_gen = load_games_for_generation(selected_run_id, selected_gen_for_agent, state_dir)
                            agent_games = [
                                g for g in games_this_gen 
                                if g.get("player_A_id") == selected_agent_id or g.get("player_B_id") == selected_agent_id
                            ]
                            if agent_games:
                                games_display_data = []
                                for game in agent_games:
                                    outcome_str = game.get("adjudication_result", "Unknown")                                          
                                    wealth_c = game.get("wealth_changes") or {}
                                    agent_wealth_change = wealth_c.get("player_A_wealth_change") if game.get("player_A_id") == selected_agent_id else wealth_c.get("player_B_wealth_change")
                                    opponent_id = game.get("player_B_id") if game.get("player_A_id") == selected_agent_id else game.get("player_A_id")
                                    games_display_data.append({
                                        "Game ID": game.get("game_id"),
                                        "Opponent ID": opponent_id,
                                        "Outcome": outcome_str,
                                        "Wealth Change": agent_wealth_change,
                                        "Scenario (Start)": game.get("scenario_text", "")[:100]+"..."
                                    })
                                st.dataframe(pd.DataFrame(games_display_data), use_container_width=True)
                            else:
                                st.info("No game records found for this agent in this generation's game file.")
                        else:
                            st.error("Selected agent data not found.")
            else:
                st.warning(f"No population data for Generation {selected_gen_for_agent} to select an agent.")

# --- Tab 4: Game Viewer ---
with tab4:
    st.header("Game Viewer")
    if latest_gen_num_for_run is None:
        st.warning("No generation data available to select games from.")
    else:
        gen_numbers_for_games = list(range(1, latest_gen_num_for_run + 1))
        selected_gen_for_games = st.selectbox(
            "Select Generation (for Games)", 
            options=gen_numbers_for_games, 
            format_func=lambda x: f"Generation {x}",
            index=len(gen_numbers_for_games)-1, # Default to latest
            key="game_viewer_gen_select"
        )

        if selected_gen_for_games:
            games_data = load_games_for_generation(selected_run_id, selected_gen_for_games, state_dir)
            if not games_data:
                st.info(f"No game records found for Generation {selected_gen_for_games}.")
            else:
                game_ids = [g.get("game_id", f"UnknownGame_{i}") for i, g in enumerate(games_data)]
                selected_game_id = st.selectbox("Select Game ID", options=game_ids, key="game_viewer_game_select")
                
                if selected_game_id:
                    # Find the game data. Handle cases where game_id might be "UnknownGame_i"
                    game_to_display = None
                    if selected_game_id.startswith("UnknownGame_"):
                        try:
                            idx = int(selected_game_id.split("_")[-1])
                            game_to_display = games_data[idx]
                        except: pass
                    else:
                         game_to_display = next((g for g in games_data if g.get("game_id") == selected_game_id), None)

                    if game_to_display:
                        st.subheader(f"Details for Game: {game_to_display.get('game_id')}")
                        
                        st.markdown(f"""
                        - **Players:** {game_to_display.get('player_A_id')} (as {game_to_display.get('player_A_game_role')}) vs {game_to_display.get('player_B_id')} (as {game_to_display.get('player_B_game_role')})
                        - **Scenario Proposer:** {game_to_display.get('proposer_agent_id')}
                        - **Timestamp:** {game_to_display.get('timestamp_start')} to {game_to_display.get('timestamp_end')}
                        """)

                        with st.expander("Scenario Text", expanded=True):
                            st.text_area("Scenario", value=game_to_display.get("scenario_text", "N/A"), height=200, disabled=True)

                        st.markdown("##### Conversation Transcript")
                        transcript = game_to_display.get("transcript", [])
                        if transcript:
                            for turn in transcript:
                                role = turn.get('role', 'Unknown Role')
                                agent_id_speaker = turn.get('agent_id', 'Unknown Agent')
                                content = turn.get('content', '')
                                with st.chat_message(name=role, avatar="🧑‍💻" if role == game_to_display.get("player_A_game_role") else "🤖"): # Basic avatar differentiation
                                    st.write(f"**{role} ({agent_id_speaker[:8]}...)**: {content}")
                        else:
                            st.info("No transcript available for this game.")

                        st.markdown("##### Adjudication & Outcome")
                        st.markdown(f"- **Adjudication Result:** `{game_to_display.get('adjudication_result', 'N/A')}`")
                        if game_to_display.get('defaulted_to_tie_reason'):
                            st.caption(f"Defaulted Reason: {game_to_display.get('defaulted_to_tie_reason')}")
                        
                        bets = game_to_display.get("betting_details") or {}
                        st.markdown(f"- **Player A Bet:** {bets.get('player_A_bet', 'N/A')}, **Player B Bet:** {bets.get('player_B_bet', 'N/A')}")

                        wealth_c = game_to_display.get("wealth_changes") or {}
                        st.markdown(f"- **Player A Wealth Change:** {wealth_c.get('player_A_wealth_change', 'N/A')}")
                        st.markdown(f"- **Player B Wealth Change:** {wealth_c.get('player_B_wealth_change', 'N/A')}")

                    else:
                        st.error("Could not find details for the selected game ID.")
