import streamlit as st
import pandas as pd
import json
from pathlib import Path
import altair as alt
import re
from collections import Counter

# --- Configuration ---
DEFAULT_STATE_BASE_DIR = "simulation_state"
CONFIG_FILENAME = "config_snapshot.json"
LATEST_GEN_TRACKER_FILENAME = "_latest_generation_number.txt"

# --- Utility Functions ---

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Key for natural sorting (e.g., gen_2 before gen_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

@st.cache_data(ttl=30) # Shorter TTL for more frequent updates if sim is running
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
    return sorted(run_ids, reverse=True)

@st.cache_data(ttl=30)
def load_config_snapshot(run_id: str, state_base_dir: str) -> dict | None:
    if not run_id: return None
    config_path = Path(state_base_dir) / run_id / CONFIG_FILENAME
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading config for {run_id}: {e}")
    return None

@st.cache_data(ttl=30)
def get_latest_generation_number(run_id: str, state_base_dir: str) -> int | None:
    if not run_id: return None
    tracker_path = Path(state_base_dir) / run_id / LATEST_GEN_TRACKER_FILENAME
    if tracker_path.exists():
        try:
            with open(tracker_path, 'r') as f:
                return int(f.read().strip())
        except Exception as e:
            st.error(f"Error reading latest generation tracker for {run_id}: {e}")
    return None

@st.cache_data(ttl=30)
def load_generation_data(run_id: str, generation_number: int, state_base_dir: str) -> dict | None:
    if not run_id: return None
    gen_file_path = Path(state_base_dir) / run_id / f"generation_{generation_number:04d}.json"
    if gen_file_path.exists():
        try:
            with open(gen_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading data for generation {generation_number} of run {run_id}: {e}")
    return None

@st.cache_data(ttl=30)
def load_all_generation_summary_data(run_id: str, state_base_dir: str) -> pd.DataFrame:
    if not run_id: return pd.DataFrame()
    
    latest_gen = get_latest_generation_number(run_id, state_base_dir)
    if latest_gen is None:
        return pd.DataFrame()

    all_summaries = []
    for gen_num in range(1, latest_gen + 1):
        gen_data = load_generation_data(run_id, gen_num, state_base_dir)
        if gen_data:
            summary = gen_data.get("generation_summary_metrics", {})
            summary["generation_number"] = gen_data.get("generation_number", gen_num)
            summary["timestamp_completed"] = gen_data.get("timestamp_completed")
            
            # Calculate approximate genome diversity
            population_state = gen_data.get("population_state", [])
            if population_state:
                genome_strings = [json.dumps(agent.get("genome", {}), sort_keys=True) for agent in population_state]
                summary["unique_genomes_approx"] = len(set(genome_strings))
            else:
                summary["unique_genomes_approx"] = 0
            all_summaries.append(summary)
    
    return pd.DataFrame(all_summaries)

@st.cache_data(ttl=30)
def load_games_for_generation(run_id: str, generation_number: int, state_base_dir: str) -> list[dict]:
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

def get_agent_display_name(agent_id: str) -> str:
    return f"Agent {agent_id[:8]}..." if agent_id else "N/A"

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="SAEvolution Dashboard")
st.title("🚀 Socio-Cultural Agent Evolution Dashboard")

# Initialize session state for active tab and game viewer pre-selection
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📈 Overview & Curves"
if 'selected_game_id_from_agent_detail' not in st.session_state:
    st.session_state.selected_game_id_from_agent_detail = None
if 'selected_gen_for_game_viewer' not in st.session_state:
    st.session_state.selected_gen_for_game_viewer = None


# --- Sidebar ---
st.sidebar.header("Simulation Controls")
state_dir = st.sidebar.text_input("State Directory Path", value=DEFAULT_STATE_BASE_DIR)
available_runs = get_available_simulation_runs(state_dir)

if not available_runs:
    st.sidebar.warning("No simulation runs found. simulations have run and saved state.")
    st.info("No simulation data to display. Please check the 'State Directory Path' and run a simulation.")
    st.stop()

selected_run_id = st.sidebar.selectbox(
    "Select Simulation Run ID",
    options=available_runs,
    index=0,
    help="Choose a simulation run to inspect."
)

if st.sidebar.button("🔄 Refresh Data", help="Reload all data from the selected run."):
    st.cache_data.clear() # Clear all cached data
    # Reset potentially stale selections from a previous run/state
    st.session_state.selected_game_id_from_agent_detail = None
    st.session_state.selected_gen_for_game_viewer = None
    st.rerun()

if not selected_run_id:
    st.info("Please select a Simulation Run ID from the sidebar.")
    st.stop()

# --- Load Core Data ---
config_data = load_config_snapshot(selected_run_id, state_dir)
latest_gen_num_for_run = get_latest_generation_number(selected_run_id, state_dir)
all_gen_summary_df = load_all_generation_summary_data(selected_run_id, state_dir)

if latest_gen_num_for_run is None and selected_run_id:
    st.warning(f"No generation data found for run '{selected_run_id}'. The simulation might be in progress or encountered an issue.")
    # Allow dashboard to load, but tabs will show 'no data' messages.


# --- Tab Navigation ---
tabs_list = ["📈 Overview & Curves", "🔬 Generation Explorer", "👤 Agent Detail", "📜 Game Viewer"]
# Allow active_tab to be set by buttons, otherwise default or keep current
if 'active_tab_choice' not in st.session_state: # To handle direct selection via tabs UI
    st.session_state.active_tab_choice = st.session_state.active_tab

# This creates the actual tab UI. The `active_tab` session state should ideally control this.
# However, st.tabs itself doesn't directly take an active_tab parameter.
# We use session_state to manage the content visibility or default selections within tabs.

active_tab_ui = st.tabs(tabs_list)

# --- Tab 1: Overview & Curves ---
with active_tab_ui[0]: # Corresponds to "📈 Overview & Curves"
    st.header(f"Run Overview: {selected_run_id}")
    
    col_overview1, col_overview2 = st.columns(2)
    with col_overview1:
        if config_data:
            with st.expander("Simulation Configuration Snapshot", expanded=False):
                st.json(config_data, expanded=False)
        else:
            st.warning("Configuration snapshot not found for this run.")
    with col_overview2:
        st.metric("Generations Completed", latest_gen_num_for_run if latest_gen_num_for_run is not None else "N/A")

    if not all_gen_summary_df.empty:
        st.subheader("Performance Curves Over Generations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Fitness Metrics")
            fitness_metrics_to_plot = ["avg_fitness", "min_fitness", "max_fitness"]
            available_fitness_metrics = [m for m in fitness_metrics_to_plot if m in all_gen_summary_df.columns]
            if available_fitness_metrics:
                fitness_df = all_gen_summary_df[["generation_number"] + available_fitness_metrics].melt(
                    id_vars=["generation_number"], var_name="Metric", value_name="Fitness"
                )
                fitness_chart = alt.Chart(fitness_df).mark_line(point=True).encode(
                    x=alt.X('generation_number:Q', title='Generation'),
                    y=alt.Y('Fitness:Q', title='Fitness Value', scale=alt.Scale(zero=False)),
                    color='Metric:N',
                    tooltip=['generation_number', 'Metric', alt.Tooltip('Fitness:Q', format='.3f')]
                ).interactive()
                st.altair_chart(fitness_chart, use_container_width=True)
            else:
                st.caption("Fitness data (avg, min, max) not available in summaries.")

        with col2:
            st.markdown("##### Wealth Metrics")
            if "avg_wealth" in all_gen_summary_df.columns:
                wealth_df = all_gen_summary_df[["generation_number", "avg_wealth"]].copy()
                wealth_df.rename(columns={"avg_wealth": "Average Wealth"}, inplace=True)
                
                wealth_chart = alt.Chart(wealth_df).mark_line(point=True).encode(
                    x=alt.X('generation_number:Q', title='Generation'),
                    y=alt.Y('Average Wealth:Q', title='Wealth Value', scale=alt.Scale(zero=False)),
                    tooltip=['generation_number', alt.Tooltip('Average Wealth:Q', format='.2f')]
                ).interactive()
                st.altair_chart(wealth_chart, use_container_width=True)
            else:
                st.caption("Average wealth data not available in summaries.")
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### Game Statistics")
            if "total_games_played_in_generation" in all_gen_summary_df.columns:
                games_chart = alt.Chart(all_gen_summary_df).mark_bar().encode(
                    x=alt.X('generation_number:Q', title='Generation', bin=alt.Bin(maxbins=max(1, latest_gen_num_for_run or 1))),
                    y=alt.Y('total_games_played_in_generation:Q', title='Games Played'),
                    tooltip=['generation_number', 'total_games_played_in_generation']
                ).interactive()
                st.altair_chart(games_chart, use_container_width=True)
            else:
                st.caption("Total games played data not available.")
        
        with col4:
            st.markdown("##### Genome Diversity (Approx.)")
            if "unique_genomes_approx" in all_gen_summary_df.columns:
                diversity_chart = alt.Chart(all_gen_summary_df).mark_line(point=alt.OverlayMarkDef(filled=False, fill="white")).encode(
                    x=alt.X('generation_number:Q', title='Generation'),
                    y=alt.Y('unique_genomes_approx:Q', title='Unique Genomes (Approx.)'),
                    tooltip=['generation_number', 'unique_genomes_approx']
                ).interactive()
                st.altair_chart(diversity_chart, use_container_width=True)
            else:
                st.caption("Approximate unique genomes data not available.")

        st.subheader("Latest Generation Snapshot")
        if latest_gen_num_for_run is not None:
            latest_gen_data = load_generation_data(selected_run_id, latest_gen_num_for_run, state_dir)
            if latest_gen_data and "population_state" in latest_gen_data:
                pop_state = latest_gen_data["population_state"]
                latest_gen_summary = latest_gen_data.get("generation_summary_metrics", {})

                df_latest_pop = pd.DataFrame([{
                    "wealth": agent.get("wealth"),
                    "fitness": agent.get("fitness_score", latest_gen_summary.get("fitness_scores_map", {}).get(agent.get("agent_id")))
                } for agent in pop_state])
                
                col_dist1, col_dist2 = st.columns(2)
                with col_dist1:
                    st.markdown("##### Wealth Distribution (Latest Gen)")
                    if not df_latest_pop.empty and "wealth" in df_latest_pop.columns and df_latest_pop["wealth"].notna().any():
                        wealth_hist = alt.Chart(df_latest_pop).mark_bar().encode(
                            alt.X("wealth:Q", bin=alt.Bin(maxbins=20), title="Wealth"),
                            alt.Y("count()", title="Number of Agents"),
                            tooltip=[alt.Tooltip("wealth:Q", title="Wealth Bin"), alt.Tooltip("count():Q", title="Agents")]
                        ).interactive()
                        st.altair_chart(wealth_hist, use_container_width=True)
                    else:
                        st.caption("Wealth data for distribution unavailable for the latest generation.")
                
                with col_dist2:
                    st.markdown("##### Fitness Distribution (Latest Gen)")
                    if not df_latest_pop.empty and "fitness" in df_latest_pop.columns and df_latest_pop["fitness"].notna().any():
                        fitness_hist = alt.Chart(df_latest_pop).mark_bar().encode(
                            alt.X("fitness:Q", bin=alt.Bin(maxbins=20), title="Fitness Score"),
                            alt.Y("count()", title="Number of Agents"),
                            tooltip=[alt.Tooltip("fitness:Q", title="Fitness Bin", format=".3f"), alt.Tooltip("count():Q", title="Agents")]
                        ).interactive()
                        st.altair_chart(fitness_hist, use_container_width=True)
                    else:
                        st.caption("Fitness data for distribution unavailable for the latest generation.")
            else:
                st.info("Data for the latest generation is not available for distribution plots.")
        else:
            st.info("No completed generations to show distribution for.")
    else:
        st.info("No generation summary data found to plot curves for this run.")

# --- Tab 2: Generation Explorer ---
with active_tab_ui[1]: # Corresponds to "🔬 Generation Explorer"
    st.header("Generation Explorer")
    if latest_gen_num_for_run is None:
        st.warning("No generation data available for this run.")
    else:
        gen_numbers = list(range(1, latest_gen_num_for_run + 1))
        # index is valid if gen_numbers is empty (though caught by latest_gen_num_for_run check)
        default_idx_gen_explorer = len(gen_numbers) - 1 if gen_numbers else 0
        selected_gen_num_explorer = st.selectbox(
            "Select Generation to Explore", 
            options=gen_numbers, 
            format_func=lambda x: f"Generation {x}",
            index=default_idx_gen_explorer,
            key="gen_explorer_gen_select"
        )

        if selected_gen_num_explorer:
            gen_data = load_generation_data(selected_run_id, selected_gen_num_explorer, state_dir)
            if gen_data:
                st.subheader(f"Details for Generation {selected_gen_num_explorer}")
                
                summary_metrics = gen_data.get("generation_summary_metrics", {})
                cols_metrics_gen = st.columns(4)
                cols_metrics_gen[0].metric("Avg Fitness", f"{summary_metrics.get('avg_fitness', 0):.3f}" if summary_metrics.get('avg_fitness') is not None else "N/A")
                cols_metrics_gen[1].metric("Max Fitness", f"{summary_metrics.get('max_fitness', 0):.3f}" if summary_metrics.get('max_fitness') is not None else "N/A")
                cols_metrics_gen[2].metric("Avg Wealth", f"{summary_metrics.get('avg_wealth', 0):.2f}" if summary_metrics.get('avg_wealth') is not None else "N/A")
                cols_metrics_gen[3].metric("Games Played", summary_metrics.get('total_games_played_in_generation', "N/A"))

                st.markdown("##### Agents in this Generation")
                population_state = gen_data.get("population_state", [])
                if population_state:
                    agents_df_data = []
                    fitness_map = summary_metrics.get("fitness_scores_map", {})
                    for agent_s in population_state:
                        agent_id = agent_s.get("agent_id")
                        fitness = agent_s.get("fitness_score", fitness_map.get(agent_id))
                        agents_df_data.append({
                            "Agent ID": agent_id,
                            "Display ID": get_agent_display_name(agent_id),
                            "Model ID": agent_s.get("model_id"),
                            "Wealth": agent_s.get("wealth"),
                            "Fitness": fitness,
                            "Genome Size": len(agent_s.get("genome", {})),
                        })
                    agents_df = pd.DataFrame(agents_df_data)
                    
                    if not agents_df.empty and "Fitness" in agents_df.columns:
                         agents_df["Rank (by Fitness)"] = agents_df["Fitness"].rank(method="dense", ascending=False).astype(int)
                         # Select and order columns for display
                         display_columns = ["Rank (by Fitness)", "Display ID", "Wealth", "Fitness", "Genome Size", "Model ID", "Agent ID"]
                         # Filter out columns that might not exist if data is partial
                         display_columns = [col for col in display_columns if col in agents_df.columns]
                         st.dataframe(agents_df[display_columns], use_container_width=True, hide_index=True)
                    elif not agents_df.empty:
                        st.dataframe(agents_df, use_container_width=True, hide_index=True) # Show without rank if fitness is missing
                    else:
                        st.info("No agent details to display in table.")

                else:
                    st.info("No agent population data found for this generation.")
            else:
                st.error(f"Could not load data for Generation {selected_gen_num_explorer}.")

# --- Tab 3: Agent Detail ---
with active_tab_ui[2]: # Corresponds to "👤 Agent Detail"
    st.header("Agent Detail Viewer")
    if latest_gen_num_for_run is None:
        st.warning("No generation data available to select agents from.")
    else:
        gen_numbers_for_agent_detail = list(range(1, latest_gen_num_for_run + 1))
        default_idx_agent_detail_gen = len(gen_numbers_for_agent_detail) - 1 if gen_numbers_for_agent_detail else 0
        
        selected_gen_for_agent_detail = st.selectbox(
            "Select Generation (for Agent)", 
            options=gen_numbers_for_agent_detail, 
            format_func=lambda x: f"Generation {x}",
            index=default_idx_agent_detail_gen,
            key="agent_detail_gen_select"
        )

        if selected_gen_for_agent_detail:
            gen_data_for_agent = load_generation_data(selected_run_id, selected_gen_for_agent_detail, state_dir)
            if gen_data_for_agent and "population_state" in gen_data_for_agent:
                agent_ids_in_gen = [a.get("agent_id", f"UnknownAgent_{i}") for i, a in enumerate(gen_data_for_agent["population_state"])]
                
                if not agent_ids_in_gen:
                    st.info("No agents found in the selected generation.")
                else:
                    # Try to pre-select agent if navigated from Game Viewer (future feature) or keep current
                    default_agent_idx = 0 # Simple default
                    
                    selected_agent_id_detail = st.selectbox(
                        "Select Agent ID", 
                        options=agent_ids_in_gen,
                        format_func=get_agent_display_name,
                        index=default_agent_idx,
                        key="agent_detail_agent_select"
                    )
                    
                    if selected_agent_id_detail:
                        agent_data = next((a for a in gen_data_for_agent["population_state"] if a.get("agent_id") == selected_agent_id_detail), None)
                        if agent_data:
                            st.subheader(f"Agent: {get_agent_display_name(selected_agent_id_detail)} (Generation {selected_gen_for_agent_detail})")
                            
                            cols_agent_metrics = st.columns(3)
                            cols_agent_metrics[0].metric("Wealth", f"{agent_data.get('wealth', 0):.2f}" if agent_data.get('wealth') is not None else "N/A")
                            
                            fitness_val = agent_data.get("fitness_score")
                            if fitness_val is None:
                                summary_metrics = gen_data_for_agent.get("generation_summary_metrics", {})
                                fitness_val = summary_metrics.get("fitness_scores_map", {}).get(selected_agent_id_detail)
                            cols_agent_metrics[1].metric("Fitness", f"{fitness_val:.3f}" if fitness_val is not None else "N/A")
                            cols_agent_metrics[2].metric("Model ID", agent_data.get("model_id", "N/A"))

                            st.markdown("##### Genome Activations")
                            genome = agent_data.get("genome", {})
                            if genome:
                                genome_df = pd.DataFrame(list(genome.items()), columns=['Feature UUID', 'Activation Value']).sort_values(by="Activation Value", ascending=False)
                                
                                col_genome_table, col_genome_chart = st.columns(2)
                                with col_genome_table:
                                    st.dataframe(genome_df, height=250, use_container_width=True, hide_index=True)
                                
                                with col_genome_chart:
                                    if not genome_df.empty:
                                        # Show top N and bottom N features for better readability
                                        n_features_to_show = 10
                                        top_n = genome_df.head(n_features_to_show)
                                        bottom_n = genome_df.tail(n_features_to_show) if len(genome_df) > n_features_to_show else pd.DataFrame() # Avoid duplicates if few features
                                        
                                        chart_data = pd.concat([top_n, bottom_n]).drop_duplicates()
                                        chart_data['Feature UUID'] = chart_data['Feature UUID'].apply(lambda x: x[:15] + "..." if len(x) > 15 else x) # Shorten long UUIDs

                                        genome_chart = alt.Chart(chart_data).mark_bar().encode(
                                            x=alt.X('Activation Value:Q'),
                                            y=alt.Y('Feature UUID:N', sort='-x', title="Feature (Shortened UUID)"),
                                            tooltip=['Feature UUID', alt.Tooltip('Activation Value:Q', format='.3f')],
                                            color=alt.condition(
                                                alt.datum['Activation Value'] > 0,
                                                alt.value('steelblue'), # Positive activations
                                                alt.value('orange')    # Negative activations
                                            )
                                        ).properties(
                                            title=f"Top/Bottom {n_features_to_show} Genome Activations"
                                        ).interactive()
                                        st.altair_chart(genome_chart, use_container_width=True)
                                    else:
                                        st.caption("No genome features to chart.")
                            else:
                                st.info("No genome data for this agent.")

                            st.markdown(f"##### Games Played by {get_agent_display_name(selected_agent_id_detail)} (Generation {selected_gen_for_agent_detail})")
                            games_this_gen_for_agent = load_games_for_generation(selected_run_id, selected_gen_for_agent_detail, state_dir)
                            agent_games = [
                                g for g in games_this_gen_for_agent 
                                if selected_agent_id_detail in [g.get("player_A_id"), g.get("player_B_id")]
                            ]
                            if agent_games:
                                games_display_data = []
                                for idx, game in enumerate(agent_games):
                                    is_player_A = game.get("player_A_id") == selected_agent_id_detail
                                    opponent_id = game.get("player_B_id") if is_player_A else game.get("player_A_id")
                                    agent_role = game.get("player_A_game_role") if is_player_A else game.get("player_B_game_role")
                                    
                                    wealth_c = game.get("wealth_changes") or {}
                                    agent_wealth_change = wealth_c.get("player_A_wealth_change") if is_player_A else wealth_c.get("player_B_wealth_change")
                                    
                                    # Determine win/loss/tie from agent's perspective
                                    outcome_str = game.get("adjudication_result", "Unknown")
                                    agent_outcome = "N/A"
                                    if outcome_str != "Unknown" and outcome_str != "Error" and outcome_str != "Critical Game Error":
                                        if outcome_str == "Tie" or "Tie" in outcome_str :
                                            agent_outcome = "Tie"
                                        elif (outcome_str == "Role A Wins" and agent_role == "Role A") or \
                                             (outcome_str == "Role B Wins" and agent_role == "Role B"):
                                            agent_outcome = "Win"
                                        elif (outcome_str == "Role A Wins" and agent_role == "Role B") or \
                                             (outcome_str == "Role B Wins" and agent_role == "Role A"):
                                            agent_outcome = "Loss"
                                        else:
                                            if agent_wealth_change is not None:
                                                if agent_wealth_change > 0: agent_outcome = "Win (Implied)"
                                                elif agent_wealth_change < 0: agent_outcome = "Loss (Implied)"
                                                else: agent_outcome = "Tie (Implied)"
                                    
                                    games_display_data.append({
                                        "Game ID": game.get("game_id"),
                                        "Opponent": get_agent_display_name(opponent_id),
                                        "Agent Role": agent_role,
                                        "Outcome (Agent)": agent_outcome,
                                        "Wealth Change": f"{agent_wealth_change:.2f}" if agent_wealth_change is not None else "N/A",
                                        "Scenario Snippet": game.get("scenario_text", "")[:70]+"..." if game.get("scenario_text") else "N/A",
                                        "View Game Button": game.get("game_id") # For button mapping
                                    })
                                
                                df_agent_games = pd.DataFrame(games_display_data)
                                
                                # Use st.columns to create a "button column"
                                if not df_agent_games.empty:
                                    col_defs = {
                                        "Game ID": st.column_config.TextColumn(width="medium"),
                                        "Scenario Snippet": st.column_config.TextColumn(width="large")
                                    }
                                    st.markdown("###### Click 'View Game' to see full details in the Game Viewer tab:")
                                    for _, row in df_agent_games.iterrows():
                                        cols = st.columns((2,2,1,1,1,3,1))
                                        cols[0].write(row["Game ID"])
                                        cols[1].write(row["Opponent"])
                                        cols[2].write(row["Agent Role"])
                                        cols[3].write(row["Outcome (Agent)"])
                                        cols[4].write(row["Wealth Change"])
                                        cols[5].caption(row["Scenario Snippet"])
                                        button_key = f"view_game_{row['Game ID']}_{selected_gen_for_agent_detail}"
                                        if cols[6].button("👁️ View", key=button_key, help=f"View game {row['Game ID']}"):
                                            st.session_state.selected_game_id_from_agent_detail = row['Game ID']
                                            st.session_state.selected_gen_for_game_viewer = selected_gen_for_agent_detail
                                            st.session_state.active_tab = "📜 Game Viewer" # Signal to switch tab (requires rerun)
                                            st.rerun() # Force rerun to reflect tab change and pre-selection
                                else:
                                    st.info("No game records to display in table.")
                            else:
                                st.info(f"No game records found for this agent in generation {selected_gen_for_agent_detail}'s game file.")
                        else:
                            st.error("Selected agent data could not be retrieved.")
            else:
                st.warning(f"No population data for Generation {selected_gen_for_agent_detail} to select an agent.")

# --- Tab 4: Game Viewer ---
with active_tab_ui[3]: # Corresponds to "📜 Game Viewer"
    st.header("Game Viewer")

    # Handle pre-selection from Agent Detail tab
    pre_selected_gen_num = st.session_state.get('selected_gen_for_game_viewer', None)
    pre_selected_game_id = st.session_state.get('selected_game_id_from_agent_detail', None)

    if latest_gen_num_for_run is None:
        st.warning("No generation data available to select games from.")
    else:
        gen_numbers_for_games_tab = list(range(1, latest_gen_num_for_run + 1))
        default_gen_idx_games_tab = 0
        if pre_selected_gen_num and pre_selected_gen_num in gen_numbers_for_games_tab:
            default_gen_idx_games_tab = gen_numbers_for_games_tab.index(pre_selected_gen_num)
        elif gen_numbers_for_games_tab: # Default to latest if no pre-selection
            default_gen_idx_games_tab = len(gen_numbers_for_games_tab) -1
            
        selected_gen_for_games_viewer = st.selectbox(
            "Select Generation (for Games)", 
            options=gen_numbers_for_games_tab, 
            format_func=lambda x: f"Generation {x}",
            index=default_gen_idx_games_tab,
            key="game_viewer_gen_select"
        )

        if selected_gen_for_games_viewer:
            games_data_for_viewer = load_games_for_generation(selected_run_id, selected_gen_for_games_viewer, state_dir)
            if not games_data_for_viewer:
                st.info(f"No game records found for Generation {selected_gen_for_games_viewer}.")
            else:
                # Create a mapping for easier lookup if needed, or just use list of IDs
                game_id_options = {g.get("game_id", f"UnknownGame_{i}"): g for i, g in enumerate(games_data_for_viewer)}
                
                default_game_id_idx = 0
                if pre_selected_game_id and pre_selected_game_id in game_id_options and selected_gen_for_games_viewer == pre_selected_gen_num:
                    # the pre-selected game is from the currently selected generation for safety
                    try:
                        default_game_id_idx = list(game_id_options.keys()).index(pre_selected_game_id)
                    except ValueError:
                        default_game_id_idx = 0 # Fallback if ID not found (should not happen if logic is correct)
                
                # Clear the pre-selection after using it once to avoid sticky selection on manual changes
                st.session_state.selected_game_id_from_agent_detail = None
                st.session_state.selected_gen_for_game_viewer = None

                selected_game_id_viewer = st.selectbox(
                    "Select Game ID", 
                    options=list(game_id_options.keys()), 
                    index=default_game_id_idx,
                    key="game_viewer_game_select"
                    )
                
                if selected_game_id_viewer:
                    game_to_display = game_id_options.get(selected_game_id_viewer)

                    if game_to_display:
                        st.subheader(f"Details for Game: {game_to_display.get('game_id', 'N/A')}")
                        
                        pA_id = game_to_display.get('player_A_id')
                        pB_id = game_to_display.get('player_B_id')
                        pA_role = game_to_display.get('player_A_game_role', 'Role A') # Default for display
                        pB_role = game_to_display.get('player_B_game_role', 'Role B') # Default for display
                        
                        st.markdown(f"""
                        - **Player A ({pA_role}):** {get_agent_display_name(pA_id)}
                        - **Player B ({pB_role}):** {get_agent_display_name(pB_id)}
                        - **Scenario Proposer:** {get_agent_display_name(game_to_display.get('proposer_agent_id'))}
                        - **Timestamp Start:** {game_to_display.get('timestamp_start', 'N/A')}
                        - **Timestamp End:** {game_to_display.get('timestamp_end', 'N/A')}
                        """)

                        with st.expander("📜 Scenario Text", expanded=True):
                            st.text_area("Scenario", value=game_to_display.get("scenario_text", "N/A"), height=200, disabled=True, label_visibility="collapsed")

                        st.markdown("##### 💬 Conversation Transcript")
                        transcript = game_to_display.get("transcript", [])
                        if transcript:
                            for turn in transcript:
                                turn_role = turn.get('role', 'Unknown Role')
                                speaker_agent_id = turn.get('agent_id', 'Unknown Agent')
                                content = turn.get('content', '')
                                
                                # Determine avatar based on actual game roles if possible
                                avatar_symbol = "👤" # Default
                                if turn_role == pA_role : # Player A's assigned role
                                    avatar_symbol = "🧑‍💻"
                                elif turn_role == pB_role: # Player B's assigned role
                                    avatar_symbol = "🤖"
                                
                                with st.chat_message(name=turn_role, avatar=avatar_symbol):
                                    st.write(f"**{turn_role} ({get_agent_display_name(speaker_agent_id)})**: ")
                                    st.markdown(content) # Use markdown for better formatting of LLM output
                        else:
                            st.info("No transcript available for this game.")

                        st.markdown("##### ⚖️ Adjudication & Outcome")
                        adjudication_cols = st.columns(2)
                        with adjudication_cols[0]:
                            st.markdown(f"**Adjudication Result:**")
                            st.code(game_to_display.get('adjudication_result', 'N/A'), language=None)
                            if game_to_display.get('defaulted_to_tie_reason'):
                                st.caption(f"Defaulted Reason: {game_to_display.get('defaulted_to_tie_reason')}")
                        
                        with adjudication_cols[1]:
                            bets = game_to_display.get("betting_details") or {}
                            st.markdown(f"""
                            - **{pA_role} ({get_agent_display_name(pA_id)}) Bet:** {bets.get('player_A_bet', 'N/A')}
                            - **{pB_role} ({get_agent_display_name(pB_id)}) Bet:** {bets.get('player_B_bet', 'N/A')}
                            """)
                        
                        st.markdown("##### 💰 Wealth Changes")
                        wealth_c = game_to_display.get("wealth_changes") or {}
                        st.markdown(f"""
                        - **{pA_role} ({get_agent_display_name(pA_id)}) Change:** {wealth_c.get('player_A_wealth_change', 'N/A')}
                        - **{pB_role} ({get_agent_display_name(pB_id)}) Change:** {wealth_c.get('player_B_wealth_change', 'N/A')}
                        """)
                    else:
                        st.error("Could not find details for the selected game ID.")
