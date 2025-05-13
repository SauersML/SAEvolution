import streamlit as st
import pandas as pd
import json
from pathlib import Path
import altair as alt
import re
import numpy as np

# --- Configuration ---
DEFAULT_STATE_BASE_DIR = "simulation_state"
CONFIG_FILENAME = "config_snapshot.json"
LATEST_GEN_TRACKER_FILENAME = "_latest_generation_number.txt"

# --- Utility Functions ---

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """Key for natural sorting (e.g., gen_2 before gen_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

@st.cache_data(ttl=30)
def get_available_simulation_runs(state_base_dir: str) -> list[str]:
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
                content = f.read().strip()
                if content:
                    return int(content)
        except ValueError:
             st.error(f"Invalid content in latest generation tracker for {run_id}. Expected an integer.")
        except Exception as e:
            st.error(f"Error reading latest generation tracker for {run_id}: {e}")
    return None

@st.cache_data(ttl=30)
def load_generation_data(run_id: str, generation_number: int, state_base_dir: str) -> dict | None:
    if not run_id or generation_number is None: return None
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
    for gen_num in range(1, latest_gen + 1): # Ensure we iterate up to and including latest_gen
        gen_data = load_generation_data(run_id, gen_num, state_base_dir)
        if gen_data:
            summary = gen_data.get("generation_summary_metrics", {})
            summary["generation_number"] = gen_data.get("generation_number", gen_num)
            summary["timestamp_completed"] = gen_data.get("timestamp_completed")
            
            population_state = gen_data.get("population_state", [])
            if population_state:
                genome_strings = [json.dumps(agent.get("genome", {}), sort_keys=True) for agent in population_state]
                summary["unique_genomes_approx"] = len(set(genome_strings))
            else:
                summary["unique_genomes_approx"] = 0 
            all_summaries.append(summary)
    
    df = pd.DataFrame(all_summaries)
    numeric_cols = ["avg_fitness", "min_fitness", "max_fitness", "avg_wealth", "unique_genomes_approx", "total_games_played_in_generation"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data(ttl=30) # Consider reducing ttl for more "live" feel if files update frequently
def load_games_for_generation(run_id: str, generation_number: int, state_base_dir: str) -> list[dict]:
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

def get_agent_display_name(agent_id: str) -> str:
    return f"Agent {agent_id[:8]}..." if isinstance(agent_id, str) and agent_id else "N/A"

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="SAEvolution Dashboard")
st.title("Agent Evolution Dashboard")

# Initialize session state
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📈 Overview & Curves" 
if 'selected_game_id_from_agent_detail' not in st.session_state:
    st.session_state.selected_game_id_from_agent_detail = None
if 'selected_gen_for_game_viewer' not in st.session_state:
    st.session_state.selected_gen_for_game_viewer = None

# --- Sidebar ---
st.sidebar.header("Simulation Controls")
state_dir = st.sidebar.text_input("State Directory Path", value=DEFAULT_STATE_BASE_DIR, help="Path to the base directory where simulation states are saved.")
available_runs = get_available_simulation_runs(state_dir)

if not available_runs:
    st.sidebar.warning("No simulation runs found. Ensure simulations have run and saved state.")
    st.info("No simulation data to display. Please check the 'State Directory Path' and run a simulation.")
    st.stop()

selected_run_id = st.sidebar.selectbox(
    "Select Simulation Run ID",
    options=available_runs,
    index=0 if available_runs else -1, 
    help="Choose a simulation run to inspect."
)

if st.sidebar.button("🔄 Refresh Data", help="Reload all data from the selected run and clear cache."):
    st.cache_data.clear()
    st.session_state.selected_game_id_from_agent_detail = None
    st.session_state.selected_gen_for_game_viewer = None
    # st.session_state.active_tab remains as is, or reset if needed
    st.rerun()

if not selected_run_id:
    st.info("Please select a Simulation Run ID from the sidebar.")
    st.stop()

# --- Load Core Data ---
config_data = load_config_snapshot(selected_run_id, state_dir)
latest_gen_num_for_run = get_latest_generation_number(selected_run_id, state_dir)
all_gen_summary_df = load_all_generation_summary_data(selected_run_id, state_dir)

if latest_gen_num_for_run is None and selected_run_id:
    st.warning(f"No generation data found for run '{selected_run_id}'. The simulation might be in progress, errored, or has not completed a generation yet.")

# --- Tab Navigation ---
tabs_list = ["📈 Overview & Curves", "🔬 Generation Explorer", "👤 Agent Detail", "📜 Game Viewer"]
try:
    default_tab_idx = tabs_list.index(st.session_state.active_tab)
except ValueError:
    default_tab_idx = 0
    st.session_state.active_tab = tabs_list[0]

active_tab_ui_selection = st.tabs(tabs_list) # This returns a list of containers for each tab


# --- Tab 1: Overview & Curves ---
with active_tab_ui_selection[0]: 
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
        plot_columns = st.columns(2)
        
        with plot_columns[0]:
            st.markdown("##### Fitness Metrics")
            fitness_metrics_to_plot = ["avg_fitness", "min_fitness", "max_fitness"]
            available_fitness_metrics = [m for m in fitness_metrics_to_plot if m in all_gen_summary_df.columns and all_gen_summary_df[m].notna().any()]
            if available_fitness_metrics:
                fitness_df_melted = all_gen_summary_df[["generation_number"] + available_fitness_metrics].melt(
                    id_vars=["generation_number"], var_name="Metric", value_name="Fitness"
                ).dropna(subset=['Fitness']) 
                
                if not fitness_df_melted.empty:
                    fitness_chart = alt.Chart(fitness_df_melted).mark_line(point=True).encode(
                        x=alt.X('generation_number:Q', title='Generation'),
                        y=alt.Y('Fitness:Q', title='Fitness Value', scale=alt.Scale(zero=False)),
                        color='Metric:N',
                        tooltip=['generation_number', 'Metric', alt.Tooltip('Fitness:Q', format='.3f')]
                    ).interactive()
                    st.altair_chart(fitness_chart, use_container_width=True)
                else:
                    st.caption("No valid fitness data to plot.")
            else:
                st.caption("Fitness data (avg, min, max) not available or all NaN in summaries.")

        with plot_columns[1]:
            st.markdown("##### Wealth Metrics")
            if "avg_wealth" in all_gen_summary_df.columns and all_gen_summary_df["avg_wealth"].notna().any():
                wealth_df_plot = all_gen_summary_df[["generation_number", "avg_wealth"]].copy().dropna(subset=['avg_wealth'])
                wealth_df_plot.rename(columns={"avg_wealth": "Average Wealth"}, inplace=True)
                
                if not wealth_df_plot.empty:
                    wealth_chart = alt.Chart(wealth_df_plot).mark_line(point=True).encode(
                        x=alt.X('generation_number:Q', title='Generation'),
                        y=alt.Y('Average Wealth:Q', title='Wealth Value', scale=alt.Scale(zero=False)),
                        tooltip=['generation_number', alt.Tooltip('Average Wealth:Q', format='.2f')]
                    ).interactive()
                    st.altair_chart(wealth_chart, use_container_width=True)
                else:
                    st.caption("No valid average wealth data to plot.")
            else:
                st.caption("Average wealth data not available or all NaN in summaries.")
        
        plot_columns_2 = st.columns(2)
        with plot_columns_2[0]:
            st.markdown("##### Game Statistics")
            if "total_games_played_in_generation" in all_gen_summary_df.columns and all_gen_summary_df["total_games_played_in_generation"].notna().any():
                games_df_plot = all_gen_summary_df.dropna(subset=['total_games_played_in_generation'])
                if not games_df_plot.empty:
                    # FIX: Ensure maxbins is at least 2 for Altair
                    current_max_bins = max(2, latest_gen_num_for_run if latest_gen_num_for_run is not None and latest_gen_num_for_run > 0 else 2)
                    games_chart = alt.Chart(games_df_plot).mark_bar().encode(
                        x=alt.X('generation_number:Q', title='Generation', bin=alt.Bin(maxbins=current_max_bins)),
                        y=alt.Y('total_games_played_in_generation:Q', title='Games Played'),
                        tooltip=['generation_number', 'total_games_played_in_generation']
                    ).interactive()
                    st.altair_chart(games_chart, use_container_width=True)
                else:
                    st.caption("No valid game statistics data to plot.")
            else:
                st.caption("Total games played data not available or all NaN.")
        
        with plot_columns_2[1]:
            st.markdown("##### Genome Diversity (Approx.)")
            if "unique_genomes_approx" in all_gen_summary_df.columns and all_gen_summary_df["unique_genomes_approx"].notna().any():
                diversity_df_plot = all_gen_summary_df.dropna(subset=['unique_genomes_approx'])
                if not diversity_df_plot.empty:
                    diversity_chart = alt.Chart(diversity_df_plot).mark_line(point=alt.OverlayMarkDef(filled=False, fill="white")).encode(
                        x=alt.X('generation_number:Q', title='Generation'),
                        y=alt.Y('unique_genomes_approx:Q', title='Unique Genomes (Approx.)'),
                        tooltip=['generation_number', 'unique_genomes_approx']
                    ).interactive()
                    st.altair_chart(diversity_chart, use_container_width=True)
                else:
                    st.caption("No valid genome diversity data to plot.")
            else:
                st.caption("Approximate unique genomes data not available or all NaN.")

        st.subheader("Latest Generation Snapshot")
        if latest_gen_num_for_run is not None:
            latest_gen_data = load_generation_data(selected_run_id, latest_gen_num_for_run, state_dir)
            if latest_gen_data and "population_state" in latest_gen_data:
                pop_state = latest_gen_data["population_state"]
                latest_gen_summary = latest_gen_data.get("generation_summary_metrics", {})
                
                agent_metrics_latest_gen = []
                fitness_map_latest = latest_gen_summary.get("fitness_scores_map", {})
                for agent in pop_state:
                    agent_id = agent.get("agent_id")
                    fitness = agent.get("fitness_score", fitness_map_latest.get(agent_id))
                    agent_metrics_latest_gen.append({
                        "wealth": agent.get("wealth"),
                        "fitness": fitness
                    })
                df_latest_pop = pd.DataFrame(agent_metrics_latest_gen)
                
                dist_cols = st.columns(2)
                with dist_cols[0]:
                    st.markdown("##### Wealth Distribution (Latest Gen)")
                    if not df_latest_pop.empty and "wealth" in df_latest_pop.columns and df_latest_pop["wealth"].notna().any():
                        wealth_hist = alt.Chart(df_latest_pop.dropna(subset=['wealth'])).mark_bar().encode(
                            alt.X("wealth:Q", bin=alt.Bin(maxbins=20), title="Wealth"),
                            alt.Y("count()", title="Number of Agents"),
                            tooltip=[alt.Tooltip("wealth:Q", title="Wealth Bin"), alt.Tooltip("count():Q", title="Agents")]
                        ).interactive()
                        st.altair_chart(wealth_hist, use_container_width=True)
                    else:
                        st.caption("Wealth data for distribution unavailable for the latest generation.")
                
                with dist_cols[1]:
                    st.markdown("##### Fitness Distribution (Latest Gen)")
                    if not df_latest_pop.empty and "fitness" in df_latest_pop.columns and df_latest_pop["fitness"].notna().any():
                        fitness_hist = alt.Chart(df_latest_pop.dropna(subset=['fitness'])).mark_bar().encode(
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
with active_tab_ui_selection[1]:
    st.header("Generation Explorer")
    if latest_gen_num_for_run is None:
        st.warning("No generation data available for this run.")
    else:
        gen_numbers_explorer = list(range(1, latest_gen_num_for_run + 1))
        default_idx_gen_explorer = len(gen_numbers_explorer) - 1 if gen_numbers_explorer else 0
        
        selected_gen_num_explorer = st.selectbox(
            "Select Generation to Explore", 
            options=gen_numbers_explorer, 
            format_func=lambda x: f"Generation {x}",
            index=default_idx_gen_explorer,
            key="gen_explorer_gen_select"
        )

        if selected_gen_num_explorer is not None: # Check if a selection was made
            gen_data_explorer = load_generation_data(selected_run_id, selected_gen_num_explorer, state_dir)
            if gen_data_explorer:
                st.subheader(f"Details for Generation {selected_gen_num_explorer}")
                
                summary_metrics_explorer = gen_data_explorer.get("generation_summary_metrics", {})
                cols_metrics_gen_explorer = st.columns(4)
                cols_metrics_gen_explorer[0].metric("Avg Fitness", f"{summary_metrics_explorer.get('avg_fitness', 0):.3f}" if summary_metrics_explorer.get('avg_fitness') is not None else "N/A")
                cols_metrics_gen_explorer[1].metric("Max Fitness", f"{summary_metrics_explorer.get('max_fitness', 0):.3f}" if summary_metrics_explorer.get('max_fitness') is not None else "N/A")
                cols_metrics_gen_explorer[2].metric("Avg Wealth", f"{summary_metrics_explorer.get('avg_wealth', 0):.2f}" if summary_metrics_explorer.get('avg_wealth') is not None else "N/A")
                cols_metrics_gen_explorer[3].metric("Games Played", summary_metrics_explorer.get('total_games_played_in_generation', "N/A"))

                st.markdown("##### Agents in this Generation")
                population_state_explorer = gen_data_explorer.get("population_state", [])
                if population_state_explorer:
                    agents_df_data = []
                    fitness_map_explorer = summary_metrics_explorer.get("fitness_scores_map", {})
                    for agent_s in population_state_explorer:
                        agent_id = agent_s.get("agent_id")
                        fitness = agent_s.get("fitness_score", fitness_map_explorer.get(agent_id)) 
                        agents_df_data.append({
                            "Agent ID": agent_id,
                            "Display ID": get_agent_display_name(agent_id),
                            "Model ID": agent_s.get("model_id"),
                            "Wealth": agent_s.get("wealth"),
                            "Fitness": fitness, 
                            "Genome Size": len(agent_s.get("genome", {})),
                        })
                    agents_df_explorer = pd.DataFrame(agents_df_data)
                    
                    if not agents_df_explorer.empty:
                        if "Fitness" in agents_df_explorer.columns and agents_df_explorer["Fitness"].notna().any():
                            agents_df_explorer["Rank (by Fitness)"] = agents_df_explorer["Fitness"].rank(method="dense", ascending=False).astype(pd.Int64Dtype())
                        else:
                            agents_df_explorer["Rank (by Fitness)"] = pd.NA 
                        
                        display_columns = ["Rank (by Fitness)", "Display ID", "Wealth", "Fitness", "Genome Size", "Model ID", "Agent ID"]
                        display_columns = [col for col in display_columns if col in agents_df_explorer.columns] 
                        
                        st.dataframe(
                            agents_df_explorer[display_columns], 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "Wealth": st.column_config.NumberColumn(format="%.2f"),
                                "Fitness": st.column_config.NumberColumn(format="%.3f")
                            }
                        )
                    else:
                        st.info("No agent details to display in table for this generation.")
                else:
                    st.info("No agent population data found for this generation.")
            else:
                st.error(f"Could not load data for Generation {selected_gen_num_explorer}.")

# --- Tab 3: Agent Detail ---
with active_tab_ui_selection[2]:
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

        if selected_gen_for_agent_detail is not None:
            gen_data_for_agent_tab = load_generation_data(selected_run_id, selected_gen_for_agent_detail, state_dir)
            if gen_data_for_agent_tab and "population_state" in gen_data_for_agent_tab:
                population_for_agent_select = gen_data_for_agent_tab.get("population_state", [])
                agent_id_options_detail = {
                    agent.get("agent_id"): get_agent_display_name(agent.get("agent_id")) 
                    for agent in population_for_agent_select if agent.get("agent_id")
                }
                
                if not agent_id_options_detail:
                    st.info("No agents found in the selected generation.")
                else:
                    selected_agent_id_detail = st.selectbox(
                        "Select Agent ID", 
                        options=list(agent_id_options_detail.keys()),
                        format_func=lambda x: agent_id_options_detail.get(x, x), # Graceful format
                        index=0 if agent_id_options_detail else -1, 
                        key="agent_detail_agent_select"
                    )
                    
                    if selected_agent_id_detail:
                        agent_data_detail = next((a for a in population_for_agent_select if a.get("agent_id") == selected_agent_id_detail), None)
                        if agent_data_detail:
                            st.subheader(f"Agent: {get_agent_display_name(selected_agent_id_detail)} (Generation {selected_gen_for_agent_detail})")
                            
                            cols_agent_metrics_detail = st.columns(3)
                            cols_agent_metrics_detail[0].metric("Wealth", f"{agent_data_detail.get('wealth', 0):.2f}" if agent_data_detail.get('wealth') is not None else "N/A")
                            
                            fitness_val_detail = agent_data_detail.get("fitness_score")
                            if fitness_val_detail is None: 
                                summary_metrics_agent_tab = gen_data_for_agent_tab.get("generation_summary_metrics", {})
                                fitness_val_detail = summary_metrics_agent_tab.get("fitness_scores_map", {}).get(selected_agent_id_detail)
                            cols_agent_metrics_detail[1].metric("Fitness", f"{fitness_val_detail:.3f}" if fitness_val_detail is not None else "N/A")
                            cols_agent_metrics_detail[2].metric("Model ID", agent_data_detail.get("model_id", "N/A"))

                            st.markdown("##### Genome Activations")
                            genome_detail = agent_data_detail.get("genome", {})
                            if genome_detail:
                                genome_df_detail = pd.DataFrame(list(genome_detail.items()), columns=['Feature UUID', 'Activation Value']).sort_values(by="Activation Value", ascending=False)
                                
                                col_genome_table, col_genome_chart = st.columns(2)
                                with col_genome_table:
                                    st.dataframe(genome_df_detail.reset_index(drop=True), height=250, use_container_width=True)
                                
                                with col_genome_chart:
                                    if not genome_df_detail.empty:
                                        n_features_to_show = 10
                                        top_n = genome_df_detail.head(n_features_to_show)
                                        bottom_n = genome_df_detail.tail(n_features_to_show) if len(genome_df_detail) > n_features_to_show else pd.DataFrame()
                                        
                                        chart_data_genome = pd.concat([top_n, bottom_n]).drop_duplicates()
                                        chart_data_genome['Feature UUID Display'] = chart_data_genome['Feature UUID'].apply(lambda x: x[:15] + "..." if isinstance(x, str) and len(x) > 15 else x)

                                        genome_chart_detail = alt.Chart(chart_data_genome).mark_bar().encode(
                                            x=alt.X('Activation Value:Q'),
                                            y=alt.Y('Feature UUID Display:N', sort='-x', title="Feature"),
                                            tooltip=[alt.Tooltip('Feature UUID:N', title="Full UUID"), alt.Tooltip('Activation Value:Q', format='.3f')],
                                            color=alt.condition(alt.datum['Activation Value'] > 0, alt.value('steelblue'),alt.value('orange'))
                                        ).properties(title=f"Top/Bottom Genome Activations").interactive()
                                        st.altair_chart(genome_chart_detail, use_container_width=True)
                                    else:
                                        st.caption("No genome features to chart.")
                            else:
                                st.info("No genome data for this agent.")

                            st.markdown(f"##### Games Played by {get_agent_display_name(selected_agent_id_detail)} (Gen {selected_gen_for_agent_detail})")
                            games_this_gen_for_agent = load_games_for_generation(selected_run_id, selected_gen_for_agent_detail, state_dir)
                            agent_games_list = [
                                g for g in games_this_gen_for_agent 
                                if selected_agent_id_detail in [g.get("player_A_id"), g.get("player_B_id")]
                            ]
                            if agent_games_list:
                                games_display_list = []
                                for game_item in agent_games_list:
                                    is_player_A = game_item.get("player_A_id") == selected_agent_id_detail
                                    opponent_id = game_item.get("player_B_id") if is_player_A else game_item.get("player_A_id")
                                    agent_role_in_game = game_item.get("player_A_game_role") if is_player_A else game_item.get("player_B_game_role")
                                    
                                    wealth_changes_game = game_item.get("wealth_changes") or {}
                                    agent_wealth_change_val = wealth_changes_game.get("player_A_wealth_change") if is_player_A else wealth_changes_game.get("player_B_wealth_change")
                                    
                                    outcome_str_game = game_item.get("adjudication_result", "Unknown")
                                    agent_game_outcome = "N/A"
                                    if outcome_str_game not in ["Unknown", "Error", "Critical Game Error", "error"] and agent_role_in_game: # added "error"
                                        if "Tie" in outcome_str_game: agent_game_outcome = "Tie"
                                        elif (outcome_str_game == "Role A Wins" and agent_role_in_game == "Role A") or \
                                             (outcome_str_game == "Role B Wins" and agent_role_in_game == "Role B"):
                                            agent_game_outcome = "Win"
                                        else: agent_game_outcome = "Loss" 
                                    
                                    games_display_list.append({
                                        "Game ID": game_item.get("game_id"),
                                        "Opponent": get_agent_display_name(opponent_id),
                                        "Agent Role": agent_role_in_game,
                                        "Outcome (Agent)": agent_game_outcome,
                                        "Wealth Change": f"{agent_wealth_change_val:.2f}" if agent_wealth_change_val is not None else "N/A",
                                        "Scenario Snippet": (game_item.get("scenario_text", "")[:70]+"..." if game_item.get("scenario_text") else "N/A"),
                                    })
                                
                                if games_display_list:
                                    st.markdown("###### Click '👁️ View' to see full details in the Game Viewer tab:")
                                    for row_data in games_display_list:
                                        game_cols = st.columns((1.5, 1.5, 1, 1, 1, 2.5, 0.5)) 
                                        game_cols[0].write(row_data["Game ID"])
                                        game_cols[1].write(row_data["Opponent"])
                                        game_cols[2].write(row_data["Agent Role"] or "N/A")
                                        game_cols[3].write(row_data["Outcome (Agent)"])
                                        game_cols[4].write(row_data["Wealth Change"])
                                        game_cols[5].caption(row_data["Scenario Snippet"])
                                        button_key = f"view_game_btn_{row_data['Game ID']}_{selected_gen_for_agent_detail}"
                                        if game_cols[6].button("👁️", key=button_key, help=f"View game {row_data['Game ID']}"):
                                            st.session_state.selected_game_id_from_agent_detail = row_data['Game ID']
                                            st.session_state.selected_gen_for_game_viewer = selected_gen_for_agent_detail
                                            st.session_state.active_tab = "📜 Game Viewer"
                                            st.rerun()
                                else:
                                    st.info("No game records to display in table for this agent.")
                            else:
                                st.info(f"No game records found for this agent in generation {selected_gen_for_agent_detail}'s game file.")
                        else:
                            st.error("Selected agent data could not be retrieved.")
            else:
                st.warning(f"No population data for Generation {selected_gen_for_agent_detail} to select an agent.")

# --- Tab 4: Game Viewer ---
with active_tab_ui_selection[3]:
    st.header("Game Viewer")

    pre_selected_gen_gv = st.session_state.get('selected_gen_for_game_viewer', None)
    pre_selected_game_id_gv = st.session_state.get('selected_game_id_from_agent_detail', None)

    if latest_gen_num_for_run is None:
        st.warning("No generation data available to select games from.")
    else:
        gen_numbers_for_games_viewer_tab = list(range(1, latest_gen_num_for_run + 1))
        default_gen_idx_gv = 0
        if pre_selected_gen_gv and pre_selected_gen_gv in gen_numbers_for_games_viewer_tab:
            default_gen_idx_gv = gen_numbers_for_games_viewer_tab.index(pre_selected_gen_gv)
        elif gen_numbers_for_games_viewer_tab: 
            default_gen_idx_gv = len(gen_numbers_for_games_viewer_tab) -1
            
        selected_gen_for_gv = st.selectbox(
            "Select Generation (for Games)", 
            options=gen_numbers_for_games_viewer_tab, 
            format_func=lambda x: f"Generation {x}",
            index=default_gen_idx_gv,
            key="game_viewer_gen_select_key" 
        )

        if selected_gen_for_gv is not None:
            games_data_gv = load_games_for_generation(selected_run_id, selected_gen_for_gv, state_dir)
            if not games_data_gv:
                st.info(f"No game records found for Generation {selected_gen_for_gv}.")
            else:
                game_id_to_game_map_gv = {g.get("game_id", f"UnknownGame_{i}"): g for i, g in enumerate(games_data_gv)}
                game_id_options_gv = list(game_id_to_game_map_gv.keys())
                
                default_game_id_idx_gv = 0
                if pre_selected_game_id_gv and pre_selected_game_id_gv in game_id_options_gv and selected_gen_for_gv == pre_selected_gen_gv:
                    try:
                        default_game_id_idx_gv = game_id_options_gv.index(pre_selected_game_id_gv)
                    except ValueError: pass 
                
                if st.session_state.selected_game_id_from_agent_detail:
                     st.session_state.selected_game_id_from_agent_detail = None
                if st.session_state.selected_gen_for_game_viewer:
                     st.session_state.selected_gen_for_game_viewer = None

                selected_game_id_gv = st.selectbox(
                    "Select Game ID", 
                    options=game_id_options_gv, 
                    index=default_game_id_idx_gv,
                    key="game_viewer_game_select_key" 
                    )
                
                if selected_game_id_gv:
                    game_to_display_gv = game_id_to_game_map_gv.get(selected_game_id_gv)

                    if game_to_display_gv:
                        st.subheader(f"Details for Game: {game_to_display_gv.get('game_id', 'N/A')}")
                        
                        pA_id_gv = game_to_display_gv.get('player_A_id')
                        pB_id_gv = game_to_display_gv.get('player_B_id')
                        pA_role_gv = game_to_display_gv.get('player_A_game_role', 'Role A')
                        pB_role_gv = game_to_display_gv.get('player_B_game_role', 'Role B')
                        
                        st.markdown(f"""
                        - **Player A ({pA_role_gv}):** {get_agent_display_name(pA_id_gv)}
                        - **Player B ({pB_role_gv}):** {get_agent_display_name(pB_id_gv)}
                        - **Scenario Proposer:** {get_agent_display_name(game_to_display_gv.get('proposer_agent_id'))}
                        - **Timestamp Start:** {game_to_display_gv.get('timestamp_start', 'N/A')}
                        - **Timestamp End:** {game_to_display_gv.get('timestamp_end', 'N/A')}
                        """)

                        # Scenario Text and Raw Prompts
                        with st.expander("📜 Scenario Details", expanded=True):
                            st.markdown("###### Generated Scenario Text")
                            st.text_area("Scenario", value=game_to_display_gv.get("scenario_text", "N/A"), height=200, disabled=True, label_visibility="collapsed")
                            
                            scenario_gen_prompt = game_to_display_gv.get("scenario_generation_prompt")
                            if scenario_gen_prompt:
                                st.markdown("###### 🔍 Raw Scenario Generation LLM Input")
                                st.text_area("Scenario Gen Prompt", value=scenario_gen_prompt, height=150, disabled=True, label_visibility="collapsed")
                            
                            scenario_raw_llm_output = game_to_display_gv.get("scenario_raw_llm_output")
                            if scenario_raw_llm_output:
                                st.markdown("###### 📄 Raw Scenario Generation LLM Output (e.g., XML)")
                                st.text_area("Scenario LLM Raw Output", value=scenario_raw_llm_output, height=150, disabled=True, label_visibility="collapsed")


                        st.markdown("##### 💬 Conversation Transcript")
                        transcript_gv = game_to_display_gv.get("transcript", [])
                        if transcript_gv:
                            for i, turn_gv in enumerate(transcript_gv):
                                turn_role_gv = turn_gv.get('role', 'Unknown Role')
                                speaker_agent_id_gv = turn_gv.get('agent_id', 'Unknown Agent')
                                content_gv = turn_gv.get('content', '')
                                
                                avatar_symbol_gv = "👤" 
                                if turn_role_gv == pA_role_gv: avatar_symbol_gv = "🧑‍💻" # Player A
                                elif turn_role_gv == pB_role_gv: avatar_symbol_gv = "🤖" # Player B
                                
                                with st.chat_message(name=turn_role_gv, avatar=avatar_symbol_gv):
                                    st.write(f"**{turn_role_gv} ({get_agent_display_name(speaker_agent_id_gv)})**: ")
                                    st.markdown(content_gv)
                                    
                                    raw_turn_prompt = turn_gv.get('raw_llm_prompt')
                                    if raw_turn_prompt:
                                        with st.expander(f"🔍 Raw LLM Input for this Turn ({turn_role_gv})", expanded=False):
                                            st.text_area(f"LLM Input Turn {i+1}", value=raw_turn_prompt, height=150, disabled=True, label_visibility="collapsed", key=f"turn_prompt_{game_to_display_gv.get('game_id')}_{i}")
                        else:
                            st.info("No transcript available for this game.")

                        st.markdown("##### ⚖️ Adjudication & Outcome")
                        adjudication_cols_gv = st.columns(2)
                        with adjudication_cols_gv[0]:
                            st.markdown(f"**Adjudication Result:**")
                            st.code(game_to_display_gv.get('adjudication_result', 'N/A'), language=None)
                            if game_to_display_gv.get('defaulted_to_tie_reason'):
                                st.caption(f"Defaulted Reason: {game_to_display_gv.get('defaulted_to_tie_reason')}")
                            
                            adjudication_prompt = game_to_display_gv.get("adjudication_prompt")
                            if adjudication_prompt:
                                with st.expander("🔍 Raw Adjudication LLM Input", expanded=False):
                                    st.text_area("Adjudication Prompt", value=adjudication_prompt, height=150, disabled=True, label_visibility="collapsed")
                            
                            adjudication_raw_output = game_to_display_gv.get("adjudication_raw_llm_output")
                            if adjudication_raw_output:
                                with st.expander("📄 Raw Adjudication LLM Output", expanded=False):
                                    st.text_area("Adjudicator Raw LLM Output", value=adjudication_raw_output, height=100, disabled=True, label_visibility="collapsed")

                        with adjudication_cols_gv[1]:
                            bets_gv = game_to_display_gv.get("betting_details") or {}
                            st.markdown(f"""
                            - **{pA_role_gv} Bet:** {bets_gv.get('player_A_bet', 'N/A')}
                            - **{pB_role_gv} Bet:** {bets_gv.get('player_B_bet', 'N/A')}
                            """)
                        
                        st.markdown("##### 💰 Wealth Changes")
                        wealth_c_gv = game_to_display_gv.get("wealth_changes") or {}
                        st.markdown(f"""
                        - **{pA_role_gv} ({get_agent_display_name(pA_id_gv)}) Change:** {wealth_c_gv.get('player_A_wealth_change', 'N/A')}
                        - **{pB_role_gv} ({get_agent_display_name(pB_id_gv)}) Change:** {wealth_c_gv.get('player_B_wealth_change', 'N/A')}
                        """)
                    else:
                        st.error("Could not find details for the selected game ID.")
