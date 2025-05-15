"""
Analysis script for feature trajectories from multi-run simulation data.

This script:
- Looks at files across MULTIPLE different runs (but initially lumps them).
- Gathers all generations.
- Collects, for each generation, every feature that is nonzero and the average
  activation value (normed for population size) associated with it.
- Traces feature paths (trajectories over time).
- Analyzes variance over time and statistically tests its increase.
- Finds outlier activations (more suppressed/repressed than average).
- Identifies consistently activated features across different lineages.
- Makes ONE beautiful plot of feature values over time, smoothed, with
  +/- SD regions colored differently and top/bottom features labeled.
- Creates a null distribution for trajectories and performs a statistical test.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import logging
import yaml
from collections import defaultdict, Counter
from pathlib import Path
from scipy.stats import linregress
from scipy.signal import savgol_filter # For smoothing
import argparse
import random
import math

# --- Constants & Configuration ---
SIMULATION_STATE_BASE_DIR = "simulation_state"
CONFIG_FILENAME = "config_snapshot.json"
LATEST_GEN_TRACKER_FILENAME = "_latest_generation_number.txt"
GENERATION_STATE_PREFIX = "generation_"
GAME_LOG_PREFIX = "games_generation_"

# Keys for accessing data within JSON structures (examples, adjust as needed)
GENOME_KEY = "genome"
ACTIVATION_KEY = "activation"
LABEL_KEY = "label"
POPULATION_STATE_KEY = "population_state"
AGENT_ID_KEY = "agent_id"
PARENT_ID_KEY = "parent_id"

# Config paths (relative to the root of the config dict)
LEARNING_RATE_CONFIG_PATH = "evolution.learning_rate"
ACTIVATION_MIN_CONFIG_PATH = "evolution.activation_min"
ACTIVATION_MAX_CONFIG_PATH = "evolution.activation_max"
POPULATION_SIZE_CONFIG_PATH = "simulation.population_size"

# Plotting
DEFAULT_PLOT_FILENAME = "feature_trajectories.png"
SMOOTHING_WINDOW_SIZE = 5 # For Savitzky-Golay, must be odd
SMOOTHING_POLYORDER = 2   # For Savitzky-Golay
PLOT_TOP_N_LABEL = 5
PLOT_BOTTOM_N_LABEL = 5
NULL_MODEL_SIMULATIONS = 1000

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions ---

def smooth_trajectory(trajectory_data: list, window_size: int, polyorder: int) -> np.ndarray:
    """
    Smooths a trajectory using Savitzky-Golay filter.
    Handles NaNs by interpolating them before smoothing.
    """
    if not isinstance(trajectory_data, list):
        logger.warning("Invalid trajectory data type for smoothing. Expected list.")
        return np.array([])

    y = np.array(trajectory_data, dtype=float)

    # Interpolate NaNs
    nans = np.isnan(y)
    if np.any(nans):
        x = np.arange(len(y))
        try:
            # Interpolate only if there are non-NaN points to interpolate from
            if not np.all(nans):
                y[nans] = np.interp(x[nans], x[~nans], y[~nans])
            else: # All NaNs, return as is or array of NaNs
                logger.debug("Trajectory consists entirely of NaNs. Cannot smooth.")
                return y # Return array of NaNs
        except ValueError as e:
            logger.warning(f"Interpolation failed for trajectory: {e}. Returning original with NaNs.")
            return np.array(trajectory_data, dtype=float) # return original if interp fails

    # Savitzky-Golay filter requires window_size to be odd and smaller than data length
    if len(y) < window_size:
        logger.debug(f"Trajectory length ({len(y)}) is less than window_size ({window_size}). Skipping smoothing.")
        return y

    if window_size % 2 == 0: # window_size is odd
        window_size += 1
        logger.debug(f"Adjusted smoothing window size to be odd: {window_size}")

    if len(y) <= polyorder: # Polyorder must be less than window_size and data length
        logger.debug(f"Trajectory length or window size too small for polyorder {polyorder}. Using simpler smoothing or returning raw.")
        # Fallback to simpler rolling mean if Savgol constraints are not met
        if len(y) > 1:
            return pd.Series(y).rolling(window=min(window_size, len(y)), min_periods=1, center=True).mean().to_numpy()
        return y


    try:
        smoothed = savgol_filter(y, window_size, polyorder)
        return smoothed
    except ValueError as e:
        logger.warning(f"Savitzky-Golay smoothing failed: {e}. Returning original (interpolated) data.")
        return y

def get_config_value(config_dict: dict, path_string: str, default_value: any = None) -> any:
    """
    Safely accesses nested configuration values using a dot-separated path string.
    """
    keys = path_string.split('.')
    current_val = config_dict
    for key in keys:
        if isinstance(current_val, dict) and key in current_val:
            current_val = current_val[key]
        else:
            return default_value
    return current_val

# --- Data Loading and Preprocessing Functions ---

def find_simulation_runs(base_dir_path: Path) -> list[Path]:
    """Locates valid run directories."""
    valid_runs = []
    if not base_dir_path.is_dir():
        logger.error(f"Base directory {base_dir_path} does not exist.")
        return []
    for item in base_dir_path.iterdir():
        if item.is_dir():
            if (item / CONFIG_FILENAME).exists() and \
               (item / LATEST_GEN_TRACKER_FILENAME).exists():
                valid_runs.append(item)
            else:
                logger.debug(f"Skipping directory {item.name}: missing config or latest_gen_tracker.")
    logger.info(f"Found {len(valid_runs)} valid simulation run(s) in {base_dir_path}.")
    return valid_runs

def load_run_config(run_path: Path) -> dict:
    """Loads config_snapshot.json for a run."""
    config_file = run_path / CONFIG_FILENAME
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config snapshot {config_file} not found for run {run_path.name}.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_file} for run {run_path.name}.")
    return {}

def _parse_genome_data(genome_dict: dict, agent_id_for_log: str = "UnknownAgent") -> dict:
    """
    Standardizes genome entry parsing.
    activation is float and label is string.
    """
    parsed_genome = {}
    if not isinstance(genome_dict, dict):
        logger.warning(f"Agent {agent_id_for_log}: Genome data is not a dict: {type(genome_dict)}. Skipping.")
        return {}

    for feature_uuid, feature_data in genome_dict.items():
        activation = 0.0
        label = f"Feature_{feature_uuid[:8]}" # Default label

        if isinstance(feature_data, dict): # New format
            activation = float(feature_data.get(ACTIVATION_KEY, 0.0))
            label = str(feature_data.get(LABEL_KEY, label))
        elif isinstance(feature_data, (int, float)): # Old format (direct activation value)
            activation = float(feature_data)
            # Label remains default
        else:
            logger.warning(f"Agent {agent_id_for_log}: Genome feature {feature_uuid} has unexpected data type: {type(feature_data)}. Using default activation 0.0.")

        parsed_genome[str(feature_uuid)] = {ACTIVATION_KEY: activation, LABEL_KEY: label}
    return parsed_genome

def load_and_process_run_data(run_path: Path, run_id: str, run_config: dict) -> tuple[
    defaultdict, defaultdict, defaultdict, int
]:
    """
    Loads all generation files for a single run, extracts agent genomes,
    parent IDs, and populates data structures for that run.
    Returns: (run_feature_activations, run_population_counts, run_agent_lineages, max_gen_this_run)
    """
    run_feature_activations = defaultdict(lambda: defaultdict(lambda: {'activations': [], 'labels': set()}))
    run_population_counts = defaultdict(int)
    run_agent_lineages = defaultdict(list) # gen -> list of (run_id, agent_id, progenitor_id, genome_dict)
    progenitor_map_this_run = {} # agent_id -> progenitor_id

    max_gen_this_run = 0
    tracker_file = run_path / LATEST_GEN_TRACKER_FILENAME
    try:
        with open(tracker_file, 'r') as f:
            max_gen_this_run = int(f.read().strip())
    except Exception as e:
        logger.error(f"Run {run_id}: Could not read {LATEST_GEN_TRACKER_FILENAME}: {e}. Cannot process this run.")
        return run_feature_activations, run_population_counts, run_agent_lineages, 0

    logger.info(f"Run {run_id}: Processing up to generation {max_gen_this_run}.")

    for gen_num in range(1, max_gen_this_run + 1):
        gen_file = run_path / f"{GENERATION_STATE_PREFIX}{gen_num:04d}.json"
        if not gen_file.exists():
            logger.warning(f"Run {run_id}: Generation file {gen_file.name} not found. Stopping processing for this run at gen {gen_num-1}.")
            max_gen_this_run = gen_num -1
            break
        try:
            with open(gen_file, 'r') as f:
                generation_data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Run {run_id}: Error decoding JSON from {gen_file.name}. Skipping this generation.")
            continue

        population_state = generation_data.get(POPULATION_STATE_KEY, [])
        run_population_counts[gen_num] = len(population_state)

        for agent_dict in population_state:
            agent_id = agent_dict.get(AGENT_ID_KEY)
            if not agent_id:
                logger.warning(f"Run {run_id}, Gen {gen_num}: Agent data missing '{AGENT_ID_KEY}'. Skipping agent.")
                continue

            parent_id = agent_dict.get(PARENT_ID_KEY)
            progenitor_id = None
            if gen_num == 1:
                progenitor_id = agent_id
            elif parent_id:
                progenitor_id = progenitor_map_this_run.get(parent_id)
                if progenitor_id is None:
                    logger.warning(f"Run {run_id}, Gen {gen_num}, Agent {agent_id}: Parent {parent_id} not in progenitor map. Treating agent as new progenitor.")
                    progenitor_id = agent_id # Fallback: treat as new lineage root
            else: # gen_num > 1 but no parent_id (should not happen for valid data)
                logger.warning(f"Run {run_id}, Gen {gen_num}, Agent {agent_id}: No parent_id found. Treating agent as new progenitor.")
                progenitor_id = agent_id

            progenitor_map_this_run[agent_id] = progenitor_id

            raw_genome = agent_dict.get(GENOME_KEY, {})
            parsed_genome = _parse_genome_data(raw_genome, agent_id)

            run_agent_lineages[gen_num].append((run_id, agent_id, progenitor_id, parsed_genome))

            for feature_uuid, feat_details in parsed_genome.items():
                activation_val = feat_details.get(ACTIVATION_KEY, 0.0)
                if activation_val != 0.0: # Only store non-zero activations
                    run_feature_activations[gen_num][feature_uuid]['activations'].append(activation_val)
                    run_feature_activations[gen_num][feature_uuid]['labels'].add(feat_details[LABEL_KEY])

    return run_feature_activations, run_population_counts, run_agent_lineages, max_gen_this_run

def collate_all_runs_data(base_dir_str: str) -> dict:
    """Orchestrates loading for all runs and "lumps" the data."""
    base_dir_path = Path(base_dir_str)
    simulation_run_paths = find_simulation_runs(base_dir_path)

    if not simulation_run_paths:
        logger.error("No valid simulation runs found. Exiting.")
        return {}

    # Collated data structures
    all_feature_activations_by_gen = defaultdict(lambda: defaultdict(lambda: {'activations': [], 'labels': set()}))
    population_counts_by_gen = defaultdict(int) # Total population for a given gen_num across all runs
    agent_lineage_info_by_gen = defaultdict(list) # gen_num -> list of (run_id, agent_id, prog_id, genome)
    global_config = {}
    max_generation_observed = 0

    first_run_config_loaded = False
    for run_path in simulation_run_paths:
        run_id = run_path.name
        logger.info(f"Processing run: {run_id}")
        run_config = load_run_config(run_path)
        if not run_config:
            logger.warning(f"Skipping run {run_id} due to config loading error.")
            continue

        if not first_run_config_loaded:
            global_config[LEARNING_RATE_CONFIG_PATH.split('.')[-1]] = get_config_value(run_config, LEARNING_RATE_CONFIG_PATH, 0.1)
            global_config[ACTIVATION_MIN_CONFIG_PATH.split('.')[-1]] = get_config_value(run_config, ACTIVATION_MIN_CONFIG_PATH, -5.0)
            global_config[ACTIVATION_MAX_CONFIG_PATH.split('.')[-1]] = get_config_value(run_config, ACTIVATION_MAX_CONFIG_PATH, 5.0)
            global_config[POPULATION_SIZE_CONFIG_PATH.split('.')[-1]] = get_config_value(run_config, POPULATION_SIZE_CONFIG_PATH, 25) # Example default
            first_run_config_loaded = True
        # else: Consider checking for consistency if needed, for now, first run sets global config values

        r_feat_act, r_pop_counts, r_agent_lineages, r_max_gen = load_and_process_run_data(run_path, run_id, run_config)

        # Merge data
        for gen, features_data in r_feat_act.items():
            for feat_uuid, data_dict in features_data.items():
                all_feature_activations_by_gen[gen][feat_uuid]['activations'].extend(data_dict['activations'])
                all_feature_activations_by_gen[gen][feat_uuid]['labels'].update(data_dict['labels'])

        for gen, count in r_pop_counts.items():
            population_counts_by_gen[gen] += count

        for gen, lineage_list in r_agent_lineages.items():
            agent_lineage_info_by_gen[gen].extend(lineage_list)

        if r_max_gen > max_generation_observed:
            max_generation_observed = r_max_gen

    # Sanitize labels (pick one if multiple, though usually consistent)
    for gen_data in all_feature_activations_by_gen.values():
        for feat_data in gen_data.values():
            if feat_data['labels']:
                feat_data['label'] = list(feat_data['labels'])[0]
            else:
                feat_data['label'] = "Unknown_Label"
            del feat_data['labels'] # Remove the set after picking one

    return {
        "all_feature_activations_by_gen": all_feature_activations_by_gen,
        "population_counts_by_gen": population_counts_by_gen,
        "agent_lineage_info_by_gen": agent_lineage_info_by_gen,
        "global_config": global_config,
        "max_generation_observed": max_generation_observed
    }

# --- Analysis Functions ---

def calculate_feature_trajectories(
    all_feature_activations_by_gen: defaultdict,
    population_counts_by_gen: defaultdict,
    max_generation_observed: int
) -> dict:
    """
    Calculates average activation per feature per generation, normed by total population size.
    """
    logger.info("Calculating feature trajectories (average activations)...")
    feature_trajectories_data = defaultdict(lambda: {
        'label': 'Unknown',
        'avg_trajectory': [None] * max_generation_observed
    })
    all_feature_uuids_seen = set()
    for gen_num in range(1, max_generation_observed + 1):
        if gen_num in all_feature_activations_by_gen:
            for feat_uuid in all_feature_activations_by_gen[gen_num].keys():
                all_feature_uuids_seen.add(feat_uuid)

    for feat_uuid in all_feature_uuids_seen:
        first_label_found = "Unknown"
        # Find a label for this feature (can appear in later generations)
        for gen_num_for_label in range(1, max_generation_observed + 1):
            if gen_num_for_label in all_feature_activations_by_gen and \
               feat_uuid in all_feature_activations_by_gen[gen_num_for_label]:
                first_label_found = all_feature_activations_by_gen[gen_num_for_label][feat_uuid]['label']
                break
        feature_trajectories_data[feat_uuid]['label'] = first_label_found

        for gen_num in range(1, max_generation_observed + 1):
            total_pop_in_gen = population_counts_by_gen.get(gen_num, 0)
            if total_pop_in_gen == 0:
                feature_trajectories_data[feat_uuid]['avg_trajectory'][gen_num - 1] = 0.0 # Or None, but 0 might be better for plotting means
                continue

            feature_data_for_gen = all_feature_activations_by_gen.get(gen_num, {}).get(feat_uuid)
            if feature_data_for_gen:
                sum_activations = sum(feature_data_for_gen['activations'])
                # Normed for population size: sum of activations / total agents in generation
                # (agents not having the feature contribute 0 to this sum)
                avg_activation = sum_activations / total_pop_in_gen
                feature_trajectories_data[feat_uuid]['avg_trajectory'][gen_num - 1] = avg_activation
            else: # Feature not present or no activations recorded for it in this generation
                feature_trajectories_data[feat_uuid]['avg_trajectory'][gen_num - 1] = 0.0 # Treat as 0 if not active

    logger.info(f"Calculated trajectories for {len(feature_trajectories_data)} features.")
    return dict(feature_trajectories_data)


def analyze_feature_variance_trends(
    all_feature_activations_by_gen: defaultdict,
    population_counts_by_gen: defaultdict,
    max_generation_observed: int
) -> dict:
    """
    Calculates variance of activations for each feature within each generation.
    Performs statistical test (linear regression) on the trend of these variances.
    """
    logger.info("Analyzing feature variance trends...")
    feature_variance_data = defaultdict(lambda: {
        'label': 'Unknown',
        'variance_trajectory': [None] * max_generation_observed,
        'trend_stats_full': {},
        'trend_stats_late': {}
    })
    all_feature_uuids_seen = set() # Collect all unique feature UUIDs first
    for gen_num in range(1, max_generation_observed + 1):
        if gen_num in all_feature_activations_by_gen:
            for feat_uuid in all_feature_activations_by_gen[gen_num].keys():
                all_feature_uuids_seen.add(feat_uuid)

    for feat_uuid in all_feature_uuids_seen:
        first_label_found = "Unknown"
        for gen_num_for_label in range(1, max_generation_observed + 1):
            if gen_num_for_label in all_feature_activations_by_gen and \
               feat_uuid in all_feature_activations_by_gen[gen_num_for_label]:
                first_label_found = all_feature_activations_by_gen[gen_num_for_label][feat_uuid]['label']
                break
        feature_variance_data[feat_uuid]['label'] = first_label_found

        for gen_num in range(1, max_generation_observed + 1):
            total_pop_in_gen = population_counts_by_gen.get(gen_num, 0)
            if total_pop_in_gen <= 1: # Variance is undefined or 0 for 1 or 0 elements
                feature_variance_data[feat_uuid]['variance_trajectory'][gen_num - 1] = None # Or 0.0, None is better for stats
                continue

            feature_gen_data = all_feature_activations_by_gen.get(gen_num, {}).get(feat_uuid)
            if feature_gen_data:
                activations_present = feature_gen_data['activations']
                # Full distribution for variance includes agents not expressing the feature (activation 0)
                full_distribution_for_variance = activations_present + [0.0] * (total_pop_in_gen - len(activations_present))
                variance_val = np.var(full_distribution_for_variance)
                feature_variance_data[feat_uuid]['variance_trajectory'][gen_num - 1] = variance_val
            else:
                feature_variance_data[feat_uuid]['variance_trajectory'][gen_num - 1] = 0.0 # If feature not active at all in this gen, variance of activations is 0

    # Perform linear regression on variance trajectories
    for feature_uuid, data in feature_variance_data.items():
        generations = np.arange(1, max_generation_observed + 1)
        variances = np.array([v if v is not None else np.nan for v in data['variance_trajectory']])

        # Full trajectory
        valid_indices_full = ~np.isnan(variances)
        if np.sum(valid_indices_full) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(generations[valid_indices_full], variances[valid_indices_full])
            data['trend_stats_full'] = {'slope': slope, 'p_value': p_value, 'r_squared': r_value**2}
            logger.info(f"Feature {data['label']} ({feature_uuid}) variance trend (full): slope={slope:.4e}, p={p_value:.4f}, R^2={r_value**2:.4f}")

        # Late trajectory (last half of generations)
        late_start_gen_idx = max_generation_observed // 2
        generations_late = generations[late_start_gen_idx:]
        variances_late = variances[late_start_gen_idx:]
        valid_indices_late = ~np.isnan(variances_late)
        if np.sum(valid_indices_late) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(generations_late[valid_indices_late], variances_late[valid_indices_late])
            data['trend_stats_late'] = {'slope': slope, 'p_value': p_value, 'r_squared': r_value**2}
            logger.info(f"Feature {data['label']} ({feature_uuid}) variance trend (late): slope={slope:.4e}, p={p_value:.4f}, R^2={r_value**2:.4f}")

    return dict(feature_variance_data)

def find_activation_outliers(
    all_feature_activations_by_gen: defaultdict,
    population_counts_by_gen: defaultdict,
    sd_threshold: float = 2.0
):
    """
    Identifies features/generations with agents having outlier activations.
    This function primarily prints results.
    """
    logger.info(f"Finding activation outliers (threshold: {sd_threshold} SDs)...")
    outlier_summary = defaultdict(list)

    for gen_num, features_in_gen_data in all_feature_activations_by_gen.items():
        total_pop_in_gen = population_counts_by_gen.get(gen_num, 0)
        if total_pop_in_gen == 0:
            continue

        for feature_uuid, data in features_in_gen_data.items():
            feature_label = data.get('label', f"UUID_{feature_uuid[:8]}")
            activations_present = data.get('activations', [])
            if not activations_present: # Should not happen if data exists for feature
                continue

            # Consider the distribution of activations for this feature across the *entire population*
            # (agents without the feature implicitly have activation 0 for this calculation)
            full_dist_for_stats = activations_present + [0.0] * (total_pop_in_gen - len(activations_present))
            mean_act = np.mean(full_dist_for_stats)
            std_act = np.std(full_dist_for_stats)

            if std_act > 1e-6: # Avoid division by zero or near-zero std
                found_outliers_for_this_feature_gen = []
                # We check outliers only among agents *actually expressing* the feature
                for act_val in activations_present:
                    if abs(act_val - mean_act) > sd_threshold * std_act:
                        found_outliers_for_this_feature_gen.append(round(act_val, 3))

                if found_outliers_for_this_feature_gen:
                    msg = (f"Gen {gen_num}, Feature '{feature_label}' ({feature_uuid[:8]}): "
                           f"Outlier activations {found_outliers_for_this_feature_gen} "
                           f"(Mean across pop: {mean_act:.3f}, SD: {std_act:.3f})")
                    logger.info(msg)
                    outlier_summary[gen_num].append({
                        'feature_uuid': feature_uuid,
                        'label': feature_label,
                        'outliers': found_outliers_for_this_feature_gen,
                        'mean': mean_act,
                        'std': std_act
                    })
    return dict(outlier_summary) # Or just print/log

def find_consistently_activated_features(
    agent_lineage_info_by_gen: defaultdict,
    min_distinct_lineages: int = 2
) -> dict:
    """
    Identifies features activated in multiple distinct lineages.
    A lineage is defined by (run_id, progenitor_id).
    """
    logger.info(f"Finding consistently activated features (min distinct lineages: {min_distinct_lineages})...")
    feature_activation_by_lineage = defaultdict(set) # feature_uuid -> set of (run_id, progenitor_id)
    feature_labels_map = {} # feature_uuid -> label (to store one representative label)

    for gen_num, agents_in_gen_list in agent_lineage_info_by_gen.items():
        for run_id, agent_id, progenitor_id, genome_dict in agents_in_gen_list:
            if progenitor_id is None: # Should be handled during loading, but defensive
                logger.debug(f"Gen {gen_num}, Agent {agent_id}: Progenitor ID is None. Skipping for lineage analysis.")
                continue

            for feature_uuid, feat_details in genome_dict.items():
                if feat_details.get(ACTIVATION_KEY, 0.0) != 0.0: # Feature is active
                    feature_activation_by_lineage[feature_uuid].add((run_id, progenitor_id))
                    if feature_uuid not in feature_labels_map:
                        feature_labels_map[feature_uuid] = feat_details.get(LABEL_KEY, f"UUID_{feature_uuid[:8]}")

    consistent_features = {}
    for feature_uuid, lineages_set in feature_activation_by_lineage.items():
        if len(lineages_set) >= min_distinct_lineages:
            label = feature_labels_map.get(feature_uuid, f"UUID_{feature_uuid[:8]}")
            consistent_features[feature_uuid] = {
                'label': label,
                'distinct_lineage_count': len(lineages_set),
                'activating_lineages': lineages_set # set of (run_id, progenitor_id) tuples
            }
            logger.info(f"Consistent Feature: '{label}' ({feature_uuid[:8]}) "
                        f"activated in {len(lineages_set)} distinct lineages.")
    if not consistent_features:
        logger.info("No features found that meet the consistent activation criteria.")
    return consistent_features


def generate_null_model_trajectories(
    actual_trajectories: dict,
    global_config: dict,
    max_generation_observed: int,
    num_null_sims: int = NULL_MODEL_SIMULATIONS
) -> dict:
    """
    Generates random walk trajectories based on observed change rates.
    """
    logger.info(f"Generating {num_null_sims} null model trajectories for each feature...")
    if not actual_trajectories:
        logger.warning("No actual trajectories to base null model on. Skipping.")
        return {}

    learning_rate = global_config.get('learning_rate', 0.1)
    activation_min = global_config.get('activation_min', -5.0)
    activation_max = global_config.get('activation_max', 5.0)

    # Calculate overall change rate from actual_trajectories
    total_changes = 0
    total_possible_steps = 0
    for feat_uuid, data in actual_trajectories.items():
        traj = [v for v in data['avg_trajectory'] if v is not None]
        if len(traj) > 1:
            for i in range(1, len(traj)):
                if abs(traj[i] - traj[i-1]) > 1e-7: # Small epsilon for float changes
                    total_changes += 1
                total_possible_steps += 1
    
    observed_change_rate = (total_changes / total_possible_steps) if total_possible_steps > 0 else 0.05 # Default if no changes
    logger.info(f"Null Model: Observed change rate in actual data: {observed_change_rate:.4f}")

    null_distributions = defaultdict(list)
    for feature_uuid, data in actual_trajectories.items():
        # Initial activation is the average activation in Gen 1, or 0 if not present/None
        initial_activation_val = data['avg_trajectory'][0] if data['avg_trajectory'] and data['avg_trajectory'][0] is not None else 0.0

        for _ in range(num_null_sims):
            null_traj_points = [initial_activation_val]
            current_val = initial_activation_val
            for _ in range(1, max_generation_observed): # For subsequent generations
                if random.random() < observed_change_rate:
                    change_direction = random.choice([-1, 1])
                    current_val += change_direction * learning_rate
                    current_val = max(activation_min, min(activation_max, current_val))
                null_traj_points.append(current_val)
            null_distributions[feature_uuid].append(null_traj_points)
    return dict(null_distributions)


def perform_trajectory_statistical_test(
    actual_trajectories: dict,
    null_model_trajectories: dict
) -> dict:
    """
    Compares actual trajectories against the null model using a test statistic.
    Example test statistic: Sum of absolute changes.
    """
    logger.info("Performing statistical test of actual trajectories against null model...")
    if not actual_trajectories or not null_model_trajectories:
        logger.warning("Missing actual or null trajectories for statistical test. Skipping.")
        return {}

    def calculate_trajectory_statistic(trajectory: list) -> float:
        """Calculates sum of absolute changes for a trajectory."""
        if not trajectory or len(trajectory) < 2:
            return 0.0
        return sum(abs(trajectory[i] - trajectory[i-1]) for i in range(1, len(trajectory)))

    trajectory_test_results = {}
    for feature_uuid, actual_data in actual_trajectories.items():
        actual_traj_values = [v for v in actual_data['avg_trajectory'] if v is not None]
        if not actual_traj_values or len(actual_traj_values) < 2:
            continue

        stat_actual = calculate_trajectory_statistic(actual_traj_values)
        null_stats_for_feature = []
        if feature_uuid in null_model_trajectories:
            for null_single_traj in null_model_trajectories[feature_uuid]:
                null_stats_for_feature.append(calculate_trajectory_statistic(null_single_traj))

        if null_stats_for_feature:
            # P-value: proportion of null stats that are as extreme or more extreme than actual
            num_more_extreme = sum(1 for s_null in null_stats_for_feature if abs(s_null) >= abs(stat_actual))
            p_value = num_more_extreme / len(null_stats_for_feature)
            
            label = actual_data.get('label', f"UUID_{feature_uuid[:8]}")
            logger.info(f"Stat Test - Feature '{label}' ({feature_uuid[:8]}): Actual Stat={stat_actual:.3f}, p-value={p_value:.4f}")
            trajectory_test_results[feature_uuid] = {
                'label': label,
                'actual_stat': stat_actual,
                'p_value': p_value,
                'null_stat_mean': np.mean(null_stats_for_feature) if null_stats_for_feature else None
            }
        else:
            logger.warning(f"No null statistics generated for feature {feature_uuid} for statistical test.")

    return trajectory_test_results


# --- Plotting Function ---

def plot_feature_activation_trajectories(
    feature_trajectories_data: dict,
    output_filename: str = DEFAULT_PLOT_FILENAME,
    top_n_label: int = PLOT_TOP_N_LABEL,
    bottom_n_label: int = PLOT_BOTTOM_N_LABEL,
    smoothing_window: int = SMOOTHING_WINDOW_SIZE,
    smoothing_order: int = SMOOTHING_POLYORDER
):
    """
    Makes ONE beautiful plot of the feature values over time.
    Smooths it. Anything +/- SD away from the mean of THAT FEATURE'S TRAJECTORY,
    make it colorful. Everything within an SD should be gray.
    Label the top N and bottom N features by their final activation value.
    """
    logger.info(f"Generating feature activation trajectory plot: {output_filename}")
    if not feature_trajectories_data:
        logger.warning("No feature trajectory data to plot.")
        return

    num_features = len(feature_trajectories_data)
    max_gen = 0
    if feature_trajectories_data:
        # Get max_gen from the length of the first trajectory list
        first_traj_key = list(feature_trajectories_data.keys())[0]
        max_gen = len(feature_trajectories_data[first_traj_key]['avg_trajectory'])
    if max_gen == 0:
        logger.warning("Max generation is 0, cannot plot trajectories.")
        return

    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Determine unique labels for colormap, assign consistent colors
    sorted_feature_items = sorted(feature_trajectories_data.items(), key=lambda item: item[1]['label'])
    
    # Create a colormap for distinct features if they go outside SD
    # Using a qualitative colormap like 'tab20' or 'Paired'
    colors = plt.cm.get_cmap('tab20', num_features)
    feature_color_map = {item[0]: colors(i) for i, item in enumerate(sorted_feature_items)}

    final_values_for_labeling = {} # feature_uuid -> (final_value, label)

    for i, (feature_uuid, data) in enumerate(sorted_feature_items):
        raw_traj = data['avg_trajectory']
        label = data.get('label', f"UUID_{feature_uuid[:8]}")

        # Convert to numpy array with NaNs for smoothing
        traj_np = np.array([val if val is not None else np.nan for val in raw_traj], dtype=float)
        
        smoothed_traj = smooth_trajectory(list(traj_np), smoothing_window, smoothing_order)
        
        # Store final non-NaN value for labeling
        final_val = np.nan
        for val in reversed(smoothed_traj):
            if not np.isnan(val):
                final_val = val
                break
        if not np.isnan(final_val):
            final_values_for_labeling[feature_uuid] = (final_val, label)

        mean_of_this_feature_traj = np.nanmean(smoothed_traj)
        std_of_this_feature_traj = np.nanstd(smoothed_traj)

        x_axis = np.arange(max_gen)
        
        # Create segments for LineCollection
        points = np.array([x_axis, smoothed_traj]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        line_colors = []
        for k in range(len(smoothed_traj) - 1):
            # Color based on the starting point of the segment
            val_k = smoothed_traj[k]
            if np.isnan(val_k) or np.isnan(mean_of_this_feature_traj) or np.isnan(std_of_this_feature_traj) or std_of_this_feature_traj < 1e-6:
                line_colors.append('gray') # Default to gray if stats are undefined
            elif val_k > mean_of_this_feature_traj + std_of_this_feature_traj or \
                 val_k < mean_of_this_feature_traj - std_of_this_feature_traj:
                line_colors.append(feature_color_map.get(feature_uuid, 'blue')) # Colorful
            else:
                line_colors.append('lightgray') # Within SD, gray

        lc = LineCollection(segments, colors=line_colors, linewidths=1.2, alpha=0.8, capstyle='round')
        ax.add_collection(lc)

    # Labeling Top N and Bottom N features
    if final_values_for_labeling:
        sorted_by_final_val = sorted(final_values_for_labeling.items(), key=lambda item: item[1][0])
        
        top_features = [item[0] for item in sorted_by_final_val[-top_n_label:]]
        bottom_features = [item[0] for item in sorted_by_final_val[:bottom_n_label]]
        features_to_label = set(top_features + bottom_features)

        for feature_uuid_to_label in features_to_label:
            if feature_uuid_to_label in final_values_for_labeling:
                final_val, label_text = final_values_for_labeling[feature_uuid_to_label]
                # Find the trajectory again for plotting position
                original_data = feature_trajectories_data[feature_uuid_to_label]
                traj_np_label = np.array([val if val is not None else np.nan for val in original_data['avg_trajectory']], dtype=float)
                smoothed_traj_label = smooth_trajectory(list(traj_np_label), smoothing_window, smoothing_order)
                
                # Find last non-NaN x,y for annotation
                last_x, last_y = -1, np.nan
                for x_idx, y_val in reversed(list(enumerate(smoothed_traj_label))):
                    if not np.isnan(y_val):
                        last_x, last_y = x_idx, y_val
                        break
                
                if last_x != -1:
                    ax.text(last_x + 0.5, last_y, label_text, fontsize=9,
                            color=feature_color_map.get(feature_uuid_to_label, 'black'),
                            verticalalignment='center')

    ax.set_xlim(0, max_gen -1 if max_gen > 0 else 1)
    ax.set_xlabel("Generation", fontsize=14)
    ax.set_ylabel("Average Activation (Smoothed)", fontsize=14)
    ax.set_title("Feature Activation Trajectories Over Generations", fontsize=16, fontweight='bold')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    try:
        plt.savefig(output_filename, dpi=300)
        logger.info(f"Plot saved to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    plt.close(fig)


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze feature evolution from simulation data.")
    parser.add_argument(
        "--sim_base_dir",
        type=str,
        default=SIMULATION_STATE_BASE_DIR,
        help="Base directory where simulation run folders are stored."
    )
    parser.add_argument(
        "--output_plot_file",
        type=str,
        default=DEFAULT_PLOT_FILENAME,
        help="Filename for the output plot."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    parser.add_argument(
        "--min_lineages_consistent",
        type=int,
        default=2,
        help="Minimum number of distinct lineages for a feature to be 'consistently activated'."
    )
    parser.add_argument(
        "--outlier_sd_threshold",
        type=float,
        default=2.0,
        help="SD threshold for identifying outlier activations."
    )

    args = parser.parse_args()

    # Update logger level
    logger.setLevel(args.log_level.upper())
    for handler in logger.handlers: # Also update handlers if already configured
        handler.setLevel(args.log_level.upper())

    logger.info("Starting feature analysis script.")

    collated_data = collate_all_runs_data(args.sim_base_dir)

    if not collated_data:
        logger.critical("Failed to load or collate any run data. Exiting.")
    else:
        all_feat_act_by_gen = collated_data["all_feature_activations_by_gen"]
        pop_counts_by_gen = collated_data["population_counts_by_gen"]
        agent_lineage_info = collated_data["agent_lineage_info_by_gen"]
        global_cfg = collated_data["global_config"]
        max_gen_obs = collated_data["max_generation_observed"]

        if max_gen_obs == 0:
            logger.warning("Maximum observed generation is 0. No data to analyze.")
        else:
            # 1. Calculate feature trajectories
            trajectories = calculate_feature_trajectories(
                all_feat_act_by_gen, pop_counts_by_gen, max_gen_obs
            )

            # 2. Analyze feature variance trends
            variance_analysis = analyze_feature_variance_trends(
                all_feat_act_by_gen, pop_counts_by_gen, max_gen_obs
            )
            # (Results are logged within the function)

            # 3. Find activation outliers
            outlier_analysis = find_activation_outliers(
                all_feat_act_by_gen, pop_counts_by_gen, args.outlier_sd_threshold
            )
            # (Results are logged within the function)

            # 4. Find consistently activated features
            consistent_features_info = find_consistently_activated_features(
                agent_lineage_info, args.min_lineages_consistent
            )
            # (Results are logged within the function)

            # 5. Null model and statistical test
            if trajectories:
                null_distributions = generate_null_model_trajectories(
                    trajectories, global_cfg, max_gen_obs
                )
                if null_distributions:
                    stat_test_results = perform_trajectory_statistical_test(
                        trajectories, null_distributions
                    )
                    # (Results are logged within the function)

            # 6. Plotting
            if trajectories:
                plot_feature_activation_trajectories(
                    trajectories,
                    args.output_plot_file,
                    top_n_label=PLOT_TOP_N_LABEL,
                    bottom_n_label=PLOT_BOTTOM_N_LABEL,
                    smoothing_window=SMOOTHING_WINDOW_SIZE,
                    smoothing_order=SMOOTHING_POLYORDER
                )
            else:
                logger.warning("No trajectories calculated, skipping plot generation.")

    logger.info("Feature analysis script finished.")
