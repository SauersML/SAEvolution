"""
This module isolates all direct communication with external Large Language
Model APIs. It provides the mechanisms for generating text responses while
incorporating behavioral steering based on an agent's internal feature state,
performing contrastive analyses to identify features differentiating between
interaction sets, and requesting external judgments on game outcomes. It also
handles low-level details like API key management, request/response formatting,
and error handling for these external service interactions.
"""

import goodfire
from goodfire import exceptions as goodfire_exceptions
import logging
import os
import time
import random
import hashlib
import json
import copy
import re
import uuid # For unique seed in scenario generation

_client_instance = None
_last_client_config_hash = None

def _get_config_hash(config: dict) -> str:
    relevant_config = {
        'api_key_env_var': config.get('goodfire', {}).get('api_key_env_var'),
        'base_url': config.get('goodfire', {}).get('base_url')
    }
    return hashlib.sha256(json.dumps(relevant_config, sort_keys=True).encode()).hexdigest()

def get_goodfire_client(config: dict) -> goodfire.Client:
    """
    Initializes and returns a Goodfire client instance.
    Manages a global client instance, re-initializing if config changes.
    """
    global _client_instance
    global _last_client_config_hash

    if not isinstance(config, dict):
        logging.critical("get_goodfire_client: Invalid config type provided (must be a dictionary).")
        raise TypeError("Configuration must be a dictionary.")

    current_config_hash = _get_config_hash(config)

    if _client_instance is not None and current_config_hash == _last_client_config_hash:
        return _client_instance

    goodfire_config_dict = config.get('goodfire', {})
    api_key_env_var = goodfire_config_dict.get('api_key_env_var', 'GOODFIRE_API_KEY')
    api_key = os.environ.get(api_key_env_var)
    base_url = goodfire_config_dict.get('base_url')

    if not api_key:
        logging.critical(f"Goodfire API key not found in environment variable: {api_key_env_var}")
        raise ValueError(f"Environment variable {api_key_env_var} must be set for Goodfire API.")

    try:
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        _client_instance = goodfire.Client(**client_args)
        logging.info(f"Goodfire client initialized (Base URL: {base_url if base_url else 'Default Goodfire API URL'}).")
        _last_client_config_hash = current_config_hash
        return _client_instance
    except Exception as e:
        logging.critical(f"Failed to initialize Goodfire client: {e}", exc_info=True)
        _client_instance = None # Reset on failure
        _last_client_config_hash = None
        raise RuntimeError("Could not initialize Goodfire client.") from e

def _execute_api_call(api_call_func: callable, *args, **kwargs):
    """
    Executes an API call with retry logic for specified errors.
    """
    retry_config = kwargs.pop('retry_config', {})
    max_retries = retry_config.get('max_retries', 3)
    initial_delay = retry_config.get('initial_delay', 1.0)
    backoff_factor = retry_config.get('backoff_factor', 2.0)

    if not callable(api_call_func):
         raise TypeError("api_call_func must be callable.")

    current_delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            result = api_call_func(*args, **kwargs)
            return result
        except goodfire_exceptions.APIStatusError as e: # Catch HTTP status errors from Goodfire SDK
            status_code = e.status_code
            if status_code in [429, 500, 502, 503, 504]: # Retriable HTTP status codes
                logging.warning(
                    f"Retriable API Status Error (Status: {status_code}) on attempt {attempt + 1}/{max_retries} "
                    f"for {getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
                )
                last_exception = e
            else:
                logging.error(
                    f"Non-retriable API Status Error (Status: {status_code}) for "
                    f"{getattr(api_call_func, '__name__', 'API call')}: {e}."
                )
                raise # Re-raise non-retriable Goodfire API status errors
        except goodfire_exceptions.APITimeoutError as e: # Catch specific timeout errors from Goodfire SDK
            logging.warning(
                f"Goodfire API Timeout Error on attempt {attempt + 1}/{max_retries} "
                f"for {getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
            )
            last_exception = e
        except goodfire_exceptions.GoodfireError as e: # Catch other Goodfire-specific errors
            # Decide if generic GoodfireErrors are retriable. For now, let's assume they might be.
            logging.warning(
                f"Goodfire API Error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries} "
                f"for {getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
            )
            last_exception = e
        except requests.exceptions.Timeout as e: # Catch timeouts from the underlying requests library
            logging.warning(
                f"HTTP Request Timeout on attempt {attempt + 1}/{max_retries} for "
                f"{getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
            )
            last_exception = e
        except requests.exceptions.RequestException as e: # Catch other network errors from requests library
             logging.warning(
                f"HTTP Request Exception on attempt {attempt + 1}/{max_retries} for "
                f"{getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
            )
             last_exception = e
        except Exception as e: # Catch any other unexpected errors
            logging.error(
                f"Unexpected error during API call attempt {attempt + 1}/{max_retries} for "
                f"{getattr(api_call_func, '__name__', 'API call')}: {e}", exc_info=True
            )
            last_exception = e
            if attempt == max_retries - 1: # If it's the last attempt, re-raise
                raise

        time.sleep(current_delay)
        current_delay *= backoff_factor
        # Add a small jitter to the delay
        current_delay += random.uniform(0, initial_delay * 0.1)

    logging.error(f"API call {getattr(api_call_func, '__name__', 'API call')} failed after {max_retries} retries.")
    # Re-raise the last exception that occurred, or a generic TimeoutError if last_exception is None
    if last_exception:
        raise TimeoutError(f"API call failed after {max_retries} retries.") from last_exception
    else: # Should not happen if loop runs at least once and fails
        raise TimeoutError(f"API call failed after {max_retries} retries, but no specific exception was captured.")


def generate_scenario(proposer_agent, config: dict) -> tuple[dict | None, str | None]:
    """
    Generates a game scenario using the proposer agent and configuration.
    Returns:
        A tuple:
        (dict: {'scenario_text': str, 'role_assignment': dict, 'raw_output': str} or None on failure,
         str: The prompt_text used for generation or None on early failure)
    """
    from manager import Agent # Local import to avoid circular dependency issues

    if not isinstance(proposer_agent, Agent):
        logging.error("generate_scenario: Invalid proposer_agent provided (must be an Agent instance).")
        return None, None
    if not isinstance(config, dict):
        logging.error("generate_scenario: Invalid config provided (must be a dictionary).")
        return None, None

    client = get_goodfire_client(config)
    gen_config = config.get('generation', {}).get('scenario', {})
    model_id = proposer_agent.model_id
    prompt_text_template = gen_config.get('prompt', "Error: Scenario generation prompt template not found in config.")
    max_tokens = gen_config.get('max_tokens', 1000)
    temperature = gen_config.get('temperature', 0.7)
    api_retry_config = config.get('api_retries', {})

    # Add diversification seed to the prompt to prevent caching
    # Using agent_id and a random UUID for uniqueness per specific call.
    diversification_seed = f"agent_id:{proposer_agent.agent_id}_call_id:{uuid.uuid4().hex[:8]}"
    final_prompt_text = f"{prompt_text_template}\n[{diversification_seed}]"

    variant = goodfire.Variant(model_id)
    variant_edits = proposer_agent.genome
    if isinstance(variant_edits, dict) and variant_edits:
        try:
            variant.set(variant_edits)
        except Exception as e: # Catch potential errors from variant.set()
            logging.error(f"Error setting variant edits for agent {proposer_agent.agent_id} in generate_scenario: {e}", exc_info=True)
            # Continue with base variant if edits fail to apply

    messages = [{"role": "user", "content": final_prompt_text}]
    raw_content = None # Initialize for logging in case of early failure

    try:
        logging.debug(f"Attempting scenario generation for agent {proposer_agent.agent_id} with prompt ending: ...{final_prompt_text[-100:]}")
        response = _execute_api_call(
            client.chat.completions.create,
            messages=messages,
            model=variant,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False, # Ensure stream is False for single response
            retry_config=api_retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
             logging.error(f"Invalid or empty response structure received from API for scenario generation (Agent: {proposer_agent.agent_id}).")
             return None, final_prompt_text

        raw_content = response.choices[0].message.get('content', '').strip()
        logging.debug(f"Raw LLM output for scenario (Agent {proposer_agent.agent_id}):\n{raw_content}")

        if not raw_content:
             logging.warning(f"Empty content received from API for scenario generation (Agent: {proposer_agent.agent_id}).")
             return None, final_prompt_text

        tags_content = {}
        # Ensure tag names exactly match those in the prompt
        tag_names = ["context", "roles", "objectives", "win_criteria", "tie_criteria", "proposer_role"]
        
        parsing_successful = True
        for tag_name in tag_names:
            # Using re.IGNORECASE might be too lenient if strict XML case is expected by the LLM.
            # Using re.DOTALL is crucial for multi-line content within tags.
            match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", raw_content, re.DOTALL | re.IGNORECASE)
            if match:
                tags_content[tag_name] = match.group(1).strip()
            else:
                logging.warning(f"Required tag <{tag_name}> not found in LLM output for scenario (Agent: {proposer_agent.agent_id}). Raw content snippet: {raw_content[:500]}...")
                parsing_successful = False
                break # Stop parsing if a required tag is missing
        
        if not parsing_successful:
            return None, final_prompt_text

        # Validate content constraints
        if "objective" not in tags_content.get("objectives", "").lower():
            logging.warning(f"Keyword 'objective' not found in <objectives> content (Agent: {proposer_agent.agent_id}). Content: '{tags_content.get('objectives', '')}'")
            return None, final_prompt_text
        if "win criteria" not in tags_content.get("win_criteria", "").lower():
            logging.warning(f"Phrase 'win criteria' not found in <win_criteria> content (Agent: {proposer_agent.agent_id}). Content: '{tags_content.get('win_criteria', '')}'")
            return None, final_prompt_text
        if "tie criteria" not in tags_content.get("tie_criteria", "").lower():
            logging.warning(f"Phrase 'tie criteria' not found in <tie_criteria> content (Agent: {proposer_agent.agent_id}). Content: '{tags_content.get('tie_criteria', '')}'")
            return None, final_prompt_text
        
        proposer_role_text = tags_content.get("proposer_role", "")
        if proposer_role_text not in ["Role A", "Role B"]:
            logging.warning(f"Invalid content for <proposer_role>: '{proposer_role_text}'. Expected 'Role A' or 'Role B'. (Agent: {proposer_agent.agent_id})")
            return None, final_prompt_text

        assembled_scenario_text = (
            f"Context:\n{tags_content['context']}\n\n"
            f"Roles:\n{tags_content['roles']}\n\n"
            f"Objective(s):\n{tags_content['objectives']}\n\n"
            f"Win Criteria:\n{tags_content['win_criteria']}\n\n"
            f"Tie Criteria:\n{tags_content['tie_criteria']}"
        )
        logging.debug(f"Assembled scenario text (Agent {proposer_agent.agent_id}):\n{assembled_scenario_text}")

        role_assignment = {proposer_agent.agent_id: proposer_role_text}
        scenario_info_dict = {
            'scenario_text': assembled_scenario_text,
            'role_assignment': role_assignment,
            'raw_output': raw_content # Store the direct XML-style output from the LLM
        }
        logging.info(f"Scenario generated and parsed successfully for agent {proposer_agent.agent_id}.")
        return scenario_info_dict, final_prompt_text

    except TimeoutError: # This catches the TimeoutError raised by _execute_api_call after retries
         logging.error(f"Timeout generating scenario for agent {proposer_agent.agent_id} after all retries.")
         return None, final_prompt_text
    except goodfire_exceptions.GoodfireError as e: # Catch Goodfire specific errors that weren't retried or exhausted retries
        logging.error(f"Goodfire API error generating scenario for agent {proposer_agent.agent_id}: {e}", exc_info=True)
        return None, final_prompt_text
    except Exception as e: # Catch any other unexpected error during the process
        logging.critical(f"Unexpected critical error generating scenario for agent {proposer_agent.agent_id}: {e}", exc_info=True)
        # Log the raw content if available, it might give clues
        if raw_content is not None:
            logging.error(f"Raw content at time of critical error (Agent {proposer_agent.agent_id}): {raw_content[:1000]}...")
        return None, final_prompt_text


def generate_agent_response(agent, scenario_data: dict, transcript: list, current_role: str, config: dict) -> tuple[str | None, str | None]:
    """
    Generates an agent's response in a game turn.
    Returns:
        A tuple: (str: agent's response text or None on failure,
                  str: The prompt_text used for generation or None on early failure)
    """
    from manager import Agent # Local import

    if not all(isinstance(arg, exp_type) for arg, exp_type in [
        (agent, Agent), (scenario_data, dict), (transcript, list),
        (current_role, str), (config, dict)
    ]):
        logging.error(f"generate_agent_response: Invalid argument types provided. Agent: {type(agent)}, Scenario: {type(scenario_data)}, etc.")
        return None, None

    client = get_goodfire_client(config)
    gen_config = config.get('generation', {}).get('response', {})
    model_id = agent.model_id
    scenario_text = scenario_data.get('scenario_text', '[Scenario Text Missing from input scenario_data]')

    prompt_template = gen_config.get('prompt_template',
        "Scenario:\n{scenario}\n\nYour Role: {role}\n\nConversation History:\n{history}\n\n{role} (Respond according to your role and objective):")
    max_tokens = gen_config.get('max_tokens', 150)
    temperature = gen_config.get('temperature', 0.6)
    api_retry_config = config.get('api_retries', {})

    history_lines = []
    for msg in transcript: # Iterate safely
        role = msg.get('role', 'UnknownRole')
        content = msg.get('content', '[empty_content]')
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "[No conversation history yet]"

    prompt_text_for_llm = None
    try:
        prompt_text_for_llm = prompt_template.format(scenario=scenario_text, role=current_role, history=history_text)
    except KeyError as ke:
        logging.error(f"Missing key '{ke}' in prompt_template for agent response. Template: '{prompt_template}' Data: scenario_text_present={bool(scenario_text)}, role_present={bool(current_role)}, history_present={bool(history_text)}")
        return None, prompt_template # Return template itself if formatting fails

    variant = goodfire.Variant(model_id)
    variant_edits = agent.genome
    if isinstance(variant_edits, dict) and variant_edits:
        try:
            variant.set(variant_edits)
        except Exception as e:
            logging.error(f"Error setting variant edits for agent {agent.agent_id} during response generation: {e}", exc_info=True)

    messages = [{"role": "user", "content": prompt_text_for_llm}]

    try:
        logging.debug(f"Attempting agent response for {agent.agent_id} ({current_role}). Prompt: {prompt_text_for_llm[:500]}...")
        response = _execute_api_call(
            client.chat.completions.create,
            messages=messages,
            model=variant,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            retry_config=api_retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
             logging.error(f"Invalid or empty response structure from API for agent {agent.agent_id} response.")
             return None, prompt_text_for_llm

        response_text = response.choices[0].message.get('content', '').strip()
        if not response_text: # Allow empty responses if the model generates them, but log it.
             logging.info(f"Agent {agent.agent_id} generated an empty string response (raw output was empty or whitespace).")
             # return None, prompt_text_for_llm # DECISION: Is empty string a valid response or a failure? Let's treat it as valid for now.

        logging.debug(f"Response generated by agent {agent.agent_id}: '{response_text[:100]}...'")
        return response_text, prompt_text_for_llm

    except TimeoutError:
         logging.error(f"Timeout generating response for agent {agent.agent_id} after all retries.")
         return None, prompt_text_for_llm
    except goodfire_exceptions.GoodfireError as e:
        logging.error(f"Goodfire API error generating response for agent {agent.agent_id}: {e}", exc_info=True)
        return None, prompt_text_for_llm
    except Exception as e:
        logging.critical(f"Unexpected critical error generating response for agent {agent.agent_id}: {e}", exc_info=True)
        return None, prompt_text_for_llm


def adjudicate_interaction(scenario_data: dict, transcript: list, config: dict) -> tuple[str, str | None]:
    """
    Adjudicates the game interaction.
    Returns:
        A tuple: (str: adjudication outcome e.g., 'Role A Wins', 'Tie', 'error',
                  str: The prompt_text used for adjudication or None on early failure)
    """
    if not all(isinstance(arg, exp_type) for arg, exp_type in [
        (scenario_data, dict), (transcript, list), (config, dict)
    ]):
        logging.error(f"adjudicate_interaction: Invalid argument types. Scenario: {type(scenario_data)}, etc.")
        return "error", None

    client = get_goodfire_client(config)
    adjudicator_config = config.get('adjudicator', {})
    model_id = adjudicator_config.get('model_id')
    prompt_template = adjudicator_config.get('prompt_template',
        "Analyze the interaction based strictly on the rules and objectives defined in the scenario description.\n\nScenario Description:\n{scenario}\n\nInteraction Transcript:\n{transcript}\n\nTask: Based ONLY on the scenario's win criteria and the interaction, determine the outcome. Respond with ONLY ONE of the following exact phrases: 'Role A Wins', 'Role B Wins', or 'Tie'.")
    max_tokens = adjudicator_config.get('max_tokens', 20)
    temperature = adjudicator_config.get('temperature', 0.0) # Low temp for deterministic adjudication
    api_retry_config = config.get('api_retries', {})

    if not model_id:
        logging.error("Adjudicator model ID not specified in configuration. Cannot adjudicate.")
        return "error", None

    scenario_text = scenario_data.get('scenario_text', '[Scenario text missing from input scenario_data]')
    transcript_text = "\n".join([f"{msg.get('role', 'UnknownRole')}: {msg.get('content', '[empty_content]')}" for msg in transcript]) \
                      if transcript else "[No transcript available for adjudication]"

    adjudication_prompt_text = None
    try:
        adjudication_prompt_text = prompt_template.format(scenario=scenario_text, transcript=transcript_text)
    except KeyError as ke:
        logging.error(f"Missing key '{ke}' in prompt_template for adjudication. Template: '{prompt_template}'")
        return "error", prompt_template

    messages = [{"role": "user", "content": adjudication_prompt_text}]
    raw_adjudicator_llm_output = "[Adjudicator LLM output not captured]"

    try:
        logging.debug(f"Attempting adjudication. Prompt: {adjudication_prompt_text[:500]}...")
        response = _execute_api_call(
            client.chat.completions.create,
            messages=messages,
            model=model_id, # Adjudicator uses a base model_id, not a variant
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            retry_config=api_retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
             logging.error("Invalid or empty response structure received from adjudicator API.")
             return "error", adjudication_prompt_text

        raw_adjudicator_llm_output = response.choices[0].message.get('content', '')
        result_text = raw_adjudicator_llm_output.strip().replace('.', '') # Remove periods for exact match

        valid_outcomes = ['Role A Wins', 'Role B Wins', 'Tie']
        if result_text in valid_outcomes:
            logging.info(f"Adjudication successful: {result_text}")
            return result_text, adjudication_prompt_text
        else:
            logging.warning(f"Adjudicator returned non-standard text: '{result_text}' (Raw: '{raw_adjudicator_llm_output}'). Attempting relaxed matching.")
            # Relaxed matching
            result_text_lower = result_text.lower()
            if "role a wins" in result_text_lower:
                logging.info("Adjudication relaxed match: 'Role A Wins'")
                return 'Role A Wins', adjudication_prompt_text
            if "role b wins" in result_text_lower:
                logging.info("Adjudication relaxed match: 'Role B Wins'")
                return 'Role B Wins', adjudication_prompt_text
            if "tie" in result_text_lower or "draw" in result_text_lower : # Common synonyms
                logging.info("Adjudication relaxed match: 'Tie'")
                return 'Tie', adjudication_prompt_text
            
            logging.error(f"Adjudicator response cleanup failed. Original: '{result_text}', Raw: '{raw_adjudicator_llm_output}'. Defaulting to error.")
            return "error", adjudication_prompt_text # Return the actual non-standard text for logging if needed

    except TimeoutError:
         logging.error("Timeout during adjudication after all retries.")
         return "error", adjudication_prompt_text
    except goodfire_exceptions.GoodfireError as e:
        logging.error(f"Goodfire API error during adjudication: {e}", exc_info=True)
        return "error", adjudication_prompt_text
    except Exception as e:
        logging.critical(f"Unexpected critical error during adjudication: {e}", exc_info=True)
        return "error", adjudication_prompt_text


def perform_contrastive_analysis(dataset_1: list, dataset_2: list, agent_variant_or_model_id, top_k: int, config: dict) -> tuple[list | None, list | None]:
    """
    Performs contrastive analysis between two datasets.
    Returns a tuple of (features_for_dataset1, features_for_dataset2), or (None, None) on failure.
    """
    from manager import Agent # Local import

    if not all(isinstance(arg, exp_type) for arg, exp_type in [
        (dataset_1, list), (dataset_2, list), (top_k, int), (config, dict)
    ]):
        logging.error(f"perform_contrastive_analysis: Invalid argument types. dataset_1: {type(dataset_1)}, etc.")
        return None, None
    if not isinstance(agent_variant_or_model_id, (Agent, str)):
        logging.error("perform_contrastive_analysis: agent_variant_or_model_id must be Agent instance or model_id string.")
        return None, None

    client = get_goodfire_client(config)
    api_retry_config = config.get('api_retries', {})

    if not dataset_1 or not dataset_2:
        logging.info("perform_contrastive_analysis: One or both datasets are empty. Skipping analysis.")
        return None, None # Return tuple of Nones

    model_arg = None
    agent_id_for_log = "N/A"
    if isinstance(agent_variant_or_model_id, Agent):
        agent_id_for_log = agent_variant_or_model_id.agent_id
        variant = goodfire.Variant(agent_variant_or_model_id.model_id)
        if isinstance(agent_variant_or_model_id.genome, dict) and agent_variant_or_model_id.genome:
            try:
                variant.set(agent_variant_or_model_id.genome)
                model_arg = variant
            except Exception as e:
                logging.error(f"Error setting variant edits for contrast analysis on agent {agent_id_for_log}: {e}. Using base model ID.")
                model_arg = agent_variant_or_model_id.model_id # Fallback to base model ID string
        else:
             model_arg = variant # Use variant even if genome is empty (it represents the base model)
    elif isinstance(agent_variant_or_model_id, str):
         model_arg = agent_variant_or_model_id
         agent_id_for_log = f"model_id: {agent_variant_or_model_id}"


    if model_arg is None: # Should not happen if logic above is correct
        logging.error("perform_contrastive_analysis: model_arg was not set. This is a bug.")
        return None, None

    try:
        logging.debug(f"Attempting contrastive analysis for {agent_id_for_log}. Datasets sizes: {len(dataset_1)} vs {len(dataset_2)}.")
        # The Goodfire docs for contrast() return: tuple[FeatureGroup, FeatureGroup]
        # dataset_1 is for behavior to avoid, dataset_2 for behavior to encourage.
        # contrast returns (features_explaining_dataset_1, features_explaining_dataset_2)
        features_explaining_d1, features_explaining_d2 = _execute_api_call(
            client.features.contrast,
            dataset_1=dataset_1,
            dataset_2=dataset_2,
            model=model_arg,
            top_k=top_k,
            retry_config=api_retry_config
        )

        num_d1 = len(features_explaining_d1) if features_explaining_d1 else 0
        num_d2 = len(features_explaining_d2) if features_explaining_d2 else 0

        logging.info(
            f"Contrastive analysis for {agent_id_for_log} completed. "
            f"Got {num_d1} features for dataset_1 (to avoid/less like) and "
            f"{num_d2} features for dataset_2 (to encourage/more like)."
        )
        return features_explaining_d1, features_explaining_d2

    except AttributeError as ae:
        logging.critical(f"Goodfire client object is missing 'features.contrast' method. Error: {ae}", exc_info=True)
        raise # This is a critical SDK or setup issue
    except TimeoutError:
        logging.error(f"Timeout during contrastive analysis for {agent_id_for_log} after all retries.")
        return None, None
    except goodfire_exceptions.GoodfireError as e:
        logging.error(f"Goodfire API error during contrastive analysis for {agent_id_for_log}: {e}", exc_info=True)
        return None, None
    except Exception as e:
        logging.critical(f"Unexpected critical error during contrastive analysis for {agent_id_for_log}: {e}", exc_info=True)
        return None, None
