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
import logging
import os
import time
import random
import hashlib
import json
import copy
import re # Added for XML-style tag parsing

_client_instance = None
_last_client_config_hash = None

def _get_config_hash(config):
    relevant_config = {
        'api_key_env_var': config.get('goodfire', {}).get('api_key_env_var'),
        'base_url': config.get('goodfire', {}).get('base_url')
    }
    return hashlib.sha256(json.dumps(relevant_config, sort_keys=True).encode()).hexdigest()

def get_goodfire_client(config):
    global _client_instance
    global _last_client_config_hash

    if not isinstance(config, dict):
        logging.critical("get_goodfire_client: Invalid config type provided.")
        raise TypeError("Configuration must be a dictionary.")

    current_config_hash = _get_config_hash(config)

    if _client_instance is not None and current_config_hash == _last_client_config_hash:
        return _client_instance

    goodfire_config = config.get('goodfire', {})
    api_key_env_var = goodfire_config.get('api_key_env_var', 'GOODFIRE_API_KEY')
    api_key = os.environ.get(api_key_env_var)
    base_url = goodfire_config.get('base_url')

    if not api_key:
        logging.critical(f"Goodfire API key not found in environment variable: {api_key_env_var}")
        raise ValueError(f"Environment variable {api_key_env_var} must be set.")

    try:
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url

        _client_instance = goodfire.Client(**client_args)
        logging.info(f"Goodfire client initialized (Base URL: {base_url if base_url else 'Default'}).")
        _last_client_config_hash = current_config_hash
        return _client_instance
    except Exception as e:
        logging.critical(f"Failed to initialize Goodfire client: {e}", exc_info=True)
        _client_instance = None
        _last_client_config_hash = None
        raise RuntimeError("Could not initialize Goodfire client.") from e

def _execute_api_call(api_call_func, *args, **kwargs):
    retry_config = kwargs.pop('retry_config', {})
    max_retries = retry_config.get('max_retries', 3)
    initial_delay = retry_config.get('initial_delay', 1.0)
    backoff_factor = retry_config.get('backoff_factor', 2.0)

    if not callable(api_call_func):
         raise TypeError("api_call_func must be callable")

    current_delay = initial_delay
    last_exception = None
    for attempt in range(max_retries):
        try:
            result = api_call_func(*args, **kwargs)
            return result
        except goodfire.APIError as e:
             status_code = getattr(e, 'status_code', None)
             if status_code in [429, 500, 502, 503, 504]: 
                  logging.warning(f"Retriable API Error (Status: {status_code}) on attempt {attempt + 1}. Retrying in {current_delay:.2f} seconds. Error: {e}")
                  last_exception = e
             else: 
                  logging.error(f"Non-retriable API Error (Status: {status_code}): {e}.")
                  raise e
        except Exception as e:
             logging.error(f"Unexpected error during API call attempt {attempt + 1}: {e}", exc_info=True)
             last_exception = e
             if attempt == max_retries - 1:
                 raise e

        time.sleep(current_delay)
        current_delay *= backoff_factor
        current_delay += random.uniform(0, initial_delay * 0.1)

    logging.error(f"API call failed after {max_retries} retries.")
    raise TimeoutError(f"API call failed after {max_retries} retries.") from last_exception


def generate_scenario(proposer_agent, config: dict) -> tuple[dict | None, str | None]:
    """
    Generates a game scenario using the proposer agent and configuration.
    Returns:
        A tuple: 
        (dict: {'scenario_text': str, 'role_assignment': dict, 'raw_output': str} or None on failure,
         str: The prompt_text used for generation or None on early failure)
    """
    from manager import Agent # Local import to avoid circular dependency issues at module load time
    if not isinstance(proposer_agent, Agent):
        logging.error("generate_scenario: Invalid proposer_agent provided.")
        return None, None
    if not isinstance(config, dict):
        logging.error("generate_scenario: Invalid config provided.")
        return None, None

    client = get_goodfire_client(config)
    gen_config = config.get('generation', {}).get('scenario', {})
    model_id = proposer_agent.model_id
    prompt_text = gen_config.get('prompt', "Error: Scenario generation prompt not found in config.")
    max_tokens = gen_config.get('max_tokens', 1000) 
    temperature = gen_config.get('temperature', 0.7)
    retry_config = config.get('api_retries', {})

    variant = goodfire.Variant(model_id)
    variant_edits = proposer_agent.genome
    if isinstance(variant_edits, dict) and variant_edits:
        try:
            variant.set(variant_edits)
        except Exception as e:
            logging.error(f"Error setting variant edits for agent {proposer_agent.agent_id}: {e}", exc_info=True)

    messages = [{"role": "user", "content": prompt_text}]

    try:
        response = _execute_api_call(
            client.chat.completions.create,
            messages=messages,
            model=variant,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            retry_config=retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
             logging.error(f"Invalid response structure received from API for agent {proposer_agent.agent_id}.")
             return None, prompt_text

        raw_content = response.choices[0].message.get('content', '').strip()
        if not raw_content:
             logging.warning(f"Empty content received from API for agent {proposer_agent.agent_id}.")
             return None, prompt_text

        tags_content = {}
        tag_names = ["context", "roles", "objectives", "win_criteria", "tie_criteria", "proposer_role"]
        
        for tag_name in tag_names:
            match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", raw_content, re.DOTALL | re.IGNORECASE)
            if match:
                tags_content[tag_name] = match.group(1).strip()
            else:
                logging.warning(f"Tag <{tag_name}> not found in LLM output for agent {proposer_agent.agent_id}. Raw content: {raw_content[:500]}")
                return None, prompt_text
        
        if "objective" not in tags_content["objectives"].lower():
            logging.warning(f"Keyword 'objective' not found in <objectives> content for agent {proposer_agent.agent_id}. Content: {tags_content['objectives']}")
            return None, prompt_text
        if "win criteria" not in tags_content["win_criteria"].lower():
            logging.warning(f"Keyword 'win criteria' not found in <win_criteria> content for agent {proposer_agent.agent_id}. Content: {tags_content['win_criteria']}")
            return None, prompt_text
        if "tie criteria" not in tags_content["tie_criteria"].lower():
            logging.warning(f"Keyword 'tie criteria' not found in <tie_criteria> content for agent {proposer_agent.agent_id}. Content: {tags_content['tie_criteria']}")
            return None, prompt_text
        
        proposer_role_text = tags_content["proposer_role"]
        if proposer_role_text not in ["Role A", "Role B"]:
            logging.warning(f"Invalid content for <proposer_role>: '{proposer_role_text}'. Expected 'Role A' or 'Role B'. Agent: {proposer_agent.agent_id}")
            return None, prompt_text

        assembled_scenario_text = (
            f"Context:\n{tags_content['context']}\n\n"
            f"Roles:\n{tags_content['roles']}\n\n"
            f"Objective(s):\n{tags_content['objectives']}\n\n"
            f"Win Criteria:\n{tags_content['win_criteria']}\n\n"
            f"Tie Criteria:\n{tags_content['tie_criteria']}"
        )
        role_assignment = {proposer_agent.agent_id: proposer_role_text}
        scenario_info_dict = {
            'scenario_text': assembled_scenario_text, 
            'role_assignment': role_assignment,
            'raw_output': raw_content # The direct XML-style output from the LLM
        }
        logging.info(f"Scenario generated and parsed successfully using XML-style tags for agent {proposer_agent.agent_id}")
        return scenario_info_dict, prompt_text

    except TimeoutError:
         logging.error(f"Timeout generating scenario for agent {proposer_agent.agent_id} after retries.")
         return None, prompt_text # Return prompt_text even on timeout if it was defined
    except Exception as e:
        logging.error(f"Error generating scenario for agent {proposer_agent.agent_id}: {e}", exc_info=True)
        return None, prompt_text # Return prompt_text on other errors if defined

def generate_agent_response(agent, scenario_data: dict, transcript: list, current_role: str, config: dict) -> tuple[str | None, str | None]:
    """
    Generates an agent's response in a game turn.
    Returns:
        A tuple: (str: agent's response text or None on failure,
                  str: The prompt_text used for generation or None on early failure)
    """
    from manager import Agent # Local import
    if not isinstance(agent, Agent) or not isinstance(scenario_data, dict) or \
       not isinstance(transcript, list) or not isinstance(current_role, str) or \
       not isinstance(config, dict):
        logging.error("generate_agent_response: Invalid arguments provided.")
        return None, None

    client = get_goodfire_client(config)
    gen_config = config.get('generation', {}).get('response', {})
    model_id = agent.model_id
    scenario_text = scenario_data.get('scenario_text', '[Scenario Missing]')
    prompt_template = gen_config.get('prompt_template',
        "Scenario:\n{scenario}\n\nYour Role: {role}\n\nConversation History:\n{history}\n\n{role} (Respond according to your role and objective):")
    max_tokens = gen_config.get('max_tokens', 150)
    temperature = gen_config.get('temperature', 0.6)
    retry_config = config.get('api_retries', {})

    history_lines = []
    for msg in transcript:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    # This is the full prompt sent to the LLM for this turn
    prompt_text_for_llm = prompt_template.format(scenario=scenario_text, role=current_role, history=history_text)

    variant = goodfire.Variant(model_id)
    variant_edits = agent.genome
    if isinstance(variant_edits, dict) and variant_edits:
        try:
            variant.set(variant_edits)
        except Exception as e:
            logging.error(f"Error setting variant edits for agent {agent.agent_id} during response generation: {e}", exc_info=True)

    messages = [{"role": "user", "content": prompt_text_for_llm}]

    try:
        response = _execute_api_call(
            client.chat.completions.create,
            messages=messages,
            model=variant,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            retry_config=retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
             logging.error(f"Invalid response structure received from API for agent {agent.agent_id}.")
             return None, prompt_text_for_llm

        response_text = response.choices[0].message.get('content', '').strip()
        if not response_text:
             logging.warning(f"Agent {agent.agent_id} generated an empty response.")
             return None, prompt_text_for_llm 

        logging.debug(f"Response generated successfully by agent {agent.agent_id}")
        return response_text, prompt_text_for_llm

    except TimeoutError:
         logging.error(f"Timeout generating response for agent {agent.agent_id} after retries.")
         return None, prompt_text_for_llm
    except Exception as e:
        logging.error(f"Error generating response for agent {agent.agent_id}: {e}", exc_info=True)
        return None, prompt_text_for_llm

def adjudicate_interaction(scenario_data: dict, transcript: list, config: dict) -> tuple[str, str | None]:
    """
    Adjudicates the game interaction.
    Returns:
        A tuple: (str: adjudication outcome e.g., 'Role A Wins', 'Tie', 'error',
                  str: The prompt_text used for adjudication or None on early failure)
    """
    if not isinstance(scenario_data, dict) or not isinstance(transcript, list) or not isinstance(config, dict):
        logging.error("adjudicate_interaction: Invalid arguments provided.")
        return "error", None 

    client = get_goodfire_client(config)
    adjudicator_config = config.get('adjudicator', {})
    model_id = adjudicator_config.get('model_id')
    prompt_template = adjudicator_config.get('prompt_template',
        "Analyze the interaction based strictly on the rules and objectives defined in the scenario description.\n\nScenario Description:\n{scenario}\n\nInteraction Transcript:\n{transcript}\n\nTask: Based ONLY on the scenario's win criteria and the interaction, determine the outcome. Respond with ONLY ONE of the following exact phrases: 'Role A Wins', 'Role B Wins', or 'Tie'.")
    max_tokens = adjudicator_config.get('max_tokens', 20)
    temperature = adjudicator_config.get('temperature', 0.0)
    retry_config = config.get('api_retries', {})

    if not model_id:
        logging.error("Adjudicator model ID not specified in configuration.")
        return "error", None

    scenario_text = scenario_data.get('scenario_text', '[Scenario text missing]')
    transcript_text = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in transcript])

    # This is the full prompt sent to the adjudicator LLM
    adjudication_prompt_text = prompt_template.format(scenario=scenario_text, transcript=transcript_text)
    messages = [{"role": "user", "content": adjudication_prompt_text}]
    
    # We will also include the raw LLM output from the adjudicator in the main return, if available
    raw_adjudicator_llm_output = None

    try:
        response = _execute_api_call(
            client.chat.completions.create,
            messages=messages,
            model=model_id,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            retry_config=retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
             logging.error(f"Invalid response structure received from adjudicator API.")
             return "error", adjudication_prompt_text

        raw_adjudicator_llm_output = response.choices[0].message.get('content', '') # Store raw output
        result_text = raw_adjudicator_llm_output.strip().replace('.', '') 

        valid_outcomes = ['Role A Wins', 'Role B Wins', 'Tie']
        if result_text in valid_outcomes:
            logging.info(f"Adjudication successful: {result_text}")
            return result_text, adjudication_prompt_text # raw_adjudicator_llm_output is implicitly part of result_text here
        else:
            logging.warning(f"Adjudicator returned non-standard text: '{result_text}'. Attempting cleanup.")
            result_text_lower = result_text.lower()
            if "role a wins" in result_text_lower:
                logging.info("Adjudication cleanup: Matched 'Role A Wins'")
                return 'Role A Wins', adjudication_prompt_text
            if "role b wins" in result_text_lower:
                logging.info("Adjudication cleanup: Matched 'Role B Wins'")
                return 'Role B Wins', adjudication_prompt_text
            if "tie" in result_text_lower:
                logging.info("Adjudication cleanup: Matched 'Tie'")
                return 'Tie', adjudication_prompt_text
            
            logging.error(f"Adjudicator response cleanup failed for text: '{result_text}'. Returning error.")
            return "error", adjudication_prompt_text

    except TimeoutError:
         logging.error(f"Timeout during adjudication after retries.")
         return "error", adjudication_prompt_text
    except Exception as e:
        logging.error(f"Error during adjudication: {e}", exc_info=True)
        return "error", adjudication_prompt_text

def perform_contrastive_analysis(dataset_1: list, dataset_2: list, agent_variant_or_model_id, top_k: int, config: dict):
    from manager import Agent 
    if not isinstance(dataset_1, list) or not isinstance(dataset_2, list) or \
       not isinstance(top_k, int) or not isinstance(config, dict):
        logging.error("perform_contrastive_analysis: Invalid arguments provided.")
        return None, None

    client = get_goodfire_client(config)
    retry_config = config.get('api_retries', {})

    if not dataset_1 or not dataset_2:
        logging.info("perform_contrastive_analysis: One or both datasets are empty. Skipping analysis.")
        return None, None

    model_arg = None
    if isinstance(agent_variant_or_model_id, Agent):
        variant = goodfire.Variant(agent_variant_or_model_id.model_id)
        if isinstance(agent_variant_or_model_id.genome, dict) and agent_variant_or_model_id.genome:
            try:
                variant.set(agent_variant_or_model_id.genome)
                model_arg = variant
            except Exception as e:
                logging.error(f"Error setting variant edits for contrast analysis on agent {agent_variant_or_model_id.agent_id}: {e}. Using base model ID.")
                model_arg = agent_variant_or_model_id.model_id 
        else:
             model_arg = agent_variant_or_model_id.model_id
    elif isinstance(agent_variant_or_model_id, str):
         model_arg = agent_variant_or_model_id
    else:
         logging.error("Invalid type for agent_variant_or_model_id in contrastive analysis.")
         return None, None

    try:
        result = _execute_api_call(
            client.features.contrast,
            dataset_1=dataset_1,
            dataset_2=dataset_2,
            model=model_arg,
            top_k=top_k,
            retry_config=retry_config
        )

        if isinstance(result, tuple) and len(result) == 2:
            features_d1, features_d2 = result
            is_list_like = lambda x: hasattr(x, '__iter__') and not isinstance(x, (str, bytes))
            if is_list_like(features_d1) and is_list_like(features_d2):
                 features_d1_list = list(features_d1)
                 features_d2_list = list(features_d2)
                 logging.info(f"Contrastive analysis completed. Got {len(features_d1_list)} features for dataset_1 (to avoid) and {len(features_d2_list)} features for dataset_2 (to encourage).")
                 return features_d1_list, features_d2_list
            else:
                 logging.error(f"Contrastive analysis returned tuple, but contents are not list-like: Type1={type(features_d1)}, Type2={type(features_d2)}")
                 return None, None
        else:
            logging.error(f"Contrastive analysis returned unexpected result type: {type(result)}. Expected tuple of two lists.")
            return None, None

    except AttributeError as ae: 
        logging.critical(f"Goodfire client object is missing the 'features.contrast' method. Error: {ae}", exc_info=True)
        raise 
    except TimeoutError:
        logging.error(f"Timeout during contrastive analysis after retries.")
        return None, None
    except Exception as e:
        logging.error(f"Error during contrastive analysis: {e}", exc_info=True)
        return None, None
