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
from goodfire import exceptions as goodfire_exceptions, AsyncClient
import logging
import requests
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
_async_client_instance = None # For the asynchronous client
_last_async_client_config_hash = None # For the asynchronous client's config hash

def _get_config_hash(config: dict) -> str:
    """Helper function to create a hash of relevant client configuration parts."""
    relevant_config = {
        'api_key_env_var': config.get('goodfire', {}).get('api_key_env_var'),
        'base_url': config.get('goodfire', {}).get('base_url')
    }
    return hashlib.sha256(json.dumps(relevant_config, sort_keys=True).encode()).hexdigest()

def get_goodfire_async_client(config: dict) -> goodfire.AsyncClient:
    """
    Initializes and returns an asynchronous Goodfire client instance.
    Manages a global async client instance, re-initializing if config changes.
    """
    global _async_client_instance
    global _last_async_client_config_hash

    if not isinstance(config, dict):
        logging.critical("get_goodfire_async_client: Invalid config type provided (must be a dictionary).")
        raise TypeError("Configuration must be a dictionary for async client.")

    current_config_hash = _get_config_hash(config)

    if _async_client_instance is not None and current_config_hash == _last_async_client_config_hash:
        return _async_client_instance

    goodfire_config_dict = config.get('goodfire', {})
    api_key_env_var = goodfire_config_dict.get('api_key_env_var', 'GOODFIRE_API_KEY')
    api_key = os.environ.get(api_key_env_var)
    base_url = goodfire_config_dict.get('base_url') # Can be None

    if not api_key:
        logging.critical(f"Goodfire API key not found in environment variable: {api_key_env_var} for async client.")
        raise ValueError(f"Environment variable {api_key_env_var} must be set for Goodfire API (async client).")

    try:
        client_args = {"api_key": api_key}
        if base_url: # Only add base_url if it's provided and not None/empty
            client_args["base_url"] = base_url

        _async_client_instance = goodfire.AsyncClient(**client_args)
        logging.info(f"Goodfire AsyncClient initialized (Base URL: {base_url if base_url else 'Default Goodfire API URL'}).")
        _last_async_client_config_hash = current_config_hash
        return _async_client_instance
    except Exception as e:
        logging.critical(f"Failed to initialize Goodfire AsyncClient: {e}", exc_info=True)
        _async_client_instance = None # Reset on failure
        _last_async_client_config_hash = None
        raise RuntimeError("Could not initialize Goodfire AsyncClient.") from e

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
        except goodfire_exceptions.GoodfireBaseException as e: # Catch other Goodfire-specific base errors
            # Decide if generic GoodfireBaseExceptions are retriable. For now, let's assume they might be.
            logging.warning(
                f"Goodfire API Base Error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries} "
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
    Employs lenient parsing for XML-like tags, focusing on extracting content
    between expected tag pairs in sequence, while tolerating junk outside
    and between these pairs.
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
    
    prompt_text_template = gen_config.get('prompt')
    if prompt_text_template is None:
        err_msg = "Configuration error: 'generation.scenario.prompt' is missing from config.yaml."
        logging.critical(err_msg + f" (Agent: {proposer_agent.agent_id})")
        # Return None for scenario_info and None for prompt_text, signaling failure.
        return None, None

    max_tokens = gen_config.get('max_tokens', 1000)
    temperature = gen_config.get('temperature', 0.7)
    api_retry_config = config.get('api_retries', {})

    diversification_seed = f"agent_id:{proposer_agent.agent_id}_call_id:{uuid.uuid4().hex[:8]}"
    # final_prompt_text will be used for the actual API call and returned if an API error occurs.
    # It is constructed here using the validated prompt_text_template.
    final_prompt_text = f"{prompt_text_template}\n[{diversification_seed}]"

    variant = goodfire.Variant(model_id)
    raw_genome_from_proposer = proposer_agent.genome
    genome_for_goodfire_api = {}
    # Transform the agent's genome (dict of dicts) to the flat dict expected by goodfire.Variant().set()
    if isinstance(raw_genome_from_proposer, dict):
        for feature_uuid, feature_data_dict in raw_genome_from_proposer.items():
            if isinstance(feature_data_dict, dict) and 'activation' in feature_data_dict:
                # genome format: {'activation': float, 'label': str}
                genome_for_goodfire_api[feature_uuid] = feature_data_dict['activation']
            elif isinstance(feature_data_dict, (int, float)):
                # This case might occur if loading very old data or if there's mixed format.
                logging.debug(f"Agent {proposer_agent.agent_id} genome for feature {feature_uuid} in generate_scenario appears to be in old format (direct activation value). Using it directly.")
                genome_for_goodfire_api[feature_uuid] = float(feature_data_dict)
            else:
                logging.warning(f"Agent {proposer_agent.agent_id} genome for feature {feature_uuid} in generate_scenario has unexpected structure: {feature_data_dict}. Skipping this feature for Goodfire variant.")

    if genome_for_goodfire_api: # Check if the transformed genome has any features to set
        try:
            variant.set(genome_for_goodfire_api)
        except Exception as e:
            # Log error specific to setting the transformed genome
            logging.error(f"Error setting transformed variant edits for agent {proposer_agent.agent_id} in generate_scenario: {e}", exc_info=True)

    messages = [{"role": "user", "content": final_prompt_text}]
    llm_raw_output_text = None 

    # Define expected tags locally. XML tags are case-sensitive.
    expected_tag_sequence = ["context", "roles", "objectives", "win_criteria", "tie_criteria", "proposer_role"]

    # The 'final_prompt_text' variable, used for the API call and error reporting, 
    # is already defined earlier if prompt_text_template was successfully loaded.
    # 'messages' is also defined earlier using this 'final_prompt_text'.

    try:
        logging.debug(f"Attempting scenario generation for agent {proposer_agent.agent_id} with prompt ending: ...{final_prompt_text[-100:]}")
        response = _execute_api_call(
            client.chat.completions.create,
            messages=messages, # messages was formed using the validated final_prompt_text
            model=variant,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            retry_config=api_retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
             logging.error(f"Invalid or empty response structure received from API for scenario generation (Agent: {proposer_agent.agent_id}).")
             # Return the prompt that was attempted, even if the response was bad.
             return None, final_prompt_text

        llm_raw_output_text = response.choices[0].message.get('content', '')
        logging.debug(f"Raw LLM output for scenario (Agent {proposer_agent.agent_id}):\n{llm_raw_output_text}")

        if not llm_raw_output_text.strip():
             logging.warning(f"Empty content received from API for scenario generation (Agent: {proposer_agent.agent_id}).")
             # Return the prompt that was attempted.
             return None, final_prompt_text

        tags_content = {}
        # Use the raw output directly for extremely lenient parsing, ignore complex pre-cleaning/slicing.
        search_text = llm_raw_output_text 
        current_search_position = 0 # Tracks our position in search_text
        parsing_successful = True

        for i, tag_name_to_find in enumerate(expected_tag_sequence):
            # 1. Find the opening marker for the current tag.
            #    This is the tag name itself, treated as a whole word (case-insensitive).
            #    We allow optional non-alphanumeric characters (like '<', '>', '.') before/after it.
            #    The regex finds `junk + tag_name + junk_until_content_starts`.
            #    Pattern: (junk symbols)? TAG_NAME (junk symbols like >)? CAPTURE_GROUP_FOR_JUNK_AND_TAG
            #    We want to find the end of this entire opening marker pattern.
            open_marker_regex_str = r"\b" + re.escape(tag_name_to_find) + r"\b" # \b for whole word
            # Search for this tag name, possibly preceded by < or . and followed by >
            # More lenient: look for the tag name, then account for a possible ">" or other chars before content.
            # Example: "<context>", "context>", ".context>" should all find "context" and advance past ">"
            
            # Find the tag name itself.
            # Compile the regex for searching with a starting position.
            compiled_open_marker_regex = re.compile(open_marker_regex_str, re.IGNORECASE)
            open_tag_name_match = compiled_open_marker_regex.search(search_text, pos=current_search_position)

            if not open_tag_name_match:
                logging.warning(f"Scenario parsing: Start of tag '{tag_name_to_find}' not found. Agent: {proposer_agent.agent_id}. Search text from pos {current_search_position}: '{search_text[current_search_position:current_search_position+200]}...'")
                parsing_successful = False
                break
            
            # Content starts after the matched tag name and any immediately following non-content characters (like '>')
            # Advance past the tag name itself.
            content_start_index = open_tag_name_match.end()
            optional_gt_match = re.match(r"\s*(?:>)?", search_text[content_start_index:])
            if optional_gt_match:
                content_start_index += optional_gt_match.end()

            # 2. Determine the end of the content for tag_name_to_find.
            content_end_index = len(search_text) # Default to end of string if no other delimiter found
            found_explicit_closer = False

            # Option A: Look for the specific closing tag "/tag_name_to_find"
            # Example: "</context>", "/context>", "/context"
            close_marker_regex_str = r"(?:<)?\s*/\s*\b" + re.escape(tag_name_to_find) + r"\b"
            # Compile the regex for searching with a starting position.
            compiled_close_marker_regex = re.compile(close_marker_regex_str, re.IGNORECASE)
            close_tag_match = compiled_close_marker_regex.search(
                search_text,
                pos=content_start_index # Search after the opening tag's content starts
            )
            if close_tag_match:
                content_end_index = close_tag_match.start() # Content ends before the closing marker starts
                # Advance current_search_position past this closing tag for the next iteration
                # Find the full extent of the closing marker (e.g. including a '>')
                # Compile the regex for searching with a starting position.
                compiled_full_closing_marker_regex = re.compile(close_marker_regex_str + r"\s*(?:>)?", re.IGNORECASE) # Add optional '>'
                full_closing_marker_match = compiled_full_closing_marker_regex.search(
                    search_text,
                    pos=close_tag_match.start() # Start search from where the basic closer was found
                )
                current_search_position = full_closing_marker_match.end() if full_closing_marker_match else close_tag_match.end()
                found_explicit_closer = True
            
            # Option B: If not the last tag, look for the start of the *next* tag in sequence.
            # The content ends before the next tag's opening marker if that comes first.
            if i + 1 < len(expected_tag_sequence):
                next_tag_name = expected_tag_sequence[i+1]
                # This regex looks for an opening angle bracket '<', followed by optional whitespace,
                # then the tag name, and then ensures it's followed by either a closing angle bracket '>' or whitespace.
                # This makes it more specific to finding an actual tag opening rather than just the tag name as a word.
                next_tag_open_marker_regex_str = r"<\s*" + re.escape(next_tag_name) + r"(?:>|\s)"
                
                # Compile the regex for searching with a starting position.
                compiled_next_tag_open_marker_regex = re.compile(next_tag_open_marker_regex_str, re.IGNORECASE)
                next_tag_open_name_match = compiled_next_tag_open_marker_regex.search(
                    search_text,
                    pos=content_start_index # Search after current tag's content starts
                )
                
                if next_tag_open_name_match:
                    # If next tag starts before the current tag's explicit closer (or if closer wasn't found)
                    if not found_explicit_closer or next_tag_open_name_match.start() < content_end_index:
                        content_end_index = next_tag_open_name_match.start()
                        # For the next iteration, start searching from where this next tag was found
                        current_search_position = next_tag_open_name_match.start() 
            
            # Extract content
            extracted_content = search_text[content_start_index:content_end_index].strip()
            tags_content[tag_name_to_find] = extracted_content
            logging.debug(f"Scenario parsing: Tag '{tag_name_to_find}' -> '{extracted_content[:100]}...'. Next search pos: {current_search_position}")

            if not found_explicit_closer and (i + 1 >= len(expected_tag_sequence)): 
                # If it was the last tag and no explicit closer, we took till end of string.
                # No more text to parse.
                break 
            
            if current_search_position >= len(search_text) and i + 1 < len(expected_tag_sequence):
                # Ran out of text before finding all tags
                logging.warning(f"Scenario parsing: Ran out of text after parsing '{tag_name_to_find}', but more tags expected. Agent: {proposer_agent.agent_id}.")
                parsing_successful = False # Mark as failure if not all tags found
                break


        # Final check: all expected tags were found
        if len(tags_content) != len(expected_tag_sequence):
            logging.warning(f"Scenario parsing: Failed to extract all {len(expected_tag_sequence)} expected tags. Found {len(tags_content)}: {list(tags_content.keys())}. Agent: {proposer_agent.agent_id}. Raw text: '{llm_raw_output_text[:500]}...'")
            parsing_successful = False

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
            'raw_output': llm_raw_output_text 
        }
        logging.info(f"Scenario generated and parsed successfully for agent {proposer_agent.agent_id}.")
        return scenario_info_dict, final_prompt_text

    except TimeoutError:
         logging.error(f"Timeout generating scenario for agent {proposer_agent.agent_id} after all retries.")
         return None, final_prompt_text
    except goodfire_exceptions.GoodfireBaseException as e: # Catching the base Goodfire exception
        logging.error(f"Goodfire API error generating scenario for agent {proposer_agent.agent_id}: {e}", exc_info=True)
        return None, final_prompt_text
    except Exception as e:
        logging.critical(f"Unexpected critical error generating scenario for agent {proposer_agent.agent_id}: {e}", exc_info=True)
        if llm_raw_output_text is not None:
            logging.error(f"Raw content at time of critical error (Agent {proposer_agent.agent_id}): {llm_raw_output_text[:1000]}...")
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

    prompt_template_str = gen_config.get('prompt_template')
    if prompt_template_str is None:
        err_msg = "Configuration error: 'generation.response.prompt_template' is missing from config.yaml."
        logging.critical(err_msg + f" (Agent: {agent.agent_id}, Role: {current_role})")
        # Return None for response_text and None for prompt_text, signaling failure.
        return None, None

    max_tokens = gen_config.get('max_tokens', 150)
    temperature = gen_config.get('temperature', 0.6)
    api_retry_config = config.get('api_retries', {})

    history_lines = []
    for msg in transcript: # Iterate safely
        role = msg.get('role', 'UnknownRole')
        content = msg.get('content', '[empty_content]')
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "[No conversation history yet]"

    prompt_text_for_llm = None # This will hold the fully formatted prompt sent to LLM.
    try:
        # Use the validated prompt_template_str from config.
        prompt_text_for_llm = prompt_template_str.format(scenario=scenario_text, role=current_role, history=history_text)
    except KeyError as ke:
        logging.error(f"Missing key '{ke}' in prompt_template (from config) for agent response. Template: '{prompt_template_str}' Data: scenario_text_present={bool(scenario_text)}, role_present={bool(current_role)}, history_present={bool(history_text)}")
        # Return None for response_text, but return the problematic template string for debugging.
        return None, prompt_template_str 

    variant = goodfire.Variant(model_id)
    raw_genome_from_agent = agent.genome
    genome_for_goodfire_api = {}
    # Transform the agent's genome (dict of dicts) to the flat dict expected by goodfire.Variant().set()
    if isinstance(raw_genome_from_agent, dict):
        for feature_uuid, feature_data_dict in raw_genome_from_agent.items():
            if isinstance(feature_data_dict, dict) and 'activation' in feature_data_dict:
                # genome format: {'activation': float, 'label': str}
                genome_for_goodfire_api[feature_uuid] = feature_data_dict['activation']
            elif isinstance(feature_data_dict, (int, float)):
                # Fallback for potential old genome format: float activation value
                logging.debug(f"Agent {agent.agent_id} genome for feature {feature_uuid} in generate_agent_response appears to be in old format. Using it directly.")
                genome_for_goodfire_api[feature_uuid] = float(feature_data_dict)
            else:
                logging.warning(f"Agent {agent.agent_id} genome for feature {feature_uuid} in generate_agent_response has unexpected structure: {feature_data_dict}. Skipping.")

    if genome_for_goodfire_api: # Check if the transformed genome has any features to set
        try:
            variant.set(genome_for_goodfire_api)
        except Exception as e:
            logging.error(f"Error setting transformed variant edits for agent {agent.agent_id} during response generation: {e}", exc_info=True)
            
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
    except goodfire_exceptions.GoodfireBaseException as e: # Catching the base Goodfire exception
        logging.error(f"Goodfire API error generating response for agent {agent.agent_id}: {e}", exc_info=True)
        return None, prompt_text_for_llm
    except Exception as e:
        logging.critical(f"Unexpected critical error generating response for agent {agent.agent_id}: {e}", exc_info=True)
        return None, prompt_text_for_llm

    # 3. Adjudication
    logging.debug(f"Game {game_id}: Requesting adjudication.")
    adjudication_final_text, adjudication_llm_prompt = adjudicate_interaction(
        scenario_info, current_transcript, config
    )
    game_details_dict["adjudication_result"] = adjudication_final_text
    game_details_dict["adjudication_prompt"] = adjudication_llm_prompt

    if adjudication_final_text not in ['Role A Wins', 'Role B Wins', 'Tie', 'error']:
        game_details_dict["adjudication_raw_llm_output"] = adjudication_final_text

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
        raw_genome_for_contrast = agent_variant_or_model_id.genome
        genome_for_goodfire_api_contrast = {}
        # Transform the agent's genome for Goodfire API
        if isinstance(raw_genome_for_contrast, dict):
            for feature_uuid, feature_data_dict in raw_genome_for_contrast.items():
                if isinstance(feature_data_dict, dict) and 'activation' in feature_data_dict:
                    genome_for_goodfire_api_contrast[feature_uuid] = feature_data_dict['activation']
                elif isinstance(feature_data_dict, (int, float)):
                    logging.debug(f"Agent {agent_id_for_log} genome for feature {feature_uuid} in perform_contrastive_analysis (Agent type) appears to be in old format. Using it directly.")
                    genome_for_goodfire_api_contrast[feature_uuid] = float(feature_data_dict)
                else:
                    logging.warning(f"Agent {agent_id_for_log} genome for feature {feature_uuid} in perform_contrastive_analysis (Agent type) has unexpected structure: {feature_data_dict}. Skipping.")

        if genome_for_goodfire_api_contrast: # If there are valid edits
            try:
                variant.set(genome_for_goodfire_api_contrast)
                model_arg = variant # Use the variant with edits
            except Exception as e:
                logging.error(f"Error setting transformed variant edits for contrast analysis on agent {agent_id_for_log}: {e}. Using base model ID as fallback.")
                model_arg = agent_variant_or_model_id.model_id # Fallback to base model ID string if set fails
        else:
             # No genome edits to apply, or genome was empty/invalid.
             # Use the un-edited variant (which represents the base model)
             model_arg = variant
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
    except goodfire_exceptions.GoodfireBaseException as e: # Catching the base Goodfire exception
        logging.error(f"Goodfire API error during contrastive analysis for {agent_id_for_log}: {e}", exc_info=True)
        return None, None
    except Exception as e:
        logging.critical(f"Unexpected critical error during contrastive analysis for {agent_id_for_log}: {e}", exc_info=True)
        return None, None

import re

def _extract_xml_tag_content(tag_name: str, xml_text: str, default_value: str | None = None) -> str | None:
    """
    Helper to extract content from an XML-like tag, with lenient delimiters
    and exclusion of content within <scratchpad>...</scratchpad> blocks.

    Lenient delimiters mean '<' can be '.', and '>' can be '.'.
    For example, "<tag>content</tag>", ".tag.content</tag>", "<tag.content./tag.",
    or ".tag.content./tag." are all potentially parsable for the target tag.

    Content within any scratchpad block (e.g., "<scratchpad>text<target>ignored</target></scratchpad>"
    or ".scratchpad.text./scratchpad.") is ignored for target tag searching.
    The scratchpad block delimiters themselves also follow the lenient rule.
    """
    if not xml_text: # Handle empty or None xml_text
        return default_value

    # 1. Define lenient patterns for scratchpad tags to remove their content first.
    #    The pattern allows for attributes or other junk within the tag delimiters.
    #    [^>.]*? matches any characters that are not '>' or '.', non-greedily.
    #    This is applied between the 'scratchpad' name and its closing delimiter char ('>' or '.').
    scratch_open_delimiters = r"[<.]"  # Matches '<' or '.'
    scratch_close_delimiters = r"[>.]" # Matches '>' or '.'
    
    # Pattern for the opening scratchpad tag, e.g., <scratchpad ...> or .scratchpad ... .
    scratch_open_pattern = rf"{scratch_open_delimiters}\s*scratchpad(?:[^>.]*?)\s*{scratch_close_delimiters}"
    # Pattern for the closing scratchpad tag, e.g., </scratchpad> or ./scratchpad.
    scratch_close_pattern = rf"{scratch_open_delimiters}\s*/\s*scratchpad\s*{scratch_close_delimiters}"
    
    # Full pattern to find and identify scratchpad blocks. Content (group 1) is non-greedy.
    scratchpad_block_pattern_for_removal = rf"({scratch_open_pattern})(.*?)({scratch_close_pattern})"
    
    temp_xml_text = xml_text
    cleaned_xml_text = ""
    while True:
        # Find the first scratchpad block from the current start of temp_xml_text
        match_scratch = re.search(scratchpad_block_pattern_for_removal, temp_xml_text, re.DOTALL | re.IGNORECASE)
        if match_scratch:
            # Add text before the scratchpad block
            cleaned_xml_text += temp_xml_text[:match_scratch.start()]
            # Advance temp_xml_text past this entire matched scratchpad block
            temp_xml_text = temp_xml_text[match_scratch.end():]
        else:
            # No more scratchpad blocks found, add the rest of the text
            cleaned_xml_text += temp_xml_text
            break
    
    # 2. Define lenient pattern for the target tag using the cleaned_xml_text
    escaped_tag_name = re.escape(tag_name) # Safety for tag_name content

    # Delimiters for the target tag (can be '<' or '.')
    tag_open_delimiters = r"[<.]"
    tag_close_delimiters = r"[>.]"
    
    target_tag_pattern = rf"{tag_open_delimiters}\s*{escaped_tag_name}(?:[^>.]*?)\s*{tag_close_delimiters}(.*?)(?:{tag_open_delimiters}\s*/\s*{escaped_tag_name}\s*{tag_close_delimiters}|$)"
    
    match_target = re.search(target_tag_pattern, cleaned_xml_text, re.DOTALL | re.IGNORECASE)
    
    if match_target:
        # group(1) is the (.*?) content part
        return match_target.group(1).strip()
        
    return default_value

def adjudicate_interaction(scenario_info: dict, transcript_with_ids: str, config: dict) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None, str | None, str | None]:
    """
    Adjudicates a game interaction using two LLM calls: one for scratchpad, one for outcome & message IDs.
    The scratchpad content from the first call is used as input for the second call.

    Args:
        scenario_info: Dictionary containing scenario details, including 'scenario_text'.
        transcript_with_ids: The game transcript formatted with message IDs (e.g., "A1: ...").
        config: The main simulation configuration dictionary.

    Returns:
        A tuple: (final_parsed_outcome, win_message_id, lose_message_id, scratchpad_content,
                  prompt_for_scratchpad, raw_output_scratchpad,
                  prompt_for_outcome_ids, raw_output_outcome_ids)
    """
    # Default return values for critical failure, matching the new signature
    default_failure_return = ("error", None, None, None, None, None, None, None)

    if not all(isinstance(arg, exp_type) for arg, exp_type in [
        (scenario_info, dict), (transcript_with_ids, str), (config, dict)
    ]):
        logging.error("adjudicate_interaction: Invalid argument types.")
        return default_failure_return

    client = get_goodfire_client(config)
    adj_config = config.get('adjudicator', {})
    model_id = adj_config.get('model_id')
    api_retry_config = config.get('api_retries', {})
    scenario_text_for_prompt = scenario_info.get('scenario_text', '[Scenario Text Missing from input scenario_data]')

    # --- Adjudication Call 1: Scratchpad ---
    prompt_template_scratchpad = adj_config.get('prompt_template_scratchpad')
    max_tokens_scratchpad = adj_config.get('max_tokens_scratchpad', 2500)
    temp_scratchpad = adj_config.get('temperature_scratchpad', 0.0)
    
    scratchpad_content: str | None = None # This will store the *parsed* content of the <scratchpad> tag
    prompt_for_scratchpad: str | None = None
    raw_output_scratchpad: str | None = None # This will store the full raw XML output from the first call

    if not model_id:
        logging.error("Adjudicator model_id missing in config for scratchpad call.")
        # Cannot proceed with either call if model_id is missing.
        return default_failure_return
        
    if not prompt_template_scratchpad:
        logging.error("Adjudicator prompt_template_scratchpad missing in config.")
        prompt_for_scratchpad = f"Scenario: {scenario_text_for_prompt}\nTranscript: {transcript_with_ids}" # Basic prompt for return
        # Allow to proceed to second call, but scratchpad_content will be None.
    else:
        try:
            prompt_for_scratchpad = prompt_template_scratchpad.format(
                scenario=scenario_text_for_prompt,
                transcript_with_ids=transcript_with_ids
            )
            messages_scratchpad = [{"role": "user", "content": prompt_for_scratchpad}]
            logging.debug(f"Attempting adjudication (scratchpad). Prompt: {prompt_for_scratchpad[:500]}...")
            
            response_scratchpad = _execute_api_call(
                client.chat.completions.create,
                messages=messages_scratchpad,
                model=model_id,
                max_completion_tokens=max_tokens_scratchpad,
                temperature=temp_scratchpad,
                stream=False,
                retry_config=api_retry_config
            )

            if response_scratchpad and response_scratchpad.choices and isinstance(response_scratchpad.choices[0].message, dict):
                raw_output_scratchpad = response_scratchpad.choices[0].message.get('content', '').strip()
                if raw_output_scratchpad:
                    # Extract only the content *inside* the <scratchpad>...</scratchpad> tags
                    scratchpad_content = _extract_xml_tag_content("scratchpad", raw_output_scratchpad)
                    if not scratchpad_content:
                        logging.warning(f"Could not parse content within <scratchpad> tag from adjudicator's first response. Raw output: {raw_output_scratchpad[:300]}")
                else:
                    logging.warning("Empty content received from API for adjudication scratchpad.")
            else:
                logging.error("Invalid or empty response structure from API for adjudication scratchpad.")
                # raw_output_scratchpad remains None
        except TimeoutError:
            logging.error("Timeout during adjudication (scratchpad) after all retries.")
        except goodfire_exceptions.GoodfireBaseException as e:
            logging.error(f"Goodfire API error during adjudication (scratchpad): {e}", exc_info=True)
        except KeyError as ke: # Catch missing keys in prompt formatting
            logging.error(f"Missing key '{ke}' in adjudicator prompt_template_scratchpad. Template: '{prompt_template_scratchpad}'")
            # prompt_for_scratchpad would be the problematic template here if assigned before error
        except Exception as e:
            logging.critical(f"Unexpected critical error during adjudication (scratchpad): {e}", exc_info=True)

    # --- Adjudication Call 2: Outcome and Message IDs ---
    prompt_template_outcome_ids = adj_config.get('prompt_template_outcome_ids')
    max_tokens_outcome_ids = adj_config.get('max_tokens_outcome_ids', 500)
    temp_outcome_ids = adj_config.get('temperature_outcome_ids', 0.0)

    raw_outcome_tag_content: str | None = None # Parsed content of <outcome>
    win_message_id: str | None = None
    lose_message_id: str | None = None
    final_parsed_outcome: str = "error" # Default to error for the game outcome
    prompt_for_outcome_ids: str | None = None
    raw_output_outcome_ids: str | None = None # Full raw XML output from the second call

    if not model_id: # Already checked, but defensive for this block
        logging.error("Adjudicator model_id missing in config for outcome_ids call.")
        return ("error", None, None, scratchpad_content, prompt_for_scratchpad, raw_output_scratchpad, None, None)

    if not prompt_template_outcome_ids:
        logging.error("Adjudicator prompt_template_outcome_ids missing in config.")
        # Construct a basic prompt string for returning, even if the call won't be made well
        scratchpad_input_for_debug_prompt = scratchpad_content if scratchpad_content else "[Scratchpad Content Missing]"
        prompt_for_outcome_ids = f"Scenario: {scenario_text_for_prompt}\nTranscript: {transcript_with_ids}\nScratchpad: {scratchpad_input_for_debug_prompt}"
        # Fall through, final_parsed_outcome remains "error"
    else:
        try:
            # scratchpad_content (parsed from first call) is a string for formatting.
            # This is the content *inside* the <scratchpad> tags.
            scratchpad_analysis_input = scratchpad_content if scratchpad_content else "[No scratchpad analysis provided or parsed from the first step]"
            
            prompt_for_outcome_ids = prompt_template_outcome_ids.format(
                scenario=scenario_text_for_prompt,
                transcript_with_ids=transcript_with_ids,
                scratchpad_analysis=scratchpad_analysis_input # Feed parsed scratchpad content here
            )
            messages_outcome_ids = [{"role": "user", "content": prompt_for_outcome_ids}]
            logging.debug(f"Attempting adjudication (outcome/IDs). Input scratchpad snippet: {scratchpad_analysis_input[:200]}... Prompt: {prompt_for_outcome_ids[:500]}...")

            response_outcome_ids = _execute_api_call(
                client.chat.completions.create,
                messages=messages_outcome_ids,
                model=model_id,
                max_completion_tokens=max_tokens_outcome_ids,
                temperature=temp_outcome_ids,
                stream=False,
                retry_config=api_retry_config
            )

            if response_outcome_ids and response_outcome_ids.choices and isinstance(response_outcome_ids.choices[0].message, dict):
                raw_output_outcome_ids = response_outcome_ids.choices[0].message.get('content', '').strip()
                logging.debug(f"Raw LLM output for adjudication (outcome/IDs):\n{raw_output_outcome_ids}")

                if raw_output_outcome_ids:
                    # Parse the <outcome> tag from the second response
                    raw_outcome_tag_content = _extract_xml_tag_content("outcome", raw_output_outcome_ids)
                    
                    # Try parsing primary message ID tag names, then fall back to alternatives
                    win_message_id_temp = _extract_xml_tag_content("win_message_id", raw_output_outcome_ids)
                    if not win_message_id_temp: # If primary tag not found or empty, try the alternative
                        logging.debug("Primary tag <win_message_id> not found or empty, trying <message_id_win>.")
                        win_message_id_temp = _extract_xml_tag_content("message_id_win", raw_output_outcome_ids)

                    lose_message_id_temp = _extract_xml_tag_content("lose_message_id", raw_output_outcome_ids)
                    if not lose_message_id_temp: # If primary tag not found or empty, try the alternative
                        logging.debug("Primary tag <lose_message_id> not found or empty, trying <message_id_loss>.")
                        lose_message_id_temp = _extract_xml_tag_content("message_id_loss", raw_output_outcome_ids)

                    valid_msg_id_pattern = r"^[AB][1-3]$"
                    
                    is_win_msg_id_valid = bool(win_message_id_temp and re.match(valid_msg_id_pattern, win_message_id_temp))
                    is_lose_msg_id_valid = bool(lose_message_id_temp and re.match(valid_msg_id_pattern, lose_message_id_temp))

                    if not raw_outcome_tag_content:
                        logging.warning(f"Could not parse <outcome> tag from adjudicator's second response. Raw: {raw_output_outcome_ids[:300]}. Current final_parsed_outcome: '{final_parsed_outcome}'.")
                        # final_parsed_outcome remains "error", will be checked by override logic
                    else:
                        # Normalize and check outcome tag content more leniently
                        outcome_text_normalized = raw_outcome_tag_content.strip().lower()
                        if outcome_text_normalized == "role a wins" or outcome_text_normalized == "role a":
                            final_parsed_outcome = "Role A Wins"
                        elif outcome_text_normalized == "role b wins" or outcome_text_normalized == "role b":
                            final_parsed_outcome = "Role B Wins"
                        elif outcome_text_normalized == "tie":
                            final_parsed_outcome = "Tie"
                        else:
                            # Outcome tag content is not one of the expected exact or shortened phrases.
                            logging.warning(f"Non-canonical or unexpected content in <outcome> tag: '{raw_outcome_tag_content}'. Current final_parsed_outcome: '{final_parsed_outcome}'. Will rely on message IDs for override if possible.")
                            # final_parsed_outcome remains "error" (its default) or its value from a previous step if logic changes,
                            # allowing the override logic below to take effect if applicable.
                    
                    # Override outcome based on win_message_id if outcome is still "error" (e.g. outcome tag missing or completely unparseable)
                    # or if the outcome tag had content that was not one of the recognized win/loss/tie states.
                    # This ensures that if a valid win_message_id is present, it can dictate a win.
                    if final_parsed_outcome == "error" or \
                       (raw_outcome_tag_content and final_parsed_outcome not in ["Role A Wins", "Role B Wins", "Tie"]):
                        if is_win_msg_id_valid:
                            if win_message_id_temp.startswith("A"):
                                final_parsed_outcome = "Role A Wins"
                                logging.info(f"Outcome set to 'Role A Wins' based on valid win_message_id: {win_message_id_temp} (original outcome tag was '{raw_outcome_tag_content}').")
                            elif win_message_id_temp.startswith("B"):
                                final_parsed_outcome = "Role B Wins"
                                logging.info(f"Outcome set to 'Role B Wins' based on valid win_message_id: {win_message_id_temp} (original outcome tag was '{raw_outcome_tag_content}').")
                            # If win_message_id is valid but doesn't start with A/B, it's an issue with pattern or logic
                        else:
                             logging.warning(f"Original outcome tag was '{raw_outcome_tag_content}' (or missing), and no valid win_message_id found to determine winner. Final outcome remains '{final_parsed_outcome}'.")
                    
                    # Assign win/lose message IDs if the final_parsed_outcome is a win/loss
                    if final_parsed_outcome in ["Role A Wins", "Role B Wins"]:
                        if is_win_msg_id_valid:
                            win_message_id = win_message_id_temp
                        else:
                            logging.warning(f"Outcome is '{final_parsed_outcome}' but win_message_id ('{win_message_id_temp}') is missing or invalid. Setting to None.")
                            win_message_id = None 
                        
                        if is_lose_msg_id_valid:
                            lose_message_id = lose_message_id_temp
                        else:
                            logging.warning(f"Outcome is '{final_parsed_outcome}' but lose_message_id ('{lose_message_id_temp}') is missing or invalid. Setting to None.")
                            lose_message_id = None
                    else: # Tie or error
                        win_message_id = None
                        lose_message_id = None
                else: # raw_output_outcome_ids was empty
                    logging.warning("Empty content received from API for adjudication (outcome/IDs).")
                    # final_parsed_outcome remains "error"
            else: # Bad response structure from API
                logging.error("Invalid or empty response structure from API for adjudication (outcome/IDs).")
                # final_parsed_outcome remains "error", raw_output_outcome_ids might be None

        except TimeoutError:
             logging.error("Timeout during adjudication (outcome/IDs) after all retries.")
             # final_parsed_outcome remains "error"
        except goodfire_exceptions.GoodfireBaseException as e:
            logging.error(f"Goodfire API error during adjudication (outcome/IDs): {e}", exc_info=True)
            # final_parsed_outcome remains "error"
        except KeyError as ke: # Catch missing keys in prompt formatting
            logging.error(f"Missing key '{ke}' in adjudicator prompt_template_outcome_ids. Template: '{prompt_template_outcome_ids}'")
            # prompt_for_outcome_ids might be the problematic template string if assigned before error
        except Exception as e:
            logging.critical(f"Unexpected critical error during adjudication (outcome/IDs): {e}", exc_info=True)
            # final_parsed_outcome remains "error"

    logging.info(f"Adjudication result: Outcome='{final_parsed_outcome}', WinMsg='{win_message_id}', LoseMsg='{lose_message_id}', Scratchpad content (parsed from 1st call) present: {bool(scratchpad_content)}.")
    return (final_parsed_outcome, win_message_id, lose_message_id, scratchpad_content,
            prompt_for_scratchpad, raw_output_scratchpad,
            prompt_for_outcome_ids, raw_output_outcome_ids)
