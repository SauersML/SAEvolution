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
import asyncio
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

async def _execute_api_call_async(api_call_func: callable, *args, **kwargs):
    """
    Executes an asynchronous API call with retry logic for specified errors.
    The api_call_func is expected to be an awaitable (e.g., a method of AsyncClient).
    """
    retry_config = kwargs.pop('retry_config', {})
    max_retries = retry_config.get('max_retries', 3)
    initial_delay = retry_config.get('initial_delay', 1.0)
    backoff_factor = retry_config.get('backoff_factor', 2.0)

    if not callable(api_call_func):
        raise TypeError("api_call_func must be callable for async execution.")

    current_delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            result = await api_call_func(*args, **kwargs) # Await the API call
            return result
        except goodfire_exceptions.APIStatusError as e: # Catch HTTP status errors from Goodfire SDK
            status_code = e.status_code
            if status_code in [429, 500, 502, 503, 504]: # Retriable HTTP status codes
                logging.warning(
                    f"Retriable API Status Error (Status: {status_code}) on async attempt {attempt + 1}/{max_retries} "
                    f"for {getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
                )
                last_exception = e
            else:
                logging.error(
                    f"Non-retriable API Status Error (Status: {status_code}) for async "
                    f"{getattr(api_call_func, '__name__', 'API call')}: {e}."
                )
                raise # Re-raise non-retriable Goodfire API status errors
        except goodfire_exceptions.APITimeoutError as e: # Catch specific timeout errors from Goodfire SDK
            logging.warning(
                f"Goodfire API Timeout Error on async attempt {attempt + 1}/{max_retries} "
                f"for {getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
            )
            last_exception = e
        except goodfire_exceptions.GoodfireBaseException as e: # Catch other Goodfire-specific base errors
            logging.warning(
                f"Goodfire API Base Error ({type(e).__name__}) on async attempt {attempt + 1}/{max_retries} "
                f"for {getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
            )
            last_exception = e
        except requests.exceptions.Timeout as e: # Catch timeouts from the underlying requests library (might be less common with pure async client)
            logging.warning(
                f"HTTP Request Timeout on async attempt {attempt + 1}/{max_retries} for "
                f"{getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
            )
            last_exception = e
        except requests.exceptions.RequestException as e: # Catch other network errors (might be less common with pure async client)
            logging.warning(
                f"HTTP Request Exception on async attempt {attempt + 1}/{max_retries} for "
                f"{getattr(api_call_func, '__name__', 'API call')}. Retrying in {current_delay:.2f}s. Error: {e}"
            )
            last_exception = e
        except Exception as e: # Catch any other unexpected errors
            logging.error(
                f"Unexpected error during async API call attempt {attempt + 1}/{max_retries} for "
                f"{getattr(api_call_func, '__name__', 'API call')}: {e}", exc_info=True
            )
            last_exception = e
            if attempt == max_retries - 1: # If it's the last attempt, re-raise
                raise

        await asyncio.sleep(current_delay) # Use asyncio.sleep for async context
        current_delay *= backoff_factor
        # Add a small jitter to the delay
        current_delay += random.uniform(0, initial_delay * 0.1)

    logging.error(f"Async API call {getattr(api_call_func, '__name__', 'API call')} failed after {max_retries} retries.")
    if last_exception:
        raise TimeoutError(f"Async API call failed after {max_retries} retries.") from last_exception
    else:
        raise TimeoutError(f"Async API call failed after {max_retries} retries, but no specific exception was captured.")

async def generate_scenario(proposer_agent, config: dict) -> tuple[dict | None, str | None]:
    """
    Generates a game scenario asynchronously using the proposer agent and configuration.
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
        logging.error("generate_scenario_async: Invalid proposer_agent provided (must be an Agent instance).")
        return None, None
    if not isinstance(config, dict):
        logging.error("generate_scenario_async: Invalid config provided (must be a dictionary).")
        return None, None

    client = get_goodfire_async_client(config) # Use async client
    gen_config = config.get('generation', {}).get('scenario', {})
    model_id = proposer_agent.model_id
    
    prompt_text_template = gen_config.get('prompt')
    if prompt_text_template is None:
        err_msg = "Configuration error: 'generation.scenario.prompt' is missing from config.yaml."
        logging.critical(err_msg + f" (Agent: {proposer_agent.agent_id})")
        return None, None

    max_tokens = gen_config.get('max_tokens', 1000)
    temperature = gen_config.get('temperature', 0.7)
    api_retry_config = config.get('api_retries', {})

    diversification_seed = f"agent_id:{proposer_agent.agent_id}_call_id:{uuid.uuid4().hex[:8]}"
    final_prompt_text = f"{prompt_text_template}\n[{diversification_seed}]"

    variant = goodfire.Variant(model_id)
    raw_genome_from_proposer = proposer_agent.genome
    genome_for_goodfire_api = {}
    if isinstance(raw_genome_from_proposer, dict):
        for feature_uuid, feature_data_dict in raw_genome_from_proposer.items():
            if isinstance(feature_data_dict, dict) and 'activation' in feature_data_dict:
                genome_for_goodfire_api[feature_uuid] = feature_data_dict['activation']
            elif isinstance(feature_data_dict, (int, float)):
                logging.debug(f"Agent {proposer_agent.agent_id} genome for feature {feature_uuid} in generate_scenario_async appears to be in old format. Using it directly.")
                genome_for_goodfire_api[feature_uuid] = float(feature_data_dict)
            else:
                logging.warning(f"Agent {proposer_agent.agent_id} genome for feature {feature_uuid} in generate_scenario_async has unexpected structure: {feature_data_dict}. Skipping.")

    if genome_for_goodfire_api:
        try:
            variant.set(genome_for_goodfire_api) # This is a synchronous operation on a local object
        except Exception as e:
            logging.error(f"Error setting transformed variant edits for agent {proposer_agent.agent_id} in generate_scenario_async: {e}", exc_info=True)

    messages = [{"role": "user", "content": final_prompt_text}]
    llm_raw_output_text = None 
    expected_tag_sequence = ["context", "roles", "objectives", "win_criteria", "tie_criteria", "proposer_role"]

    try:
        logging.debug(f"Attempting async scenario generation for agent {proposer_agent.agent_id} with prompt ending: ...{final_prompt_text[-100:]}")
        response = await _execute_api_call_async( # Use await and async version
            client.chat.completions.create,
            messages=messages,
            model=variant,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            retry_config=api_retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
            logging.error(f"Invalid or empty response structure received from async API for scenario generation (Agent: {proposer_agent.agent_id}).")
            return None, final_prompt_text

        llm_raw_output_text = response.choices[0].message.get('content', '')
        logging.debug(f"Raw LLM output for async scenario (Agent {proposer_agent.agent_id}):\n{llm_raw_output_text}")

        if not llm_raw_output_text.strip():
            logging.warning(f"Empty content received from async API for scenario generation (Agent: {proposer_agent.agent_id}).")
            return None, final_prompt_text

        tags_content = {}
        search_text = llm_raw_output_text 
        current_search_position = 0
        parsing_successful = True

        for i, tag_name_to_find in enumerate(expected_tag_sequence):
            open_marker_regex_str = r"\b" + re.escape(tag_name_to_find) + r"\b"
            compiled_open_marker_regex = re.compile(open_marker_regex_str, re.IGNORECASE)
            open_tag_name_match = compiled_open_marker_regex.search(search_text, pos=current_search_position)

            if not open_tag_name_match:
                logging.warning(f"Async scenario parsing: Start of tag '{tag_name_to_find}' not found. Agent: {proposer_agent.agent_id}.")
                parsing_successful = False
                break
            
            content_start_index = open_tag_name_match.end()
            optional_gt_match = re.match(r"\s*(?:>)?", search_text[content_start_index:])
            if optional_gt_match:
                content_start_index += optional_gt_match.end()

            content_end_index = len(search_text)
            found_explicit_closer = False
            close_marker_regex_str = r"(?:<)?\s*/\s*\b" + re.escape(tag_name_to_find) + r"\b"
            compiled_close_marker_regex = re.compile(close_marker_regex_str, re.IGNORECASE)
            close_tag_match = compiled_close_marker_regex.search(search_text, pos=content_start_index)
            
            if close_tag_match:
                content_end_index = close_tag_match.start()
                compiled_full_closing_marker_regex = re.compile(close_marker_regex_str + r"\s*(?:>)?", re.IGNORECASE)
                full_closing_marker_match = compiled_full_closing_marker_regex.search(search_text, pos=close_tag_match.start())
                current_search_position = full_closing_marker_match.end() if full_closing_marker_match else close_tag_match.end()
                found_explicit_closer = True
            
            if i + 1 < len(expected_tag_sequence):
                next_tag_name = expected_tag_sequence[i+1]
                next_tag_open_marker_regex_str = r"<\s*" + re.escape(next_tag_name) + r"(?:>|\s)"
                compiled_next_tag_open_marker_regex = re.compile(next_tag_open_marker_regex_str, re.IGNORECASE)
                next_tag_open_name_match = compiled_next_tag_open_marker_regex.search(search_text, pos=content_start_index)
                
                if next_tag_open_name_match:
                    if not found_explicit_closer or next_tag_open_name_match.start() < content_end_index:
                        content_end_index = next_tag_open_name_match.start()
                        current_search_position = next_tag_open_name_match.start() 
            
            extracted_content = search_text[content_start_index:content_end_index].strip()
            tags_content[tag_name_to_find] = extracted_content
            logging.debug(f"Async scenario parsing: Tag '{tag_name_to_find}' -> '{extracted_content[:100]}...'. Next search pos: {current_search_position}")

            if not found_explicit_closer and (i + 1 >= len(expected_tag_sequence)): 
                break 
            
            if current_search_position >= len(search_text) and i + 1 < len(expected_tag_sequence):
                logging.warning(f"Async scenario parsing: Ran out of text after parsing '{tag_name_to_find}'. Agent: {proposer_agent.agent_id}.")
                parsing_successful = False
                break

        if len(tags_content) != len(expected_tag_sequence):
            logging.warning(f"Async scenario parsing: Failed to extract all {len(expected_tag_sequence)} tags. Found {len(tags_content)}. Agent: {proposer_agent.agent_id}.")
            parsing_successful = False

        if not parsing_successful:
            return None, final_prompt_text

        if "objective" not in tags_content.get("objectives", "").lower():
            logging.warning(f"Keyword 'objective' not found in <objectives> (async). Agent: {proposer_agent.agent_id}.")
            return None, final_prompt_text
        if "win criteria" not in tags_content.get("win_criteria", "").lower():
            logging.warning(f"Phrase 'win criteria' not found in <win_criteria> (async). Agent: {proposer_agent.agent_id}.")
            return None, final_prompt_text
        if "tie criteria" not in tags_content.get("tie_criteria", "").lower():
            logging.warning(f"Phrase 'tie criteria' not found in <tie_criteria> (async). Agent: {proposer_agent.agent_id}.")
            return None, final_prompt_text
        
        proposer_role_text = tags_content.get("proposer_role", "")
        if proposer_role_text not in ["Role A", "Role B"]:
            logging.warning(f"Invalid content for <proposer_role> (async): '{proposer_role_text}'. Agent: {proposer_agent.agent_id}.")
            return None, final_prompt_text

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
            'raw_output': llm_raw_output_text 
        }
        logging.info(f"Async scenario generated and parsed successfully for agent {proposer_agent.agent_id}.")
        return scenario_info_dict, final_prompt_text

    except TimeoutError:
        logging.error(f"Timeout generating async scenario for agent {proposer_agent.agent_id} after all retries.")
        return None, final_prompt_text
    except goodfire_exceptions.GoodfireBaseException as e:
        logging.error(f"Goodfire API error generating async scenario for agent {proposer_agent.agent_id}: {e}", exc_info=True)
        return None, final_prompt_text
    except Exception as e:
        logging.critical(f"Unexpected critical error generating async scenario for agent {proposer_agent.agent_id}: {e}", exc_info=True)
        if llm_raw_output_text is not None:
            logging.error(f"Raw content at time of critical error (Agent {proposer_agent.agent_id}, async): {llm_raw_output_text[:1000]}...")
        return None, final_prompt_text


async def generate_agent_response(agent, scenario_data: dict, transcript: list, current_role: str, config: dict) -> tuple[str | None, str | None]:
    """
    Generates an agent's response asynchronously in a game turn.
    Returns:
        A tuple: (str: agent's response text or None on failure,
                  str: The prompt_text used for generation or None on early failure)
    """
    from manager import Agent # Local import

    if not all(isinstance(arg, exp_type) for arg, exp_type in [
        (agent, Agent), (scenario_data, dict), (transcript, list),
        (current_role, str), (config, dict)
    ]):
        logging.error(f"generate_agent_response_async: Invalid argument types provided. Agent: {type(agent)}, Scenario: {type(scenario_data)}, etc.")
        return None, None

    client = get_goodfire_async_client(config) # Use async client
    gen_config = config.get('generation', {}).get('response', {})
    model_id = agent.model_id
    scenario_text = scenario_data.get('scenario_text', '[Scenario Text Missing from input scenario_data]')

    prompt_template_str = gen_config.get('prompt_template')
    if prompt_template_str is None:
        err_msg = "Configuration error: 'generation.response.prompt_template' is missing from config.yaml."
        logging.critical(err_msg + f" (Agent: {agent.agent_id}, Role: {current_role})")
        return None, None

    max_tokens = gen_config.get('max_tokens', 150)
    temperature = gen_config.get('temperature', 0.6)
    api_retry_config = config.get('api_retries', {})

    history_lines = []
    for msg in transcript:
        role = msg.get('role', 'UnknownRole')
        content = msg.get('content', '[empty_content]')
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "[No conversation history yet]"

    prompt_text_for_llm = None
    try:
        prompt_text_for_llm = prompt_template_str.format(scenario=scenario_text, role=current_role, history=history_text)
    except KeyError as ke:
        logging.error(f"Missing key '{ke}' in prompt_template for async agent response. Template: '{prompt_template_str}'")
        return None, prompt_template_str 

    variant = goodfire.Variant(model_id)
    raw_genome_from_agent = agent.genome
    genome_for_goodfire_api = {}
    if isinstance(raw_genome_from_agent, dict):
        for feature_uuid, feature_data_dict in raw_genome_from_agent.items():
            if isinstance(feature_data_dict, dict) and 'activation' in feature_data_dict:
                genome_for_goodfire_api[feature_uuid] = feature_data_dict['activation']
            elif isinstance(feature_data_dict, (int, float)):
                logging.debug(f"Agent {agent.agent_id} genome for feature {feature_uuid} in generate_agent_response_async (old format). Using directly.")
                genome_for_goodfire_api[feature_uuid] = float(feature_data_dict)
            else:
                logging.warning(f"Agent {agent.agent_id} genome for feature {feature_uuid} in generate_agent_response_async (unexpected structure): {feature_data_dict}. Skipping.")

    if genome_for_goodfire_api:
        try:
            variant.set(genome_for_goodfire_api) # Synchronous local operation
        except Exception as e:
            logging.error(f"Error setting variant edits for agent {agent.agent_id} during async response generation: {e}", exc_info=True)
            
    messages = [{"role": "user", "content": prompt_text_for_llm}]

    try:
        logging.debug(f"Attempting async agent response for {agent.agent_id} ({current_role}). Prompt: {prompt_text_for_llm[:500]}...")
        response = await _execute_api_call_async( # Use await and async version
            client.chat.completions.create,
            messages=messages,
            model=variant,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            retry_config=api_retry_config
        )

        if not response or not response.choices or not isinstance(response.choices[0].message, dict):
            logging.error(f"Invalid or empty response structure from async API for agent {agent.agent_id} response.")
            return None, prompt_text_for_llm

        response_text = response.choices[0].message.get('content', '').strip()
        if not response_text:
            logging.info(f"Agent {agent.agent_id} generated an empty string response (async).")

        logging.debug(f"Async response generated by agent {agent.agent_id}: '{response_text[:100]}...'")
        return response_text, prompt_text_for_llm

    except TimeoutError:
        logging.error(f"Timeout generating async response for agent {agent.agent_id} after all retries.")
        return None, prompt_text_for_llm
    except goodfire_exceptions.GoodfireBaseException as e:
        logging.error(f"Goodfire API error generating async response for agent {agent.agent_id}: {e}", exc_info=True)
        return None, prompt_text_for_llm
    except Exception as e:
        logging.critical(f"Unexpected critical error generating async response for agent {agent.agent_id}: {e}", exc_info=True)
        return None, prompt_text_for_llm


def _extract_xml_tag_content(tag_name: str, xml_text: str, default_value: str | None = None) -> str | None:
    """
    Helper to extract content related to an XML-like tag.
    For 'scratchpad', it extracts the entire block from the first occurrence
    to a corresponding (potentially malformed) end, or to the end of the string.
    Requires 'scratchpad' to appear on the first line for it to be processed.

    For other tags, it extracts content *between* the tag and its closer,
    after removing any 'scratchpad' blocks.

    All matching is case-insensitive and lenient with delimiters like '<', '.', '>'.
    """
    if not xml_text:
        return default_value

    target_tag_name_lower = tag_name.lower()

    if target_tag_name_lower == "scratchpad":
        # 1. Check if "scratchpad" (case-insensitive) is present on the first line.
        first_line = xml_text.split('\n', 1)[0]
        if not re.search(r"scratchpad", first_line, re.IGNORECASE):
            logging.debug(f"_extract_xml_tag_content: 'scratchpad' not found on the first line for tag_name 'scratchpad'. Raw first line: '{first_line[:100]}'")
            return default_value

        # 2. Find the start of the first scratchpad-like tag structure.
        # Regex: ([<.]?\s*) captures optional opening delimiter (e.g., '<', '.') and spaces.
        #        (scratchpad) captures the word "scratchpad".
        open_tag_match = re.search(r"([<.]?\s*)(scratchpad)", xml_text, re.IGNORECASE)
        if not open_tag_match:
            # This case should be rare if the first-line check passed and was reasonably aligned with this regex.
            logging.debug(f"_extract_xml_tag_content: Could not find a starting 'scratchpad' pattern for tag_name 'scratchpad' despite passing first-line check. Raw text: '{xml_text[:200]}'")
            return default_value
        
        start_extraction_index = open_tag_match.start(1)  # Start of the captured opening delimiter part.
        search_pos_for_end_tag = open_tag_match.end(2)  # Position right after the word "scratchpad" in the opening tag.

        # 3. Find the end of the scratchpad block.
        # Attempt to find a structured closing tag first.
        # Regex: ([<.\/]\s*) captures delimiter like '<', '.', '/'.
        #        (/\s*)? optionally captures a slash if separated by space e.g. '< / scratchpad'.
        #        (scratchpad) captures the word "scratchpad".
        #        (\s*[>.]?) captures optional space and closing delimiter like '>' or '.'.
        # Search from search_pos_for_end_tag to avoid re-matching the opening tag immediately.
        structured_close_match = re.search(r"([<.\/]\s*)(/\s*)?(scratchpad)(\s*[>.]?)", xml_text[search_pos_for_end_tag:], re.IGNORECASE)
        
        if structured_close_match:
            # end_extraction_index is after the matched closing tag part, relative to the whole xml_text.
            end_extraction_index = search_pos_for_end_tag + structured_close_match.end(0)
            logging.debug(f"_extract_xml_tag_content (scratchpad): Found structured closing tag. Extraction from index {start_extraction_index} to {end_extraction_index}.")
            return xml_text[start_extraction_index:end_extraction_index]

        # If no structured closing tag, check for a simple "scratchpad" word occurrence at/near the end as a malformed closer.
        # Find all occurrences of "scratchpad" after the opening one.
        all_later_scratchpad_occurrences = []
        for m in re.finditer(r"scratchpad", xml_text[search_pos_for_end_tag:], re.IGNORECASE):
            # Store start index relative to original xml_text
            all_later_scratchpad_occurrences.append(search_pos_for_end_tag + m.start()) 
        
        if all_later_scratchpad_occurrences:
            last_occurrence_start_index = all_later_scratchpad_occurrences[-1]
            
            # Check if this last "scratchpad" word (optionally followed by > or .) is near the end of the string.
            # Regex for malformed end: (scratchpad) followed by optional space and > or .
            potential_malformed_end_tag_match = re.match(r"(scratchpad)(\s*[>.]?)", xml_text[last_occurrence_start_index:], re.IGNORECASE)
            if potential_malformed_end_tag_match:
                end_of_this_malformed_tag = last_occurrence_start_index + potential_malformed_end_tag_match.end(0)
                
                # Heuristic: consider it a closer if it's in the last part of the string.
                # Define "near the end" as being within a certain number of characters
                chars_after_malformed_end = len(xml_text) - end_of_this_malformed_tag
                threshold_chars = 1000
                if chars_after_malformed_end < threshold_chars :
                    logging.debug(f"_extract_xml_tag_content (scratchpad): Found malformed closing 'scratchpad' near end. Extraction from index {start_extraction_index} to {end_of_this_malformed_tag}.")
                    return xml_text[start_extraction_index:end_of_this_malformed_tag]

        # If no clear structured or convincing malformed closing tag is found, extract to the end of the string.
        logging.debug(f"_extract_xml_tag_content (scratchpad): No clear closing tag found. Extracting from start of first tag (index {start_extraction_index}) to end of string.")
        return xml_text[start_extraction_index:]

    else: # Logic for tags other than "scratchpad"
        # 1. Remove any scratchpad blocks first to avoid interference.
        temp_xml_text_for_others = xml_text
        cleaned_xml_text_for_others = ""
        
        scratch_open_delimiters_others = r"[<.]"
        scratch_close_delimiters_others = r"[>.]"
        scratch_open_pattern_others = rf"{scratch_open_delimiters_others}\s*scratchpad(?:[^>.]*?)\s*{scratch_close_delimiters_others}"
        scratch_close_pattern_others = rf"{scratch_open_delimiters_others}\s*/\s*scratchpad\s*{scratch_close_delimiters_others}"
        scratchpad_block_pattern_for_removal_others = rf"({scratch_open_pattern_others})(.*?)({scratch_close_pattern_others})"
        
        current_pos_others = 0
        for match_scratch in re.finditer(scratchpad_block_pattern_for_removal_others, temp_xml_text_for_others, re.DOTALL | re.IGNORECASE):
            cleaned_xml_text_for_others += temp_xml_text_for_others[current_pos_others:match_scratch.start()]
            current_pos_others = match_scratch.end()
        cleaned_xml_text_for_others += temp_xml_text_for_others[current_pos_others:]
        
        # 2. Define lenient pattern for the target tag to extract content *between* delimiters.
        escaped_tag_name = re.escape(tag_name) 
        tag_open_delimiters_others = r"[<.]"
        tag_close_delimiters_others = r"[>.]"
        # Pattern: <tag_name ...> (content_group) </tag_name ...> or end of string
        # (?:[^>.]*?) allows for attributes or junk within the opening tag delimiters.
        target_tag_pattern_others = (
            rf"{tag_open_delimiters_others}\s*{escaped_tag_name}(?:[^>.]*?)\s*{tag_close_delimiters_others}"  # Opening tag part
            r"(.*?)"  # Group 1: The content to extract (non-greedy)
            r"(?:"  # Start of non-capturing group for closing tag or end of string
            rf"{tag_open_delimiters_others}\s*/\s*{escaped_tag_name}\s*{tag_close_delimiters_others}"  # Closing tag
            r"|$"  # OR end of string
            r")"
        )
        
        match_target = re.search(target_tag_pattern_others, cleaned_xml_text_for_others, re.DOTALL | re.IGNORECASE)
        
        if match_target:
            return match_target.group(1).strip() # group(1) is the content between the tags
            
        return default_value

async def adjudicate_interaction(scenario_info: dict, transcript_with_ids: str, config: dict) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None, str | None, str | None]:
    """
    Adjudicates a game interaction asynchronously using two LLM calls: one for scratchpad, one for outcome & message IDs.
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
    default_failure_return = ("error", None, None, None, None, None, None, None)

    if not all(isinstance(arg, exp_type) for arg, exp_type in [
        (scenario_info, dict), (transcript_with_ids, str), (config, dict)
    ]):
        logging.error("adjudicate_interaction_async: Invalid argument types.")
        return default_failure_return

    client = get_goodfire_async_client(config) # Use async client
    adj_config = config.get('adjudicator', {})
    model_id = adj_config.get('model_id')
    api_retry_config = config.get('api_retries', {})
    scenario_text_for_prompt = scenario_info.get('scenario_text', '[Scenario Text Missing from input scenario_data]')

    # --- Adjudication Call 1: Scratchpad ---
    prompt_template_scratchpad = adj_config.get('prompt_template_scratchpad')
    max_tokens_scratchpad = adj_config.get('max_tokens_scratchpad', 2500)
    temp_scratchpad = adj_config.get('temperature_scratchpad', 0.0)
    
    scratchpad_content: str | None = None
    prompt_for_scratchpad: str | None = None
    raw_output_scratchpad: str | None = None

    if not model_id:
        logging.error("Adjudicator model_id missing in config for async scratchpad call.")
        return default_failure_return
        
    if not prompt_template_scratchpad:
        logging.error("Adjudicator prompt_template_scratchpad missing in config for async call.")
        prompt_for_scratchpad = f"Scenario: {scenario_text_for_prompt}\nTranscript: {transcript_with_ids}"
    else:
        try:
            prompt_for_scratchpad = prompt_template_scratchpad.format(
                scenario=scenario_text_for_prompt,
                transcript_with_ids=transcript_with_ids
            )
            messages_scratchpad = [{"role": "user", "content": prompt_for_scratchpad}]
            logging.debug(f"Attempting async adjudication (scratchpad). Prompt: {prompt_for_scratchpad[:500]}...")
            
            response_scratchpad = await _execute_api_call_async( # Use await and async version
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
                    scratchpad_content = _extract_xml_tag_content("scratchpad", raw_output_scratchpad)
                    if not scratchpad_content:
                        logging.warning(f"Could not parse <scratchpad> tag from async adjudicator's first response. Raw: {raw_output_scratchpad[:300]}")
                else:
                    logging.warning("Empty content received from async API for adjudication scratchpad.")
            else:
                logging.error("Invalid or empty response structure from async API for adjudication scratchpad.")
        except TimeoutError:
            logging.error("Timeout during async adjudication (scratchpad) after all retries.")
        except goodfire_exceptions.GoodfireBaseException as e:
            logging.error(f"Goodfire API error during async adjudication (scratchpad): {e}", exc_info=True)
        except KeyError as ke:
            logging.error(f"Missing key '{ke}' in adjudicator prompt_template_scratchpad (async). Template: '{prompt_template_scratchpad}'")
        except Exception as e:
            logging.critical(f"Unexpected critical error during async adjudication (scratchpad): {e}", exc_info=True)

    # --- Adjudication Call 2: Outcome and Message IDs ---
    prompt_template_outcome_ids = adj_config.get('prompt_template_outcome_ids')
    max_tokens_outcome_ids = adj_config.get('max_tokens_outcome_ids', 500)
    temp_outcome_ids = adj_config.get('temperature_outcome_ids', 0.0)

    raw_outcome_tag_content: str | None = None
    win_message_id: str | None = None
    lose_message_id: str | None = None
    final_parsed_outcome: str = "error"
    prompt_for_outcome_ids: str | None = None
    raw_output_outcome_ids: str | None = None

    if not model_id: # Already checked
        logging.error("Adjudicator model_id missing in config for async outcome_ids call.")
        return ("error", None, None, scratchpad_content, prompt_for_scratchpad, raw_output_scratchpad, None, None)

    if not prompt_template_outcome_ids:
        logging.error("Adjudicator prompt_template_outcome_ids missing in config for async call.")
        scratchpad_input_for_debug_prompt = scratchpad_content if scratchpad_content else "[Scratchpad Content Missing]"
        prompt_for_outcome_ids = f"Scenario: {scenario_text_for_prompt}\nTranscript: {transcript_with_ids}\nScratchpad: {scratchpad_input_for_debug_prompt}"
    else:
        try:
            scratchpad_analysis_input = scratchpad_content if scratchpad_content else "[No scratchpad analysis provided or parsed from the first step]"
            prompt_for_outcome_ids = prompt_template_outcome_ids.format(
                scenario=scenario_text_for_prompt,
                transcript_with_ids=transcript_with_ids,
                scratchpad_analysis=scratchpad_analysis_input
            )
            messages_outcome_ids = [{"role": "user", "content": prompt_for_outcome_ids}]
            logging.debug(f"Attempting async adjudication (outcome/IDs). Prompt: {prompt_for_outcome_ids[:500]}...")

            response_outcome_ids = await _execute_api_call_async( # Use await and async version
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
                logging.debug(f"Raw LLM output for async adjudication (outcome/IDs):\n{raw_output_outcome_ids}")

                if raw_output_outcome_ids:
                    raw_outcome_tag_content = _extract_xml_tag_content("outcome", raw_output_outcome_ids)
                    win_message_id_temp = _extract_xml_tag_content("win_message_id", raw_output_outcome_ids)
                    if not win_message_id_temp:
                        win_message_id_temp = _extract_xml_tag_content("message_id_win", raw_output_outcome_ids)
                    lose_message_id_temp = _extract_xml_tag_content("lose_message_id", raw_output_outcome_ids)
                    if not lose_message_id_temp:
                        lose_message_id_temp = _extract_xml_tag_content("message_id_loss", raw_output_outcome_ids)

                    valid_msg_id_pattern = r"^[AB][1-3]$"
                    is_win_msg_id_valid = bool(win_message_id_temp and re.match(valid_msg_id_pattern, win_message_id_temp))
                    is_lose_msg_id_valid = bool(lose_message_id_temp and re.match(valid_msg_id_pattern, lose_message_id_temp))

                    if not raw_outcome_tag_content:
                        logging.warning(f"Could not parse <outcome> tag from async adjudicator's second response. Raw: {raw_output_outcome_ids[:300]}.")
                    else:
                        outcome_text_normalized = raw_outcome_tag_content.strip().lower()
                        if outcome_text_normalized == "role a wins" or outcome_text_normalized == "role a":
                            final_parsed_outcome = "Role A Wins"
                        elif outcome_text_normalized == "role b wins" or outcome_text_normalized == "role b":
                            final_parsed_outcome = "Role B Wins"
                        elif outcome_text_normalized == "tie":
                            final_parsed_outcome = "Tie"
                        else:
                            logging.warning(f"Non-canonical content in <outcome> (async): '{raw_outcome_tag_content}'.")
                    
                    if final_parsed_outcome == "error" or \
                       (raw_outcome_tag_content and final_parsed_outcome not in ["Role A Wins", "Role B Wins", "Tie"]):
                        if is_win_msg_id_valid:
                            if win_message_id_temp.startswith("A"): final_parsed_outcome = "Role A Wins"
                            elif win_message_id_temp.startswith("B"): final_parsed_outcome = "Role B Wins"
                            if final_parsed_outcome != "error": logging.info(f"Outcome set via win_message_id (async): {final_parsed_outcome}")
                        else:
                            logging.warning(f"Outcome unclear and no valid win_message_id (async). Original tag: '{raw_outcome_tag_content}'.")

                    if final_parsed_outcome in ["Role A Wins", "Role B Wins"]:
                        win_message_id = win_message_id_temp if is_win_msg_id_valid else None
                        lose_message_id = lose_message_id_temp if is_lose_msg_id_valid else None
                        if not is_win_msg_id_valid: logging.warning(f"Outcome '{final_parsed_outcome}' but win_id '{win_message_id_temp}' invalid (async).")
                        if not is_lose_msg_id_valid: logging.warning(f"Outcome '{final_parsed_outcome}' but lose_id '{lose_message_id_temp}' invalid (async).")
                    else: # Tie or error
                        win_message_id = None
                        lose_message_id = None
                else:
                    logging.warning("Empty content from async API for adjudication (outcome/IDs).")
            else:
                logging.error("Invalid response structure from async API for adjudication (outcome/IDs).")
        except TimeoutError:
            logging.error("Timeout during async adjudication (outcome/IDs) after all retries.")
        except goodfire_exceptions.GoodfireBaseException as e:
            logging.error(f"Goodfire API error during async adjudication (outcome/IDs): {e}", exc_info=True)
        except KeyError as ke:
            logging.error(f"Missing key '{ke}' in adjudicator prompt_template_outcome_ids (async).")
        except Exception as e:
            logging.critical(f"Unexpected critical error during async adjudication (outcome/IDs): {e}", exc_info=True)

    logging.info(f"Async adjudication result: Outcome='{final_parsed_outcome}', WinMsg='{win_message_id}', LoseMsg='{lose_message_id}'.")
    return (final_parsed_outcome, win_message_id, lose_message_id, scratchpad_content,
            prompt_for_scratchpad, raw_output_scratchpad,
            prompt_for_outcome_ids, raw_output_outcome_ids)
