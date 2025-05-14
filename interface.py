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
    prompt_text_template = gen_config.get('prompt', "Error: Scenario generation prompt template not found in config.")
    max_tokens = gen_config.get('max_tokens', 1000)
    temperature = gen_config.get('temperature', 0.7)
    api_retry_config = config.get('api_retries', {})

    diversification_seed = f"agent_id:{proposer_agent.agent_id}_call_id:{uuid.uuid4().hex[:8]}"
    final_prompt_text = f"{prompt_text_template}\n[{diversification_seed}]"

    variant = goodfire.Variant(model_id)
    variant_edits = proposer_agent.genome
    if isinstance(variant_edits, dict) and variant_edits:
        try:
            variant.set(variant_edits)
        except Exception as e:
            logging.error(f"Error setting variant edits for agent {proposer_agent.agent_id} in generate_scenario: {e}", exc_info=True)

    messages = [{"role": "user", "content": final_prompt_text}]
    llm_raw_output_text = None 

    # Define expected tags locally. XML tags are case-sensitive.
    expected_tag_sequence = ["context", "roles", "objectives", "win_criteria", "tie_criteria", "proposer_role"]

    try:
        logging.debug(f"Attempting scenario generation for agent {proposer_agent.agent_id} with prompt ending: ...{final_prompt_text[-100:]}")
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
             logging.error(f"Invalid or empty response structure received from API for scenario generation (Agent: {proposer_agent.agent_id}).")
             return None, final_prompt_text

        llm_raw_output_text = response.choices[0].message.get('content', '')
        logging.debug(f"Raw LLM output for scenario (Agent {proposer_agent.agent_id}):\n{llm_raw_output_text}")

        if not llm_raw_output_text.strip():
             logging.warning(f"Empty content received from API for scenario generation (Agent: {proposer_agent.agent_id}).")
             return None, final_prompt_text

        content_to_parse = llm_raw_output_text.strip()

        # Fix common LLM quirk: extraneous periods before tags
        for tag_name_to_sanitize in expected_tag_sequence:
            # Period at line start: . <tag>
            content_to_parse = re.sub(rf"^\s*\.\s*(<{tag_name_to_sanitize}>)", r"\1", content_to_parse, flags=re.MULTILINE)
            # Period after a previous tag: </prev_tag> . <tag>
            content_to_parse = re.sub(rf"(>\s*)\.\s*(<{tag_name_to_sanitize}>)", r"\1\2", content_to_parse)
        
        # Try to isolate the main block of XML from the first expected tag to the last.
        # This discards prefix/suffix junk outside the core tag sequence.
        first_tag_opener_str = f"<{expected_tag_sequence[0]}>"
        last_tag_closer_str = f"</{expected_tag_sequence[-1]}>"

        # Find the first occurrence of the first tag's opener
        first_tag_match = re.search(re.escape(first_tag_opener_str), content_to_parse)
        start_slice_index = 0
        if first_tag_match:
            start_slice_index = first_tag_match.start()
        else:
            logging.warning(f"First expected tag '{first_tag_opener_str}' not found in content by sanitizer. Parsing will proceed on dot-fixed content. Agent: {proposer_agent.agent_id}.")
            # No change to start_slice_index, will parse from beginning of dot-fixed string.

        # Find the last occurrence of the last tag's closer
        # We search in the full string (after dot fixing) to find the true end of the block.
        end_slice_index = len(content_to_parse)
        # Find all matches and take the one that ends latest.
        last_tag_matches = list(re.finditer(re.escape(last_tag_closer_str), content_to_parse))
        if last_tag_matches:
            # Get the match object that has the maximum end position
            final_match = max(last_tag_matches, key=lambda m: m.end())
            end_slice_index = final_match.end()
        else:
            logging.warning(f"Last expected tag '{last_tag_closer_str}' not found in content by sanitizer. Parsing will proceed on potentially un-truncated content. Agent: {proposer_agent.agent_id}.")
            # No change to end_slice_index, will parse up to end of dot-fixed string.
        
        # Slice the content to the identified block
        content_to_parse = content_to_parse[start_slice_index:end_slice_index].strip()

        if not content_to_parse:
            logging.warning(f"Content for agent {proposer_agent.agent_id} became empty after sanitization/isolation. Original raw: '{llm_raw_output_text[:200]}...'")
            return None, final_prompt_text
        
        if content_to_parse != llm_raw_output_text.strip() and \
           (llm_raw_output_text.strip().startswith(".") or not (llm_raw_output_text.strip().startswith(first_tag_opener_str) and llm_raw_output_text.strip().endswith(last_tag_closer_str))):
            # Log if sanitization made a substantive change (beyond simple stripping)
             logging.info(f"LLM output was pre-processed/sanitized for agent {proposer_agent.agent_id}. Using: '{content_to_parse[:100]}...' to '{content_to_parse[-100:]}'")

        tags_content = {}
        current_parse_offset = 0 
        parsing_successful = True

        for tag_name in expected_tag_sequence:
            # Regex to find <tag_name>content</tag_name>:
            # - Allows optional leading whitespace.
            # - Allows an optional single "junk" character (not whitespace, word character, or '<'). This handles stray punctuation like periods.
            # - Allows optional whitespace after the junk char and before the tag.
            # - Matches the tag itself and its content non-greedily.
            # - Tags are case-sensitive. Content can span multiple lines (re.DOTALL).
            pattern_str = rf"\s*[^\s\w<]?\s*<{tag_name}>(.*?)</{tag_name}>"
            pattern = re.compile(pattern_str, re.DOTALL)
            match = pattern.search(content_to_parse, pos=current_parse_offset)
            
            if match:
                tags_content[tag_name] = match.group(1).strip()
                current_parse_offset = match.end() # Next search starts after this entire matched tag
            else:
                logging.warning(
                    f"Required tag <{tag_name}> not found in sequence in pre-processed LLM output "
                    f"(Agent: {proposer_agent.agent_id}). Search started at offset {current_parse_offset}. "
                    f"Relevant content snippet: '{content_to_parse[current_parse_offset : current_parse_offset+150]}...'"
                )
                parsing_successful = False
                break
        
        if not parsing_successful:
            logging.debug(f"Pre-processed content at time of parsing failure (Agent {proposer_agent.agent_id}):\n{content_to_parse}")
            return None, final_prompt_text
        
        # Check for significant text after all expected tags have been parsed from the isolated block.
        remaining_text_after_tags = content_to_parse[current_parse_offset:].strip()
        if remaining_text_after_tags:
            logging.warning(
                f"Extraneous content found within the isolated tag block after all expected tags were parsed (Agent: {proposer_agent.agent_id}). "
                f"Remaining: '{remaining_text_after_tags[:200]}...'. This might violate 'ONLY these tags' rule."
            )
            # For maximum leniency on trailing junk within the identified block, we don't fail here.

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
    except goodfire_exceptions.GoodfireError as e:
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
        A tuple: (str: adjudication outcome e.g., 'Role A Wins', 'Tie', 'error' (though 'error' becomes 'Tie' here in most cases),
                  str: The prompt_text used for adjudication or None on early failure)
    """

    if not all(isinstance(arg, exp_type) for arg, exp_type in [
        (scenario_data, dict), (transcript, list), (config, dict)
    ]):
        logging.error(f"adjudicate_interaction: Invalid argument types. Scenario: {type(scenario_data)}, Transcript: {type(transcript)}, Config: {type(config)}")
        return "Tie", None # Default to Tie on bad input types, log error

    client = get_goodfire_client(config)
    adjudicator_config = config.get('adjudicator', {})
    model_id = adjudicator_config.get('model_id')
    prompt_template = adjudicator_config.get('prompt_template',
        "Analyze the interaction based strictly on the rules and objectives defined in the scenario description.\n\nScenario Description:\n{scenario}\n\nInteraction Transcript:\n{transcript}\n\nTask: Based ONLY on the scenario's win criteria and the interaction, determine the outcome. Respond with ONLY ONE of the following exact phrases: 'Role A Wins', 'Role B Wins', or 'Tie'.")
    max_tokens = adjudicator_config.get('max_tokens', 20)
    temperature = adjudicator_config.get('temperature', 0.7)
    api_retry_config = config.get('api_retries', {})

    if not model_id:
        logging.error("Adjudicator model ID not specified in configuration. Cannot adjudicate. Defaulting to Tie.")
        return "Tie", None

    scenario_text = scenario_data.get('scenario_text', '[Scenario text missing from input scenario_data]')
    transcript_text = "\n".join([f"{msg.get('role', 'UnknownRole')}: {msg.get('content', '[empty_content]')}" for msg in transcript]) \
                      if transcript else "[No transcript available for adjudication]"

    adjudication_prompt_text = None
    try:
        adjudication_prompt_text = prompt_template.format(scenario=scenario_text, transcript=transcript_text)
    except KeyError as ke:
        logging.error(f"Missing key '{ke}' in prompt_template for adjudication. Template: '{prompt_template}'. Defaulting to Tie.")
        return "Tie", prompt_template # Return template itself if formatting fails

    messages = [{"role": "user", "content": adjudication_prompt_text}]
    raw_adjudicator_llm_output = "[Adjudicator LLM output not captured before potential error]"

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
             logging.error("Invalid or empty response structure received from adjudicator API. Defaulting to Tie.")
             return "Tie", adjudication_prompt_text

        raw_adjudicator_llm_output = response.choices[0].message.get('content', '')
        cleaned_output = raw_adjudicator_llm_output.strip()

        if not cleaned_output:
            logging.error(f"Adjudicator returned empty or whitespace-only output. Raw: '{raw_adjudicator_llm_output}'. Defaulting to Tie.")
            return "Tie", adjudication_prompt_text

        output_lower = cleaned_output.lower()

        # Define keywords for each outcome (case-insensitive)
        # Using lists of keywords for flexibility (e.g., "role a wins", "a wins")
        keywords_a_wins = ["role a wins", "a wins", "player a wins"]
        keywords_b_wins = ["role b wins", "b wins", "player b wins"]
        keywords_tie = ["tie", "draw", "stalemate"]

        # Check presence of keywords
        contains_a_wins = any(kw in output_lower for kw in keywords_a_wins)
        contains_b_wins = any(kw in output_lower for kw in keywords_b_wins)
        contains_tie = any(kw in output_lower for kw in keywords_tie)

        # Exclusive "contains one and not the others" logic
        if contains_a_wins and not contains_b_wins and not contains_tie:
            logging.info(f"Adjudication determined: 'Role A Wins'. Raw: '{raw_adjudicator_llm_output}'")
            return 'Role A Wins', adjudication_prompt_text
        elif contains_b_wins and not contains_a_wins and not contains_tie:
            logging.info(f"Adjudication determined: 'Role B Wins'. Raw: '{raw_adjudicator_llm_output}'")
            return 'Role B Wins', adjudication_prompt_text
        elif contains_tie and not contains_a_wins and not contains_b_wins:
            logging.info(f"Adjudication determined: 'Tie'. Raw: '{raw_adjudicator_llm_output}'")
            return 'Tie', adjudication_prompt_text
        else:
            # This case covers:
            # 1. Contains none of the keywords.
            # 2. Contains conflicting keywords (e.g., "a wins" and "tie").
            logging.warning(
                f"Adjudicator response is ambiguous or does not clearly indicate a single outcome. "
                f"Output: '{cleaned_output}' (Raw: '{raw_adjudicator_llm_output}'). "
                f"Detected flags: A_wins={contains_a_wins}, B_wins={contains_b_wins}, Tie={contains_tie}. Defaulting to Tie."
            )
            return "Tie", adjudication_prompt_text

    except TimeoutError:
         logging.error(f"Timeout during adjudication after all retries. Defaulting to Tie. Raw LLM output (if captured): '{raw_adjudicator_llm_output}'")
         return "Tie", adjudication_prompt_text
    except goodfire_exceptions.GoodfireError as e:
        logging.error(f"Goodfire API error during adjudication: {e}. Defaulting to Tie. Raw LLM output (if captured): '{raw_adjudicator_llm_output}'", exc_info=True)
        return "Tie", adjudication_prompt_text
    except Exception as e:
        logging.critical(f"Unexpected critical error during adjudication: {e}. Defaulting to Tie. Raw LLM output (if captured): '{raw_adjudicator_llm_output}'", exc_info=True)
        return "Tie", adjudication_prompt_text

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
