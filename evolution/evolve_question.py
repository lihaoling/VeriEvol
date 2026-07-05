#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolve questions in JSONL data by calling the MLLM API, turning simple questions into more complex and challenging ones
Uses approach one of vision_api (VisionAPIClient and call_vision_api)

Error handling and resume support:
- Failed records (API call failure or invalid returned content) are not saved to the output file
- The program automatically skips records already processed successfully in the output file (based on record ID)
- Re-running the program automatically reprocesses previously failed records

Concurrent processing:
- By default uses 10 threads to process API calls concurrently, improving throughput
- The number of concurrent threads can be adjusted via the --max-workers parameter
- Setting it to 0 or 1 disables concurrency and uses sequential processing

Usage examples:
    # Basic usage
    python evolve_question.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 http://<VLLM_ENDPOINT>/v1 \
        --api-key sk-abc123 \
        --model-name qwen_3

    # Resume (start from the 100th record, based on line number)
    python evolve_question.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --start-index 100

    # Process only the first 50 records
    python evolve_question.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --max-items 50

    # Re-run to process failed records (automatically skips already successful ones)
    python evolve_question.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1

    # Use concurrent processing (default 10 threads)
    python evolve_question.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --max-workers 20

    # Disable concurrency (sequential processing)
    python evolve_question.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --max-workers 1
"""

import json
import sys
import argparse
import threading
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from vision_api import VisionAPIClient, call_vision_api
from prompt import (
    OCR_prompt,
    Image_Description_prompt,
    Detection_prompt,
    Analysis_prompt,
    Content_Creation_prompt,
    Suggestion_prompt,
    Summarization_prompt,
    Logical_Reasoning_prompt,
    Scientific_Related_prompt,
    Concept_Extraction_prompt,
    Medical_Image_Analysis_prompt,
    Scene_Understanding_prompt
)

# Default prompt template for question evolution (used when there is no topic or the topic does not match)
EVOLUTION_PROMPT = """You are an AI expert specializing in visual-linguistic data augmentation. Your mission is to transform simple, descriptive questions about images into complex, challenging ones that test the advanced capabilities of next-generation AI models.

You will be provided with an image and a simple, existing question-answer pair. Your task is to generate a more new, sophisticated question-answer pairs based on the same image.

//-- GUIDING PRINCIPLES FOR QUESTION EVOLUTION --//

Your new questions must adhere to one or more of the following principles to increase their difficulty and value:

### Require Multi-Step Reasoning: The question should demand a sequence of cognitive steps. For example: locate multiple objects, compare their attributes, and then form a conclusion.

### Incorporate Common Sense & World Knowledge: The question should require the model to connect visual cues to general knowledge about the world (e.g., physics, social norms, typical causalities).

### Probe Deeper Reasoning (Why/How/What if): Go beyond simple identification ("What is...") to inferential and counterfactual reasoning.

### Synthesize Information Across the Image: The question should not be answerable by looking at a single, isolated object. It must require integrating details from different parts of the image.

//-- STRICT OUTPUT REQUIREMENTS --//

1. Consistency: Maintain the number of sub-questions in both new and original questions. Ensure consistency between subjective and objective questions in both new and original questions.
2. Visual Dependency: The question must be unanswerable without the image. The visual details are critical, not incidental.
3. Style: New questions should adopt diverse styles such as colloquial or role-playing formats, and must avoid rigid, formulaic questioning.
4. Format: Your output MUST be a valid JSON object with the following structure. Do not include any other text, explanations, markdown code blocks, or formatting outside of the JSON object.

Required JSON format:
{{
  "evol_question": "your evolved question here"
}}

Important:
- Output ONLY the JSON object, nothing else
- The JSON must be valid and parseable
- The "evol_question" field must contain the evolved question as a string
- Do not wrap the JSON in markdown code blocks (```json or ```)
- Do not add any explanatory text before or after the JSON

//-- YOUR TASK BEGINS NOW --//

INPUT:

Image: [Image data will be provided]

Original Question: "{original_question}"

YOUR OUTPUT:

"""

# Mapping from Topic to Prompt
TOPIC_TO_PROMPT = {
    "OCR": OCR_prompt,
    "Image Description": Image_Description_prompt,
    "Detection": Detection_prompt,
    "Analysis": Analysis_prompt,
    "Content Creation": Content_Creation_prompt,
    "Suggestions": Suggestion_prompt,
    "Summarization": Summarization_prompt,
    "Logical Reasoning": Logical_Reasoning_prompt,
    "Science-Related": Scientific_Related_prompt,
    "Concept Extraction": Concept_Extraction_prompt,
    "Medical Imaging Analysis": Medical_Image_Analysis_prompt,
    "Scene Understanding": Scene_Understanding_prompt,
}


def extract_original_question(conversations: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract the original question from the conversation (human's value, stripped of the <image>\n prefix)

    Args:
        conversations: list of conversation turns

    Returns:
        The original question text, or None if not found
    """
    for conv in conversations:
        if conv.get("from") == "human":
            value = conv.get("value", "")
            # Strip the <image>\n prefix (if present)
            if value.startswith("<image>\n"):
                value = value[8:]  # strip "<image>\n" (8 characters)
            elif value.startswith("<image>"):
                value = value[7:]  # strip "<image>" (7 characters)
            # Strip leading and trailing whitespace
            value = value.strip()
            return value if value else None
    return None


def build_evolution_prompt(original_question: str, topic: Optional[str] = None, use_topic_prompt: bool = True, question_type: Optional[str] = None) -> str:
    """
    Build the prompt for question evolution

    Args:
        original_question: the original question text
        topic: topic category (optional); if provided and matched, the corresponding dedicated prompt is used
        use_topic_prompt: whether to use the dedicated prompt for the topic (default True)
        question_type: question type ("objective" or "subjective"); if provided, the prompt emphasizes keeping the type unchanged

    Returns:
        The prompt text for the evolution task
    """
    # If topic prompt is enabled and a topic is provided and present in the mapping, use the corresponding prompt
    if use_topic_prompt and topic and topic in TOPIC_TO_PROMPT:
        prompt_template = TOPIC_TO_PROMPT[topic]
        logger.debug(f"Using the dedicated prompt for topic '{topic}'")
    else:
        # Use the default prompt
        prompt_template = EVOLUTION_PROMPT
        if not use_topic_prompt:
            logger.debug("Topic prompt disabled, using the default prompt")
        elif topic:
            logger.debug(f"Unknown topic: {topic}, using the default prompt")
        else:
            logger.debug("No topic provided, using the default prompt")

    # Format the prompt, replacing placeholders
    # Note: prompts in prompt.py use {{original_question}} and {{question_type}} as placeholders
    # They need to be replaced with the actual values

    # Replace the original_question placeholder (handle both quoted and unquoted cases)
    prompt = prompt_template.replace(
        '"{{original_question}}"', f'"{original_question}"')
    prompt = prompt.replace("{{original_question}}", original_question)
    # Also handle the case without double curly braces (if some prompts use single braces)
    prompt = prompt.replace("{original_question}", original_question)

    # Replace the question_type placeholder (if a question type is provided)
    if question_type:
        prompt = prompt.replace("{{question_type}}", question_type)
        logger.debug(f"Replaced question_type placeholder with: {question_type}")
    else:
        # If no question_type is provided, remove lines containing {{question_type}}
        # This avoids leaving an unreplaced placeholder in the prompt
        lines = prompt.split('\n')
        prompt = '\n'.join(
            [line for line in lines if '{{question_type}}' not in line])

    return prompt


def evolve_single_record(
    api_client: VisionAPIClient,
    record: Dict[str, Any],
    max_retries: int = 3,
    use_topic_prompt: bool = True
) -> Optional[str]:
    """
    Evolve the question of a single record

    Args:
        api_client: VisionAPIClient instance
        record: a single JSONL record
        max_retries: maximum number of retries
        use_topic_prompt: whether to use the dedicated prompt for the topic (default True)

    Returns:
        The evolved question text, or None on failure
    """
    # Extract image information
    image_info = record.get("image")
    if not image_info:
        logger.warning(f"Record {record.get('id', 'unknown')} is missing the image field")
        return None

    # Extract conversation information
    conversations = record.get("conversations", [])
    if not conversations:
        logger.warning(f"Record {record.get('id', 'unknown')} is missing the conversations field")
        return None

    # Extract the original question
    original_question = extract_original_question(conversations)
    if not original_question:
        logger.warning(f"Record {record.get('id', 'unknown')} has no human question")
        return None

    # Extract topic (if present)
    topic = record.get("topic") if use_topic_prompt else None

    # Extract question type (if present)
    question_type = record.get("question_type")
    if question_type:
        logger.debug(f"Record {record.get('id', 'unknown')} question type: {question_type}")

    # Build the evolution prompt (choose the prompt based on topic, and consider the question type)
    if topic:
        logger.debug(f"Record {record.get('id', 'unknown')} using topic: {topic}")
    question = build_evolution_prompt(
        original_question, topic=topic, use_topic_prompt=use_topic_prompt, question_type=question_type)

    # Call the API
    for attempt in range(max_retries):
        try:
            response = call_vision_api(
                api_client=api_client,
                image=image_info,
                question=question
            )

            if response:
                # Clean the response and extract the new question
                evolved_question = clean_evolved_question(response)
                if evolved_question:
                    return evolved_question
                else:
                    logger.warning(
                        f"Record {record.get('id', 'unknown')} could not extract a question from the response: {response[:100] if len(str(response)) > 100 else response}"
                    )
                    # If the response is non-empty but no question can be extracted, do not retry (may be a model format issue)
                    return None
            else:
                # response is None, meaning the API call failed
                logger.warning(
                    f"Record {record.get('id', 'unknown')} API call returned None (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Record {record.get('id', 'unknown')} question evolution failed: API call returned None")
                    return None

        except Exception as e:
            logger.warning(
                f"Record {record.get('id', 'unknown')} API call exception (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"Record {record.get('id', 'unknown')} question evolution failed: {e}")
                return None

    return None


def clean_evolved_question(response: str) -> Optional[str]:
    """
    Clean the model response and extract the evolved question
    Prefer parsing JSON format; if that fails, fall back to text extraction logic

    Args:
        response: the response text returned by the model

    Returns:
        The cleaned question text, or None if extraction is not possible
    """
    if not response:
        return None

    # Strip leading and trailing whitespace
    response = response.strip()
    if not response:
        return None

    # Method 1: Try to extract and parse JSON (preferred method)
    # First, try to extract JSON from a markdown code block
    json_match = re.search(
        r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            data = json.loads(json_str)
            # Prefer looking for the evol_question field
            if isinstance(data, dict) and 'evol_question' in data:
                question = str(data['evol_question']).strip()
                if question:
                    return question
        except json.JSONDecodeError:
            pass

    # Method 2: Try to parse the entire response as JSON directly
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            # Look for the question field in priority order
            question_fields = ['evol_question', 'question',
                               'new_question', 'evolved_question', 'output']
            for field in question_fields:
                if field in data and data[field]:
                    question = str(data[field]).strip()
                    if question:
                        return question
        elif isinstance(data, str):
            # If the JSON itself is a string, return it directly
            return data.strip()
    except json.JSONDecodeError:
        pass

    # Method 3: Try to extract the first complete JSON object (handles cases where there may be other text before/after the JSON)
    # Find the first { and its matching }
    if '{' in response:
        brace_count = 0
        json_start = -1
        json_end = -1

        for i, char in enumerate(response):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start >= 0:
                    json_end = i + 1
                    break

        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    question_fields = ['evol_question', 'question',
                                       'new_question', 'evolved_question', 'output']
                    for field in question_fields:
                        if field in data and data[field]:
                            question = str(data[field]).strip()
                            if question:
                                return question
            except json.JSONDecodeError:
                pass

    # Method 4: Fallback - text extraction logic (if all JSON parsing fails)
    lines = response.split('\n')
    first_line = lines[0].strip() if lines else ""

    # Strip possible quotes
    first_line = first_line.strip('"').strip("'").strip()

    # Strip possible prompt prefixes
    prefixes_to_remove = [
        "question:",
        "new question:",
        "evolved question:",
        "your output:",
        "output:",
        "evol_question:",
    ]
    for prefix in prefixes_to_remove:
        if first_line.lower().startswith(prefix):
            first_line = first_line[len(prefix):].strip()
            break

    # If the first line is non-empty and not a lone bracket, return it
    if first_line and first_line not in ['{', '}', '[', ']']:
        return first_line

    # If the first line is empty or just a bracket, try the entire response
    cleaned_response = response.strip().strip('"').strip("'").strip()
    if cleaned_response and cleaned_response not in ['{', '}', '[', ']']:
        return cleaned_response

    return None


def update_record_with_evolved_question(
    record: Dict[str, Any],
    evolved_question: str
) -> Dict[str, Any]:
    """
    Update the record by adding the evolved question to the evol_question field (without modifying the existing conversations)

    Args:
        record: the original record
        evolved_question: the evolved question

    Returns:
        The updated record
    """
    # Deep-copy the record
    updated_record = json.loads(json.dumps(record))

    # Add the evol_question field (without modifying the existing conversations)
    updated_record["evol_question"] = evolved_question

    return updated_record


def get_processed_ids(output_file: str) -> set:
    """
    Read the IDs of records already successfully processed from the output file (records that have the evol_question field)

    Args:
        output_file: output file path

    Returns:
        The set of IDs of already-processed records
    """
    processed_ids = set()
    output_path = Path(output_file)

    if not output_path.exists():
        return processed_ids

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Check for the evol_question field (we add this field when saving)
                    if "evol_question" in data and data["evol_question"]:
                        record_id = data.get("id") or data.get("berlin_id")
                        if record_id:
                            processed_ids.add(str(record_id))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Error reading the output file: {e}")

    return processed_ids


def process_single_record_for_concurrent(
    record_data: tuple,
    api_client: VisionAPIClient,
    processed_ids: set,
    output_file: str,
    output_lock: threading.Lock,
    skip_processed: bool,
    verbose: bool,
    use_topic_prompt: bool = True
) -> Dict[str, Any]:
    """
    Process a single record concurrently (for the thread pool)

    Args:
        record_data: (line_num, data, record_id) tuple
        api_client: VisionAPIClient instance
        processed_ids: set of already-processed record IDs
        output_file: output file path
        output_lock: file write lock
        skip_processed: whether to skip already-processed records
        verbose: whether to print detailed information
        use_topic_prompt: whether to use the dedicated prompt for the topic (default True)

    Returns:
        A result dictionary
    """
    line_num, data, record_id = record_data

    result = {
        "record_id": record_id,
        "line_num": line_num,
        "success": False,
        "evolved_question": None,
        "skipped": False,
        "failed": False
    }

    # Check whether it has already been processed
    if skip_processed and record_id in processed_ids:
        result["skipped"] = True
        return result

    # Perform question evolution
    evolved_question = evolve_single_record(
        api_client, data, use_topic_prompt=use_topic_prompt)

    if evolved_question:
        # Evolution succeeded
        updated_record = update_record_with_evolved_question(
            data, evolved_question)

        result["success"] = True
        result["evolved_question"] = evolved_question

        # Write to the file in a thread-safe manner
        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(updated_record, ensure_ascii=False) + '\n')
    else:
        # Evolution failed
        result["failed"] = True
        if verbose:
            logger.warning(f"Record {record_id} question evolution failed, skipping without saving")

    return result


def process_jsonl_file(
    input_file: str,
    output_file: str,
    api_client: VisionAPIClient,
    start_index: int = 0,
    max_items: Optional[int] = None,
    verbose: bool = True,
    skip_processed: bool = True,
    max_workers: int = 10,
    use_topic_prompt: bool = True
) -> Dict[str, Any]:
    """
    Process a JSONL file, evolving the question of each record

    Args:
        input_file: input JSONL file path
        output_file: output JSONL file path
        api_client: VisionAPIClient instance
        start_index: which record to start processing from (for resume, based on line number)
        max_items: maximum number of records to process (None means process all)
        verbose: whether to print detailed information
        skip_processed: whether to automatically skip already successfully processed records (based on ID)
        max_workers: number of concurrent threads (0 or 1 disables concurrency)
        use_topic_prompt: whether to use the dedicated prompt for the topic (default True)

    Returns:
        A statistics dictionary
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the IDs of already-processed records (for resume)
    processed_ids = set()
    if skip_processed and output_path.exists():
        processed_ids = get_processed_ids(output_file)
        if verbose and processed_ids:
            logger.info(f"Found {len(processed_ids)} already successfully processed records, will skip them automatically")

    stats = {
        "total_lines": 0,
        "processed_lines": 0,
        "successful_evolutions": 0,
        "failed_evolutions": 0,
        "skipped_lines": 0,
        "already_processed": 0,
    }

    if verbose:
        logger.info(f"Starting to process file: {input_file}")
        logger.info(f"Output file: {output_file}")
        if start_index > 0:
            logger.info(f"Starting from record {start_index + 1} (based on line number)")
        if max_items:
            logger.info(f"Processing at most {max_items} records")
        if skip_processed:
            logger.info(f"Automatically skipping already successfully processed records (based on ID)")
        if max_workers and max_workers > 1:
            logger.info(f"Using concurrent processing, threads: {max_workers}")
        else:
            logger.info(f"Using sequential processing")
        if use_topic_prompt:
            logger.info(f"Topic prompt enabled (will select the dedicated prompt based on the record's topic field)")
        else:
            logger.info(f"Topic prompt disabled (will use the default prompt)")

    # Initialize the output file (if it's a new file, clear it; if in append mode, keep existing content)
    output_mode = 'a' if output_path.exists() and skip_processed else 'w'
    if output_mode == 'w':
        output_path.write_text('', encoding='utf-8')

    # File write lock (for concurrent mode)
    output_lock = threading.Lock()

    try:
        # Step 1: read all records that need to be processed
        records_to_process = []
        processed_count = 0

        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                stats["total_lines"] += 1

                # Skip the preceding records (line-number-based resume)
                if line_num <= start_index:
                    stats["skipped_lines"] += 1
                    continue

                # Check whether the maximum number of records has been reached
                if max_items and processed_count >= max_items:
                    break

                try:
                    # Parse the JSON line
                    data = json.loads(line)

                    if not isinstance(data, dict):
                        if verbose:
                            logger.warning(f"Line {line_num} is not in dictionary format, skipping")
                        stats["skipped_lines"] += 1
                        continue

                    record_id = str(data.get("id") or data.get(
                        "berlin_id") or line_num)

                    # Check whether it has already been processed
                    if skip_processed and record_id in processed_ids:
                        stats["already_processed"] += 1
                        continue

                    # Check whether there is a human question
                    original_question = extract_original_question(
                        data.get("conversations", []))
                    if not original_question:
                        if verbose:
                            logger.warning(f"Record {record_id} has no human question, skipping")
                        stats["skipped_lines"] += 1
                        continue

                    # Add to the to-process list
                    records_to_process.append((line_num, data, record_id))
                    processed_count += 1

                except json.JSONDecodeError as e:
                    if verbose:
                        logger.error(f"Line {line_num} JSON parsing failed: {e}")
                    stats["skipped_lines"] += 1
                    continue
                except Exception as e:
                    if verbose:
                        logger.error(f"Line {line_num} processing failed: {e}")
                    stats["skipped_lines"] += 1
                    continue

        # Step 2: process records concurrently
        if not records_to_process:
            if verbose:
                logger.info("No records to process")
        elif max_workers and max_workers > 1:
            # Use concurrent processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_record = {
                    executor.submit(
                        process_single_record_for_concurrent,
                        record_data,
                        api_client,
                        processed_ids,
                        output_file,
                        output_lock,
                        skip_processed,
                        verbose,
                        use_topic_prompt
                    ): record_data
                    for record_data in records_to_process
                }

                # Use a progress bar to show progress
                with tqdm(total=len(records_to_process), desc="Progress", disable=not verbose) as pbar:
                    for future in as_completed(future_to_record):
                        try:
                            result = future.result()
                            record_data = future_to_record[future]

                            if result["skipped"]:
                                stats["already_processed"] += 1
                            elif result["success"]:
                                stats["processed_lines"] += 1
                                stats["successful_evolutions"] += 1
                                if verbose:
                                    logger.info(
                                        f"Record {result['record_id']} question evolution succeeded")
                            else:
                                stats["failed_evolutions"] += 1

                            pbar.update(1)
                        except Exception as e:
                            if verbose:
                                logger.error(f"Error while processing record: {e}")
                            stats["failed_evolutions"] += 1
                            pbar.update(1)
        else:
            # Sequential processing
            for record_data in tqdm(records_to_process, desc="Progress", disable=not verbose):
                result = process_single_record_for_concurrent(
                    record_data,
                    api_client,
                    processed_ids,
                    output_file,
                    output_lock,
                    skip_processed,
                    verbose,
                    use_topic_prompt
                )

                if result["skipped"]:
                    stats["already_processed"] += 1
                elif result["success"]:
                    stats["processed_lines"] += 1
                    stats["successful_evolutions"] += 1
                    if verbose:
                        logger.info(f"Record {result['record_id']} question evolution succeeded")
                else:
                    stats["failed_evolutions"] += 1

        # Print the statistics
        if verbose:
            logger.info("\n" + "="*60)
            logger.info("Processing complete! Statistics:")
            logger.info("="*60)
            logger.info(f"Total lines: {stats['total_lines']}")
            logger.info(f"Processed lines: {stats['processed_lines']}")
            logger.info(f"Successful evolutions: {stats['successful_evolutions']}")
            logger.info(f"Failed evolutions: {stats['failed_evolutions']}")
            logger.info(f"Skipped lines: {stats['skipped_lines']}")
            if stats.get('already_processed', 0) > 0:
                logger.info(f"Already processed (skipped): {stats['already_processed']}")

            logger.info("="*60)
            logger.info(f"Output file saved to: {output_file}")
            if stats['failed_evolutions'] > 0:
                logger.warning(
                    f"Note: {stats['failed_evolutions']} records failed question evolution "
                    f"and were not saved to the output file. You can re-run the program to process these records."
                )

        return stats

    except Exception as e:
        logger.error(f"An error occurred while processing the file: {e}")
        raise


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description='Evolve questions in JSONL data by calling the MLLM API'
    )
    parser.add_argument('input', help='input JSONL file path')
    parser.add_argument('-o', '--output', required=True, help='output JSONL file path')
    parser.add_argument('--api-endpoints', nargs='+', required=True,
                        help='list of API endpoints, e.g.: http://<VLLM_ENDPOINT>/v1 http://<VLLM_ENDPOINT>/v1')
    parser.add_argument('--api-key', default='sk-abc123', help='API key')
    parser.add_argument('--model-name', default='qwen_3', help='model name')
    parser.add_argument('--temperature', type=float, default=0.7, help='temperature parameter')
    parser.add_argument('--max-tokens', type=int,
                        default=4096, help='maximum number of generated tokens')
    parser.add_argument('--client-selection', choices=['random', 'round_robin', 'localhost'],
                        default='random', help='client selection strategy')
    parser.add_argument('--start-index', type=int, default=0,
                        help='which record to start processing from (for resume, counting from 0, based on line number)')
    parser.add_argument('--max-items', type=int, default=None,
                        help='maximum number of records to process (None means process all)')
    parser.add_argument('--no-skip-processed', action='store_true',
                        help='disable automatically skipping already-processed records (enabled by default, based on record ID)')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='number of concurrent threads (default 10, set to 0 or 1 to disable concurrency)')
    parser.add_argument('--no-use-topic-prompt', action='store_true',
                        help='disable topic prompt (enabled by default, will select the dedicated prompt based on the record\'s topic field)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='quiet mode, do not print detailed information')

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Show the actual received max_workers parameter value (for debugging)
    logger.info(f"Concurrent threads parameter: {args.max_workers}")

    try:
        # Create the API client
        api_client = VisionAPIClient(
            api_endpoints=args.api_endpoints,
            api_key=args.api_key,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            client_selection=args.client_selection
        )

        # Process the file
        stats = process_jsonl_file(
            input_file=args.input,
            output_file=args.output,
            api_client=api_client,
            start_index=args.start_index,
            max_items=args.max_items,
            verbose=not args.quiet,
            skip_processed=not args.no_skip_processed,
            max_workers=args.max_workers,
            use_topic_prompt=not args.no_use_topic_prompt
        )

        # Return the status code
        sys.exit(0 if stats["failed_evolutions"] == 0 else 1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
