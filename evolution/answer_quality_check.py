#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform dual-verification quality evaluation of answers by calling the MLLM API.
Uses an LLM-as-a-judge approach, judging from two directions:
1. Whether the model's answer has errors
2. Where the errors are

Error handling and resume support:
- Failed records (API call failed or returned invalid content) are not saved to the output file
- The program automatically skips records already successfully processed in the output file (based on record ID)
- Re-running the program automatically processes previously failed records

Concurrent processing:
- By default uses 10 threads to process API calls concurrently, improving throughput
- The number of concurrent threads can be adjusted via the --max-workers argument
- Setting it to 0 or 1 disables concurrency and uses sequential processing

Usage example:
    # Basic usage
    python answer_quality_check.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --api-key sk-abc123 \
        --model-name qwen_3
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

# Dual-verification prompt template
DUAL_VERIFICATION_PROMPT = """# ROLE:
You are an expert AI Answer Quality Evaluator. Your task is to perform dual verification on a model's answer to a visual question.

# GOAL:
You will be given an `Image`, an `Evolved Question`, the model's `Reasoning Content`, and the model's `Final Answer`. 
You must evaluate the answer from TWO directions:
1. **Correctness Check**: Is the answer correct? What are the strengths?
2. **Error Detection**: Are there any errors? Where are the mistakes?

# EVALUATION CRITERIA:

## Direction 1: Correctness Check
- **Factual Accuracy**: Does the answer correctly describe what is shown in the image?
- **Completeness**: Does the answer fully address all aspects of the question?
- **Logical Consistency**: Is the reasoning process logical and coherent?
- **Image Consistency**: Does the answer align with the visual content?

## Direction 2: Error Detection
- **Factual Errors**: Are there any incorrect facts or descriptions?
- **Logical Errors**: Are there flaws in the reasoning process?
- **Missing Information**: Is any important information omitted?
- **Contradictions**: Are there any internal contradictions in the answer?
- **Hallucinations**: Does the answer contain information not present in the image?

# INPUTS:
*   `Image`: [Image will be provided here]
*   `Evolved Question`: [The question that was asked]
*   `Reasoning Content`: [The model's reasoning process]
*   `Final Answer`: [The model's final answer content]

# OUTPUT FORMAT:
You MUST provide your response in the following structured format:

**Overall Assessment:** [CORRECT / PARTIALLY_CORRECT / INCORRECT]

**Direction 1 - Correctness Analysis:**
- **Is Correct:** [YES / NO / PARTIAL]
- **Strengths:**
  * [List the main strengths of the answer]
  * [What aspects are well-answered?]
  * [What reasoning steps are sound?]

**Direction 2 - Error Detection:**
- **Has Errors:** [YES / NO]
- **Errors Found:**
  * [List specific errors found, if any]
  * [Point out where the mistakes are]
  * [Identify any hallucinations or contradictions]
- **Error Severity:** [NONE / MINOR / MODERATE / SEVERE]

**Detailed Analysis:**
[Provide a comprehensive analysis combining both directions, explaining your overall assessment]
"""


def extract_original_question(conversations: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract the original question (the human's question) from conversations

    Args:
        conversations: list of conversation turns

    Returns:
        The original question string, or None if not found
    """
    for conv in conversations:
        if conv.get('from') == 'human':
            value = conv.get('value', '')
            # Remove any possible <image> tags
            value = value.replace('<image>', '').replace(
                '<image>\n', '').strip()
            if value:
                return value
    return None


def build_dual_verification_prompt(
    evol_question: str,
    reasoning_content: str,
    content: str
) -> str:
    """
    Build the prompt for dual verification

    Args:
        evol_question: the evolved question
        reasoning_content: the model's reasoning content
        content: the model's final answer content

    Returns:
        The prompt text for the dual-verification task
    """
    prompt = f"""{DUAL_VERIFICATION_PROMPT}

# ACTUAL INPUTS:

**Evolved Question:** {evol_question}

**Reasoning Content:** {reasoning_content}

**Final Answer:** {content}

Please evaluate this answer according to the dual verification criteria above and provide your response in the required format."""
    return prompt


def answer_quality_check_single_record(
    api_client: VisionAPIClient,
    record: Dict[str, Any],
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Perform dual-verification quality evaluation on a single record

    Args:
        api_client: VisionAPIClient instance
        record: a single JSONL record
        max_retries: maximum number of retries

    Returns:
        A dict containing the evaluation result, or None on failure
        Format: {
            "overall_assessment": "CORRECT/PARTIALLY_CORRECT/INCORRECT",
            "correctness_check": {...},
            "error_detection": {...},
            "detailed_analysis": "..."
        }
    """
    # Extract image information
    image_info = record.get("image")
    if not image_info:
        logger.warning(f"Record {record.get('id', 'unknown')} is missing the image field")
        return None

    # Extract the evolved question
    evol_question = record.get("evol_question")
    if not evol_question:
        logger.warning(f"Record {record.get('id', 'unknown')} is missing the evol_question field")
        return None

    # Extract reasoning content and final answer
    reasoning_content = record.get("reasoning_content", "")
    content = record.get("content", "")

    if not reasoning_content and not content:
        logger.warning(
            f"Record {record.get('id', 'unknown')} is missing both reasoning_content and content fields")
        return None

    # Build the dual-verification prompt
    question = build_dual_verification_prompt(
        evol_question, reasoning_content, content
    )

    # Call the API
    for attempt in range(max_retries):
        try:
            response = call_vision_api(
                api_client=api_client,
                image=image_info,
                question=question
            )

            if response:
                # Parse the response and extract the evaluation result
                evaluation_result = parse_evaluation_response(response)
                if evaluation_result:
                    return evaluation_result
                else:
                    logger.warning(
                        f"Record {record.get('id', 'unknown')} could not extract evaluation result from response: {response[:100] if len(str(response)) > 100 else response}"
                    )
                    # If the response is non-empty but the result cannot be extracted, do not retry (may be a model output format issue)
                    return None
            else:
                # response is None, meaning the API call failed
                logger.warning(
                    f"Record {record.get('id', 'unknown')} API call returned None (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Record {record.get('id', 'unknown')} quality evaluation failed: API call returned None")
                    return None

        except Exception as e:
            logger.warning(
                f"Record {record.get('id', 'unknown')} API call exception (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"Record {record.get('id', 'unknown')} quality evaluation failed: {e}")
                return None

    return None


def parse_evaluation_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse the evaluation result from the model response

    Args:
        response: the response text returned by the model

    Returns:
        A dict containing the evaluation result, or None if it cannot be parsed
    """
    if not response:
        return None

    response = response.strip()

    result = {
        "overall_assessment": None,
        "correctness_check": {
            "is_correct": None,
            "strengths": []
        },
        "error_detection": {
            "has_errors": None,
            "errors_found": [],
            "error_severity": None
        },
        "detailed_analysis": "",
        "raw_response": response
    }

    # Extract Overall Assessment
    overall_patterns = [
        r'\*\*Overall Assessment:\*\*\s*(CORRECT|PARTIALLY_CORRECT|INCORRECT)',
        r'Overall Assessment:\s*(CORRECT|PARTIALLY_CORRECT|INCORRECT)',
        r'Overall Assessment\s*:\s*(CORRECT|PARTIALLY_CORRECT|INCORRECT)',
    ]
    for pattern in overall_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            result["overall_assessment"] = match.group(1).upper()
            break

    # Extract Direction 1 - Correctness Check
    correctness_section = re.search(
        r'Direction 1[^\*]*\*\*Is Correct:\*\*\s*(YES|NO|PARTIAL)',
        response, re.IGNORECASE | re.DOTALL
    )
    if correctness_section:
        result["correctness_check"]["is_correct"] = correctness_section.group(
            1).upper()

    # Extract Strengths
    strengths_section = re.search(
        r'\*\*Strengths:\*\*([^\*]+?)(?=\*\*|$)',
        response, re.IGNORECASE | re.DOTALL
    )
    if strengths_section:
        strengths_text = strengths_section.group(1).strip()
        # Split by line and extract each strength
        for line in strengths_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('*') or line.startswith('-') or line.startswith('•')):
                strength = line.lstrip('*-•').strip()
                if strength:
                    result["correctness_check"]["strengths"].append(strength)

    # Extract Direction 2 - Error Detection
    has_errors_section = re.search(
        r'Direction 2[^\*]*\*\*Has Errors:\*\*\s*(YES|NO)',
        response, re.IGNORECASE | re.DOTALL
    )
    if has_errors_section:
        result["error_detection"]["has_errors"] = has_errors_section.group(
            1).upper()

    # Extract Errors Found
    errors_section = re.search(
        r'\*\*Errors Found:\*\*([^\*]+?)(?=\*\*|$)',
        response, re.IGNORECASE | re.DOTALL
    )
    if errors_section:
        errors_text = errors_section.group(1).strip()
        # Split by line and extract each error
        for line in errors_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('*') or line.startswith('-') or line.startswith('•')):
                error = line.lstrip('*-•').strip()
                if error:
                    result["error_detection"]["errors_found"].append(error)

    # Extract Error Severity
    severity_patterns = [
        r'\*\*Error Severity:\*\*\s*(NONE|MINOR|MODERATE|SEVERE)',
        r'Error Severity:\s*(NONE|MINOR|MODERATE|SEVERE)',
    ]
    for pattern in severity_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            result["error_detection"]["error_severity"] = match.group(
                1).upper()
            break

    # Extract Detailed Analysis
    analysis_section = re.search(
        r'\*\*Detailed Analysis:\*\*([^\*]+?)(?=\*\*|$)',
        response, re.IGNORECASE | re.DOTALL
    )
    if analysis_section:
        result["detailed_analysis"] = analysis_section.group(1).strip()

    # If at least the Overall Assessment was extracted, consider parsing successful
    if result["overall_assessment"]:
        return result

    return None


def get_processed_ids(output_file: str) -> set:
    """
    Read the IDs of records already successfully processed from the output file (records that have an answer_quality_check field)

    Args:
        output_file: output file path

    Returns:
        Set of IDs of already-processed records
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
                    # If there is a non-empty answer_quality_check field, it has been successfully processed
                    if "answer_quality_check" in data and data["answer_quality_check"]:
                        record_id = data.get("id") or data.get(
                            "uid") or data.get("berlin_id")
                        if record_id:
                            processed_ids.add(str(record_id))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Error while reading output file: {e}")

    return processed_ids


def process_single_record_for_concurrent(
    record_data: tuple,
    api_client: VisionAPIClient,
    processed_ids: set,
    output_file: str,
    output_lock: threading.Lock,
    skip_processed: bool,
    verbose: bool
) -> Dict[str, Any]:
    """
    Process a single record concurrently (for use with the thread pool)

    Args:
        record_data: (line_num, data, record_id) tuple
        api_client: VisionAPIClient instance
        processed_ids: set of already-processed record IDs
        output_file: output file path
        output_lock: file write lock
        skip_processed: whether to skip already-processed records
        verbose: whether to print detailed information

    Returns:
        A dict with the processing result
    """
    line_num, data, record_id = record_data

    result = {
        "record_id": record_id,
        "line_num": line_num,
        "success": False,
        "skipped": False,
        "failed": False
    }

    # Check whether it has already been processed
    if skip_processed and record_id in processed_ids:
        result["skipped"] = True
        return result

    # Perform quality evaluation
    evaluation_result = answer_quality_check_single_record(api_client, data)

    if evaluation_result:
        # Evaluation succeeded
        updated_record = data.copy()
        updated_record["answer_quality_check"] = evaluation_result

        result["success"] = True

        # Thread-safely write to the file
        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(updated_record, ensure_ascii=False) + '\n')
    else:
        # Evaluation failed
        result["failed"] = True
        if verbose:
            logger.warning(f"Record {record_id} quality evaluation failed, skipping without saving")

    return result


def process_jsonl_file(
    input_file: str,
    output_file: str,
    api_client: VisionAPIClient,
    start_index: int = 0,
    max_items: Optional[int] = None,
    verbose: bool = True,
    skip_processed: bool = True,
    max_workers: int = 10
) -> Dict[str, Any]:
    """
    Process a JSONL file, performing dual-verification quality evaluation on each record's answer

    Args:
        input_file: input JSONL file path
        output_file: output JSONL file path
        api_client: VisionAPIClient instance
        start_index: which record to start from (for resume support, based on line number)
        max_items: maximum number of records to process (None means process all)
        verbose: whether to print detailed information
        skip_processed: whether to automatically skip already-processed records (based on ID)
        max_workers: number of concurrent threads (0 or 1 means no concurrency)

    Returns:
        A dict with statistics
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the IDs of already-processed records (for resume support)
    processed_ids = set()
    if skip_processed and output_path.exists():
        processed_ids = get_processed_ids(output_file)
        if verbose and processed_ids:
            logger.info(f"Found {len(processed_ids)} already successfully processed records, will skip automatically")

    stats = {
        "total_lines": 0,
        "processed_lines": 0,
        "successful_evaluations": 0,
        "failed_evaluations": 0,
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

    # Initialize the output file (if it's a new file, clear it; if in append mode, keep existing content)
    output_mode = 'a' if output_path.exists() and skip_processed else 'w'
    if output_mode == 'w':
        output_path.write_text('', encoding='utf-8')

    # File write lock (used in concurrent mode)
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

                # Skip earlier records (line-number-based resume)
                if line_num <= start_index:
                    stats["skipped_lines"] += 1
                    continue

                # Check whether the maximum number to process has been reached
                if max_items and processed_count >= max_items:
                    break

                try:
                    data = json.loads(line)
                    record_id = data.get('id') or data.get(
                        'uid') or data.get('berlin_id')
                    if not record_id:
                        if verbose:
                            logger.warning(f"Line {line_num} is missing the id field, skipping")
                        continue

                    # Check whether it has already been processed
                    if skip_processed and record_id in processed_ids:
                        stats["already_processed"] += 1
                        continue

                    records_to_process.append((line_num, data, record_id))
                    processed_count += 1
                except json.JSONDecodeError as e:
                    if verbose:
                        logger.warning(f"Line {line_num} JSON parsing failed: {e}")
                    continue

        if verbose:
            logger.info(f"Read {len(records_to_process)} records that need to be processed in total")

        # Step 2: process records
        if max_workers and max_workers > 1:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_record_for_concurrent,
                        record_data,
                        api_client,
                        processed_ids,
                        output_file,
                        output_lock,
                        skip_processed,
                        verbose
                    ): record_data
                    for record_data in records_to_process
                }

                # Use tqdm to show progress
                with tqdm(total=len(records_to_process), desc="Progress") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        stats["processed_lines"] += 1
                        if result["success"]:
                            stats["successful_evaluations"] += 1
                        elif result["skipped"]:
                            stats["already_processed"] += 1
                        else:
                            stats["failed_evaluations"] += 1
                        pbar.update(1)
        else:
            # Sequential processing
            with tqdm(total=len(records_to_process), desc="Progress") as pbar:
                for record_data in records_to_process:
                    result = process_single_record_for_concurrent(
                        record_data,
                        api_client,
                        processed_ids,
                        output_file,
                        output_lock,
                        skip_processed,
                        verbose
                    )
                    stats["processed_lines"] += 1
                    if result["success"]:
                        stats["successful_evaluations"] += 1
                    elif result["skipped"]:
                        stats["already_processed"] += 1
                    else:
                        stats["failed_evaluations"] += 1
                    pbar.update(1)

    except Exception as e:
        logger.error(f"Error while processing file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    if verbose:
        logger.info("=" * 50)
        logger.info("Processing complete! Statistics:")
        logger.info(f"  Total lines: {stats['total_lines']}")
        logger.info(f"  Processed: {stats['processed_lines']}")
        logger.info(f"  Succeeded: {stats['successful_evaluations']}")
        logger.info(f"  Failed: {stats['failed_evaluations']}")
        logger.info(f"  Skipped (already processed): {stats['already_processed']}")
        logger.info(f"  Skipped lines (resume): {stats['skipped_lines']}")
        logger.info("=" * 50)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Perform dual-verification quality evaluation on answers (LLM-as-a-judge)"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--api-endpoints',
        type=str,
        nargs='+',
        required=True,
        help='List of API endpoints, e.g.: http://ip:port/v1'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default='sk-abc123',
        help='API key (default: sk-abc123)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='qwen_3',
        help='Model name (default: qwen_3)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature parameter (default: 0.7)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=32768,
        help='Maximum number of generated tokens (default: 32768)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=3000,
        help='Timeout (seconds, default: 3000)'
    )
    parser.add_argument(
        '--client-selection',
        type=str,
        choices=['random', 'round_robin', 'localhost'],
        default='random',
        help='Client selection strategy (default: random)'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Which record to start from (based on line number, for resume support, default: 0)'
    )
    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help='Maximum number of records to process (default: process all)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Number of concurrent threads (0 or 1 means no concurrency, default: 10)'
    )
    parser.add_argument(
        '--no-skip-processed',
        action='store_true',
        help='Do not skip already-processed records (skipped automatically by default)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Do not print detailed information'
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Initialize the API client
    api_client = VisionAPIClient(
        api_endpoints=args.api_endpoints,
        api_key=args.api_key,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        client_selection=args.client_selection,
        timeout=args.timeout
    )

    # Process the file
    try:
        stats = process_jsonl_file(
            input_file=args.input_file,
            output_file=args.output,
            api_client=api_client,
            start_index=args.start_index,
            max_items=args.max_items,
            verbose=not args.quiet,
            skip_processed=not args.no_skip_processed,
            max_workers=args.max_workers
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
