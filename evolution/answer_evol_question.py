#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call the model to answer questions based on the "evol_question" field, and save reasoning_content and content.

Use --output-prefix to add a prefix to the output fields (e.g. --output-prefix sft_ writes
sft_reasoning_content / sft_content, used for SFT answer generation).

Usage example:
    python answer_evol_question.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --api-key sk-abc123 \
        --model-name qwen_3 [--output-prefix sft_]
"""

import json
import sys
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from vision_api import VisionAPIClient, prepare_image_message

# Prefix for the output fields; overridden by --output-prefix at runtime.
OUTPUT_PREFIX = ""


def answer_single_record(
    api_client: VisionAPIClient,
    record: Dict[str, Any],
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Call the API to answer a question for a single record using evol_question

    Args:
        api_client: VisionAPIClient instance
        record: a single JSONL record
        max_retries: maximum number of retries

    Returns:
        A dict containing reasoning_content and content; returns None on failure
    """
    # Extract evol_question
    evol_question = record.get("evol_question")
    if not evol_question:
        logger.warning(f"Record {record.get('id', 'unknown')} is missing the evol_question field")
        return None

    # Extract image information
    image_info = record.get("image")
    if not image_info:
        logger.warning(f"Record {record.get('id', 'unknown')} is missing the image field")
        return None

    # Prepare the image message
    try:
        if isinstance(image_info, dict):
            image_url = prepare_image_message(image_info)
        elif isinstance(image_info, str):
            image_url = image_info
        else:
            logger.error(
                f"Record {record.get('id', 'unknown')} has an unsupported image format: {type(image_info)}")
            return None
    except Exception as e:
        logger.error(f"Record {record.get('id', 'unknown')} image processing failed: {e}")
        return None

    # Build the message list
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": evol_question,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        ]
    }]

    # Call the API
    for attempt in range(max_retries):
        try:
            result = api_client.call_with_reasoning(messages)
            if result:
                return result
            else:
                logger.warning(
                    f"Record {record.get('id', 'unknown')} API call returned None (attempt {attempt + 1}/{max_retries})"
                )
        except Exception as e:
            logger.warning(
                f"Record {record.get('id', 'unknown')} API call raised an exception (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                logger.error(f"Record {record.get('id', 'unknown')} failed to answer the question: {e}")
                return None

    return None


def get_processed_ids(output_file: str) -> set:
    """
    Read the IDs of records that were successfully processed from the output file (records that have reasoning_content and content fields)

    Args:
        output_file: output file path

    Returns:
        Set of processed record IDs
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
                    record_id = data.get('id')
                    # Check whether reasoning_content and content fields exist
                    if record_id and f'{OUTPUT_PREFIX}reasoning_content' in data and f'{OUTPUT_PREFIX}content' in data:
                        processed_ids.add(record_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Error while reading processed records: {e}")

    return processed_ids


def process_single_record(
    record_data: tuple,
    api_client: VisionAPIClient,
    processed_ids: set,
    output_file: str,
    output_lock: threading.Lock,
    skip_processed: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Process a single record (used for concurrent processing)

    Args:
        record_data: (line_num, data, record_id) tuple
        api_client: VisionAPIClient instance
        processed_ids: set of processed record IDs
        output_file: output file path
        output_lock: file write lock
        skip_processed: whether to skip already processed records
        verbose: whether to print detailed information

    Returns:
        Processing result dict
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

    # Call the API to answer the question
    api_result = answer_single_record(api_client, data)

    if api_result:
        # Call succeeded, save the result
        updated_record = data.copy()
        updated_record[f"{OUTPUT_PREFIX}reasoning_content"] = api_result.get(
            "reasoning_content", "")
        updated_record[f"{OUTPUT_PREFIX}content"] = api_result.get("content", "")

        result["success"] = True

        # Write to the file in a thread-safe manner
        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(updated_record, ensure_ascii=False) + '\n')
    else:
        # Call failed
        result["failed"] = True
        if verbose:
            logger.warning(f"Record {record_id} failed to answer the question, skipping without saving")

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
    Process a JSONL file, calling the API to answer the question for each record using evol_question

    Args:
        input_file: input JSONL file path
        output_file: output JSONL file path
        api_client: VisionAPIClient instance
        start_index: which record to start processing from (for resuming, based on line number)
        max_items: maximum number of records to process (None means process all)
        verbose: whether to print detailed information
        skip_processed: whether to automatically skip already successfully processed records (based on ID)
        max_workers: number of concurrent threads (0 or 1 means no concurrency)

    Returns:
        Statistics dict
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the IDs of processed records (for resuming)
    processed_ids = set()
    if skip_processed and output_path.exists():
        processed_ids = get_processed_ids(output_file)
        if verbose and processed_ids:
            logger.info(f"Found {len(processed_ids)} successfully processed records, which will be skipped automatically")

    stats = {
        "total_lines": 0,
        "processed_lines": 0,
        "successful_answers": 0,
        "failed_answers": 0,
        "skipped_lines": 0,
        "already_processed": 0,
    }

    if verbose:
        logger.info(f"Start processing file: {input_file}")
        logger.info(f"Output file: {output_file}")
        if start_index > 0:
            logger.info(f"Start processing from record {start_index + 1} (based on line number)")
        if max_items:
            logger.info(f"Process at most {max_items} records")
        if skip_processed:
            logger.info(f"Automatically skip already successfully processed records (based on ID)")
        if max_workers and max_workers > 1:
            logger.info(f"Using concurrent processing, number of threads: {max_workers}")
        else:
            logger.info(f"Using sequential processing")

    # Initialize the output file (if it is a new file, clear it; if in append mode, keep existing content)
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

                # Skip the earlier records (line-number-based resuming)
                if line_num <= start_index:
                    stats["skipped_lines"] += 1
                    continue

                # Check whether the maximum number of records to process has been reached
                if max_items and processed_count >= max_items:
                    break

                try:
                    data = json.loads(line)
                    record_id = data.get('id')
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
            logger.info(f"Read a total of {len(records_to_process)} records that need to be processed")

        # Step 2: process the records
        if max_workers and max_workers > 1:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_record,
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
                            stats["successful_answers"] += 1
                        elif result["skipped"]:
                            stats["already_processed"] += 1
                        else:
                            stats["failed_answers"] += 1
                        pbar.update(1)
        else:
            # Sequential processing
            with tqdm(total=len(records_to_process), desc="Progress") as pbar:
                for record_data in records_to_process:
                    result = process_single_record(
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
                        stats["successful_answers"] += 1
                    elif result["skipped"]:
                        stats["already_processed"] += 1
                    else:
                        stats["failed_answers"] += 1
                    pbar.update(1)

    except Exception as e:
        logger.error(f"Error while processing the file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    if verbose:
        logger.info("=" * 50)
        logger.info("Processing complete! Statistics:")
        logger.info(f"  Total lines: {stats['total_lines']}")
        logger.info(f"  Processed: {stats['processed_lines']}")
        logger.info(f"  Successful: {stats['successful_answers']}")
        logger.info(f"  Failed: {stats['failed_answers']}")
        logger.info(f"  Skipped (already processed): {stats['already_processed']}")
        logger.info(f"  Skipped lines (resuming): {stats['skipped_lines']}")
        logger.info("=" * 50)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Call the model to answer questions based on the evol_question field, and save reasoning_content and content"
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
        help='Which record to start processing from (based on line number, for resuming, default: 0)'
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
        help='Do not skip already processed records (skipped automatically by default)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Do not print detailed information'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='',
        help="Output field prefix (e.g. sft_ writes sft_reasoning_content / sft_content)"
    )

    args = parser.parse_args()

    global OUTPUT_PREFIX
    OUTPUT_PREFIX = args.output_prefix

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
