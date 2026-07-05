#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Consistency voting scheme: provide the model with multiple candidate answers and let it vote for the most consistent answer as ground truth

Usage example:
    python vote_consistency.py input.jsonl \
        --answer-files answer1.jsonl answer2.jsonl answer3.jsonl \
        -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --api-key sk-abc123 \
        --model-name qwen_3
"""

import json
import sys
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from vision_api import VisionAPIClient


# Self-Consistency voting prompt template
VOTE_PROMPT_TEMPLATE = """# ROLE:
You are an expert AI judge. Your task is to evaluate multiple candidate answers for a question and select the most consistent and accurate one.

# GOAL:
You will be given a `Question` and multiple `Candidate Answers`. You must determine which answer is the most consistent, accurate, and best addresses the question.

# EVALUATION CRITERIA:

1. **Factual Accuracy:**
   - The answer must be factually correct and consistent.
   - It should not contain contradictory information.

2. **Completeness:**
   - The answer should fully address the question.
   - It should cover all relevant aspects.

3. **Clarity and Coherence:**
   - The answer should be clear, well-organized, and easy to understand.
   - It should be logically coherent.

4. **Consistency:**
   - Among all candidates, find the answer that is most consistent with the majority.
   - If multiple answers agree on key points, prefer the most comprehensive one.

# INPUTS:

**Question:** {question}

**Candidate Answers:**
{candidate_answers}

# OUTPUT FORMAT:
You MUST provide your response in the following JSON format:

```json
{{
    "selected_index": <index of the best answer, 1-based>,
    "reason": "<brief explanation of why this answer was selected>",
    "confidence": "<HIGH/MEDIUM/LOW>"
}}
```

Please evaluate all candidate answers and select the best one."""


def load_answer_files(answer_files: List[str], verbose: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load multiple answer files and build a dict keyed by uid
    Some files may be missing certain uids without raising an error

    Args:
        answer_files: list of answer file paths
        verbose: whether to print detailed information

    Returns:
        A dict keyed by uid with a list of answers as value
    """
    answers_by_id: Dict[str, List[Dict[str, Any]]] = {}
    file_stats = {}  # count of records loaded from each file

    for file_idx, file_path in enumerate(answer_files):
        path = Path(file_path)
        if not path.exists():
            if verbose:
                logger.warning(f"File does not exist: {file_path}, skipping")
            continue

        file_record_count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Prefer uid as the unique ID
                        record_id = str(data.get('uid') or data.get(
                            'id') or data.get('berlin_id') or '')
                        if not record_id:
                            if verbose:
                                logger.debug(
                                    f"File {file_path} line {line_num} is missing uid/id field, skipping")
                            continue

                        if record_id not in answers_by_id:
                            answers_by_id[record_id] = []

                        # Add source file information
                        data['_source_file'] = file_path
                        data['_source_index'] = file_idx
                        answers_by_id[record_id].append(data)
                        file_record_count += 1

                    except json.JSONDecodeError as e:
                        if verbose:
                            logger.debug(
                                f"File {file_path} line {line_num} JSON parsing failed: {e}")
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise

        file_stats[file_path] = file_record_count
        if verbose:
            logger.info(
                f"  File {file_idx + 1}/{len(answer_files)} ({Path(file_path).name}): {file_record_count} records")

    return answers_by_id


def build_vote_messages(
    question: str,
    candidate_answers: List[str]
) -> List[Dict[str, Any]]:
    """
    Build the message list for a voting request

    Args:
        question: question text
        candidate_answers: list of candidate answers

    Returns:
        Message list in OpenAI format
    """
    # Format candidate answers
    formatted_answers = ""
    for idx, answer in enumerate(candidate_answers, 1):
        formatted_answers += f"\n**Answer {idx}:**\n{answer}\n"

    # Format the prompt
    prompt = VOTE_PROMPT_TEMPLATE.format(
        question=question,
        candidate_answers=formatted_answers
    )

    # Build the message list (plain text, no images)
    messages = [{
        "role": "user",
        "content": prompt
    }]

    return messages


def vote_single_record(
    api_client: VisionAPIClient,
    record: Dict[str, Any],
    candidate_answers: List[str],
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Vote on a single record and select the most consistent answer

    Args:
        api_client: VisionAPIClient instance
        record: a single JSONL record (contains question information)
        candidate_answers: list of candidate answers
        max_retries: maximum number of retries

    Returns:
        A dict containing selected_index, reason, confidence; returns None on failure
    """
    # Prefer evol_question as the question
    question = record.get("evol_question") or record.get("question", "")
    if not question:
        # Try to extract from conversations
        conversations = record.get("conversations", [])
        for conv in conversations:
            if conv.get('from') == 'human':
                question = conv.get('value', '').replace('<image>', '').strip()
                break

    # Use uid as the record identifier
    record_uid = record.get('uid') or record.get('id', 'unknown')

    if not question:
        logger.warning(f"Record {record_uid} is missing evol_question field")
        return None

    # Check candidate answers
    valid_answers = [a for a in candidate_answers if a and a.strip()]
    if len(valid_answers) < 2:
        # If there is only one valid answer, select it directly
        if len(valid_answers) == 1:
            return {
                "selected_index": candidate_answers.index(valid_answers[0]) + 1,
                "reason": "Only one valid answer",
                "confidence": "HIGH"
            }
        else:
            logger.warning(f"Record {record_uid} has no valid candidate answers")
            return None

    # Build voting messages (no images)
    try:
        messages = build_vote_messages(question, candidate_answers)
    except Exception as e:
        logger.error(f"Record {record_uid} failed to build voting messages: {e}")
        return None

    # Call the API
    for attempt in range(max_retries):
        try:
            result = api_client.call(messages)
            if result:
                # Try to parse the JSON result
                try:
                    result_text = result.strip()
                    # If the result contains a JSON code block, extract it
                    if "```json" in result_text:
                        start = result_text.find("```json") + 7
                        end = result_text.find("```", start)
                        if end != -1:
                            result_text = result_text[start:end].strip()
                    elif "```" in result_text:
                        start = result_text.find("```") + 3
                        end = result_text.find("```", start)
                        if end != -1:
                            result_text = result_text[start:end].strip()

                    # Try to find the JSON object
                    if "{" in result_text and "}" in result_text:
                        start = result_text.find("{")
                        end = result_text.rfind("}") + 1
                        result_text = result_text[start:end]

                    vote_result = json.loads(result_text)

                    # Validate the result format
                    if "selected_index" not in vote_result:
                        # Try to infer from text
                        import re
                        match = re.search(
                            r'(?:answer|select|index)\s*[:#]?\s*(\d+)', result.lower())
                        if match:
                            vote_result["selected_index"] = int(match.group(1))
                        else:
                            # Default to the first one
                            vote_result["selected_index"] = 1

                    # Ensure selected_index is valid
                    selected_idx = int(vote_result["selected_index"])
                    if selected_idx < 1 or selected_idx > len(candidate_answers):
                        logger.warning(
                            f"Record {record_uid} selected_index is invalid: {selected_idx}, defaulting to 1")
                        vote_result["selected_index"] = 1
                    else:
                        vote_result["selected_index"] = selected_idx

                    # Ensure the reason field exists
                    if "reason" not in vote_result:
                        vote_result["reason"] = result[:200] if len(
                            result) > 200 else result

                    # Ensure the confidence field exists
                    if "confidence" not in vote_result:
                        vote_result["confidence"] = "MEDIUM"

                    return vote_result

                except json.JSONDecodeError as e:
                    logger.warning(f"Record {record_uid} failed to parse voting result JSON: {e}")
                    # Try to infer the selection from text
                    import re
                    match = re.search(
                        r'(?:answer|select|index|selected)\s*[:#]?\s*(\d+)', result.lower())
                    if match:
                        selected_idx = int(match.group(1))
                        if 1 <= selected_idx <= len(candidate_answers):
                            return {
                                "selected_index": selected_idx,
                                "reason": result[:500] if len(result) > 500 else result,
                                "confidence": "LOW"
                            }
                    # Default to the first one
                    return {
                        "selected_index": 1,
                        "reason": "Unable to parse voting result, defaulting to the first one",
                        "confidence": "LOW"
                    }
        except Exception as e:
            logger.warning(
                f"Record {record_uid} voting API call exception (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                logger.error(f"Record {record_uid} voting failed: {e}")
                return None

    return None


def process_single_record(
    record_data: tuple,
    api_client: VisionAPIClient,
    answers_by_id: Dict[str, List[Dict[str, Any]]],
    output_file: str,
    output_lock: threading.Lock,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Process a single record (used for concurrent processing)

    Args:
        record_data: (record_id, record) tuple
        api_client: VisionAPIClient instance
        answers_by_id: answer dict keyed by ID
        output_file: output file path
        output_lock: file write lock
        verbose: whether to print detailed information

    Returns:
        Processing result dict
    """
    record_id, record = record_data

    result = {
        "record_id": record_id,
        "success": False,
        "failed": False,
        "skipped": False,
        "selected_index": None
    }

    # Get all candidate answers for this record (some files may be missing this uid)
    answer_records = answers_by_id.get(record_id, [])

    if len(answer_records) == 0:
        # If no file has an answer for this uid, skip (no error)
        result["skipped"] = True
        if verbose:
            logger.debug(f"Record {record_id} does not exist in any answer file, skipping")
        return result

    if len(answer_records) == 1:
        # If only one file has an answer, use it directly (no error)
        selected_record = answer_records[0].copy()
        selected_content = selected_record.get("content", "")

        # Merge information from the original input record (such as the question)
        if record:
            # Keep key fields of the original record, but prefer fields from the answer record
            for key in ["evol_question", "question", "conversations", "image", "images", "source", "topic", "question_type"]:
                if key not in selected_record and key in record:
                    selected_record[key] = record[key]

        selected_record["vote_selected_index"] = 1
        selected_record["vote_reason"] = "Only one candidate answer (this uid does not exist in other files)"
        selected_record["vote_confidence"] = "HIGH"
        selected_record["vote_num_candidates"] = 1
        # Explicitly output the final selected answer content
        selected_record["selected_content"] = selected_content

        # Remove internal fields
        selected_record.pop('_source_file', None)
        selected_record.pop('_source_index', None)

        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(selected_record,
                        ensure_ascii=False) + '\n')

        result["success"] = True
        result["selected_index"] = 1
        if verbose:
            logger.debug(f"Record {record_id} has only one candidate answer, using it directly")
        return result

    # Extract the content of candidate answers
    candidate_answers = []
    for ar in answer_records:
        content = ar.get("content", "")
        candidate_answers.append(content)

    # Perform voting
    vote_result = vote_single_record(api_client, record, candidate_answers)

    if vote_result:
        selected_idx = vote_result["selected_index"] - 1  # convert to 0-based

        if 0 <= selected_idx < len(answer_records):
            selected_record = answer_records[selected_idx].copy()
        else:
            # Invalid index, select the first one
            selected_record = answer_records[0].copy()
            selected_idx = 0

        # Merge information from the original input record (such as the question)
        if record:
            # Keep key fields of the original record, but prefer fields from the answer record
            for key in ["evol_question", "question", "conversations", "image", "images", "source", "topic", "question_type"]:
                if key not in selected_record and key in record:
                    selected_record[key] = record[key]

        # Get the final selected answer content
        selected_content = selected_record.get("content", "")

        # Add voting information
        selected_record["vote_selected_index"] = selected_idx + 1
        selected_record["vote_reason"] = vote_result.get("reason", "")
        selected_record["vote_confidence"] = vote_result.get(
            "confidence", "MEDIUM")
        selected_record["vote_num_candidates"] = len(candidate_answers)
        # Explicitly output the final selected answer content
        selected_record["selected_content"] = selected_content

        # Remove internal fields
        selected_record.pop('_source_file', None)
        selected_record.pop('_source_index', None)

        result["success"] = True
        result["selected_index"] = selected_idx + 1

        # Write to file in a thread-safe manner
        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(selected_record, ensure_ascii=False) + '\n')
    else:
        # Voting failed, default to the first answer
        selected_record = answer_records[0].copy()

        # Merge information from the original input record (such as the question)
        if record:
            # Keep key fields of the original record, but prefer fields from the answer record
            for key in ["evol_question", "question", "conversations", "image", "images", "source", "topic", "question_type"]:
                if key not in selected_record and key in record:
                    selected_record[key] = record[key]

        selected_content = selected_record.get("content", "")
        selected_record["vote_selected_index"] = 1
        selected_record["vote_reason"] = "Voting API call failed, defaulting to the first answer"
        selected_record["vote_confidence"] = "LOW"
        selected_record["vote_num_candidates"] = len(candidate_answers)
        # Explicitly output the final selected answer content
        selected_record["selected_content"] = selected_content

        # Remove internal fields
        selected_record.pop('_source_file', None)
        selected_record.pop('_source_index', None)

        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(selected_record, ensure_ascii=False) + '\n')

        result["success"] = True
        result["selected_index"] = 1
        if verbose:
            logger.warning(f"Record {record_id} voting failed, defaulting to the first answer")

    return result


def vote_consistency(
    input_file: str,
    answer_files: List[str],
    output_file: str,
    api_client: VisionAPIClient,
    verbose: bool = True,
    max_workers: int = 10,
    start_index: int = 0,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform self-consistency voting over multiple answer files

    Args:
        input_file: original input file (contains question information)
        answer_files: list of answer files
        output_file: output file path
        api_client: VisionAPIClient instance
        verbose: whether to print detailed information
        max_workers: number of concurrent threads (0 or 1 means no concurrency)
        start_index: which record to start from (based on line number, for resuming, default: 0)
        max_items: maximum number of records to process (None means process all)

    Returns:
        Statistics dict
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_records": 0,
        "processed_records": 0,
        "successful_votes": 0,
        "failed_votes": 0,
        "skipped_records": 0,
        "selection_distribution": {},
    }

    if verbose:
        logger.info(f"Starting Self-Consistency voting task")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Number of answer files: {len(answer_files)}")
        for i, af in enumerate(answer_files):
            logger.info(f"  Answer file {i+1}: {af}")
        logger.info(f"Output file: {output_file}")
        if max_workers and max_workers > 1:
            logger.info(f"Using concurrent processing, threads: {max_workers}")
        else:
            logger.info(f"Using sequential processing")

    # Load all answer files (some files may be missing certain uids)
    if verbose:
        logger.info("Loading answer files (some files may be missing certain uids)...")
    answers_by_id = load_answer_files(answer_files, verbose=verbose)
    if verbose:
        logger.info(f"Loaded {len(answers_by_id)} unique record IDs in total")
        # Count how many answers each uid has
        answer_count_dist = {}
        for uid, answers in answers_by_id.items():
            count = len(answers)
            answer_count_dist[count] = answer_count_dist.get(count, 0) + 1
        logger.info(f"Answer distribution: {dict(sorted(answer_count_dist.items()))}")

    # Initialize the output file
    output_path.write_text('', encoding='utf-8')

    # File write lock (for concurrent mode)
    output_lock = threading.Lock()

    try:
        # Read the input file
        records_to_process = []
        current_index = 0
        items_processed = 0

        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                stats["total_records"] += 1

                # Skip records before start_index
                if current_index < start_index:
                    current_index += 1
                    continue

                # If max_items is reached, stop processing
                if max_items is not None and items_processed >= max_items:
                    break

                try:
                    data = json.loads(line)
                    # Prefer uid as the unique ID
                    record_id = str(data.get('uid') or data.get(
                        'id') or data.get('berlin_id') or '')
                    if not record_id:
                        if verbose:
                            logger.warning(f"Line {line_num} is missing uid/id field, skipping")
                        current_index += 1
                        continue

                    # Do not check for answer existence here; some files may be missing certain uids
                    # The case of no answer is handled in process_single_record
                    records_to_process.append((record_id, data))
                    current_index += 1
                    items_processed += 1
                except json.JSONDecodeError as e:
                    if verbose:
                        logger.warning(f"Line {line_num} JSON parsing failed: {e}")
                    current_index += 1
                    items_processed += 1
                    continue

        if verbose:
            logger.info(f"Read {len(records_to_process)} records to process in total")
            if start_index > 0 or max_items is not None:
                logger.info(
                    f"Processing range: start_index={start_index}, max_items={max_items}")

        # Process records
        if max_workers and max_workers > 1:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_record,
                        record_data,
                        api_client,
                        answers_by_id,
                        output_file,
                        output_lock,
                        verbose
                    ): record_data
                    for record_data in records_to_process
                }

                # Use tqdm to show progress
                with tqdm(total=len(records_to_process), desc="Voting progress") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        stats["processed_records"] += 1
                        if result["success"]:
                            stats["successful_votes"] += 1
                            selected_idx = result.get("selected_index")
                            if selected_idx:
                                key = f"Answer_{selected_idx}"
                                stats["selection_distribution"][key] = stats["selection_distribution"].get(
                                    key, 0) + 1
                        elif result["skipped"]:
                            stats["skipped_records"] += 1
                        else:
                            stats["failed_votes"] += 1
                        pbar.update(1)
        else:
            # Sequential processing
            with tqdm(total=len(records_to_process), desc="Voting progress") as pbar:
                for record_data in records_to_process:
                    result = process_single_record(
                        record_data,
                        api_client,
                        answers_by_id,
                        output_file,
                        output_lock,
                        verbose
                    )
                    stats["processed_records"] += 1
                    if result["success"]:
                        stats["successful_votes"] += 1
                        selected_idx = result.get("selected_index")
                        if selected_idx:
                            key = f"Answer_{selected_idx}"
                            stats["selection_distribution"][key] = stats["selection_distribution"].get(
                                key, 0) + 1
                    elif result["skipped"]:
                        stats["skipped_records"] += 1
                    else:
                        stats["failed_votes"] += 1
                    pbar.update(1)

    except Exception as e:
        logger.error(f"Error while processing file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    if verbose:
        logger.info("=" * 50)
        logger.info("Voting complete! Statistics:")
        logger.info(f"  Total records: {stats['total_records']}")
        logger.info(f"  Processed: {stats['processed_records']}")
        logger.info(f"  Successful: {stats['successful_votes']}")
        logger.info(f"  Failed: {stats['failed_votes']}")
        logger.info(f"  Skipped: {stats['skipped_records']}")
        logger.info("  Selection distribution:")
        for key, count in sorted(stats["selection_distribution"].items()):
            logger.info(f"    {key}: {count}")
        logger.info("=" * 50)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Self-Consistency voting scheme: vote to select among multiple candidate answers"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='original input file (contains question information)'
    )
    parser.add_argument(
        '--answer-files',
        type=str,
        nargs='+',
        required=True,
        help='list of answer files (JSONL format)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='output JSONL file path'
    )
    parser.add_argument(
        '--api-endpoints',
        type=str,
        nargs='+',
        required=True,
        help='list of API endpoints, e.g.: http://ip:port/v1'
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
        help='model name (default: qwen_3)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='temperature parameter (default: 0.1, voting task uses a lower temperature)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='maximum number of generated tokens (default: 2048)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=3000,
        help='timeout (seconds, default: 3000)'
    )
    parser.add_argument(
        '--client-selection',
        type=str,
        choices=['random', 'round_robin', 'localhost'],
        default='random',
        help='client selection strategy (default: random)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='number of concurrent threads (0 or 1 means no concurrency, default: 10)'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='which record to start from (based on line number, for resuming, default: 0)'
    )
    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help='maximum number of records to process (default: process all)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='do not print detailed information'
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
        stats = vote_consistency(
            input_file=args.input_file,
            answer_files=args.answer_files,
            output_file=args.output,
            api_client=api_client,
            verbose=not args.quiet,
            max_workers=args.max_workers,
            start_index=args.start_index,
            max_items=args.max_items
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
