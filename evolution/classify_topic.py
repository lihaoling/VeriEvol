#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify each record in JSONL data by calling the MLLM API, determining which topic it belongs to and whether it is an objective or subjective question
Uses approach one of vision_api (VisionAPIClient and call_vision_api)

Features:
- Topic classification: determine which topic category the data belongs to (12 categories)
- Question type classification: determine whether a question is objective (has a definite answer) or subjective (open-ended)

Error handling and resume support:
- Failed records (API call failure or invalid returned content) are not saved to the output file
- The program automatically skips records already successfully processed in the output file (based on record ID)
- Re-running the program automatically processes previously failed records

Concurrent processing:
- By default uses 10 threads to process API calls concurrently, improving processing speed
- The number of concurrent threads can be adjusted via the --max-workers argument
- Setting it to 0 or 1 disables concurrency and uses sequential processing

Usage examples:
    # Basic usage
    python classify_topic.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 http://<VLLM_ENDPOINT>/v1 \
        --api-key sk-abc123 \
        --model-name qwen_3

    # Resume (start from record 100, based on line number)
    python classify_topic.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --start-index 100

    # Process only the first 50 records
    python classify_topic.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --max-items 50

    # Re-run to process failed records (automatically skips successful ones)
    python classify_topic.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1

    # Use concurrent processing (default 10 threads)
    python classify_topic.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --max-workers 20

    # Disable concurrency (sequential processing)
    python classify_topic.py input.jsonl -o output.jsonl \
        --api-endpoints http://<VLLM_ENDPOINT>/v1 \
        --max-workers 1
"""

import json
import sys
import argparse
import random
import string
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from vision_api import VisionAPIClient, call_vision_api

# 12 topic categories and their descriptions
TOPICS = {
    "OCR": "Optical Character Recognition (OCR) involves extracting textual and symbolic information from diverse visual sources. This ranges from recognizing text in standard documents like invoices and business cards to interpreting complex formats such as charts, diagrams, handwritten notes, and musical scores for automated data extraction, validation, and structural analysis.",
    "Image Description": "Image Description involves generating textual narratives from visual content. This extends beyond literal object identification to encompass abstract concepts, emotional interpretation, cultural context, and stylistic analysis. Applications range from creating accessible alt-text to developing rich, context-aware stories that consider potential biases and factual accuracy.",
    "Detection": "Detection tasks focus on identifying and locating specific elements within visual data. This includes pinpointing single or multiple objects like vehicles and equipment, classifying entire environments such as traffic scenes, and recognizing specific states or patterns like signatures, anomalies, or out-of-stock conditions for targeted action.",
    "Analysis": "Analysis tasks involve interpreting and evaluating the deeper meaning, properties, and context of content. This extends beyond identification to assess abstract qualities like emotional tone, design composition, and political sentiment, as well as to evaluate content for source reliability, gender representation, and user personalization.",
    "Content Creation": "Content Creation involves generating a wide range of materials, from structured business documents like annual reports to creative works like children's books and art critiques. This task includes brainstorming ideas, authoring marketing collateral, developing interactive experiences, and adapting information from sources like images to produce targeted, purpose-driven content.",
    "Suggestions": "The Suggestions task focuses on providing personalized and actionable recommendations across a vast spectrum of topics. This includes offering curated ideas for lifestyle choices like home decor and fashion, entertainment such as movies and books, and personal development, including study techniques and wellness practices, to inspire and guide users.",
    "Summarization": "Summarization involves condensing information from a wide array of sources into a concise and coherent overview. This task applies to diverse content types, from structured legal and financial documents to unstructured news articles, political debates, and even customer conversations, extracting key points, themes, and outcomes efficiently.",
    "Logical Reasoning": "Logical Reasoning involves applying structured inference to solve problems and understand relationships. This task spans multiple forms, including deductive, causal, spatial, and temporal reasoning, addressing challenges from simple sequencing to complex scenarios like analyzing data charts, reconstructing 3D scenes, and navigating ethical dilemmas.",
    "Science-Related": "Science-Related tasks involve the automated analysis and interpretation of scientific information. This includes reasoning over charts and visual data, detecting anomalies, generating hypotheses, and evaluating scientific arguments. It also encompasses analyzing citation trends and extracting conclusions from data to support the research process.",
    "Concept Extraction": "Concept Extraction focuses on identifying and structuring semantic information from data. This involves determining object attributes like function and texture, and extracting complex relationships, such as causal links or agent-action pairings. The goal is to distill content into keyphrases, summaries, and contextual connections between modalities.",
    "Medical Imaging Analysis": "Medical Imaging Analysis involves the automated interpretation of scans to aid the entire clinical pathway. This includes detecting anomalies like lesions for disease diagnosis, planning treatments, and predicting patient outcomes. It also extends to generating structured reports, providing surgical assistance, and matching patients to clinical trials.",
    "Scene Understanding": "Scene Understanding involves a holistic interpretation of visual context beyond simple object identification. It focuses on recognizing dynamic elements such as activities, gestures, and expressions, and analyzing the complex relationships between humans, objects, and their environment, often applying spatial and contextual reasoning to classify the scene."
}


def build_classification_prompt(conversations: List[Dict[str, Any]]) -> str:
    """
    Build the prompt used for classification

    Args:
        conversations: list of conversations, containing human and gpt turns

    Returns:
        prompt text for the classification task
    """
    # Extract conversation content
    conversation_text = ""
    for conv in conversations:
        role = conv.get("from", "unknown")
        value = conv.get("value", "")
        if role == "human":
            conversation_text += f"User: {value}\n"
        elif role in ["gpt", "assistant"]:
            conversation_text += f"Assistant: {value}\n"

    # Build the category list (shuffle order randomly to avoid ordering bias)
    topics_items = list(TOPICS.items())
    random.shuffle(topics_items)
    topics_list = "\n".join([
        f"- {topic_name}: {description}"
        for topic_name, description in topics_items
    ])

    prompt = f"""You are a topic classification expert. Analyze the given image and conversation to determine which topic category this data belongs to.

Conversation:
{conversation_text}

Available topic categories:
{topics_list}

Instructions:
1. Carefully analyze both the image content and the conversation
2. Select the most appropriate topic category that best matches the task
3. Respond with ONLY the exact topic name (e.g., "OCR", "Image Description", "Detection", etc.)
4. Do not include any explanations, reasoning, or additional text

Your response (topic name only):"""
    return prompt


def build_question_type_prompt(conversations: List[Dict[str, Any]]) -> str:
    """
    Build the prompt used for classifying objective/subjective questions

    Args:
        conversations: list of conversations, containing human and gpt turns

    Returns:
        prompt text for classifying objective/subjective questions
    """
    # Extract conversation content
    conversation_text = ""
    for conv in conversations:
        role = conv.get("from", "unknown")
        value = conv.get("value", "")
        if role == "human":
            conversation_text += f"User: {value}\n"
        elif role in ["gpt", "assistant"]:
            conversation_text += f"Assistant: {value}\n"

    prompt = f"""You are a question type classification expert. Analyze the given image and conversation to determine whether the question is objective or subjective.

Conversation:
{conversation_text}

Question Type Definitions:
- Objective Question (Objective): Questions that have a definite, specific answer. These include factual questions, calculations, identification tasks, and questions with clear right or wrong answers.
- Subjective Question (Subjective): Questions that are open-ended and do not have a single correct answer. These include opinion-based questions, creative tasks, interpretation tasks, and questions that allow for multiple valid perspectives.

Instructions:
1. Carefully analyze both the image content and the conversation
2. Determine whether the question has a definite answer (objective) or is open-ended (subjective)
3. Respond with ONLY one word: "objective" or "subjective"
4. Do not include any explanations, reasoning, or additional text

Your response (one word only):"""
    return prompt


def classify_single_record(
    api_client: VisionAPIClient,
    record: Dict[str, Any],
    max_retries: int = 3
) -> Optional[str]:
    """
    Classify a single record

    Args:
        api_client: VisionAPIClient instance
        record: a single JSONL record
        max_retries: maximum number of retries

    Returns:
        classification result (topic name), returns None on failure
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

    # Build the classification prompt
    question = build_classification_prompt(conversations)

    # Call the API
    for attempt in range(max_retries):
        try:
            response = call_vision_api(
                api_client=api_client,
                image=image_info,
                question=question
            )

            if response:
                # Clean the response and extract the category name
                topic = extract_topic_from_response(response)
                if topic:
                    return topic
                else:
                    logger.warning(
                        f"Record {record.get('id', 'unknown')} could not extract topic from response: {response[:100] if len(str(response)) > 100 else response}"
                    )
                    # If the response is non-empty but topic cannot be extracted, do not retry (possibly a model output format issue)
                    return None
            else:
                # response is None, meaning the API call failed
                logger.warning(
                    f"Record {record.get('id', 'unknown')} API call returned None (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Record {record.get('id', 'unknown')} classification failed: API call returned None")
                    return None

        except Exception as e:
            logger.warning(
                f"Record {record.get('id', 'unknown')} API call exception (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"Record {record.get('id', 'unknown')} classification failed: {e}")
                return None

    return None


def classify_question_type(
    api_client: VisionAPIClient,
    record: Dict[str, Any],
    max_retries: int = 3
) -> Optional[str]:
    """
    Classify a single record as objective/subjective question

    Args:
        api_client: VisionAPIClient instance
        record: a single JSONL record
        max_retries: maximum number of retries

    Returns:
        classification result ("objective" or "subjective"), returns None on failure
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

    # Build the classification prompt
    question = build_question_type_prompt(conversations)

    # Call the API
    for attempt in range(max_retries):
        try:
            response = call_vision_api(
                api_client=api_client,
                image=image_info,
                question=question
            )

            if response:
                # Clean the response and extract the question type
                question_type = extract_question_type_from_response(response)
                if question_type:
                    return question_type
                else:
                    logger.warning(
                        f"Record {record.get('id', 'unknown')} could not extract question type from response: {response[:100] if len(str(response)) > 100 else response}"
                    )
                    # If the response is non-empty but question type cannot be extracted, do not retry (possibly a model output format issue)
                    return None
            else:
                # response is None, meaning the API call failed
                logger.warning(
                    f"Record {record.get('id', 'unknown')} question type classification API call returned None (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Record {record.get('id', 'unknown')} question type classification failed: API call returned None")
                    return None

        except Exception as e:
            logger.warning(
                f"Record {record.get('id', 'unknown')} question type classification API call exception (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"Record {record.get('id', 'unknown')} question type classification failed: {e}")
                return None

    return None


def extract_topic_from_response(response: str) -> Optional[str]:
    """
    Extract the topic name from the model response

    Only accepts an exactly matching topic name (case-insensitive, ignoring leading/trailing whitespace); no fuzzy matching

    Args:
        response: response text returned by the model

    Returns:
        topic name, or None if no exact match is found
    """
    if not response:
        return None

    # Clean the response text, removing leading/trailing whitespace and quotes
    response = response.strip().strip('"').strip("'").strip()

    # Only perform exact matching (case-insensitive)
    # First try to match the entire response directly
    for topic_name in TOPICS.keys():
        if response.lower() == topic_name.lower():
            return topic_name

    # If the response has multiple lines, try to match the first line
    first_line = response.split('\n')[0].strip().strip('"').strip("'").strip()
    for topic_name in TOPICS.keys():
        if first_line.lower() == topic_name.lower():
            return topic_name

    # If the response contains punctuation, try matching after removing punctuation
    cleaned_response = response.strip(string.punctuation).strip()
    for topic_name in TOPICS.keys():
        if cleaned_response.lower() == topic_name.lower():
            return topic_name

    # If the first line still does not match after cleaning, return None (strict matching)
    return None


def extract_question_type_from_response(response: str) -> Optional[str]:
    """
    Extract the question type (objective or subjective) from the model response

    Args:
        response: response text returned by the model

    Returns:
        "objective" or "subjective", or None if no match is found
    """
    if not response:
        return None

    # Clean the response text, removing leading/trailing whitespace and quotes
    response = response.strip().strip('"').strip("'").strip().lower()

    # If the response has multiple lines, take only the first line
    first_line = response.split('\n')[0].strip().strip(
        '"').strip("'").strip().lower()

    # Remove punctuation
    cleaned_response = first_line.strip(string.punctuation).strip()

    # Match "objective" or "subjective"
    if cleaned_response == "objective":
        return "objective"
    elif cleaned_response == "subjective":
        return "subjective"

    # If the first line still does not match after cleaning, return None
    return None


def get_processed_ids(output_file: str) -> set:
    """
    Read the IDs of records already successfully processed from the output file (records that have a topic field)

    Args:
        output_file: output file path

    Returns:
        set of IDs of already-processed records
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
                    # If it has a non-empty topic field, it was successfully processed
                    if "topic" in data and data["topic"]:
                        record_id = data.get("id") or data.get("berlin_id")
                        if record_id:
                            processed_ids.add(str(record_id))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Error reading output file: {e}")

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
    Process a single record concurrently (used by the thread pool)

    Args:
        record_data: (line_num, data, record_id) tuple
        api_client: VisionAPIClient instance
        processed_ids: set of already-processed record IDs
        output_file: output file path
        output_lock: file write lock
        skip_processed: whether to skip already-processed records
        verbose: whether to print detailed information

    Returns:
        processing result dictionary
    """
    line_num, data, record_id = record_data

    result = {
        "record_id": record_id,
        "line_num": line_num,
        "success": False,
        "topic": None,
        "question_type": None,
        "skipped": False,
        "failed": False
    }

    # Check whether it has already been processed
    if skip_processed and record_id in processed_ids:
        result["skipped"] = True
        return result

    # If the input data already has a non-empty topic field
    if "topic" in data and data["topic"]:
        if record_id in processed_ids:
            result["skipped"] = True
            return result
        # Needs to be written to the output file (if question_type is missing, classify it)
        if "question_type" not in data or not data["question_type"]:
            question_type = classify_question_type(api_client, data)
            if question_type:
                data["question_type"] = question_type
                result["question_type"] = question_type
        else:
            result["question_type"] = data["question_type"]
        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        result["success"] = True
        result["topic"] = data["topic"]
        return result

    # Perform classification (topic and question_type)
    topic = classify_single_record(api_client, data)
    question_type = classify_question_type(api_client, data)

    # Only save when topic classification succeeds (question_type failure does not affect saving)
    if topic:
        # Classification succeeded
        data["topic"] = topic
        if question_type:
            data["question_type"] = question_type
            result["question_type"] = question_type
        result["success"] = True
        result["topic"] = topic

        # Write to file in a thread-safe manner
        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    else:
        # Classification failed
        result["failed"] = True
        if verbose:
            logger.warning(f"Record {record_id} classification failed, skipping without saving")

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
    Process a JSONL file, adding topic classification and question type classification to each record

    Args:
        input_file: input JSONL file path
        output_file: output JSONL file path
        api_client: VisionAPIClient instance
        start_index: which record to start processing from (for resume, based on line number)
        max_items: maximum number of records to process (None means process all)
        verbose: whether to print detailed information
        skip_processed: whether to automatically skip already successfully processed records (based on ID)
        max_workers: number of concurrent threads (0 or 1 means no concurrency)

    Returns:
        statistics dictionary (including topic_distribution and question_type_distribution)
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
            logger.info(f"Found {len(processed_ids)} already successfully processed records, will skip automatically")

    stats = {
        "total_lines": 0,
        "processed_lines": 0,
        "successful_classifications": 0,
        "failed_classifications": 0,
        "skipped_lines": 0,
        "already_processed": 0,
        "topic_distribution": {},
        "question_type_distribution": {}
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
            logger.info(f"Using concurrent processing, number of threads: {max_workers}")
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

                # Check whether the maximum processing count has been reached
                if max_items and processed_count >= max_items:
                    break

                try:
                    # Parse the JSON line
                    data = json.loads(line)

                    if not isinstance(data, dict):
                        if verbose:
                            logger.warning(f"Line {line_num} is not in dict format, skipping")
                        stats["skipped_lines"] += 1
                        continue

                    record_id = str(data.get("id") or data.get(
                        "berlin_id") or line_num)

                    # Check whether it has already been processed
                    if skip_processed and record_id in processed_ids:
                        stats["already_processed"] += 1
                        continue

                    # If the input data already has a non-empty topic field
                    if "topic" in data and data["topic"]:
                        if record_id in processed_ids:
                            stats["already_processed"] += 1
                            continue
                        # If question_type is missing, classification is needed
                        if "question_type" not in data or not data["question_type"]:
                            question_type = classify_question_type(
                                api_client, data)
                            if question_type:
                                data["question_type"] = question_type
                        # Write directly (no API call needed)
                        with output_lock:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(
                                    data, ensure_ascii=False) + '\n')
                        stats["processed_lines"] += 1
                        stats["successful_classifications"] += 1
                        topic = data["topic"]
                        stats["topic_distribution"][topic] = stats["topic_distribution"].get(
                            topic, 0) + 1
                        if "question_type" in data and data["question_type"]:
                            qtype = data["question_type"]
                            stats["question_type_distribution"][qtype] = stats["question_type_distribution"].get(
                                qtype, 0) + 1
                        continue

                    # Add to the list of records to process
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
                        verbose
                    ): record_data
                    for record_data in records_to_process
                }

                # Use a progress bar to display progress
                with tqdm(total=len(records_to_process), desc="Progress", disable=not verbose) as pbar:
                    for future in as_completed(future_to_record):
                        try:
                            result = future.result()
                            record_data = future_to_record[future]

                            if result["skipped"]:
                                stats["already_processed"] += 1
                            elif result["success"]:
                                stats["processed_lines"] += 1
                                stats["successful_classifications"] += 1
                                topic = result["topic"]
                                stats["topic_distribution"][topic] = stats["topic_distribution"].get(
                                    topic, 0) + 1
                                if result.get("question_type"):
                                    qtype = result["question_type"]
                                    stats["question_type_distribution"][qtype] = stats["question_type_distribution"].get(
                                        qtype, 0) + 1
                                if verbose:
                                    logger.info(
                                        f"Record {result['record_id']} classified as: {topic}" +
                                        (f", question type: {result.get('question_type', 'N/A')}" if result.get('question_type') else ""))
                            else:
                                stats["failed_classifications"] += 1

                            pbar.update(1)
                        except Exception as e:
                            if verbose:
                                logger.error(f"Error while processing record: {e}")
                            stats["failed_classifications"] += 1
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
                    verbose
                )

                if result["skipped"]:
                    stats["already_processed"] += 1
                elif result["success"]:
                    stats["processed_lines"] += 1
                    stats["successful_classifications"] += 1
                    topic = result["topic"]
                    stats["topic_distribution"][topic] = stats["topic_distribution"].get(
                        topic, 0) + 1
                    if result.get("question_type"):
                        qtype = result["question_type"]
                        stats["question_type_distribution"][qtype] = stats["question_type_distribution"].get(
                            qtype, 0) + 1
                    if verbose:
                        logger.info(f"Record {result['record_id']} classified as: {topic}" +
                                    (f", question type: {result.get('question_type', 'N/A')}" if result.get('question_type') else ""))
                else:
                    stats["failed_classifications"] += 1

        # Print statistics
        if verbose:
            logger.info("\n" + "="*60)
            logger.info("Processing complete! Statistics:")
            logger.info("="*60)
            logger.info(f"Total lines: {stats['total_lines']}")
            logger.info(f"Processed lines: {stats['processed_lines']}")
            logger.info(f"Successful classifications: {stats['successful_classifications']}")
            logger.info(f"Failed classifications: {stats['failed_classifications']}")
            logger.info(f"Skipped lines: {stats['skipped_lines']}")
            if stats.get('already_processed', 0) > 0:
                logger.info(f"Already processed (skipped): {stats['already_processed']}")

            if stats["topic_distribution"]:
                logger.info("\nTopic distribution:")
                for topic, count in sorted(stats["topic_distribution"].items(), key=lambda x: -x[1]):
                    logger.info(f"  {topic}: {count}")

            if stats["question_type_distribution"]:
                logger.info("\nQuestion type distribution:")
                for qtype, count in sorted(stats["question_type_distribution"].items(), key=lambda x: -x[1]):
                    qtype_name = "Objective" if qtype == "objective" else "Subjective" if qtype == "subjective" else qtype
                    logger.info(f"  {qtype_name} ({qtype}): {count}")

            logger.info("="*60)
            logger.info(f"Output file saved to: {output_file}")
            if stats['failed_classifications'] > 0:
                logger.warning(
                    f"Note: {stats['failed_classifications']} records failed classification and "
                    f"were not saved to the output file. Re-run the program to process these records."
                )

        return stats

    except Exception as e:
        logger.error(f"An error occurred while processing the file: {e}")
        raise


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description='Classify JSONL data by topic by calling the MLLM API'
    )
    parser.add_argument('input', help='input JSONL file path')
    parser.add_argument('-o', '--output', required=True, help='output JSONL file path')
    parser.add_argument('--api-endpoints', nargs='+', required=True,
                        help='list of API endpoints, e.g.: http://<VLLM_ENDPOINT>/v1 http://<VLLM_ENDPOINT>/v1')
    parser.add_argument('--api-key', default='sk-abc123', help='API key')
    parser.add_argument('--model-name', default='qwen_3', help='model name')
    parser.add_argument('--temperature', type=float, default=0.7, help='temperature parameter')
    parser.add_argument('--max-tokens', type=int,
                        default=16384, help='maximum number of generated tokens')
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

    # Show the actually received max_workers argument value (for debugging)
    logger.info(f"Concurrent threads argument: {args.max_workers}")

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
            max_workers=args.max_workers
        )

        # Return status code
        sys.exit(0 if stats["failed_classifications"] == 0 else 1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


# python /path/to/code/scirpts/classify_topic.py /path/to/data/sft_evol/ov_800k/sft_path.jsonl -o /path/to/data/sft_evol/ov_800k/sft_topic.jsonl \
#     --api-endpoints http://<VLLM_ENDPOINT>/v1 \
#     --api-key sk-abc123 \
#     --model-name /path/to/model/Qwen3-VL-235B-A22B-Instruct-FP8 \
#     --max-workers 1


# vllm serve /path/to/model/Qwen3-VL-235B-A22B-Instruct-FP8 --tensor-parallel-size 4 --max-model-len 32768 --reasoning-parser deepseek_r1
