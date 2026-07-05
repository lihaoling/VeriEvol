OCR_prompt = """You are a ruthless difficulty engineer for document intelligence tasks. Your ONLY job is to take a simple OCR question and transform it into a SIGNIFICANTLY HARDER document reasoning problem.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. STRUCTURAL & SPATIAL REASONING: Demand understanding of the 2D layout — row/column relationships, hierarchical structure, reading order across complex layouts (tables, forms, multi-column text).

2. CROSS-DOCUMENT AGGREGATION: Require synthesizing information from multiple disparate locations on the document and performing logical or mathematical checks (e.g., "Does the sum of line items match the total?").

3. SEMANTIC ROLE INFERENCE: Ask about the *function* or *meaning* of text, not just its content — e.g., which field is a header vs. a value, what role a number plays in context.

4. MULTIMODAL ELEMENT INTERACTION: Require interpreting non-textual elements (logos, stamps, signatures, checkboxes, charts) in relation to surrounding text.

5. PRECISE EXTRACTION WITH REASONING: Demand exact values that require disambiguation (e.g., distinguishing between multiple dates, addresses, or amounts in the document).

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask for information readable from a single, obvious text region.
- The answer must require cross-referencing or reasoning across multiple document regions.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the document image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:

"""

Image_Description_prompt = """You are a ruthless difficulty engineer for image description tasks. Your ONLY job is to take a simple description question and transform it into a SIGNIFICANTLY HARDER visual interpretation challenge.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. INFERENTIAL REASONING (Why/How): Instead of "what is in the image", demand inference of intent, causality, or consequences based on visual evidence. The answer must require a chain of reasoning, not just observation.

2. COMPOSITIONAL & STYLISTIC ANALYSIS: Require analysis of how the image was created — lighting, color palette, framing, perspective, depth of field — and their effect on meaning or mood.

3. CROSS-ELEMENT NARRATIVE SYNTHESIS: Force connecting disparate visual elements (foreground + background, multiple subjects, text + imagery) into a coherent interpretation. Single-element questions are FORBIDDEN.

4. CULTURAL & CONTEXTUAL DEPTH: Demand connecting visual cues to cultural, historical, or social context that goes beyond surface-level observation.

5. COMPARATIVE / CONTRASTIVE ANALYSIS: Ask for comparison between elements within the image, or between what is shown and what is notably absent.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question or add filler words.
- NEVER ask generic questions like "describe this image" or "what do you see".
- The cognitive demand must be fundamentally higher than the original.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""


Detection_prompt = """You are a ruthless difficulty engineer for visual detection tasks. Your ONLY job is to take a simple detection/identification question and transform it into a SIGNIFICANTLY HARDER visual reasoning challenge.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. RELATIONAL & COMPOSITIONAL REASONING: Instead of identifying isolated objects, demand understanding spatial, functional, or logical relationships between multiple detected elements.

2. PRECISE COUNTING & COMPARATIVE ANALYSIS: Require exact counts, size comparisons, density estimations, or relative positioning that demands careful inspection of the entire scene.

3. CONTEXTUAL & CONSEQUENTIAL INFERENCE: Force the model to infer the state, purpose, or implications of detected elements by connecting visual evidence to real-world knowledge (e.g., "Based on the positions of X and Y, what activity is taking place?").

4. CAUSAL / COUNTERFACTUAL ANALYSIS: Ask "what would change if object X were removed?" or "what caused this arrangement?" — go beyond static identification to situational reasoning.

5. FINE-GRAINED DISCRIMINATION: Require distinguishing between visually similar objects, detecting subtle anomalies, or identifying objects partially occluded or in unusual states.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase "what is this?" or "where is X?".
- NEVER ask about a single, obvious object in isolation.
- The answer must require cross-referencing or reasoning about multiple visual elements.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""

Analysis_prompt = """You are a ruthless difficulty engineer for visual analysis tasks. Your ONLY job is to take a simple analysis question and transform it into a SIGNIFICANTLY HARDER analytical reasoning challenge.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. MULTI-STEP INFERENCE CHAIN: Demand a logical chain — identify X, infer its state from Y, combine with Z to reach a non-obvious conclusion. Single-hop observations are FORBIDDEN.

2. WORLD KNOWLEDGE INTEGRATION: Require connecting visual cues to specific domain knowledge (cultural norms, historical context, physics, economics) — not just surface-level recognition.

3. CAUSAL & COUNTERFACTUAL REASONING: Go beyond "what" to "why this happened" or "what if X were different" — probe cause-effect relationships and hypothetical scenarios grounded in the image.

4. HOLISTIC CROSS-IMAGE SYNTHESIS: The answer must require integrating information from multiple regions (foreground vs. background, left vs. right, multiple subjects).

5. DEEP ANALYSIS VECTORS (target at least one):
   - Aesthetic/Compositional: Analyze composition, lighting, color theory, framing, visual hierarchy.
   - Emotional/Narrative: Infer unspoken stories, mood, motivations, relationship dynamics.
   - Sociocultural/Symbolic: Unpack symbolism, cultural significance, social commentary.
   - Intent/Audience: Reason about creator's purpose and intended effect on viewers.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask surface-level "what is this" questions.
- The cognitive demand must be fundamentally and obviously higher.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""


Content_Creation_prompt = """You are a ruthless difficulty engineer for content creation tasks. Your ONLY job is to take a simple content creation question and transform it into a SIGNIFICANTLY HARDER creative-strategic challenge.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. STRATEGIC & GOAL-ORIENTED CONSTRAINTS: Specify a concrete target audience, brand persona, platform, or communication objective. The question must demand strategic thinking, not just description.

2. MULTI-ELEMENT NARRATIVE SYNTHESIS: Force connecting disparate visual elements (objects, expressions, lighting, background, text) into a cohesive creative output. Single-element inspiration is FORBIDDEN.

3. CULTURAL & CONTEXTUAL DEPTH: Demand connecting visual cues to cultural contexts, market trends, historical references, or domain-specific knowledge (art history, marketing theory, design principles).

4. CONSTRAINED CREATIVITY: Add specific constraints (word count, tone, format, audience) that force creative problem-solving rather than free-form generation.

5. COMPARATIVE / ADAPTATION: Ask to adapt the visual content for contrasting contexts (e.g., "How would you adapt this for audience X vs. audience Y?") requiring nuanced understanding.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask generic "write a caption" or "describe this for social media" questions.
- The creative and strategic demand must be fundamentally higher.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""


Suggestion_prompt = """You are a ruthless difficulty engineer for suggestion/recommendation tasks. Your ONLY job is to take a simple suggestion question and transform it into a SIGNIFICANTLY HARDER recommendation challenge that demands deep visual analysis and domain expertise.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. MULTI-STEP INFERENCE TO ACTION: Require analyzing multiple visual elements, inferring the user's situation/needs/constraints, and THEN providing a concrete, justified recommendation. Generic suggestions are FORBIDDEN.

2. DOMAIN-SPECIFIC EXPERTISE: Demand suggestions grounded in specific knowledge domains (interior design principles, fashion theory, nutrition science, ergonomics, etc.) — not just common sense.

3. CONSTRAINT-BASED PROBLEM SOLVING: Add realistic constraints (budget, space, time, existing conditions visible in the image) that make the suggestion task a genuine optimization problem.

4. CROSS-ELEMENT SYNTHESIS: The suggestion must require integrating various details from across the image (clothing style + room decor + visible items) — not based on a single element.

5. TRADE-OFF ANALYSIS: Ask for suggestions that involve explicit trade-offs, requiring the model to weigh pros and cons based on visual evidence.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask generic "what would you suggest" questions.
- The recommendation must require genuine visual analysis and domain reasoning.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""


Summarization_prompt = """You are a ruthless difficulty engineer for summarization and information synthesis tasks. Your ONLY job is to take a simple summarization question and transform it into a SIGNIFICANTLY HARDER analytical synthesis challenge.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. CROSS-REGION SYNTHESIS: Require synthesizing information from multiple disparate parts of the visual content — not just extracting from one section. The answer must integrate data from different regions.

2. CONTEXTUAL & INFERENTIAL REASONING: Demand connecting visible information to domain knowledge (legal principles, financial concepts, scientific methodology) or unstated implications.

3. SUBTEXT, BIAS & INTENT ANALYSIS: Go beyond "what is shown" to "why it is presented this way" — probe authorial intent, rhetorical strategy, selective presentation, or potential bias.

4. TRANSFORMATIVE RE-FRAMING: Require re-framing the information for a different purpose or audience, demanding deep understanding rather than simple condensation.

5. COMPARATIVE & EVALUATIVE SYNTHESIS: Ask for comparison of multiple data points, evaluation of consistency, or identification of contradictions within the visual content.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "summarize", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask for a simple summary or description.
- The analytical demand must be fundamentally higher than the original.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""


Logical_Reasoning_prompt = """You are a ruthless difficulty engineer for logical reasoning tasks. Your ONLY job is to take a simple visual question and transform it into a SIGNIFICANTLY HARDER logical reasoning challenge that demands structured inference, not simple recognition.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. DEDUCTIVE MULTI-STEP INFERENCE: Demand a chain of logical steps — identify visual premises (e.g., Object A is wet, window is open), combine them, and deduce a non-obvious conclusion (e.g., it rained recently). Single-hop answers are FORBIDDEN.

2. CAUSAL & COUNTERFACTUAL REASONING: Probe cause-and-effect relationships. Ask "what caused this state?" or "what would change if X were different?" Challenge the model to reason about consequences of hypothetical changes.

3. COMPLEX SPATIAL & RELATIONAL ANALYSIS: Demand deep understanding of spatial relationships, relative positions, orientations, and functional interactions between multiple elements — layouts, lines of sight, accessibility, arrangements.

4. TEMPORAL SEQUENCE & STATE INFERENCE: If the image implies a process or history, require reconstructing the timeline or inferring the next/previous logical step from static visual clues.

5. LOGICAL CONSTRAINT SATISFACTION: Pose questions where the answer must satisfy multiple visual constraints simultaneously, requiring the model to check each constraint against the image.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask questions answerable by looking at a single object.
- The answer must be the result of a genuine logical process, not pattern matching.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""


Scientific_Related_prompt = f"""You are a ruthless difficulty engineer for scientific visual reasoning tasks. Your ONLY job is to take a simple question about a scientific visual and transform it into a SIGNIFICANTLY HARDER scientific reasoning challenge.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. QUANTITATIVE & RELATIONAL REASONING: Demand calculations, trend comparisons, rate-of-change analysis, or inferring relationships between variables — not just reading a single data point.

2. FOUNDATIONAL SCIENTIFIC KNOWLEDGE: Require connecting visual data to underlying scientific principles, theories, or domain knowledge (physics, biology, chemistry, etc.). The answer must be impossible without this external knowledge.

3. INFERENTIAL & HYPOTHETICAL THINKING: Go beyond "what" to "why," "how," and "what if." Probe for causality, require anomaly detection, or ask for predictions based on counterfactual or extrapolated conditions.

4. CROSS-REPRESENTATION SYNTHESIS: Require integrating information from multiple parts of the visual — main plot, inset graphs, legends, axis labels, annotations — to form a comprehensive conclusion. Single-region extraction is FORBIDDEN.

5. ERROR ANALYSIS & METHODOLOGY: Ask about limitations, potential confounding factors, statistical significance, or experimental design implications visible in the data.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask for a single data point readable directly from the visual.
- The answer must require genuine scientific reasoning, not just data extraction.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: {{original_question}}

YOUR OUTPUT:
"""


Concept_Extraction_prompt = """You are a ruthless difficulty engineer for concept extraction tasks. Your ONLY job is to take a simple descriptive question and transform it into a SIGNIFICANTLY HARDER conceptual reasoning challenge that demands deep understanding beyond surface-level recognition.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. CAUSAL & FUNCTIONAL RELATIONSHIPS: Go beyond identifying actions to inferring their purpose and context. Probe the "why" behind an action or the "what for" of an object — demand functional reasoning.

2. ABSTRACT & LATENT ATTRIBUTE INFERENCE: Move from perceptible properties (color, count) to inferred characteristics — material, texture, function, emotional state, social role, or historical significance implied by context.

3. NARRATIVE / HYPOTHESIS SYNTHESIS: Demand connecting disparate visual elements into a coherent explanation, summary, or plausible scenario. The answer must require reasoning across multiple parts of the image.

4. COMPLEX INTER-OBJECT RELATIONS: Force defining non-obvious connections between entities — hierarchical, functional, temporal, or social relationships that go beyond simple spatial placement.

5. ABSTRACTION LEVEL SHIFTING: Require moving between concrete observations and abstract concepts (e.g., from "people are running" to "this depicts competition/urgency/escape").

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask about surface-level properties of a single object.
- The answer must require genuine conceptual reasoning, not pattern recognition.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: {{original_question}}

YOUR OUTPUT:
"""


Medical_Image_Analysis_prompt = """You are a ruthless difficulty engineer for medical image analysis tasks. Your ONLY job is to take a simple clinical question and transform it into a SIGNIFICANTLY HARDER diagnostic reasoning challenge.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. MULTI-STEP DIAGNOSTIC REASONING: Demand a diagnostic chain — identify the primary finding, characterize key features (size, margins, density/intensity), relate to adjacent anatomy, and infer a diagnosis or differential. Simple "Is there a lesion?" questions are FORBIDDEN.

2. CLINICAL & ANATOMICAL KNOWLEDGE INTEGRATION: Require connecting visual findings to specific medical knowledge — anatomy, pathology, disease presentations, physiological processes. The answer must be impossible without domain expertise.

3. PROGNOSTIC & TREATMENT IMPLICATIONS: Go beyond identification to clinical implications — staging, prognosis, treatment planning, or urgency assessment based on the findings.

4. MULTI-REGIONAL SYNTHESIS: Require synthesizing findings from multiple image regions — comparing contralateral structures, correlating findings across organ systems, or assessing the relationship between primary and secondary findings.

5. DIFFERENTIAL DIAGNOSIS WITH DISCRIMINATION: Demand distinguishing between visually similar pathologies by identifying discriminating features visible in the image.

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase "What is this finding?".
- NEVER ask about a single obvious finding without requiring deeper analysis.
- The answer must require genuine clinical reasoning, not just pattern recognition.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the medical image.
3. Style: Use diverse styles (clinical scenario, consultation role-play, board-exam format). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Medical Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""


Scene_Understanding_prompt = """You are a ruthless difficulty engineer for scene understanding tasks. Your ONLY job is to take a simple scene question and transform it into a SIGNIFICANTLY HARDER holistic scene reasoning challenge.

//-- DIFFICULTY ESCALATION RULES (apply AT LEAST TWO) --//

1. DYNAMIC INTERPRETATION: Force interpreting ongoing activities, gestures, expressions, and intent — not just identifying static objects. Ask what is happening and why, not just what is present.

2. RELATIONSHIP & INTERACTION ANALYSIS: Probe the complex interplay between people, objects, and environment. Demand analysis of functional, social, or causal relationships between scene elements.

3. NARRATIVE & SOCIAL SYNTHESIS: Require a holistic interpretation — constructing a plausible story, identifying the social context, or inferring the event type by synthesizing ALL visual cues across the entire image.

4. SPATIAL & CONTEXTUAL REASONING: Demand using spatial arrangements and environmental cues to judge the nature of the place, situation, or event — going beyond simple location identification.

5. CAUSAL & COUNTERFACTUAL INFERENCE: Push beyond observation to inference — "what caused this scene?", "what will happen next?", "how would the meaning change if X were different?"

//-- OBJECTIVE QUESTION REQUIREMENT (MANDATORY) --//
The evolved question MUST have a SINGLE, FIXED, VERIFIABLE answer (a concrete fact, number, name, choice, or short phrase).
Preferred formats: multiple-choice (A/B/C/D with plausible distractors), fill-in-the-blank, or direct short-answer.
FORBIDDEN: "describe", "explain", "what do you think", or any question where multiple valid answers exist.
If the original question is subjective, convert it into an objective question.

//-- ANTI-TRIVIAL RULES --//
- NEVER simply rephrase the original question.
- NEVER ask "what is in this image" or other surface-level questions.
- The answer must require integrating multiple scene elements and genuine reasoning.

//-- OUTPUT RULES --//

1. Consistency: Maintain the number of sub-questions as in the original.
2. Visual Dependency: The question MUST be unanswerable without the image.
3. Style: Use diverse styles (colloquial, role-playing, scenario-based). Avoid rigid, formulaic patterns.
4. Language: Keep the same language as the original question.
5. Type Consistency: The new question must be {{question_type}}.
6. Format: Output ONLY a valid JSON object:

{{
  "evol_question": "your evolved question here"
}}

No markdown, no explanation, no extra text.

//-- YOUR TASK BEGINS NOW --//

Image: [Image data will be provided]

Original Question: "{{original_question}}"

YOUR OUTPUT:
"""
