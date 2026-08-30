from __future__ import annotations

import json
import re
from loguru import logger
from src.evaluation.models import JudgeScores
from src.evaluation.llm_clients import BaseLLMClient, get_eval_llm

JUDGE_SYSTEM_PROMPT = """You are an impartial, expert evaluation judge for an academic Question-Answering RAG benchmark.
You will evaluate the quality of a generated answer given:
1. User Question
2. Ground Truth Answer
3. Retrieved Context
4. Generated Answer

Evaluate strictly across these 3 criteria on a scale of 0 to 100:

1. CORRECTNESS (0-100):
   - How accurate and complete is the generated answer compared to the Ground Truth?
   - 100: Perfectly matches ground truth facts and key concepts.
   - 75-90: Mostly accurate with minor omissions.
   - 40-70: Partially accurate, misses important points, or contains slight inaccuracies.
   - 0-30: Completely wrong, contradictory, or empty.

2. FAITHFULNESS (0-100):
   - Is the generated answer supported ONLY by the provided Retrieved Context?
   - 100: Every claim in the answer is grounded in the retrieved context (no hallucinations).
   - 70-90: Minor statements not explicitly in context but logically inferred.
   - 30-60: Significant hallucination or ungrounded external assumptions.
   - 0-20: Entirely hallucinated or contradicts the context.

3. RELEVANCE (0-100):
   - Does the generated answer directly, clearly, and concisely address the User Question?
   - 100: Directly and fully answers the question with no irrelevant fluff.
   - 70-90: Answers the question with some extra or slightly redundant information.
   - 40-60: Vaguely addresses the question.
   - 0-30: Refuses to answer or goes completely off-topic.

Compute the OVERALL score as:
overall = round(0.4 * correctness + 0.3 * faithfulness + 0.3 * relevance, 1)

You MUST reply ONLY with a valid JSON object matching this exact format:
{
  "correctness": <float 0-100>,
  "faithfulness": <float 0-100>,
  "relevance": <float 0-100>,
  "overall": <float 0-100>,
  "reason": "<1-2 sentence concise explanation>"
}
Do not include any other text before or after the JSON block."""

JUDGE_USER_PROMPT = """### User Question:
{question}

### Ground Truth:
{ground_truth}

### Retrieved Context:
{context}

### Generated Answer:
{generated_answer}

Provide your structured JSON evaluation:"""


class LLMJudge:
    """LLM-as-Judge evaluator for assessing RAG outputs."""

    def __init__(self, judge_client: BaseLLMClient | None = None, judge_model_name: str = "auto"):
        import os

        if judge_client is not None:
            self.client = judge_client
        else:
            if judge_model_name in ["auto", "default", "model_1"]:
                if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                    judge_model_name = "gemini"
                else:
                    judge_model_name = "model_1"
            self.client = get_eval_llm(judge_model_name)
        self.judge_model_name = self.client.model_name

    def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        retrieved_context: str,
        generated_answer: str,
    ) -> JudgeScores:
        """Evaluate a single sample with the LLM judge."""
        if not generated_answer or not generated_answer.strip():
            return JudgeScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason="Generated answer is empty.",
            )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth,
                    context=retrieved_context[:3000] if retrieved_context else "No context retrieved.",
                    generated_answer=generated_answer,
                ),
            },
        ]

        try:
            raw_response = self.client.invoke_messages(messages)
            return self._parse_judge_response(raw_response)
        except Exception as e:
            logger.error(f"[LLMJudge] Error during evaluation: {e}")
            return JudgeScores(
                correctness=50.0,
                faithfulness=50.0,
                relevance=50.0,
                overall=50.0,
                reason=f"Judge evaluation encountered error: {str(e)[:100]}",
            )

    def _parse_judge_response(self, raw_text: str) -> JudgeScores:
        """Parse structured JSON from judge output with fallbacks."""
        text = raw_text.strip()
        # Strip markdown code fencing if present
        if "```json" in text:
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            try:
                data = json.loads(json_str)
                c = float(data.get("correctness", 0.0))
                f = float(data.get("faithfulness", 0.0))
                r = float(data.get("relevance", 0.0))
                o = float(data.get("overall", round(0.4 * c + 0.3 * f + 0.3 * r, 1)))
                reason = str(data.get("reason", ""))
                return JudgeScores(
                    correctness=max(0.0, min(100.0, c)),
                    faithfulness=max(0.0, min(100.0, f)),
                    relevance=max(0.0, min(100.0, r)),
                    overall=max(0.0, min(100.0, o)),
                    reason=reason,
                )
            except Exception as e:
                logger.warning(f"[LLMJudge] Failed to parse JSON: {e} from text: {text[:200]}")

        # Fallback regex extraction
        c_match = re.search(r'"correctness"\s*:\s*([0-9.]+)', text)
        f_match = re.search(r'"faithfulness"\s*:\s*([0-9.]+)', text)
        r_match = re.search(r'"relevance"\s*:\s*([0-9.]+)', text)
        if c_match and f_match and r_match:
            c = float(c_match.group(1))
            f = float(f_match.group(1))
            r = float(r_match.group(1))
            o = round(0.4 * c + 0.3 * f + 0.3 * r, 1)
            return JudgeScores(correctness=c, faithfulness=f, relevance=r, overall=o, reason="Extracted via regex fallback")

        return JudgeScores(
            correctness=50.0,
            faithfulness=50.0,
            relevance=50.0,
            overall=50.0,
            reason=f"Failed to parse structured output from judge: {raw_text[:100]}",
        )
