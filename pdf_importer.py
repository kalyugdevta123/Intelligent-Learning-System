import json
import os
import re
from typing import Any

import pdfplumber
import requests

from llm_client import chat_completion


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"^_+|_+$", "", s)
    return s or "imported_topic"


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 12) -> str:
    """Extract text from a PDF (best-effort)."""
    with pdfplumber.open(io_bytes(pdf_bytes)) as pdf:
        texts: list[str] = []
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text)
        return "\n\n".join(texts).strip()


def io_bytes(b: bytes):
    import io
    return io.BytesIO(b)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # Remove ```json ... ``` wrappers if the model includes them.
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```$", "", s).strip()
    return s


def parse_questions_with_llm(
    raw_text: str,
    topic_title: str,
    max_questions: int = 15,
) -> dict[str, Any]:
    """
    Parse MCQs from raw_text using an OpenAI-compatible endpoint.

    Configure via environment variables:
      - LLM_API_BASE (default: http://127.0.0.1:1234/v1)
      - LLM_API_KEY (optional)
      - LLM_MODEL (optional; if not set, this function auto-detects a loaded model)
    """
    # Keep context bounded to reduce timeout risk on local models.
    raw_text = raw_text[:12000]

    prompt = f"""
Extract multiple-choice questions from the following quiz text.
Return ONLY valid JSON (no markdown, no backticks).

Schema:
{{
  "title": "{topic_title}",
  "difficulty_order": ["easy","medium","hard"],
  "questions": [
    {{
      "concept": "short concept tag",
      "difficulty": "easy|medium|hard",
      "text": "question statement",
      "options": ["A","B","C","D"],
      "answer_index": 0,
      "explanation": "1-2 sentences"
    }}
  ]
}}

Rules:
- Produce at most {max_questions} questions (prefer quality over quantity).
- If difficulty is not explicit, infer it from complexity.
- Ensure options has exactly 4 items and answer_index points to the correct one.

Quiz text:
{raw_text}
""".strip()

    content = chat_completion(prompt, temperature=0.2, max_tokens=1800)
    content = _strip_code_fences(content)
    return json.loads(content)


def build_imported_topic(
    llm_output: dict[str, Any],
    topic_title: str,
    topic_key: str,
) -> dict[str, Any]:
    """
    Normalize LLM output to match the app's topics.json schema.
    """
    questions = llm_output.get("questions", [])
    normalized_questions: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        try:
            options = q.get("options", [])
            if not isinstance(options, list) or len(options) != 4:
                continue

            normalized_questions.append(
                {
                    "id": f"imp_{topic_key}_{i}",
                    "difficulty": q.get("difficulty", "medium"),
                    "concept": q.get("concept", "general"),
                    "text": q.get("text", "").strip(),
                    "options": [str(x) for x in options],
                    "answer_index": int(q.get("answer_index", 0)),
                    "explanation": q.get("explanation", ""),
                }
            )
        except Exception:
            continue

    return {
        "title": topic_title,
        "difficulty_order": ["easy", "medium", "hard"],
        "questions": normalized_questions,
    }


def import_pdf_to_topic_dict(
    pdf_bytes: bytes,
    topic_title: str,
    topic_key: str,
    max_pages: int = 12,
    max_questions: int = 15,
) -> dict[str, Any]:
    raw_text = extract_text_from_pdf_bytes(pdf_bytes, max_pages=max_pages)
    if len(raw_text) < 200:
        raise RuntimeError("PDF extraction produced too little text; try another PDF.")

    llm_output = parse_questions_with_llm(
        raw_text=raw_text,
        topic_title=topic_title,
        max_questions=max_questions,
    )
    return build_imported_topic(llm_output, topic_title=topic_title, topic_key=topic_key)


def default_topic_key(topic_title: str, existing_keys: set[str]) -> str:
    base = _slugify(topic_title)
    key = base
    j = 1
    while key in existing_keys:
        key = f"{base}_{j}"
        j += 1
    return key

