import json
import time

import google.generativeai as genai

import config

genai.configure(api_key=config.GEMINI_API_KEY)

ENTITY_KEYS = ["equipment_tags", "dates", "personnel", "regulatory_refs", "document_refs"]

PROMPT_TEMPLATE = """You are an industrial document entity extractor. Extract entities from the text below.

Return ONLY valid JSON (no markdown fences, no commentary) matching this exact shape:
{{
  "equipment_tags": ["<equipment tag strings, e.g. P-101, HX-204>"],
  "dates": ["<dates as they appear in the text>"],
  "personnel": ["<person names>"],
  "regulatory_refs": ["<regulatory/standard references, e.g. OISD-STD-118>"],
  "document_refs": ["<references to other document IDs, e.g. WO-2044>"]
}}

If a category has no entities, return an empty list for it.

Text:
\"\"\"
{text}
\"\"\"
"""


def _parse_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {key: [] for key in ENTITY_KEYS}
    return {key: [str(v) for v in data.get(key, [])] for key in ENTITY_KEYS}


def extract_entities(text: str, max_retries: int = 3) -> dict:
    if not text.strip():
        return {key: [] for key in ENTITY_KEYS}

    model = genai.GenerativeModel(config.GEMINI_MODEL)
    prompt = PROMPT_TEMPLATE.format(text=text[:6000])

    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return _parse_response(response.text)
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    print(f"entity_extractor: giving up after {max_retries} attempts: {last_error}")
    return {key: [] for key in ENTITY_KEYS}
