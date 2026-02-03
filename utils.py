import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

def parse_llm_json(raw_text: str) -> Optional[Any]:
    """
    Robustly parses JSON from LLM responses, handling markdown blocks and extra text.
    """
    if not raw_text:
        return None

    # Step 1: Try direct parsing
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Step 2: Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    # This regex looks for code blocks and captures the content inside.
    # It tries to find 'json' tag first, then falls back to any code block.
    code_block_patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```"
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, raw_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Step 3: Last ditch attempt - find the first '{' and last '}'
    try:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = raw_text[start_idx:end_idx + 1]
            return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        pass

    logger.error(f"Failed to parse JSON from LLM response: {raw_text[:200]}...")
    return None
