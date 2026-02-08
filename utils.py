import json
import re
import logging
import sys
import time
import random
import threading
from typing import Any, Optional, List

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

class ThinkingSpinner:
    """Animated spinner that cycles through themed words while waiting for AI."""

    THINKING = [
        "Pondering", "Divining", "Conjuring", "Scheming", "Weaving fate",
        "Consulting the stars", "Scrying", "Channeling", "Meditating",
        "Communing with spirits", "Deciphering", "Contemplating",
        "Gazing into the aether", "Reading the runes", "Unraveling threads",
        "Whispering to shadows", "Sifting through omens", "Peering beyond the veil",
    ]
    COOKING = [
        "Stirring the pot", "Adding a pinch of magic", "Simmering",
        "Tasting the broth", "Chopping ingredients", "Seasoning",
        "Letting it bubble", "Adjusting the flame",
    ]
    ENTERING = [
        "Pushing the door", "Stepping inside", "Crossing the threshold",
        "Peering into the darkness", "Entering cautiously",
    ]
    STUDYING = [
        "Studying the text", "Tracing the runes", "Decoding symbols",
        "Turning pages carefully", "Squinting at faded ink",
    ]

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, words: List[str] = None, newline: bool = True):
        self._words = words or self.THINKING
        self._newline = newline
        self._stop = threading.Event()
        self._thread = None
        self._max_len = 0

    def _animate(self):
        words = self._words[:]
        random.shuffle(words)
        idx, frame = 0, 0
        last_switch = time.time()
        prefix = "\n" if self._newline else ""

        while not self._stop.is_set():
            word = words[idx % len(words)]
            spinner = self.FRAMES[frame % len(self.FRAMES)]
            text = f"{prefix}{Colors.CYAN}{spinner} {word}...{Colors.ENDC}"
            visible_len = len(f"{spinner} {word}...") + (1 if prefix else 0)
            self._max_len = max(self._max_len, visible_len)
            sys.stdout.write(f"\r{' ' * self._max_len}\r{text}")
            sys.stdout.flush()
            prefix = ""  # Only first frame gets the newline

            frame += 1
            if time.time() - last_switch > 2.0:
                idx += 1
                last_switch = time.time()
            self._stop.wait(0.08)

        sys.stdout.write(f"\r{' ' * (self._max_len + 2)}\r")
        sys.stdout.flush()

    def __enter__(self):
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()


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
