from __future__ import annotations
from typing import Dict, List, Optional, Any
import json
import logging
import httpx
from pydantic import BaseModel, ValidationError

from models import (
    Direction, GeneralLocationType, CharacterClass, NPCRole,
    ItemType, ItemRarity, QuestType, EventScope,
    BaseLocation, NPC, BaseCharacter, PlayerCharacter,
    CompleteGameState,
)
from utils import parse_llm_json, Colors
from ai_provider import AIProvider

llm_logger = logging.getLogger("llm_responses")


class EngineResult(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


class BaseEngine:
    """Core game engine base: state, AI, context, helpers"""

    def __init__(self, ai_provider: AIProvider):
        self.ai = ai_provider
        self.game_state: Optional[CompleteGameState] = None
        self.in_combat = False
        self.combat_opponents: List[BaseCharacter] = []
        self.pending_messages: List[str] = []

    async def _generate_and_validate(self, prompt: str, model_name: str) -> Any:
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                raw_response = await self.ai.generate_response(prompt, is_content_generation=True, model_name=model_name)
                parsed = parse_llm_json(raw_response)
                if parsed is not None:
                    return parsed
                raise json.JSONDecodeError("Failed to parse JSON", raw_response, 0)
            except (httpx.RequestError, json.JSONDecodeError, ValidationError) as e:
                last_error = e
                llm_logger.error(f"Generation attempt {attempt + 1}/{max_retries} failed. Error: {e}")
        raise ConnectionError("Failed to generate content after multiple retries.") from last_error

    # ── Generic enum self-healing ──────────────────────────────────────
    _ENUM_DEFAULTS: Dict[type, Any] = {
        GeneralLocationType: GeneralLocationType.CLEARING,
        NPCRole: NPCRole.COMMONER,
        CharacterClass: CharacterClass.COMMONER,
        ItemType: ItemType.TOOL,
        ItemRarity: ItemRarity.COMMON,
        QuestType: QuestType.FETCH,
        EventScope: EventScope.LOCAL,
        Direction: Direction.NORTH,
    }

    @classmethod
    def _coerce_enum(cls, value: Any, enum_cls: type) -> str:
        """Coerce an AI-provided value into a valid enum value string.
        1. Exact match (case-insensitive, normalized).
        2. Substring match against valid values.
        3. Fall back to registered default.
        """
        if not isinstance(value, str):
            value = str(value) if value is not None else ""
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        valid = {e.value: e.value for e in enum_cls}
        if normalized in valid:
            return valid[normalized]
        for v in valid:
            if v in normalized or normalized in v:
                return v
        default = cls._ENUM_DEFAULTS.get(enum_cls)
        return default.value if default else next(iter(valid.values()))

    def _sanitize_location_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize AI-generated location data to prevent Pydantic validation errors"""
        if not isinstance(data, dict):
            return data
        string_fields = ['general_type', 'atmosphere', 'temperature', 'weather', 'name', 'description', 'short_description']
        for field in string_fields:
            if field in data and isinstance(data[field], list):
                data[field] = str(data[field][0]) if data[field] else ""
            elif field in data and data[field] is None:
                data[field] = ""
        if 'general_type' in data:
            data['general_type'] = self._coerce_enum(data['general_type'], GeneralLocationType)
        data.pop('coordinates', None)
        return data

    def _sanitize_npc_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize AI-generated NPC data to prevent Pydantic validation errors"""
        if not isinstance(data, dict):
            return data
        if 'role' in data:
            data['role'] = self._coerce_enum(data['role'], NPCRole)
        if 'character_class' in data:
            data['character_class'] = self._coerce_enum(data['character_class'], CharacterClass)
        if 'stats' in data and 'base_stats' not in data:
            data['base_stats'] = data['stats'].copy() if isinstance(data['stats'], dict) else data['stats']
        return data

    ENTERABLE_KEYWORDS = [
        "tavern", "inn", "shop", "smithy", "temple", "guild", "tower",
        "castle", "fortress", "manor", "hall", "house", "warehouse",
        "library", "church", "barracks", "store", "market", "forge",
    ]

    def _is_enterable_name(self, name: str) -> bool:
        return any(kw in name.lower() for kw in self.ENTERABLE_KEYWORDS)

    def get_current_location(self) -> BaseLocation:
        if not self.game_state or not self.game_state.session.player_character:
            raise ValueError("Not initialized")
        return self.game_state.locations[self.game_state.session.player_character.current_location_id]

    def build_context_for_ai(self, target_npc: Optional[NPC] = None) -> str:
        if not self.game_state:
            return ""
        loc = self.get_current_location()
        p = self.game_state.session.player_character
        npcs = [char for char in self.game_state.characters.values() if char.current_location_id == loc.id and char.id != p.id]

        context_parts = [
            f"WORLD: {self.game_state.session.world.name}",
            f"TIME: {self.game_state.session.game_time.time_of_day.value}",
            f"LOCATION: {loc.name} - {loc.description}",
        ]

        if loc.atmosphere:
            context_parts.append(f"ATMOSPHERE: {loc.atmosphere}")
        if loc.ambient_sounds:
            context_parts.append(f"SOUNDS: {', '.join(loc.ambient_sounds)}")
        if loc.ambient_smells:
            context_parts.append(f"SMELLS: {', '.join(loc.ambient_smells)}")
        if getattr(loc, 'weather', None):
            context_parts.append(f"WEATHER: {loc.weather}")

        context_parts.extend([
            f"NPCS HERE: {', '.join([n.name for n in npcs]) if npcs else 'None'}",
            f"PLAYER: {p.name} (Lvl {p.level} {p.character_class.value})",
        ])

        # Active quests context
        active_quests = [
            self.game_state.quests[qid].name
            for qid in p.active_quests if qid in self.game_state.quests
        ]
        if active_quests:
            context_parts.append(f"ACTIVE QUESTS: {', '.join(active_quests)}")

        # Global events context
        active_events = [e.name for e in self.game_state.session.active_global_events if e.is_active]
        if active_events:
            context_parts.append(f"WORLD EVENTS: {', '.join(active_events)}")

        if target_npc and target_npc.interaction_summary:
            context_parts.append(f"YOUR MEMORY OF {p.name}: {target_npc.interaction_summary}")

        return "\n".join(context_parts)
