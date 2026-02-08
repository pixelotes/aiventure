from datetime import datetime
from uuid import UUID, uuid4
import random
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import httpx
from pydantic import BaseModel, ValidationError
from utils import parse_llm_json, Colors

from models import (
    Direction, Coordinates, LocationConnection, LocationType, NotableFeature, 
    BaseLocation, Region, World, GeneralLocationType, GeneralLocation, CharacterStats, 
    CharacterClass, CharacterType, BaseCharacter, PlayerCharacter, NPCRole, NPCGoal,
    ServiceType, Service, NPC, ItemType, ItemRarity, Item, QuestType,
    QuestStatus, QuestObjective, QuestReward, Quest, TimeOfDay, GameTime, 
    GameSession, CompleteGameState, WorldGrid, GlobalEvent, EventScope
)
from ai_provider import AIProvider

llm_logger = logging.getLogger("llm_responses")

class EngineResult(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

class GameEngine:
    """Core game engine that manages game state and interactions"""
    def __init__(self, ai_provider: AIProvider):
        self.ai = ai_provider
        self.game_state: Optional[CompleteGameState] = None
        self.in_combat = False
        self.combat_opponents: List[BaseCharacter] = []
        self.pending_messages: List[str] = []

    def calculate_equipment_bonuses(self, character: BaseCharacter) -> CharacterStats:
        """Calculate total bonuses from equipped items"""
        bonuses = CharacterStats()
        if not isinstance(character, (PlayerCharacter, NPC)):
            return bonuses
            
        for item_id in character.equipped_items.values():
            item = self.game_state.items.get(item_id)
            if item:
                for stat_name, bonus in item.stat_modifiers.items():
                    if hasattr(bonuses, stat_name):
                        current_value = getattr(bonuses, stat_name)
                        setattr(bonuses, stat_name, current_value + bonus)
        return bonuses
    
    def apply_equipment_effects(self, character: BaseCharacter):
        """Apply equipment bonuses to character stats (reads base_stats, writes stats)"""
        if not isinstance(character, (PlayerCharacter, NPC)):
            return

        base = character.base_stats
        bonuses = self.calculate_equipment_bonuses(character)

        character.stats.strength = max(1, base.strength + bonuses.strength)
        character.stats.dexterity = max(1, base.dexterity + bonuses.dexterity)
        character.stats.constitution = max(1, base.constitution + bonuses.constitution)
        character.stats.intelligence = max(1, base.intelligence + bonuses.intelligence)
        character.stats.wisdom = max(1, base.wisdom + bonuses.wisdom)
        character.stats.charisma = max(1, base.charisma + bonuses.charisma)
        character.stats.armor_class = max(1, base.armor_class + bonuses.armor_class)
        character.stats.attack_bonus = base.attack_bonus + bonuses.attack_bonus
        character.stats.damage_bonus = base.damage_bonus + bonuses.damage_bonus

        character.stats.max_health = max(1, base.max_health + (bonuses.constitution * 5))
        character.stats.max_mana = max(1, base.max_mana + (bonuses.intelligence * 3))
    
    def add_item_to_inventory(self, character: BaseCharacter, item: Item, quantity: int = 1) -> str:
        """Add an item to character's inventory, handling stacking"""
        if item.stackable:
            for existing_item_id in character.inventory:
                existing_item = self.game_state.items.get(existing_item_id)
                if (existing_item and 
                    existing_item.name == item.name and 
                    existing_item.current_stack_size < existing_item.stack_size):
                    
                    space_available = existing_item.stack_size - existing_item.current_stack_size
                    items_to_add = min(quantity, space_available)
                    existing_item.current_stack_size += items_to_add
                    quantity -= items_to_add
                    
                    if quantity == 0:
                        return f"Added {items_to_add} {item.name}(s) to existing stack."
            
            while quantity > 0:
                new_item = Item.model_validate(item.model_dump())
                new_item.id = uuid4()
                items_in_stack = min(quantity, item.stack_size)
                new_item.current_stack_size = items_in_stack
                
                self.game_state.items[new_item.id] = new_item
                character.inventory.append(new_item.id)
                quantity -= items_in_stack
            return f"Added {item.name}(s) to inventory."
        else:
            for _ in range(quantity):
                new_item = Item.model_validate(item.model_dump())
                new_item.id = uuid4()
                self.game_state.items[new_item.id] = new_item
                character.inventory.append(new_item.id)
            return f"Added {quantity} {item.name}(s) to inventory."
    
    def remove_item_from_inventory(self, character: BaseCharacter, item_name: str, quantity: int = 1) -> Tuple[bool, str]:
        """Remove items from inventory, handling stacks"""
        items_to_remove = []
        remaining_quantity = quantity
        
        for item_id in character.inventory:
            if remaining_quantity <= 0: break
            item = self.game_state.items.get(item_id)
            if item and item_name.lower() in item.name.lower():
                if item.stackable and item.current_stack_size > 1:
                    items_from_stack = min(remaining_quantity, item.current_stack_size)
                    item.current_stack_size -= items_from_stack
                    remaining_quantity -= items_from_stack
                    if item.current_stack_size <= 0:
                        items_to_remove.append(item_id)
                else:
                    items_to_remove.append(item_id)
                    remaining_quantity -= 1
        
        for item_id in items_to_remove:
            character.inventory.remove(item_id)
            if item_id in self.game_state.items:
                del self.game_state.items[item_id]
        
        removed_count = quantity - remaining_quantity
        if removed_count > 0:
            return True, f"Removed {removed_count} {item_name}(s) from inventory."
        return False, f"You don't have enough {item_name}(s) to remove."

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

    ITEM_CATALOG = [
        # Weapons
        {"name": "Iron Sword", "description": "A sturdy iron blade.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"attack_bonus": 2, "damage_bonus": 3}, "value": 25},
        {"name": "Oak Staff", "description": "A gnarled wooden staff crackling with faint energy.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"intelligence": 2, "attack_bonus": 1}, "value": 20},
        {"name": "Steel Dagger", "description": "A sharp, lightweight dagger.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"dexterity": 1, "damage_bonus": 2}, "value": 15},
        {"name": "War Hammer", "description": "A heavy hammer that packs a devastating blow.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"strength": 2, "damage_bonus": 4}, "value": 30},
        {"name": "Hunting Bow", "description": "A recurve bow suited for both hunting and combat.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"dexterity": 2, "damage_bonus": 2}, "value": 22},
        # Armor
        {"name": "Leather Armor", "description": "Light but durable leather protection.", "item_type": ItemType.ARMOR, "equipment_slot": "armor", "stat_modifiers": {"armor_class": 2}, "value": 20},
        {"name": "Chainmail", "description": "Interlocking metal rings providing solid defense.", "item_type": ItemType.ARMOR, "equipment_slot": "armor", "stat_modifiers": {"armor_class": 4, "dexterity": -1}, "value": 40},
        {"name": "Mage Robes", "description": "Enchanted robes that enhance magical ability.", "item_type": ItemType.ARMOR, "equipment_slot": "armor", "stat_modifiers": {"armor_class": 1, "intelligence": 2, "max_mana": 10}, "value": 30},
        # Shields
        {"name": "Wooden Shield", "description": "A round wooden shield reinforced with iron.", "item_type": ItemType.ARMOR, "equipment_slot": "shield", "stat_modifiers": {"armor_class": 2}, "value": 12},
        # Consumables
        {"name": "Health Potion", "description": "A red potion that restores 30 health.", "item_type": ItemType.CONSUMABLE, "consumable": True, "self_use_effect_description": "You drink the potion and feel warmth spread through your body. (+30 HP)", "use_effects": ["heal:30"], "value": 10, "stackable": True, "stack_size": 5},
        {"name": "Stamina Tonic", "description": "A green tonic that restores 30 stamina.", "item_type": ItemType.CONSUMABLE, "consumable": True, "self_use_effect_description": "You drink the tonic and feel a surge of energy. (+30 Stamina)", "use_effects": ["stamina:30"], "value": 8, "stackable": True, "stack_size": 5},
        {"name": "Mana Elixir", "description": "A blue elixir that restores 25 mana.", "item_type": ItemType.CONSUMABLE, "consumable": True, "self_use_effect_description": "You drink the elixir and feel your magical reserves replenish. (+25 Mana)", "use_effects": ["mana:25"], "value": 12, "stackable": True, "stack_size": 5},
        # Food Ingredients (for cooking)
        {"name": "Wild Berries", "description": "A handful of sweet, plump berries.", "item_type": ItemType.MATERIAL, "value": 2, "stackable": True, "stack_size": 10, "weight": 0.2},
        {"name": "Forest Mushrooms", "description": "Earthy brown mushrooms with a rich aroma.", "item_type": ItemType.MATERIAL, "value": 3, "stackable": True, "stack_size": 10, "weight": 0.3},
        {"name": "Healing Herbs", "description": "Fragrant green herbs known for their restorative properties.", "item_type": ItemType.MATERIAL, "value": 4, "stackable": True, "stack_size": 10, "weight": 0.1},
        {"name": "Raw Meat", "description": "A cut of fresh game meat.", "item_type": ItemType.MATERIAL, "value": 5, "stackable": True, "stack_size": 5, "weight": 0.5},
        {"name": "Fresh Fish", "description": "A river fish, still glistening.", "item_type": ItemType.MATERIAL, "value": 4, "stackable": True, "stack_size": 5, "weight": 0.4},
        {"name": "Spicy Peppers", "description": "Small red peppers that tingle the tongue.", "item_type": ItemType.MATERIAL, "value": 3, "stackable": True, "stack_size": 10, "weight": 0.1},
        {"name": "Honeycomb", "description": "A golden chunk of wild honeycomb.", "item_type": ItemType.MATERIAL, "value": 5, "stackable": True, "stack_size": 5, "weight": 0.3},
        {"name": "Mountain Root", "description": "A tough, knotted root with warming properties.", "item_type": ItemType.MATERIAL, "value": 4, "stackable": True, "stack_size": 10, "weight": 0.2},
    ]

    FOOD_INGREDIENT_NAMES = {
        "Wild Berries", "Forest Mushrooms", "Healing Herbs", "Raw Meat",
        "Fresh Fish", "Spicy Peppers", "Honeycomb", "Mountain Root",
    }

    COOKING_BUFF_CAPS = {
        "strength": 5, "dexterity": 5, "constitution": 5,
        "intelligence": 5, "wisdom": 5, "charisma": 5,
        "armor_class": 3, "attack_bonus": 3, "damage_bonus": 3,
        "max_health": 30, "max_stamina": 20, "max_mana": 20,
    }

    AI_EFFECT_CAPS = {
        "max_time_minutes": 480,
        "max_gold_give": 50,
        "max_item_value": 30,
        "max_effects": 5,
    }

    # Hunt & Forage tables: terrain -> (success_chance, [(item_name, weight), ...])
    HUNT_TABLE = {
        "forest": (0.70, [("Raw Meat", 5), ("Fresh Fish", 1)]),
        "meadow": (0.65, [("Raw Meat", 4)]),
        "mountain": (0.50, [("Raw Meat", 3)]),
        "clearing": (0.60, [("Raw Meat", 4)]),
        "river": (0.75, [("Fresh Fish", 5)]),
    }
    FORAGE_TABLE = {
        "forest": (0.85, [("Wild Berries", 3), ("Forest Mushrooms", 4), ("Healing Herbs", 2)]),
        "meadow": (0.80, [("Wild Berries", 4), ("Healing Herbs", 3), ("Honeycomb", 1)]),
        "mountain": (0.60, [("Mountain Root", 4), ("Spicy Peppers", 2)]),
        "clearing": (0.80, [("Wild Berries", 3), ("Honeycomb", 2), ("Healing Herbs", 2)]),
        "river": (0.75, [("Healing Herbs", 3), ("Fresh Fish", 2)]),
    }
    HUNT_STAMINA_COST = 15
    HUNT_TIME_MINUTES = 30
    FORAGE_STAMINA_COST = 8
    FORAGE_TIME_MINUTES = 15

    PUZZLE_TYPES = [
        {
            "type": "offering",
            "templates": [
                ("Lonely Offering Shrine", "A small stone shrine with an empty offering bowl. Ancient carvings adorn its sides."),
                ("Weathered Altar", "An ancient altar with a hollow depression in its center, as if awaiting a gift."),
                ("Spirit Cairn", "A pile of stones arranged in a spiral, with a gap at the center."),
            ],
            "hint": "It looks like something should be placed here as an offering.",
            "accepted": [ItemType.CONSUMABLE, ItemType.MATERIAL],
        },
        {
            "type": "arrangement",
            "templates": [
                ("Circle of Stones with a Gap", "Several large stones form an incomplete circle. One position is clearly empty."),
                ("Unfinished Mosaic", "A beautiful tiled mosaic on the ground, missing a key piece."),
                ("Broken Statue Pedestal", "A pedestal with fragments of a statue. Something could complete it."),
            ],
            "hint": "Something seems to be missing from the pattern.",
            "accepted": [ItemType.MATERIAL, ItemType.TOOL],
        },
        {
            "type": "activation",
            "templates": [
                ("Unlit Brazier Trio", "Three ornate braziers stand in a triangle. All are cold and dark."),
                ("Sealed Runic Door", "A stone door covered in faintly glowing runes with a keyhole-shaped indent."),
                ("Dormant Crystal Obelisk", "A tall crystal obelisk, dark and lifeless. It hums faintly when touched."),
            ],
            "hint": "It seems like it needs to be activated with the right tool.",
            "accepted": [ItemType.TOOL, ItemType.WEAPON],
        },
    ]

    CLASS_STARTER_GEAR = {
        CharacterClass.WARRIOR: ["Iron Sword", "Wooden Shield", "Health Potion"],
        CharacterClass.MAGE: ["Oak Staff", "Mage Robes", "Mana Elixir"],
        CharacterClass.ROGUE: ["Steel Dagger", "Leather Armor", "Health Potion"],
        CharacterClass.CLERIC: ["War Hammer", "Leather Armor", "Mana Elixir"],
        CharacterClass.RANGER: ["Hunting Bow", "Leather Armor", "Health Potion"],
        CharacterClass.BARD: ["Steel Dagger", "Mage Robes", "Stamina Tonic"],
        CharacterClass.COMMONER: ["Health Potion"],
    }

    CLASS_STAT_MODIFIERS = {
        CharacterClass.WARRIOR: {"strength": 4, "constitution": 3, "max_health": 30, "max_stamina": 20},
        CharacterClass.MAGE: {"intelligence": 5, "wisdom": 2, "max_mana": 40, "max_health": -10},
        CharacterClass.ROGUE: {"dexterity": 5, "charisma": 1, "damage_bonus": 2},
        CharacterClass.CLERIC: {"wisdom": 4, "constitution": 2, "max_mana": 20, "max_health": 10},
        CharacterClass.RANGER: {"dexterity": 3, "wisdom": 2, "constitution": 2, "max_stamina": 20},
        CharacterClass.BARD: {"charisma": 5, "dexterity": 2, "intelligence": 1, "max_mana": 15},
    }

    def _create_catalog_item(self, name: str) -> Optional[Item]:
        """Create an item from the catalog by name"""
        template = next((t for t in self.ITEM_CATALOG if t["name"] == name), None)
        if not template: return None
        item = Item(**template)
        self.game_state.items[item.id] = item
        return item

    def _populate_shop_inventory(self, npc: NPC) -> None:
        """Give shopkeepers/merchants a starting inventory from the catalog"""
        if npc.role == NPCRole.SHOPKEEPER:
            stock = ["Iron Sword", "Leather Armor", "Wooden Shield", "Health Potion", "Stamina Tonic"]
        elif npc.role == NPCRole.MERCHANT:
            stock = ["Chainmail", "Mage Robes", "War Hammer", "Hunting Bow", "Health Potion", "Mana Elixir"]
        else:
            return
        for item_name in stock:
            item = self._create_catalog_item(item_name)
            if item:
                npc.shop_inventory.append(item.id)

    def _create_services_for_npc(self, npc: NPC) -> None:
        if npc.role == NPCRole.SHOPKEEPER:
            npc.services_offered.append(Service(service_type=ServiceType.BUY_SELL, name="General Goods", description="I buy and sell various items.", cost={"gold": 0}))
            self._populate_shop_inventory(npc)
        elif npc.role == NPCRole.MERCHANT:
            npc.services_offered.append(Service(service_type=ServiceType.BUY_SELL, name="Trade Goods", description="I deal in fine wares and exotic items.", cost={"gold": 0}))
            self._populate_shop_inventory(npc)
        elif npc.role == NPCRole.INNKEEPER:
            npc.services_offered.extend([
                Service(service_type=ServiceType.REST, name="Room for the Night", description="A warm bed and a hot meal.", cost={"gold": 2, "silver": 5}),
                Service(service_type=ServiceType.HEAL, name="Herbal Remedies", description="Basic healing herbs and tonics.", cost={"gold": 1})
            ])
        elif npc.role == NPCRole.CRAFTSMAN:
            npc.services_offered.append(Service(service_type=ServiceType.REPAIR, name="Item Repair", description="I can mend your broken equipment.", cost={"gold": 5}))

    # ── Temporary Effects & Item Effects ────────────────────────────
    def apply_temporary_effect(self, character: BaseCharacter, effect: Dict) -> None:
        """Add a temporary stat buff. Replaces existing effect from same source+stat."""
        character.temporary_effects = [
            e for e in character.temporary_effects
            if not (e.get("stat") == effect["stat"] and e.get("source") == effect.get("source"))
        ]
        character.temporary_effects.append(effect)
        stat = effect["stat"]
        bonus = effect["bonus"]
        if hasattr(character.stats, stat):
            setattr(character.stats, stat, getattr(character.stats, stat) + bonus)

    def expire_temporary_effects(self, character: BaseCharacter, minutes_elapsed: int) -> List[str]:
        """Tick down temporary effects and remove expired ones. Returns expiry messages."""
        expired_msgs = []
        still_active = []
        for effect in character.temporary_effects:
            effect["remaining_minutes"] = effect.get("remaining_minutes", 0) - minutes_elapsed
            if effect["remaining_minutes"] <= 0:
                stat = effect.get("stat", "")
                bonus = effect.get("bonus", 0)
                source = effect.get("source", "Unknown")
                if hasattr(character.stats, stat):
                    setattr(character.stats, stat, max(1, getattr(character.stats, stat) - bonus))
                expired_msgs.append(f"The effect of {source} (+{bonus} {stat}) has worn off.")
            else:
                still_active.append(effect)
        character.temporary_effects = still_active
        return expired_msgs

    def apply_item_effects(self, character: BaseCharacter, item: Item) -> str:
        """Apply use_effects from a consumable item. Returns description of effects."""
        if not item.use_effects:
            return ""
        effects_applied = []
        for effect_str in item.use_effects:
            parts = effect_str.split(":")
            if len(parts) < 2:
                continue
            if parts[0] == "heal":
                amount = int(parts[1])
                character.stats.health = min(character.stats.max_health, character.stats.health + amount)
                effects_applied.append(f"+{amount} HP")
            elif parts[0] == "stamina":
                amount = int(parts[1])
                character.stats.stamina = min(character.stats.max_stamina, character.stats.stamina + amount)
                effects_applied.append(f"+{amount} Stamina")
            elif parts[0] == "mana":
                amount = int(parts[1])
                character.stats.mana = min(character.stats.max_mana, character.stats.mana + amount)
                effects_applied.append(f"+{amount} Mana")
            elif parts[0] == "buff" and len(parts) == 4:
                self.apply_temporary_effect(character, {
                    "stat": parts[1], "bonus": int(parts[2]),
                    "remaining_minutes": int(parts[3]), "source": item.name,
                })
                effects_applied.append(f"+{parts[2]} {parts[1]} ({parts[3]}min)")
        if item.consumable:
            self.remove_item_from_inventory(character, item.name, 1)
        return ", ".join(effects_applied)

    async def create_new_game(self, player_name: str, player_id: UUID, session_name: str, model_name: str, character_class: CharacterClass = CharacterClass.WARRIOR) -> CompleteGameState:
        try:
            print("\n1/3: Conceptualizing the world and its main story...")
            world_prompt = (
                "Generate the high-level details for a new fantasy world and a main quest. "
                "Also generate a short, evocative background for the hero. "
                "Additionally, generate 3 optional side regions that expand the world beyond the main path. "
                "Each should be distinct and interesting (port city, ancient ruins with towers, "
                "countryside village with caves, abandoned castle, enchanted swamp, mining town, etc). "
                "JSON with keys: name (str), description (str), theme (str), lore_summary (str), "
                "quest_name (str), quest_description (str), player_background (str), "
                "starter_region_type (str), goal_region_type (str), "
                "optional_regions (list of 3 objects with: name (str), region_type (str), "
                "description (str), connected_to (one of: 'start', 'boundary', 'goal'))."
            )
            world_data = await self._generate_and_validate(world_prompt, model_name)
            
            def to_str(val: Any) -> str:
                if isinstance(val, list): return " ".join(str(v) for v in val)
                if isinstance(val, dict): return " ".join(str(v) for v in val.values())
                return str(val or "")

            description = to_str(world_data.get('description', ''))
            lore = to_str(world_data.get('lore_summary', ''))
                
            world = World(
                name=to_str(world_data.get('name', 'New World')), 
                description=description, 
                short_description=description[:100],
                theme=to_str(world_data.get('theme', 'Classic Fantasy')),
                lore_summary=lore,
                location_type=LocationType.WORLD
            )
            
            # Create three regions: Start, Middle (Gated), End (Goal)
            starter_type = to_str(world_data.get('starter_region_type', 'Wilderness'))
            goal_type = to_str(world_data.get('goal_region_type', 'Castle'))
            r_start = Region(name=f"The {starter_type}", description="Where your journey begins.", region_type=starter_type, location_type=LocationType.REGION, short_description="Initial region", tags=["starter"])
            r_mid = Region(name="The Forbidden Boundary", description="A heavily guarded or dangerous zone.", region_type="wilderness", location_type=LocationType.REGION, short_description="A gated passage", tags=["boundary"])
            r_end = Region(name=f"The {goal_type}", description="The place of your destiny.", region_type=goal_type, location_type=LocationType.REGION, short_description="Goal region", tags=["goal"])
            
            # Link main regions (linear East-West path)
            r_start.connections_to_regions[Direction.EAST] = r_mid.id
            r_mid.connections_to_regions[Direction.WEST] = r_start.id
            r_mid.connections_to_regions[Direction.EAST] = r_end.id
            r_end.connections_to_regions[Direction.WEST] = r_mid.id

            # Create optional side regions from AI response
            optional_regions = []
            main_region_map = {"start": r_start, "boundary": r_mid, "goal": r_end}
            side_directions = [
                Direction.NORTH, Direction.SOUTH, Direction.NORTHEAST,
                Direction.NORTHWEST, Direction.SOUTHEAST, Direction.SOUTHWEST,
            ]
            opposite_map = {
                Direction.NORTH: Direction.SOUTH, Direction.SOUTH: Direction.NORTH,
                Direction.NORTHEAST: Direction.SOUTHWEST, Direction.SOUTHWEST: Direction.NORTHEAST,
                Direction.NORTHWEST: Direction.SOUTHEAST, Direction.SOUTHEAST: Direction.NORTHWEST,
            }
            side_dir_idx = 0
            for opt in world_data.get('optional_regions', [])[:3]:
                if not isinstance(opt, dict):
                    continue
                r_name = to_str(opt.get('name', 'Unknown Land'))
                r_type = to_str(opt.get('region_type', 'wilderness'))
                r_desc = to_str(opt.get('description', 'A mysterious side area.'))
                connected_key = to_str(opt.get('connected_to', 'start')).lower()

                parent = main_region_map.get(connected_key, r_start)

                # Find an available direction on the parent region
                connect_dir = None
                for d in side_directions[side_dir_idx:]:
                    if d not in parent.connections_to_regions:
                        connect_dir = d
                        side_dir_idx = side_directions.index(d) + 1
                        break
                if not connect_dir:
                    continue

                r_opt = Region(
                    name=r_name, description=r_desc, region_type=r_type,
                    location_type=LocationType.REGION, short_description=r_desc[:100],
                    tags=["optional"],
                )
                # Bidirectional link
                parent.connections_to_regions[connect_dir] = r_opt.id
                r_opt.connections_to_regions[opposite_map[connect_dir]] = parent.id
                optional_regions.append(r_opt)

            all_regions = [r_start, r_mid, r_end] + optional_regions
            world.regions = {r.id: r for r in all_regions}
            world.starting_region_id = r_start.id
            
            session = GameSession(session_name=session_name, player_id=player_id, world=world, current_region_id=r_start.id)
            game_state = CompleteGameState(session=session)
            self.game_state = game_state

            # Generate the first region grid
            print(f"2/3: Generating {r_start.name}...")
            await self.generate_region_grid(r_start.id, model_name)
            
            # Setup Starting Player
            start_grid = session.region_grids[r_start.id]
            start_location_id = start_grid.get_location_id(0, 0) # Top-left corner for now
            
            player_char = PlayerCharacter(
                name=player_name,
                player_id=player_id,
                current_location_id=start_location_id,
                character_class=character_class,
                description=f"{player_name}, a {character_class.value} in {world.name}.",
                background_lore=world_data.get('player_background', f"{player_name} is a traveler seeking adventure."),
                currency={"gold": 50},
                inventory=[]
            )

            # Apply class stat modifiers
            for stat, mod in self.CLASS_STAT_MODIFIERS.get(character_class, {}).items():
                current = getattr(player_char.stats, stat, 0)
                setattr(player_char.stats, stat, max(1, current + mod))
            player_char.stats.health = player_char.stats.max_health
            player_char.stats.mana = player_char.stats.max_mana
            player_char.stats.stamina = player_char.stats.max_stamina
            player_char.base_stats = player_char.stats.model_copy()

            # Give starter gear
            for item_name in self.CLASS_STARTER_GEAR.get(character_class, []):
                item = self._create_catalog_item(item_name)
                if item:
                    player_char.inventory.append(item.id)
                    if item.equipment_slot and item.equipment_slot not in player_char.equipped_items:
                        player_char.equipped_items[item.equipment_slot] = item.id
            self.apply_equipment_effects(player_char)

            game_state.characters[player_char.id] = player_char
            session.player_character = player_char

            # Track starting location discovery
            player_char.discovered_locations.add(start_location_id)
            player_char.locations_discovered = 1

            # Setup Main Quest
            quest = Quest(
                name=world_data.get('quest_name', 'The Main Quest'),
                description=world_data.get('quest_description', 'Fulfill your destiny.'),
                quest_type=QuestType.MAIN_STORY,
                status=QuestStatus.ACTIVE,
                objectives=[QuestObjective(description=f"Reach the {r_end.name} and fulfill your destiny.", objective_type="reach_location", target=str(r_end.id))]
            )
            game_state.quests[quest.id] = quest
            player_char.active_quests.append(quest.id)
            
            # Create the Gate Key in r_start
            gate_key = Item(name="Pass Permit", description="A signed document allowed travel through the forbidden boundary.", item_type=ItemType.QUEST_ITEM)
            game_state.items[gate_key.id] = gate_key
            # Place it in the starting location
            game_state.locations[start_location_id].items.append(gate_key.id)
            
            print("3/3: Adventurer ready.")
            return self.game_state
            
        except Exception as e:
            llm_logger.exception("Hierarchical world generation failed")
            raise ConnectionError(f"Failed to generate world: {e}")

    async def generate_region_grid(self, region_id: UUID, model_name: str) -> None:
        region = self.game_state.session.world.regions.get(region_id)
        if not region: return
        
        # Adjust grid size based on region type
        r_type_lower = region.region_type.lower()
        is_city = any(t in r_type_lower for t in ["city", "port", "town", "capital"])
        is_ruins = any(t in r_type_lower for t in ["ruins", "tower", "castle", "dungeon", "fortress"])

        if is_city:
            width, height = random.randint(3, 4), random.randint(3, 4)
        elif is_ruins:
            width, height = random.randint(2, 3), random.randint(2, 3)
        else:
            width, height = random.randint(2, 3), random.randint(2, 3)

        region.width, region.height = width, height
        grid = WorldGrid(width=width, height=height, grid=[[None]*width for _ in range(height)])
        self.game_state.session.region_grids[region_id] = grid

        allowed_loc_types = [t.value for t in GeneralLocationType]

        for y in range(height):
            for x in range(width):
                print(f"   - Building {region.name} cell ({x}, {y})...", end="", flush=True)

                # Contextual prompt based on region type
                if is_city:
                    prompt = (
                        f"Generate a city district in '{region.name}' ({region.description}). "
                        f"JSON with: name, description, short_description, "
                        f"general_type (one of: city_center, market_district, residential_district, plaza), "
                        f"atmosphere, buildings (list of: name, type, description), "
                        f"and optional 'npc' object (name, description, race, role)."
                    )
                elif is_ruins:
                    prompt = (
                        f"Generate a location within ancient ruins/structure '{region.name}' ({region.description}). "
                        f"Include crumbling halls, towers, underground passages, or overgrown courtyards. "
                        f"JSON with: name, description, short_description, "
                        f"general_type (one of: ruins, cave, clearing, crossroads), "
                        f"atmosphere, notable_features (list of name/desc), "
                        f"and optional 'npc' object (name, description, race, role)."
                    )
                else:
                    prompt = (
                        f"Generate a location in the {region.region_type} region '{region.name}' "
                        f"({region.description}) of {self.game_state.session.world.name} at ({x},{y}). "
                        f"JSON with: name, description, short_description, "
                        f"general_type (one of: {allowed_loc_types}), atmosphere, "
                        f"notable_features (list name/desc), and optional 'npc' object (name, description, race, role)."
                    )

                loc_data = await self._generate_and_validate(prompt, model_name)
                loc_data = self._sanitize_location_data(loc_data)

                buildings = loc_data.pop('buildings', [])
                notable = loc_data.pop('notable_features', [])
                # NPC spawn rates: cities always, ruins 30%, wilderness 20%
                raw_npc = loc_data.pop('npc', None)
                if is_city:
                    npc_info = raw_npc
                elif is_ruins:
                    npc_info = raw_npc if random.random() < 0.3 else None
                else:
                    npc_info = raw_npc if random.random() < 0.2 else None
                
                loc = GeneralLocation.model_validate(loc_data)
                loc.coordinates = Coordinates(x=x, y=y)
                loc.parent_id = region_id
                
                # Add buildings or features
                for b in buildings:
                    if isinstance(b, dict):
                        b_name = b.get('name', 'A building')
                        b_type = b.get('type', 'building')
                        b_desc = b.get('description', 'A generic building.')
                        loc.notable_features.append(NotableFeature(
                            name=b_name,
                            detailed_description=f"A {b_type}: {b_desc}",
                            metadata={"enterable": True, "building_type": b_type, "building_location_id": None},
                        ))
                    elif isinstance(b, str):
                        loc.notable_features.append(NotableFeature(
                            name=b,
                            metadata={"enterable": True, "building_type": "building", "building_location_id": None},
                        ))

                for n in notable:
                    if isinstance(n, dict):
                        n_name = n.get('name', 'Feature')
                        n_desc = n.get('description')
                        meta = {}
                        if self._is_enterable_name(n_name):
                            meta = {"enterable": True, "building_type": n_name.split()[-1].lower(), "building_location_id": None}
                        loc.notable_features.append(NotableFeature(name=n_name, detailed_description=n_desc, metadata=meta))
                    elif isinstance(n, str):
                        meta = {}
                        if self._is_enterable_name(n):
                            meta = {"enterable": True, "building_type": n.split()[-1].lower(), "building_location_id": None}
                        loc.notable_features.append(NotableFeature(name=n, metadata=meta))
                
                self.game_state.locations[loc.id] = loc
                grid.set_location_id(x, y, loc.id)

                # Scatter food ingredients, campfires, and puzzles in wilderness
                if not is_city:
                    self._maybe_place_food_ingredients(loc)
                    self._maybe_add_campfire(loc)
                    self._maybe_place_puzzle(loc)

                if npc_info:
                    npc_info = self._sanitize_npc_data(npc_info)
                    npc_info.update({'id': str(uuid4()), 'current_location_id': loc.id, 'home_location_id': loc.id})
                    npc = NPC.model_validate(npc_info)
                    self._create_services_for_npc(npc)
                    self.game_state.characters[npc.id] = npc

                # Chance to generate sub-levels (dungeons/buildings)
                has_sublevel_keyword = any(tag in loc.name.lower() or tag in loc.description.lower() for tag in ["cave", "tower", "cellar", "ruins", "dungeon"])
                if has_sublevel_keyword or is_ruins:
                    sublevel_chance = 0.9 if is_ruins else 0.7
                    sublevel_depth = random.randint(2, 3) if is_ruins else random.randint(1, 2)
                    if random.random() < sublevel_chance:
                        await self.generate_sub_levels(loc, depth=sublevel_depth, model_name=model_name)

                print(" ✓")
        
        # Link internal cells
        for y in range(height):
            for x in range(width):
                loc = self.game_state.locations[grid.get_location_id(x, y)]
                neighbors = {Direction.NORTH: (x, y-1), Direction.SOUTH: (x, y+1), Direction.EAST: (x+1, y), Direction.WEST: (x-1, y)}
                for d, (nx, ny) in neighbors.items():
                    target_id = grid.get_location_id(nx, ny)
                    if target_id:
                        loc.connections.append(LocationConnection(target_location_id=target_id, direction=d, description=f"You can go {d.value} to another part of {region.name}."))
                    elif d in region.connections_to_regions:
                        # World Exit!
                        target_region_id = region.connections_to_regions[d]
                        target_region = self.game_state.session.world.regions[target_region_id]
                        
                        # Logic Gate: If entering the Goal Region, require the Pass Permit
                        reqs = []
                        if "goal" in target_region.tags and "boundary" in region.tags:
                            reqs = ["Pass Permit"]

                        loc.connections.append(LocationConnection(
                            target_location_id=target_region_id, # Actually points to region ID, we'll handle this in move_player
                            direction=d, 
                            description=f"A path leads out of {region.name} towards {target_region.name}.",
                            requirements=reqs
                        ))
        
        region.is_generated = True

    async def generate_sub_levels(self, parent_location: BaseLocation, depth: int, model_name: str) -> None:
        """Procedurally generate levels above or below a location"""
        current_parent = parent_location
        for i in range(depth):
            direction = Direction.DOWN if "cellar" in current_parent.name.lower() or "cave" in current_parent.name.lower() or "dungeon" in current_parent.name.lower() else Direction.UP
            opposite = Direction.UP if direction == Direction.DOWN else Direction.DOWN
            
            level_name = f"{current_parent.name} - Level {i+1}"
            prompt = f"Generate a sub-level called '{level_name}' connected to '{current_parent.name}'. JSON with: name, description, short_description, atmosphere, notable_features (list name/desc), and optional 'npc' object."
            
            try:
                loc_data = await self._generate_and_validate(prompt, model_name)
                loc_data = self._sanitize_location_data(loc_data)
                
                notable = loc_data.pop('notable_features', [])
                npc_info = loc_data.pop('npc', None)
                
                sub_loc = GeneralLocation.model_validate(loc_data)
                sub_loc.parent_id = parent_location.parent_id # Same region
                sub_loc.coordinates = Coordinates(x=current_parent.coordinates.x, y=current_parent.coordinates.y, z=current_parent.coordinates.z + (1 if direction == Direction.UP else -1))
                
                for n in notable:
                    if isinstance(n, dict):
                        sub_loc.notable_features.append(NotableFeature(name=n.get('name', 'Feature'), detailed_description=n.get('description')))
                    elif isinstance(n, str):
                        sub_loc.notable_features.append(NotableFeature(name=n))
                
                self.game_state.locations[sub_loc.id] = sub_loc
                
                # Link them
                current_parent.connections.append(LocationConnection(target_location_id=sub_loc.id, direction=direction, description=f"A passage leads {direction.value} to {sub_loc.name}."))
                sub_loc.connections.append(LocationConnection(target_location_id=current_parent.id, direction=opposite, description=f"A passage leads {opposite.value} back to {current_parent.name}."))
                
                if npc_info:
                    npc_info = self._sanitize_npc_data(npc_info)
                    npc_info.update({'id': str(uuid4()), 'current_location_id': sub_loc.id, 'home_location_id': sub_loc.id})
                    npc = NPC.model_validate(npc_info)
                    self.game_state.characters[npc.id] = npc
                
                current_parent = sub_loc
                print(f"     + Added sub-level: {sub_loc.name}")
            except Exception as e:
                llm_logger.error(f"Error generating sub-level: {e}")
                break

    def get_current_location(self) -> BaseLocation:
        if not self.game_state or not self.game_state.session.player_character: raise ValueError("Not initialized")
        return self.game_state.locations[self.game_state.session.player_character.current_location_id]

    async def execute_combat_turn(self, target_id: UUID, action: str = "attack") -> Dict[str, Any]:
        """Execute a combat turn with strategic elements"""
        if not self.game_state: return {"msg": "No session"}
        player = self.game_state.session.player_character
        target = self.game_state.characters.get(target_id)
        if not target: return {"msg": "Target not found"}

        results = {"player_msg": "", "enemy_msg": "", "victory": False, "fled": False}
        
        # 1. Player Turn
        if "stunned" in player.conditions:
            results["player_msg"] = f"You are stunned and cannot act!"
            player.conditions.remove("stunned")
        elif action == "flee":
            success_rate = 0.3 + (player.stats.dexterity * 0.02)
            if random.random() < success_rate:
                results["fled"] = True
                results["player_msg"] = "You managed to flee from combat!"
                return results
            else:
                results["player_msg"] = "You tried to flee but failed!"
        elif action == "parry":
            player.conditions.append("parrying")
            results["player_msg"] = "You take a defensive stance."
        else: # Attack
            hit_chance = 0.7 + (player.stats.dexterity * 0.01) - (target.stats.dexterity * 0.005)
            if random.random() < hit_chance:
                dmg = random.randint(1, 8) + (player.stats.strength // 3) + player.stats.damage_bonus
                target.stats.health -= dmg
                results["player_msg"] = f"You hit {target.name} for {dmg} damage."
                if random.random() < 0.2: # 20% bleed chance on hit
                    target.conditions.append("bleeding")
                    results["player_msg"] += " (Target is bleeding!)"
            else:
                results["player_msg"] = f"You swing at {target.name} but miss."

        if target.stats.health <= 0:
            results["victory"] = True
            results["enemy_msg"] = f"{target.name} has been defeated!"
            self.handle_combat_reward(target)
            del self.game_state.characters[target.id]
            return results

        # 2. Enemy Turn
        if "stunned" in target.conditions:
            results["enemy_msg"] = f"{target.name} is stunned!"
            target.conditions.remove("stunned")
        else:
            enemy_hit_chance = 0.6 + (target.stats.dexterity * 0.01) - (player.stats.dexterity * 0.005)
            if random.random() < enemy_hit_chance:
                enemy_dmg = random.randint(1, 6) + (target.stats.strength // 4)
                if "parrying" in player.conditions:
                    enemy_dmg = max(1, enemy_dmg // 2)
                    results["enemy_msg"] = f"{target.name} hits your parry for {enemy_dmg} damage."
                else:
                    results["enemy_msg"] = f"{target.name} hits you for {enemy_dmg} damage."
                player.stats.health -= enemy_dmg
            else:
                results["enemy_msg"] = f"{target.name} misses you."

        # 3. Process Conditions
        if "parrying" in player.conditions: player.conditions.remove("parrying")
        
        for char in [player, target]:
            if "bleeding" in char.conditions:
                bleed_dmg = 2
                char.stats.health -= bleed_dmg
                msg = f" (Bleeding: -{bleed_dmg} HP)"
                if char.id == player.id: results["player_msg"] += msg
                else: results["enemy_msg"] += msg

        return results

    def handle_combat_reward(self, target: BaseCharacter):
        player = self.game_state.session.player_character
        exp = target.level * 20
        player.experience += exp
        self.pending_messages.append(f"Gained {exp} XP from defeating {target.name}.")

        # Loot NPC gold
        npc_gold = target.currency.get("gold", 0) if hasattr(target, 'currency') else 0
        if npc_gold > 0:
            player.currency["gold"] = player.currency.get("gold", 0) + npc_gold
            self.pending_messages.append(f"Found {npc_gold} gold on {target.name}.")

        # Create lootable corpse
        loot_items = [iid for iid in target.inventory if iid in self.game_state.items]
        loc = self.get_current_location()
        if loot_items:
            corpse = NotableFeature(
                name=f"Remains of {target.name}",
                detailed_description=f"The fallen body of {target.name}. Something might be worth searching.",
                contained_items=loot_items,
                metadata={"corpse": True, "original_npc_name": target.name},
            )
            loc.notable_features.append(corpse)
            self.pending_messages.append(f"You notice the remains of {target.name}. Try 'examine remains' to search.")

        lvl_msg = self.check_level_up()
        if lvl_msg:
            self.pending_messages.append(lvl_msg)

    def check_level_up(self) -> Optional[str]:
        """Check and apply level-up if enough XP accumulated"""
        player = self.game_state.session.player_character
        xp_threshold = player.level * 100
        if player.experience < xp_threshold:
            return None
        player.experience -= xp_threshold
        player.level += 1
        # Boost base stats
        player.base_stats.max_health += 10
        player.base_stats.max_stamina += 5
        player.base_stats.max_mana += 5
        # Class-specific level bonus (smaller than initial)
        for stat, mod in self.CLASS_STAT_MODIFIERS.get(player.character_class, {}).items():
            if stat.startswith("max_"): continue  # already handled above
            current = getattr(player.base_stats, stat, 0)
            setattr(player.base_stats, stat, current + max(1, mod // 2))
        # Heal to new max
        self.apply_equipment_effects(player)
        player.stats.health = player.stats.max_health
        player.stats.mana = player.stats.max_mana
        player.stats.stamina = player.stats.max_stamina
        return f"LEVEL UP! You are now level {player.level}!"

    def check_player_death(self) -> Optional[str]:
        """Check if player died and handle respawn"""
        player = self.game_state.session.player_character
        if player.stats.health > 0:
            return None
        player.deaths += 1
        # Respawn at starting region (0,0)
        start_region_id = self.game_state.session.world.starting_region_id
        start_grid = self.game_state.session.region_grids.get(start_region_id)
        if start_grid:
            respawn_loc = start_grid.get_location_id(0, 0)
            if respawn_loc:
                player.current_location_id = respawn_loc
                self.game_state.session.current_region_id = start_region_id
        # Penalties
        player.stats.health = player.stats.max_health // 2
        player.stats.mana = 0
        player.stats.stamina = player.stats.max_stamina // 2
        gold_loss = player.currency.get("gold", 0) // 10
        player.currency["gold"] = max(0, player.currency.get("gold", 0) - gold_loss)
        # Clear combat
        self.in_combat = False
        self.combat_opponents = []
        return f"YOU DIED! (Death #{player.deaths}) Lost {gold_loss} gold. You awaken weakened..."

    async def move_player(self, direction: Direction, model_name: str, confirmed: bool = False) -> Tuple[bool, str]:
        if not self.game_state: return False, "No session"
        current_loc = self.get_current_location()
        conn = next((c for c in current_loc.connections if c.direction == direction), None)
        if not conn or not conn.is_visible or not conn.is_passable: return False, "Blocked"
        
        # Check requirements (item names)
        pc = self.game_state.session.player_character
        if conn.requirements:
            inventory_item_names = [self.game_state.items[iid].name.lower() for iid in pc.inventory if iid in self.game_state.items]
            for req in conn.requirements:
                if req.lower() not in inventory_item_names:
                    return False, f"You need {req} to pass."
        
        # Check if it's a Region-to-Region move
        if conn.target_location_id in self.game_state.session.world.regions:
            target_region_id = conn.target_location_id
            target_region = self.game_state.session.world.regions[target_region_id]
            
            if not confirmed:
                return False, f"TRAVEL_CONFIRM:{target_region.name}"
            
            print(f"\n🌍 Traveling to {target_region.name}...")
            if not target_region.is_generated:
                await self.generate_region_grid(target_region_id, model_name)
            
            # Switch region
            self.game_state.session.current_region_id = target_region_id
            target_grid = self.game_state.session.region_grids[target_region_id]
            
            # Find the entry point
            entry_x, entry_y = 0, 0 
            if direction == Direction.NORTH: entry_y = target_grid.height - 1
            elif direction == Direction.SOUTH: entry_y = 0
            elif direction == Direction.WEST: entry_x = target_grid.width - 1
            elif direction == Direction.EAST: entry_x = 0
            
            target_id = target_grid.get_location_id(entry_x, entry_y)
            target = self.game_state.locations.get(target_id)
            travel_time = 60 # 1 hour for region travel
        else:
            target = self.game_state.locations.get(conn.target_location_id)
            travel_time = conn.travel_time
        
        if not target: return False, "Void error"
        
        pc.previous_location_id, pc.current_location_id = pc.current_location_id, target.id
        target.visit_count += 1
        target.last_visited = datetime.now()
        if target.id not in pc.discovered_locations:
            pc.discovered_locations.add(target.id)
            pc.locations_discovered += 1
        
        # Calculate stamina cost and advance time
        stamina_cost = 5
        region = self.game_state.session.world.regions.get(self.game_state.session.current_region_id)
        if region and "Storm" in region.active_events:
            stamina_cost *= 2
            travel_time *= 2
        
        if target.weather == "rainy":
            stamina_cost += 5
        
        pc.stats.stamina = max(0, pc.stats.stamina - stamina_cost)
        
        await self.advance_time(travel_time, model_name)
        self.check_quest_progress()
        # Dynamic quest system: increment idle counter and check for encounters
        self.game_state.session.actions_since_last_quest += 1
        await self._check_dynamic_quest_trigger(model_name)
        return True, f"Moving {direction.value}... (Consumed {stamina_cost} stamina)"

    def check_quest_progress(self) -> List[str]:
        if not self.game_state: return []
        player = self.game_state.session.player_character
        msgs = []
        for qid in list(player.active_quests):
            q = self.game_state.quests.get(qid)
            if not q or q.status != QuestStatus.ACTIVE:
                continue

            # FETCH: player has the target item in inventory
            if q.quest_type == QuestType.FETCH and q.target_item_id in player.inventory:
                for obj in q.objectives:
                    if obj.objective_type == "fetch":
                        obj.completed, obj.current_progress = True, obj.required_progress

            # DELIVERY: player has item AND is at the quest giver's location
            elif q.quest_type == QuestType.DELIVERY and q.target_item_id in player.inventory:
                if q.giver_id:
                    giver = self.game_state.characters.get(q.giver_id)
                    if giver and giver.current_location_id == player.current_location_id:
                        for obj in q.objectives:
                            if obj.objective_type == "delivery":
                                obj.completed, obj.current_progress = True, obj.required_progress

            # EXPLORATION: player visited the target location
            elif q.quest_type == QuestType.EXPLORATION:
                for obj in q.objectives:
                    if obj.objective_type == "explore":
                        try:
                            target_id = UUID(obj.target)
                            if target_id in player.discovered_locations:
                                obj.completed, obj.current_progress = True, obj.required_progress
                        except ValueError:
                            pass

            # KILL: target NPC was defeated (no longer in characters)
            elif q.quest_type == QuestType.KILL:
                for obj in q.objectives:
                    if obj.objective_type == "kill":
                        try:
                            target_id = UUID(obj.target)
                            if target_id not in self.game_state.characters:
                                obj.completed, obj.current_progress = True, obj.required_progress
                        except ValueError:
                            pass

            # Check if all objectives are complete
            if all(o.completed for o in q.objectives):
                # Auto-complete if no turn-in NPC exists, or for exploration/kill quests
                can_auto_complete = (
                    q.quest_type in (QuestType.EXPLORATION, QuestType.KILL)
                    or not q.giver_id
                )
                if can_auto_complete:
                    q.status = QuestStatus.TURNED_IN
                    q.completed_at = datetime.now()
                    player.active_quests.remove(qid)
                    player.completed_quests.append(qid)
                    player.quests_completed += 1
                    player.experience += q.rewards.experience
                    for cur, amt in q.rewards.currency.items():
                        player.currency[cur] = player.currency.get(cur, 0) + amt
                    reward_msg = f" (+{q.rewards.experience} XP, +{q.rewards.currency.get('gold', 0)}g)"
                    msgs.append(f"Quest completed: {q.name}!{reward_msg}")
                    lvl_msg = self.check_level_up()
                    if lvl_msg:
                        msgs.append(lvl_msg)
                else:
                    q.status = QuestStatus.COMPLETED
                    giver = self.game_state.characters.get(q.giver_id)
                    giver_hint = f" Return to {giver.name}." if giver else ""
                    msgs.append(f"Quest ready to turn in: {q.name}.{giver_hint}")

        for m in msgs:
            self.pending_messages.append(m)
        return msgs

    async def complete_quest(self, quest_id: UUID) -> str:
        if not self.game_state: return "No session"
        q = self.game_state.quests.get(quest_id)
        if not q or q.status != QuestStatus.COMPLETED: return "Not ready"
        
        player = self.game_state.session.player_character
        if quest_id in player.active_quests: player.active_quests.remove(quest_id)
        player.completed_quests.append(quest_id)
        player.quests_completed += 1
        q.status, q.completed_at = QuestStatus.TURNED_IN, datetime.now()
        
        msg = f"Completed {q.name}!"
        if q.rewards:
            player.experience += q.rewards.experience
            for cur, amt in q.rewards.currency.items():
                player.currency[cur] = player.currency.get(cur, 0) + amt
            lvl_msg = self.check_level_up()
            if lvl_msg:
                msg += f"\n🎉 {lvl_msg}"
        return msg

    async def advance_time(self, minutes: int, model_name: str) -> None:
        """Advance game time and update the world state"""
        if not self.game_state: return
        self.game_state.session.game_time.advance_time(minutes)
        
        # Apply passive recovery bonuses
        pc = self.game_state.session.player_character
        if pc:
            # Basic recovery: 1 per 10 mins
            stamina_rec = minutes // 10
            mana_rec = minutes // 10
            
            if pc.character_class == CharacterClass.WARRIOR:
                stamina_rec = int(stamina_rec * 1.5)  # 50% extra stamina
            elif pc.character_class == CharacterClass.MAGE:
                mana_rec = int(mana_rec * 1.5)        # 50% extra mana
                
            pc.stats.stamina = min(pc.stats.max_stamina, pc.stats.stamina + stamina_rec)
            pc.stats.mana = min(pc.stats.max_mana, pc.stats.mana + mana_rec)

            # Expire temporary effects (food buffs, etc.)
            expiry_msgs = self.expire_temporary_effects(pc, minutes)
            for msg in expiry_msgs:
                self.pending_messages.append(msg)

        # Update world if time period changed or enough time passed
        await self.update_world_state(model_name)
        
    async def update_world_state(self, model_name: str) -> None:
        """Update NPCs, regional events, weather, and run Virtual DM"""
        if not self.game_state: return

        # 0. Virtual DM (action-based, replaces midnight-only heartbeat)
        await self._run_virtual_dm(model_name)

        # 1. Update NPC positions based on schedule
        self._update_npc_positions()

        # 2. Update Regional Events
        await self._update_regional_events(model_name)

        # 3. Update Local Weather (Randomly)
        for loc in self.game_state.locations.values():
            if random.random() < 0.05:
                weathers = ["sunny", "cloudy", "rainy", "foggy", "windy"]
                loc.weather = random.choice(weathers)

    async def _run_plot_heartbeat(self, model_name: str) -> None:
        """Use AI to check for and trigger major world events"""
        if not self.game_state: return
        
        # Check every 24 game hours
        gt = self.game_state.session.game_time
        if gt.minute == 0 and gt.hour == 0:
            llm_logger.info("Heartbeat: Checking for plot shifts...")
            
            lore = self.game_state.session.world.lore_summary
            decisions = "\n".join(self.game_state.session.major_decision_history) or "None yet."
            events = "\n".join([e.name for e in self.game_state.session.active_global_events]) or "No active events."
            
            prompt = (
                f"WORLD LORE: {lore}\n"
                f"RECENT PLAYER DECISIONS: {decisions}\n"
                f"ACTIVE EVENTS: {events}\n"
                "Based on this context, is it time for a major plot shift or world event? "
                "Respond with JSON: 'trigger' (bool), and if true: 'name' (str), 'description' (str), 'scope' (global/regional), 'duration_days' (int)."
            )
            
            try:
                data = await self._generate_and_validate(prompt, model_name)
                if data.get('trigger'):
                    event = GlobalEvent(
                        name=data.get('name', 'Unknown Event'),
                        description=data.get('description', 'Something has changed in the world.'),
                        scope=EventScope(self._coerce_enum(data.get('scope', 'global'), EventScope)),
                        duration_minutes=data.get('duration_days', 1) * 1440
                    )
                    self.game_state.session.active_global_events.append(event)
                    llm_logger.info(f"PLOT EVENT TRIGGERED: {event.name}")
            except Exception as e:
                llm_logger.error(f"Error in plot heartbeat: {e}")

    # ============================================================================
    # Virtual DM (Layer 3)
    # ============================================================================
    DM_ACTION_INTERVAL = 15  # Run DM check every N player actions

    async def _run_virtual_dm(self, model_name: str) -> None:
        """Virtual DM: actively manages narrative tension and world events."""
        if not self.game_state:
            return

        session = self.game_state.session
        actions_since_last = session.dm_action_counter - session.dm_last_check
        if actions_since_last < self.DM_ACTION_INTERVAL:
            return

        # Mark this check point
        session.dm_last_check = session.dm_action_counter

        # Gradually increase tension
        session.dm_tension_level = min(1.0, session.dm_tension_level + 0.05)

        player = session.player_character
        region = session.world.regions.get(session.current_region_id)
        loc = self.get_current_location()

        # Build rich context
        active_quests = [
            self.game_state.quests[qid].name
            for qid in player.active_quests if qid in self.game_state.quests
        ][:5]
        completed_quests = [
            self.game_state.quests[qid].name
            for qid in player.completed_quests if qid in self.game_state.quests
        ][-5:]
        known_npc_names = []
        for npc_id in list(player.known_npcs)[:10]:
            npc = self.game_state.characters.get(npc_id)
            if npc:
                known_npc_names.append(npc.name)

        dm_history = "\n".join(session.dm_memory[-5:]) if session.dm_memory else "No previous DM actions."
        decisions = "\n".join(session.major_decision_history[-5:]) if session.major_decision_history else "None."
        events = ", ".join(e.name for e in session.active_global_events if e.is_active) or "None."

        dm_prompt = (
            f"You are the Dungeon Master for this game. Tension level: {session.dm_tension_level:.1f}/1.0.\n"
            f"WORLD: {session.world.name} — {session.world.lore_summary[:200]}\n"
            f"PLAYER: {player.name}, Level {player.level} {player.character_class.value}, "
            f"HP {player.stats.health}/{player.stats.max_health}\n"
            f"LOCATION: {loc.name} in {region.name if region else 'unknown'}\n"
            f"ACTIVE QUESTS: {active_quests or 'None'}\n"
            f"COMPLETED QUESTS: {completed_quests or 'None'}\n"
            f"KNOWN NPCS: {known_npc_names or 'None'}\n"
            f"RECENT DECISIONS: {decisions}\n"
            f"ACTIVE EVENTS: {events}\n"
            f"DM HISTORY: {dm_history}\n\n"
            f"Choose ONE action to advance the narrative. "
            f"Actions: spawn_hunter, subplot, premonition, world_event, antagonist_move, discovery, noop.\n"
            f"JSON with: action (str), description (str), "
            f"npc_name (str or null, if spawning), npc_description (str or null, if spawning), "
            f"event_name (str or null, if world_event), "
            f"dm_note (str, your reasoning for this choice)."
        )

        try:
            data = await self._generate_and_validate(dm_prompt, model_name)
            action = str(data.get('action', 'noop')).lower()
            description = str(data.get('description', ''))
            dm_note = str(data.get('dm_note', f"DM chose: {action}"))

            # Record to DM memory
            session.dm_memory.append(dm_note)
            if len(session.dm_memory) > 20:
                session.dm_memory = session.dm_memory[-15:]

            if action == 'spawn_hunter':
                hunter_name = str(data.get('npc_name', 'Dark Hunter'))
                hunter = NPC(
                    name=hunter_name,
                    description=str(data.get('npc_description', 'A menacing figure sent by dark forces.')),
                    role=NPCRole.COMMONER,
                    current_location_id=loc.id,
                    home_location_id=loc.id,
                    mood=0.0,
                    goal=NPCGoal.ATTACK_PLAYER,
                    max_ticks=20,
                    level=max(1, player.level + random.randint(0, 2)),
                )
                hunter.stats.health = 40 + (hunter.level * 10)
                hunter.stats.max_health = hunter.stats.health
                hunter.stats.strength = 10 + hunter.level
                hunter.stats.dexterity = 8 + hunter.level
                hunter.base_stats = hunter.stats.model_copy()
                self.game_state.characters[hunter.id] = hunter
                self.pending_messages.append(
                    f"DANGER: {description or f'{hunter_name} has appeared, looking for trouble!'}"
                )
                session.dm_tension_level = max(0.0, session.dm_tension_level - 0.15)

            elif action == 'subplot':
                quest = await self.generate_dynamic_quest(model_name, trigger_reason="dm_subplot")
                if quest:
                    self.pending_messages.append(
                        f"QUEST ENCOUNTER: {description or f'A new opportunity: {quest.name}'}"
                    )
                session.dm_tension_level = max(0.0, session.dm_tension_level - 0.1)

            elif action == 'premonition':
                if description:
                    self.pending_messages.append(f"PREMONITION: {description}")

            elif action == 'world_event':
                event_name = str(data.get('event_name', 'Strange Occurrence'))
                event = GlobalEvent(
                    name=event_name,
                    description=description,
                    scope=EventScope.GLOBAL,
                    duration_minutes=1440,
                )
                session.active_global_events.append(event)
                self.pending_messages.append(f"WORLD EVENT: {event_name} — {description}")
                session.dm_tension_level = max(0.0, session.dm_tension_level - 0.2)

            elif action == 'antagonist_move':
                if description:
                    self.pending_messages.append(f"OMINOUS: {description}")
                    session.major_decision_history.append(f"Antagonist: {description}")
                session.dm_tension_level = min(1.0, session.dm_tension_level + 0.1)

            elif action == 'discovery':
                discovery_roll = random.random()
                if discovery_roll < 0.35:
                    puzzle = self.create_puzzle_feature(loc)
                    self.pending_messages.append(
                        f"DISCOVERY: You notice {puzzle.name} — {puzzle.detailed_description}"
                    )
                elif discovery_roll < 0.50:
                    if not any(f.metadata.get("campfire") for f in loc.notable_features):
                        loc.notable_features.append(NotableFeature(
                            name="Abandoned Campfire",
                            detailed_description="A circle of stones surrounds charred wood. It could be rekindled for cooking.",
                            metadata={"campfire": True},
                        ))
                    self.pending_messages.append(
                        f"DISCOVERY: You find the remains of a traveler's campfire. It could be used for cooking."
                    )
                elif description:
                    loc.notable_features.append(
                        NotableFeature(name="Strange Discovery", detailed_description=description)
                    )
                    self.pending_messages.append(f"DISCOVERY: {description}")
                session.dm_tension_level = max(0.0, session.dm_tension_level - 0.05)

            # 'noop' does nothing

            llm_logger.info(f"Virtual DM: {action} — {dm_note}")

        except Exception as e:
            llm_logger.error(f"Virtual DM error: {e}")

    async def apply_persistent_change(self, location_id: UUID, action_description: str, model_name: str) -> str:
        """Permanently change a location based on high-impact player action"""
        loc = self.game_state.locations.get(location_id)
        if not loc: return "Location not found."
        
        prompt = (
            f"LOCATION: {loc.name} - {loc.description}\n"
            f"PLAYER ACTION: {action_description}\n"
            "The player's action has permanently changed this location. "
            "Generate the new identity for this place. "
            "JSON: 'new_name' (shorter, impactful), 'new_description' (evocative), 'state_tag' (1 word status, e.g. ruined, occupied, liberated)."
        )
        
        try:
            data = await self._generate_and_validate(prompt, model_name)
            
            old_name = loc.name
            loc.name = data.get('new_name', loc.name)
            loc.description = data.get('new_description', loc.description)
            loc.current_state_tag = data.get('state_tag', 'changed')
            loc.history.append(f"Once known as {old_name}, it was changed by: {action_description}")
            
            self.game_state.session.major_decision_history.append(f"Transformed {old_name} into {loc.name} through action: {action_description}")
            
            return f"\n⚠️  {Colors.BOLD}WORLD CHANGE{Colors.ENDC}\n{old_name} has become {Colors.CYAN}{loc.name}{Colors.ENDC}.\n{loc.description}"
        except Exception as e:
            llm_logger.error(f"Error applying persistent change: {e}")
            return "The world feels different, but the change is hard to describe."

    async def process_ai_command(self, command: str, context: str, model_name: str) -> str:
        """Process a freeform AI command as the Dungeon Master, returning narrative and applying effects."""
        player = self.game_state.session.player_character
        loc = self.get_current_location()

        prompt = (
            f"FREEFORM DM COMMAND. You are the Dungeon Master processing game effects.\n"
            f"The player says: \"{command}\"\n\n"
            f"{context}\n"
            f"PLAYER STATS: HP {player.stats.health}/{player.stats.max_health}, "
            f"Stamina {player.stats.stamina}/{player.stats.max_stamina}, "
            f"Gold {player.currency.get('gold', 0)}\n"
            f"LOCATION: {loc.name}\n\n"
            "Narrate the outcome in 2-3 sentences. Then specify game effects as JSON.\n"
            "Available effect types:\n"
            "- heal: {\"type\":\"heal\", \"amount\": int}\n"
            "- damage: {\"type\":\"damage\", \"amount\": int}\n"
            "- restore_stamina: {\"type\":\"restore_stamina\", \"amount\": int}\n"
            "- advance_time: {\"type\":\"advance_time\", \"minutes\": int}\n"
            "- give_gold: {\"type\":\"give_gold\", \"amount\": int}\n"
            "- remove_gold: {\"type\":\"remove_gold\", \"amount\": int}\n"
            "- spawn_item: {\"type\":\"spawn_item\", \"name\": str, \"description\": str, \"item_type\": \"weapon|armor|consumable|tool|material\", \"value\": int}\n"
            "- add_feature: {\"type\":\"add_feature\", \"name\": str, \"description\": str}\n"
            "- spawn_npc: {\"type\":\"spawn_npc\", \"name\": str, \"description\": str, \"role\": \"commoner|merchant|guard|quest_giver\", \"mood\": float 0-1}\n"
            "- modify_location: {\"type\":\"modify_location\", \"description\": str, \"atmosphere\": str}\n\n"
            "Only include effects that logically follow from the action. Use [] for no effects (e.g. casual conversation).\n"
            "JSON: {\"narrative\": \"...\", \"effects\": [...]}"
        )

        try:
            data = await self._generate_and_validate(prompt, model_name)
            narrative = str(data.get("narrative", "Something happens, but it's hard to describe."))
            effects = data.get("effects", [])
            if not isinstance(effects, list):
                effects = []
            effects = effects[:self.AI_EFFECT_CAPS["max_effects"]]

            for effect in effects:
                if not isinstance(effect, dict) or "type" not in effect:
                    continue
                msg = self._apply_ai_effect(effect)
                if msg:
                    self.pending_messages.append(f"DM: {msg}")

            return narrative
        except Exception as e:
            llm_logger.error(f"DM command error: {e}")
            return "The world shifts around you, but nothing seems to change."

    def _apply_ai_effect(self, effect: Dict[str, Any]) -> Optional[str]:
        """Apply a single validated AI effect. Returns status message or None."""
        player = self.game_state.session.player_character
        loc = self.get_current_location()
        effect_type = str(effect.get("type", "")).lower()

        try:
            if effect_type == "heal":
                amount = max(0, int(effect.get("amount", 0)))
                actual = min(amount, player.stats.max_health - player.stats.health)
                if actual > 0:
                    player.stats.health += actual
                    return f"Recovered {actual} HP."

            elif effect_type == "damage":
                amount = max(0, int(effect.get("amount", 0)))
                actual = min(amount, player.stats.health - 1)  # Never kill
                if actual > 0:
                    player.stats.health -= actual
                    return f"Took {actual} damage."

            elif effect_type == "restore_stamina":
                amount = max(0, int(effect.get("amount", 0)))
                actual = min(amount, player.stats.max_stamina - player.stats.stamina)
                if actual > 0:
                    player.stats.stamina += actual
                    return f"Recovered {actual} stamina."

            elif effect_type == "advance_time":
                minutes = min(self.AI_EFFECT_CAPS["max_time_minutes"], max(0, int(effect.get("minutes", 0))))
                if minutes > 0:
                    self.game_state.session.game_time.advance_time(minutes)
                    # Passive recovery (same as advance_time method)
                    stamina_rec = minutes // 10
                    mana_rec = minutes // 10
                    player.stats.stamina = min(player.stats.max_stamina, player.stats.stamina + stamina_rec)
                    player.stats.mana = min(player.stats.max_mana, player.stats.mana + mana_rec)
                    expiry_msgs = self.expire_temporary_effects(player, minutes)
                    for msg in expiry_msgs:
                        self.pending_messages.append(msg)
                    h, m = divmod(minutes, 60)
                    time_str = f"{h}h {m}m" if h else f"{m}m"
                    return f"Time passes... ({time_str})"

            elif effect_type == "give_gold":
                amount = min(self.AI_EFFECT_CAPS["max_gold_give"], max(0, int(effect.get("amount", 0))))
                if amount > 0:
                    player.currency["gold"] = player.currency.get("gold", 0) + amount
                    return f"Found {amount} gold."

            elif effect_type == "remove_gold":
                current = player.currency.get("gold", 0)
                amount = min(current, max(0, int(effect.get("amount", 0))))
                if amount > 0:
                    player.currency["gold"] = current - amount
                    return f"Lost {amount} gold."

            elif effect_type == "spawn_item":
                name = str(effect.get("name", "Mysterious Object"))[:50]
                desc = str(effect.get("description", "An unusual find."))[:200]
                item_type = self._coerce_enum(effect.get("item_type", "tool"), ItemType)
                value = min(self.AI_EFFECT_CAPS["max_item_value"], max(1, int(effect.get("value", 5))))
                item = Item(name=name, description=desc, item_type=item_type, value=value)
                self.game_state.items[item.id] = item
                loc.items.append(item.id)
                return f"A {name} appears nearby."

            elif effect_type == "add_feature":
                name = str(effect.get("name", "Something unusual"))[:50]
                desc = str(effect.get("description", "You notice something new."))[:300]
                feature = NotableFeature(name=name, detailed_description=desc, is_interactive=True)
                loc.notable_features.append(feature)
                return f"You notice: {name}."

            elif effect_type == "spawn_npc":
                name = str(effect.get("name", "Stranger"))[:40]
                desc = str(effect.get("description", "A mysterious figure."))[:200]
                role = self._coerce_enum(effect.get("role", "commoner"), NPCRole)
                mood = max(0.0, min(1.0, float(effect.get("mood", 0.5))))
                npc = NPC(
                    name=name, description=desc, role=role,
                    current_location_id=loc.id, home_location_id=loc.id,
                    mood=mood, goal=NPCGoal.NONE,
                    level=min(player.level, max(1, int(effect.get("level", 1)))),
                )
                npc.base_stats = npc.stats.model_copy()
                self._create_services_for_npc(npc)
                self.game_state.characters[npc.id] = npc
                return f"{name} appears."

            elif effect_type == "modify_location":
                new_desc = effect.get("description")
                new_atmo = effect.get("atmosphere")
                if new_desc and isinstance(new_desc, str):
                    loc.description = new_desc[:500]
                if new_atmo and isinstance(new_atmo, str):
                    loc.atmosphere = new_atmo[:200]
                return None  # Silent

        except (ValueError, TypeError, KeyError) as e:
            llm_logger.warning(f"Invalid AI effect {effect_type}: {e}")
        return None

    def hunt(self) -> Tuple[bool, str]:
        """Hunt for game at the current location. Deterministic, no AI."""
        player = self.game_state.session.player_character
        loc = self.get_current_location()
        terrain = getattr(loc, 'general_type', None)
        terrain_key = terrain.value if terrain else None

        if terrain_key not in self.HUNT_TABLE:
            return False, "There's nothing to hunt here."
        if player.stats.stamina < self.HUNT_STAMINA_COST:
            return False, "You're too tired to hunt."

        player.stats.stamina -= self.HUNT_STAMINA_COST
        self.game_state.session.game_time.advance_time(self.HUNT_TIME_MINUTES)

        chance, loot_table = self.HUNT_TABLE[terrain_key]
        if random.random() > chance:
            return True, "You spent time tracking game but found nothing."

        names, weights = zip(*loot_table)
        chosen = random.choices(names, weights=weights, k=1)[0]
        item = self._create_catalog_item(chosen)
        if item:
            player.inventory.append(item.id)
            return True, f"You hunted successfully and obtained {chosen}!"
        return True, "You caught something but couldn't carry it."

    def forage(self) -> Tuple[bool, str]:
        """Forage for plants and herbs at the current location. Deterministic, no AI."""
        player = self.game_state.session.player_character
        loc = self.get_current_location()
        terrain = getattr(loc, 'general_type', None)
        terrain_key = terrain.value if terrain else None

        if terrain_key not in self.FORAGE_TABLE:
            return False, "There's nothing to forage here."
        if player.stats.stamina < self.FORAGE_STAMINA_COST:
            return False, "You're too tired to forage."

        player.stats.stamina -= self.FORAGE_STAMINA_COST
        self.game_state.session.game_time.advance_time(self.FORAGE_TIME_MINUTES)

        chance, loot_table = self.FORAGE_TABLE[terrain_key]
        if random.random() > chance:
            return True, "You searched the area but found nothing useful."

        names, weights = zip(*loot_table)
        chosen = random.choices(names, weights=weights, k=1)[0]
        item = self._create_catalog_item(chosen)
        if item:
            player.inventory.append(item.id)
            return True, f"You foraged successfully and found {chosen}!"
        return True, "You found something but couldn't carry it."

    def roll_examine_surprise(self, feature: NotableFeature) -> Optional[str]:
        """Roll for a surprise event when examining a feature. Returns message or None."""
        if feature.metadata.get("examined") or feature.metadata.get("puzzle") or \
           feature.metadata.get("campfire") or feature.metadata.get("corpse") or \
           feature.metadata.get("surprise_event"):
            return None

        feature.metadata["examined"] = True

        if random.random() > 0.20:
            return None

        player = self.game_state.session.player_character
        loc = self.get_current_location()
        feature.metadata["surprise_event"] = True

        roll = random.random()
        if roll < 0.35:
            gold = random.randint(5, 25)
            player.currency["gold"] = player.currency.get("gold", 0) + gold
            return f"DISCOVERY: Hidden among the {feature.name}, you find a small pouch containing {gold} gold!"
        elif roll < 0.65:
            reward = self._create_puzzle_reward()
            loc.items.append(reward.id)
            return f"DISCOVERY: Something glints inside the {feature.name} — it's a {reward.name}!"
        elif roll < 0.85:
            damage = random.randint(5, 15)
            actual = min(damage, player.stats.health - 1)
            if actual > 0:
                player.stats.health -= actual
                return f"DANGER: A hidden trap springs from the {feature.name}! You take {actual} damage."
            return None
        else:
            ambusher_names = ["Lurking Bandit", "Shadow Stalker", "Hidden Predator", "Cave Lurker"]
            name = random.choice(ambusher_names)
            ambusher = NPC(
                name=name,
                description=f"A hostile figure that was hiding near the {feature.name}.",
                role=NPCRole.COMMONER,
                current_location_id=loc.id, home_location_id=loc.id,
                mood=0.0, goal=NPCGoal.ATTACK_PLAYER, max_ticks=10,
                level=max(1, player.level + random.randint(-1, 1)),
            )
            ambusher.stats.health = 30 + (ambusher.level * 10)
            ambusher.stats.max_health = ambusher.stats.health
            ambusher.stats.strength = 8 + ambusher.level
            ambusher.base_stats = ambusher.stats.model_copy()
            self.game_state.characters[ambusher.id] = ambusher
            return f"DANGER: {name} was hiding near the {feature.name} and leaps out to attack!"

    def _update_npc_positions(self) -> None:
        """Move NPCs based on the time of day"""
        gt = self.game_state.session.game_time
        # Daytime: 7:00 to 19:00
        is_work_time = 7 <= gt.hour < 19
        
        for char in self.game_state.characters.values():
            if isinstance(char, NPC) and char.goal == NPCGoal.NONE:
                target_loc = char.work_location_id if (is_work_time and char.work_location_id) else char.home_location_id
                if target_loc and char.current_location_id != target_loc:
                    char.previous_location_id = char.current_location_id
                    char.current_location_id = target_loc

    # ── NPC Tick System ───────────────────────────────────────────────
    def _tick_npcs(self) -> None:
        """Tick all goal-driven NPCs. Called after each non-passive player action. No AI calls."""
        if not self.game_state:
            return
        player = self.game_state.session.player_character
        player_loc = player.current_location_id

        npcs = [c for c in list(self.game_state.characters.values())
                if isinstance(c, NPC) and c.goal != NPCGoal.NONE]

        for npc in npcs:
            if npc.id not in self.game_state.characters:
                continue
            npc.ticks_alive += 1
            same_loc = (npc.current_location_id == player_loc)

            if npc.max_ticks > 0 and npc.ticks_alive >= npc.max_ticks:
                self._despawn_npc(npc, "vanishes without a trace")
                continue

            if npc.goal == NPCGoal.ATTACK_PLAYER:
                self._tick_attack(npc, player, same_loc)
            elif npc.goal == NPCGoal.DELIVER_MESSAGE:
                self._tick_message(npc, player, same_loc)
            elif npc.goal == NPCGoal.FOLLOW_PLAYER:
                self._tick_follow(npc, player)
            elif npc.goal == NPCGoal.FLEE:
                self._tick_flee(npc, player, same_loc)

    def _tick_attack(self, npc: NPC, player: PlayerCharacter, same_loc: bool) -> None:
        if not same_loc:
            if npc.ticks_alive <= 5:
                npc.previous_location_id = npc.current_location_id
                npc.current_location_id = player.current_location_id
                self.pending_messages.append(f"DANGER: {npc.name} has tracked you down!")
            return
        if not self.in_combat:
            self.in_combat = True
            self.combat_opponents = [npc]
            self.pending_messages.append(f"DANGER: {npc.name} attacks you!")
        elif npc not in self.combat_opponents:
            self.pending_messages.append(f"DANGER: {npc.name} is circling, waiting to strike!")

    def _tick_message(self, npc: NPC, player: PlayerCharacter, same_loc: bool) -> None:
        if not same_loc:
            npc.previous_location_id = npc.current_location_id
            npc.current_location_id = player.current_location_id
            self.pending_messages.append(f"A figure approaches — {npc.name} has arrived.")
            return
        if not npc.goal_data.get("delivered"):
            msg = npc.goal_data.get("message", "...")
            self.pending_messages.append(f'{npc.name} speaks: "{msg}"')
            npc.goal_data["delivered"] = True
        else:
            if npc.is_transient:
                self._despawn_npc(npc, "slips away into the shadows")
            else:
                npc.goal = NPCGoal.NONE

    def _tick_follow(self, npc: NPC, player: PlayerCharacter) -> None:
        if npc.current_location_id != player.current_location_id:
            npc.previous_location_id = npc.current_location_id
            npc.current_location_id = player.current_location_id

    def _tick_flee(self, npc: NPC, player: PlayerCharacter, same_loc: bool) -> None:
        if same_loc:
            loc = self.game_state.locations.get(npc.current_location_id)
            if loc and loc.connections:
                valid = [c for c in loc.connections if c.is_passable]
                if valid:
                    target = random.choice(valid)
                    npc.previous_location_id = npc.current_location_id
                    npc.current_location_id = target.target_location_id
                    self.pending_messages.append(f"{npc.name} flees in panic!")
        if npc.is_transient and npc.ticks_alive >= 3:
            self._despawn_npc(npc, "has disappeared")

    def _despawn_npc(self, npc: NPC, flavor: str = "vanishes") -> None:
        self.pending_messages.append(f"{npc.name} {flavor}.")
        if npc.id in self.game_state.characters:
            del self.game_state.characters[npc.id]

    async def _update_regional_events(self, model_name: str) -> None:
        """Randomly trigger or end regional events"""
        current_region_id = self.game_state.session.current_region_id
        for region_id, region in self.game_state.session.world.regions.items():
            # Chance to end existing events
            if region.active_events and random.random() < 0.2:
                ended = region.active_events.pop(0)
                region.event_modifiers.clear()
                if region_id == current_region_id:
                    self.pending_messages.append(f"The {ended} in {region.name} has ended.")

            # Chance to start a new event if none active
            if not region.active_events and random.random() < 0.1:
                event_types = [
                    ("Market Day", "Prices are lower and the streets are crowded.", {"price_mod": 0.8}),
                    ("Festival", "Music and laughter fill the air.", {"stamina_mod": 0.5}),
                    ("Sandstorm" if "desert" in region.region_type.lower() else "Storm", "Visibility is low and travel is difficult.", {"visibility": 0, "stamina_cost": 2.0}),
                    ("Guard Inspection", "Guards are checking everyone's papers.", {"suspicion": 1.0})
                ]
                event_name, desc, mods = random.choice(event_types)
                region.active_events.append(event_name)
                region.event_modifiers.update(mods)
                if region_id == current_region_id:
                    self.pending_messages.append(f"EVENT in {region.name}: {event_name} - {desc}")

    def build_context_for_ai(self, target_npc: Optional[NPC] = None) -> str:
        if not self.game_state: return ""
        loc = self.get_current_location()
        p = self.game_state.session.player_character
        npcs = [char for char in self.game_state.characters.values() if char.current_location_id == loc.id and char.id != p.id]
        
        context_parts = [
            f"WORLD: {self.game_state.session.world.name}",
            f"TIME: {self.game_state.session.game_time.time_of_day.value}",
            f"LOCATION: {loc.name} - {loc.description}",
        ]
        
        if loc.atmosphere: context_parts.append(f"ATMOSPHERE: {loc.atmosphere}")
        if loc.ambient_sounds: context_parts.append(f"SOUNDS: {', '.join(loc.ambient_sounds)}")
        if loc.ambient_smells: context_parts.append(f"SMELLS: {', '.join(loc.ambient_smells)}")
        if getattr(loc, 'weather', None): context_parts.append(f"WEATHER: {loc.weather}")
        
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

    async def update_npc_memory(self, npc: NPC, player_msg: str, ai_msg: str, model_name: str) -> None:
        """Summarize the interaction and update NPC memory and mood"""
        current_memory = npc.interaction_summary or "No previous interactions."
        prompt = (f"Summarize the relationship and recent interaction between {npc.name} and the player. "
                  f"Previous Memory: {current_memory}\n"
                  f"New interaction: Player said '{player_msg}', {npc.name} replied '{ai_msg}'.\n"
                  "Respond with a JSON object containing: 'summary' (concise sentence) and 'sentiment' (float from -1.0 to 1.0 based on player's tone).")
        
        try:
            data = await self._generate_and_validate(prompt, model_name)
            npc.interaction_summary = data.get('summary', 'No notable interaction.')[:500]
            sentiment = data.get('sentiment', 0.0)
            # Update mood gradually
            npc.mood = max(0.0, min(1.0, npc.mood + (sentiment * 0.1)))
        except:
            # Fallback
            new_summary = await self.ai.generate_response(f"Summarize: {player_msg} -> {ai_msg}", is_content_generation=False, model_name=model_name)
            npc.interaction_summary = new_summary[:500]

    async def generate_rumor(self, npc: NPC, model_name: str) -> str:
        context = self.build_context_for_ai()
        prompt = f"The player is asking {npc.name} ({npc.description}) for a rumor. Based on the context, generate a short, intriguing rumor this NPC might know. Respond with just the rumor itself, as a single sentence."
        rumor = await self.ai.generate_response(prompt, context, model_name=model_name)
        npc.rumors.append(rumor)
        return rumor

    async def get_item_lore(self, item_id: UUID, model_name: str) -> str:
        item = self.game_state.items.get(item_id)
        if not item: return "You can't learn anything about that."
        context = self.build_context_for_ai()
        prompt = f"The player is studying '{item.name}' ({item.description}). Generate a short lore passage (2-3 sentences) about this item's history and significance."
        return await self.ai.generate_response(prompt, context, model_name=model_name)

    async def generate_quest(self, npc: NPC, model_name: str) -> Optional[Quest]:
        return await self.generate_fetch_quest(npc, model_name)

    async def generate_fetch_quest(self, npc: NPC, model_name: str) -> Optional[Quest]:
        context = self.build_context_for_ai()
        quest_prompt = f"The player is asking {npc.name} ({npc.description}) for a quest. Generate a fetch quest. JSON: name, description, item_name, item_description, location_hint."
        
        try:
            quest_data = await self._generate_and_validate(quest_prompt, model_name)
            quest_item = Item(name=quest_data.get('item_name', 'Mysterious Object'), description=quest_data.get('item_description', 'An item of unknown origin.'), item_type=ItemType.QUEST_ITEM, value=random.randint(10, 50), weight=random.uniform(0.1, 2.0))
            
            available_locations = [loc for loc in self.game_state.locations.values() if loc.id != npc.current_location_id and loc.location_type == LocationType.LOCATION]
            if not available_locations: return None
            target_location = random.choice(available_locations)
            
            if target_location.notable_features and random.random() < 0.6:
                random.choice(target_location.notable_features).contained_items.append(quest_item.id)
            else:
                target_location.items.append(quest_item.id)
            
            self.game_state.items[quest_item.id] = quest_item
            objective = QuestObjective(description=f"Find and return the {quest_item.name} to {npc.name}", objective_type="fetch", target=str(quest_item.id))
            
            quest = Quest(
                name=quest_data.get('name', 'A Task'), description=quest_data.get('description', 'Complete this task.'), quest_type=QuestType.FETCH,
                giver_id=npc.id, objectives=[objective], status=QuestStatus.ACTIVE,
                target_item_id=quest_item.id, location_hint=quest_data.get('location_hint', 'Search the area.'),
                rewards=QuestReward(experience=random.randint(50, 150), currency={"gold": random.randint(5, 25)})
            )
            
            self.game_state.quests[quest.id] = quest
            npc.available_quests.append(quest.id)
            npc.given_quests.append(quest.id)
            self.game_state.session.player_character.active_quests.append(quest.id)
            return quest
        except Exception as e:
            llm_logger.error(f"Error generating fetch quest: {e}")
            return None

    @staticmethod
    def _build_location_hint(loc_name: Optional[str], feature_name: Optional[str]) -> str:
        if loc_name and feature_name:
            return f"Search near the {feature_name} at {loc_name}."
        elif loc_name:
            return f"It was last seen somewhere around {loc_name}."
        return "Search the surrounding area."

    # ============================================================================
    # Dynamic Dungeon Generator
    # ============================================================================
    def _place_dungeon_treasure(self, location: BaseLocation, player_level: int) -> None:
        """Place a treasure reward at the deepest dungeon level."""
        equipment = [t for t in self.ITEM_CATALOG if t.get('item_type') in (ItemType.WEAPON, ItemType.ARMOR)]
        template = random.choice(equipment)
        treasure = Item(**template)
        treasure.rarity = random.choice([ItemRarity.UNCOMMON, ItemRarity.RARE])
        treasure.value = int(treasure.value * (1.5 if treasure.rarity == ItemRarity.UNCOMMON else 2.5))
        treasure.name = f"Dungeon {treasure.name}"
        treasure.lore_text = "Found in the depths of a forgotten dungeon."
        self.game_state.items[treasure.id] = treasure
        if location.notable_features:
            location.notable_features[0].contained_items.append(treasure.id)
        else:
            location.items.append(treasure.id)

    async def generate_dungeon_levels(
        self, entrance_location: BaseLocation, depth: int, dungeon_theme: str, model_name: str
    ) -> Optional[UUID]:
        """Generate a complete dungeon (2-3 levels) in a single AI call.
        Returns the UUID of the deepest level for quest targeting, or None on failure."""
        player = self.game_state.session.player_character
        region = self.game_state.session.world.regions.get(self.game_state.session.current_region_id)

        dungeon_prompt = (
            f"Generate a {depth}-level dungeon called '{dungeon_theme}' "
            f"beneath '{entrance_location.name}' in '{region.name if region else 'the wilds'}'. "
            f"Player level: {player.level}. "
            f"JSON with key 'levels': a list of {depth} objects (top to bottom). "
            f"Each level: name (str), description (str), short_description (str), "
            f"atmosphere (str), notable_features (list of {{name, description}}), "
            f"enemy (object with name, description — or null for the final level). "
            f"The final level should feel like a treasure chamber or climactic discovery."
        )

        try:
            data = await self._generate_and_validate(dungeon_prompt, model_name)
            levels_data = data.get('levels', [])
            if not levels_data:
                return None

            current_parent = entrance_location
            deepest_id = None

            for i, level_raw in enumerate(levels_data[:depth]):
                if not isinstance(level_raw, dict):
                    continue

                level_raw = self._sanitize_location_data(level_raw)
                notable_raw = level_raw.pop('notable_features', [])
                enemy_info = level_raw.pop('enemy', None)
                level_raw.pop('is_final', None)

                sub_loc = GeneralLocation.model_validate(level_raw)
                sub_loc.parent_id = entrance_location.parent_id
                sub_loc.coordinates = Coordinates(
                    x=entrance_location.coordinates.x,
                    y=entrance_location.coordinates.y,
                    z=entrance_location.coordinates.z - (i + 1),
                )
                sub_loc.tags.append("dungeon")
                sub_loc.light_level = max(10, 60 - (i * 20))

                for n in notable_raw:
                    if isinstance(n, dict):
                        sub_loc.notable_features.append(
                            NotableFeature(name=n.get('name', 'Feature'), detailed_description=n.get('description'))
                        )
                    elif isinstance(n, str):
                        sub_loc.notable_features.append(NotableFeature(name=n))

                self.game_state.locations[sub_loc.id] = sub_loc

                # Link: first connection from surface is HIDDEN (revealed by examine)
                current_parent.connections.append(LocationConnection(
                    target_location_id=sub_loc.id,
                    direction=Direction.DOWN,
                    description=f"A passage descends into {sub_loc.name}.",
                    is_visible=(i > 0),
                    is_passable=(i > 0),
                ))
                sub_loc.connections.append(LocationConnection(
                    target_location_id=current_parent.id,
                    direction=Direction.UP,
                    description=f"A passage leads back up to {current_parent.name}.",
                ))

                # Spawn enemy on non-final levels
                is_final = (i == len(levels_data[:depth]) - 1)
                if enemy_info and isinstance(enemy_info, dict) and not is_final:
                    enemy_level = max(1, player.level + i)
                    enemy = NPC(
                        name=str(enemy_info.get('name', 'Dungeon Creature')),
                        description=str(enemy_info.get('description', 'A hostile creature.')),
                        role=NPCRole.COMMONER,
                        current_location_id=sub_loc.id,
                        home_location_id=sub_loc.id,
                        mood=0.0,
                        goal=NPCGoal.ATTACK_PLAYER,
                        level=enemy_level,
                    )
                    enemy.stats.health = 30 + (enemy_level * 10)
                    enemy.stats.max_health = enemy.stats.health
                    enemy.stats.strength = 8 + enemy_level
                    enemy.base_stats = enemy.stats.model_copy()
                    self.game_state.characters[enemy.id] = enemy

                if is_final:
                    self._place_dungeon_treasure(sub_loc, player.level)

                deepest_id = sub_loc.id
                current_parent = sub_loc

            return deepest_id

        except Exception as e:
            llm_logger.error(f"Error generating dungeon levels: {e}")
            return None

    # ============================================================================
    # Building Interior Generation
    # ============================================================================
    async def generate_building_interior(
        self, district_location: BaseLocation, feature: NotableFeature, model_name: str
    ) -> Optional[UUID]:
        """Generate the interior of a building (1-3 floors + NPCs) in a single AI call.
        Returns the ground floor UUID, or None on failure."""
        player = self.game_state.session.player_character
        region = self.game_state.session.world.regions.get(self.game_state.session.current_region_id)

        building_type = feature.metadata.get("building_type", "building")
        building_name = feature.name

        # Determine floor count based on building type
        if building_type in ("tower", "castle", "fortress", "manor"):
            num_floors = random.randint(2, 3)
        elif building_type in ("tavern", "inn"):
            num_floors = 2
        else:
            num_floors = 1

        is_wilderness = building_type in ("tower", "castle", "fortress", "ruins")
        if is_wilderness:
            npc_guidance = (
                f"This is a {building_type} in the wilderness. "
                f"NPCs can be hostile creatures, bandits, trapped prisoners, or ghosts. "
                f"Include atmospheric lore, traps, or hidden treasure. "
                f"Roles: guard, commoner, antagonist."
            )
        else:
            npc_guidance = (
                f"Ground floor should have the main service NPCs. "
                f"Upper floors can have residents, storage, or special areas. "
                f"Roles: shopkeeper, innkeeper, guard, commoner, craftsman, merchant."
            )

        prompt = (
            f"Generate the interior of '{building_name}' (a {building_type}) "
            f"in '{district_location.name}' of '{region.name if region else 'the wilds'}'. "
            f"The building has {num_floors} floor(s). "
            f"JSON with key 'floors': a list of {num_floors} objects (ground floor first). "
            f"Each floor: name (str), description (str), short_description (str), "
            f"atmosphere (str), notable_features (list of {{name, description}}), "
            f"npcs (list of {{name, description, race, role}}). "
            f"{npc_guidance}"
        )

        try:
            data = await self._generate_and_validate(prompt, model_name)
            floors_data = data.get('floors', [])
            if not floors_data:
                return None

            ground_floor_id = None
            prev_floor = None

            for i, floor_raw in enumerate(floors_data[:num_floors]):
                if not isinstance(floor_raw, dict):
                    continue

                floor_raw = self._sanitize_location_data(floor_raw)
                notable_raw = floor_raw.pop('notable_features', [])
                npcs_raw = floor_raw.pop('npcs', [])
                floor_raw.pop('npc', None)

                floor_loc = GeneralLocation.model_validate(floor_raw)
                floor_loc.parent_id = district_location.parent_id
                floor_loc.location_type = LocationType.BUILDING if i == 0 else LocationType.ROOM
                floor_loc.coordinates = Coordinates(
                    x=district_location.coordinates.x,
                    y=district_location.coordinates.y,
                    z=i,
                )
                floor_loc.tags.extend(["building_interior", building_type])

                for n in notable_raw:
                    if isinstance(n, dict):
                        floor_loc.notable_features.append(
                            NotableFeature(name=n.get('name', 'Feature'), detailed_description=n.get('description'))
                        )
                    elif isinstance(n, str):
                        floor_loc.notable_features.append(NotableFeature(name=n))

                self.game_state.locations[floor_loc.id] = floor_loc

                if i == 0:
                    ground_floor_id = floor_loc.id
                    district_location.connections.append(LocationConnection(
                        target_location_id=floor_loc.id,
                        direction=Direction.IN,
                        description=f"You can enter {building_name}.",
                    ))
                    floor_loc.connections.append(LocationConnection(
                        target_location_id=district_location.id,
                        direction=Direction.OUT,
                        description=f"The exit leads back to {district_location.name}.",
                    ))
                else:
                    prev_floor.connections.append(LocationConnection(
                        target_location_id=floor_loc.id,
                        direction=Direction.UP,
                        description=f"Stairs lead up to {floor_loc.name}.",
                    ))
                    floor_loc.connections.append(LocationConnection(
                        target_location_id=prev_floor.id,
                        direction=Direction.DOWN,
                        description=f"Stairs lead down to {prev_floor.name}.",
                    ))

                for npc_raw in (npcs_raw if isinstance(npcs_raw, list) else []):
                    if not isinstance(npc_raw, dict):
                        continue
                    npc_raw = self._sanitize_npc_data(npc_raw)
                    npc_raw.update({
                        'id': str(uuid4()),
                        'current_location_id': floor_loc.id,
                        'home_location_id': floor_loc.id,
                    })
                    try:
                        npc = NPC.model_validate(npc_raw)
                        npc.base_stats = npc.stats.model_copy()
                        self._create_services_for_npc(npc)
                        self.game_state.characters[npc.id] = npc
                    except Exception as e:
                        llm_logger.error(f"Error creating building NPC: {e}")

                prev_floor = floor_loc

            if ground_floor_id:
                feature.metadata["building_location_id"] = str(ground_floor_id)

            return ground_floor_id

        except Exception as e:
            llm_logger.error(f"Error generating building interior: {e}")
            return None

    async def enter_building(self, building_name: str, model_name: str) -> Tuple[bool, str]:
        """Enter a building by name. Generates interior on first visit."""
        if not self.game_state:
            return False, "No session."

        loc = self.get_current_location()

        feature = None
        for f in loc.notable_features:
            if building_name.lower() in f.name.lower() and f.metadata.get("enterable"):
                feature = f
                break

        if not feature:
            return False, f"There is no enterable building called '{building_name}' here."

        building_loc_id_str = feature.metadata.get("building_location_id")
        if building_loc_id_str:
            building_loc_id = UUID(building_loc_id_str)
        else:
            building_loc_id = await self.generate_building_interior(loc, feature, model_name)
            if not building_loc_id:
                return False, "The door won't budge."

        target = self.game_state.locations.get(building_loc_id)
        if not target:
            return False, "The building interior could not be found."

        pc = self.game_state.session.player_character
        pc.previous_location_id = pc.current_location_id
        pc.current_location_id = target.id
        target.visit_count += 1
        target.last_visited = datetime.now()
        if target.id not in pc.discovered_locations:
            pc.discovered_locations.add(target.id)
            pc.locations_discovered += 1

        await self.advance_time(1, model_name)
        return True, f"You enter {feature.name}..."

    # ============================================================================
    # Cooking System
    # ============================================================================
    async def cook_items(self, ingredient_ids: List[UUID], model_name: str) -> Tuple[bool, str, Optional[Item]]:
        """Combine 1-3 food ingredients at a campfire to produce a meal."""
        if not self.game_state:
            return False, "No active session.", None
        player = self.game_state.session.player_character

        ingredients: List[Item] = []
        for iid in ingredient_ids:
            item = self.game_state.items.get(iid)
            if not item or iid not in player.inventory:
                return False, "Ingredient not found in your inventory.", None
            if item.name not in self.FOOD_INGREDIENT_NAMES:
                return False, f"{item.name} is not a cookable ingredient.", None
            ingredients.append(item)

        if not (1 <= len(ingredients) <= 3):
            return False, "You can cook with 1 to 3 ingredients.", None

        ingredient_names = [i.name for i in ingredients]
        prompt = (
            f"The player is cooking at a campfire with: {ingredient_names}. "
            f"Generate a fantasy meal. JSON with: name (str), description (str, 1-2 sentences), "
            f"effects (list of {{stat, bonus, duration_minutes}}), heal_amount (int, 0-50). "
            f"Valid stats: {list(self.COOKING_BUFF_CAPS.keys())}. "
            f"More ingredients = stronger effects. Bonus range 1-5, duration 30-120 minutes."
        )

        try:
            data = await self._generate_and_validate(prompt, model_name)
            meal_name = str(data.get("name", "Mystery Stew"))
            meal_desc = str(data.get("description", "A hearty meal."))
            heal_amount = min(50, max(0, int(data.get("heal_amount", 0))))

            validated_effects = []
            for eff in (data.get("effects", []) or [])[:3]:
                if not isinstance(eff, dict):
                    continue
                stat = str(eff.get("stat", "")).lower()
                if stat not in self.COOKING_BUFF_CAPS:
                    continue
                bonus = min(self.COOKING_BUFF_CAPS[stat], max(1, int(eff.get("bonus", 1))))
                duration = min(120, max(30, int(eff.get("duration_minutes", 60))))
                validated_effects.append({"stat": stat, "bonus": bonus, "duration_minutes": duration})

            use_effects_list = []
            if heal_amount > 0:
                use_effects_list.append(f"heal:{heal_amount}")
            for ve in validated_effects:
                use_effects_list.append(f"buff:{ve['stat']}:{ve['bonus']}:{ve['duration_minutes']}")

            effect_parts = []
            if heal_amount > 0:
                effect_parts.append(f"+{heal_amount} HP")
            for ve in validated_effects:
                effect_parts.append(f"+{ve['bonus']} {ve['stat']} for {ve['duration_minutes']}min")
            effect_summary = ", ".join(effect_parts) if effect_parts else "a simple meal"

            meal = Item(
                name=meal_name,
                description=f"{meal_desc} ({effect_summary})",
                item_type=ItemType.CONSUMABLE,
                consumable=True,
                use_effects=use_effects_list,
                self_use_effect_description=f"You eat the {meal_name}. {effect_summary}.",
                value=sum(i.value for i in ingredients) * 2,
            )
            self.game_state.items[meal.id] = meal

            for ingredient in ingredients:
                self.remove_item_from_inventory(player, ingredient.name, 1)

            player.inventory.append(meal.id)
            return True, f"You cooked {meal_name}! ({effect_summary})", meal

        except Exception as e:
            llm_logger.error(f"Error in cook_items: {e}")
            return False, "The cooking attempt failed.", None

    # ============================================================================
    # Environmental Puzzles
    # ============================================================================
    def _create_puzzle_reward(self) -> Item:
        """Create a reward item for solving a puzzle."""
        if random.random() < 0.6:
            equipment = [t for t in self.ITEM_CATALOG if t.get("item_type") in (ItemType.WEAPON, ItemType.ARMOR)]
            template = random.choice(equipment)
            reward = Item(**template)
            reward.id = uuid4()
            reward.rarity = random.choice([ItemRarity.UNCOMMON, ItemRarity.RARE])
            reward.value = int(reward.value * (1.5 if reward.rarity == ItemRarity.UNCOMMON else 2.5))
            reward.name = f"Ancient {reward.name}"
        else:
            stat = random.choice(["strength", "dexterity", "constitution", "intelligence", "wisdom"])
            reward = Item(
                name="Essence Vial",
                description=f"A shimmering vial that permanently boosts {stat}.",
                item_type=ItemType.CONSUMABLE,
                consumable=True,
                rarity=ItemRarity.RARE,
                value=75,
                use_effects=[f"buff:{stat}:2:9999"],
                self_use_effect_description=f"You drink the essence and feel permanently stronger. (+2 {stat})",
            )
        self.game_state.items[reward.id] = reward
        return reward

    def create_puzzle_feature(self, location: BaseLocation) -> NotableFeature:
        """Generate an environmental puzzle and place it in a location."""
        puzzle_def = random.choice(self.PUZZLE_TYPES)
        template_name, template_desc = random.choice(puzzle_def["templates"])
        reward = self._create_puzzle_reward()

        puzzle_feature = NotableFeature(
            name=template_name,
            detailed_description=template_desc,
            metadata={
                "puzzle": True,
                "puzzle_type": puzzle_def["type"],
                "solution_hint": puzzle_def["hint"],
                "accepted_item_types": [t.value for t in puzzle_def["accepted"]],
                "solved": False,
                "reward_item_id": str(reward.id),
                "reward_gold": random.randint(10, 50),
                "reward_xp": random.randint(30, 100),
            },
        )
        location.notable_features.append(puzzle_feature)
        return puzzle_feature

    def attempt_solve_puzzle(self, feature: NotableFeature, item: Item) -> Tuple[bool, str]:
        """Attempt to solve a puzzle feature with an item."""
        if not feature.metadata.get("puzzle"):
            return False, "This is not a puzzle."
        if feature.metadata.get("solved"):
            return False, "This puzzle has already been solved."

        accepted_types = feature.metadata.get("accepted_item_types", [])
        puzzle_type = feature.metadata.get("puzzle_type", "offering")

        if item.item_type.value not in accepted_types:
            hint = feature.metadata.get("solution_hint", "")
            return False, f"The {item.name} doesn't seem right for this. {hint}"

        feature.metadata["solved"] = True
        player = self.game_state.session.player_character
        loc = self.get_current_location()

        if puzzle_type in ("offering", "arrangement"):
            self.remove_item_from_inventory(player, item.name, 1)

        reward_msgs = []
        reward_item_id_str = feature.metadata.get("reward_item_id")
        if reward_item_id_str:
            reward_item = self.game_state.items.get(UUID(reward_item_id_str))
            if reward_item:
                loc.items.append(reward_item.id)
                reward_msgs.append(f"A {reward_item.name} materializes before you!")

        reward_gold = feature.metadata.get("reward_gold", 0)
        if reward_gold > 0:
            player.currency["gold"] = player.currency.get("gold", 0) + reward_gold
            reward_msgs.append(f"+{reward_gold} gold")

        reward_xp = feature.metadata.get("reward_xp", 0)
        if reward_xp > 0:
            player.experience += reward_xp
            reward_msgs.append(f"+{reward_xp} XP")
            lvl_msg = self.check_level_up()
            if lvl_msg:
                reward_msgs.append(lvl_msg)

        reward_summary = " ".join(reward_msgs)

        narratives = {
            "offering": f"You place the {item.name} on the {feature.name}. The shrine glows with ethereal light!",
            "arrangement": f"You use the {item.name} to complete the {feature.name}. The pattern clicks into place!",
            "activation": f"You apply the {item.name} to the {feature.name}. Ancient mechanisms whir to life!",
        }
        narrative = narratives.get(puzzle_type, f"The {feature.name} reacts to the {item.name}!")

        return True, f"{narrative}\n{reward_summary}"

    def _maybe_place_food_ingredients(self, location) -> None:
        """Randomly scatter food ingredients in outdoor locations."""
        gt = getattr(location, 'general_type', None)
        if not gt or gt.value not in ("forest", "meadow", "clearing", "river", "mountain", "crossroads"):
            return
        if random.random() < 0.4:
            food_templates = [t for t in self.ITEM_CATALOG if t["name"] in self.FOOD_INGREDIENT_NAMES]
            for _ in range(random.randint(1, 2)):
                template = random.choice(food_templates)
                item = Item(**template)
                item.id = uuid4()
                item.current_stack_size = random.randint(1, 3)
                self.game_state.items[item.id] = item
                if location.notable_features and random.random() < 0.5:
                    random.choice(location.notable_features).contained_items.append(item.id)
                else:
                    location.items.append(item.id)

    def _maybe_add_campfire(self, location) -> None:
        """Chance to add a campfire feature to wilderness locations."""
        gt = getattr(location, 'general_type', None)
        if not gt or gt.value not in ("forest", "meadow", "clearing", "river", "mountain", "crossroads", "ruins"):
            return
        if random.random() < 0.25:
            names = ["Campfire Ring", "Smoldering Campfire", "Stone Fire Pit", "Traveler's Hearth", "Old Campfire"]
            location.notable_features.append(NotableFeature(
                name=random.choice(names),
                detailed_description="A circle of stones surrounds charred wood and ash. It could be rekindled for cooking.",
                metadata={"campfire": True},
            ))

    def _maybe_place_puzzle(self, location) -> None:
        """Chance to place an environmental puzzle in a location during world gen."""
        gt = getattr(location, 'general_type', None)
        if not gt or gt.value not in ("ruins", "clearing", "mountain", "cave", "graveyard"):
            return
        if random.random() < 0.15:
            self.create_puzzle_feature(location)

    # ============================================================================
    # Dynamic Quest Generator (Layer 2)
    # ============================================================================
    async def generate_dynamic_quest(self, model_name: str, trigger_reason: str = "exploration") -> Optional[Quest]:
        """Generate a varied quest (any type) based on current context."""
        if not self.game_state:
            return None

        player = self.game_state.session.player_character
        loc = self.get_current_location()
        region = self.game_state.session.world.regions.get(self.game_state.session.current_region_id)

        active_quest_names = [
            self.game_state.quests[qid].name
            for qid in player.active_quests if qid in self.game_state.quests
        ]
        npcs_here = [
            c.name for c in self.game_state.characters.values()
            if c.current_location_id == loc.id and c.id != player.id
        ]

        quest_prompt = (
            f"Generate a quest encounter for the player at '{loc.name}' ({loc.description}), "
            f"in the '{region.name}' region ({region.region_type}). "
            f"Player level: {player.level}. Active quests: {active_quest_names or 'None'}. "
            f"NPCs here: {npcs_here or 'None'}. Trigger: {trigger_reason}. "
            f"Choose ONE quest type: fetch, kill, escort, delivery, exploration, puzzle. "
            f"JSON with: quest_type (str), name (str), description (str), "
            f"npc_name (str or null), npc_description (str or null), "
            f"npc_role (one of: quest_giver, merchant, guard, commoner), "
            f"objective_description (str), "
            f"item_name (str or null, if quest involves an item), "
            f"item_description (str or null), "
            f"target_location_hint (str), "
            f"reward_gold (int), reward_xp (int)."
        )

        try:
            data = await self._generate_and_validate(quest_prompt, model_name)
            quest_type_str = str(data.get('quest_type', 'fetch')).lower()
            type_map = {
                'fetch': QuestType.FETCH, 'kill': QuestType.KILL,
                'escort': QuestType.ESCORT, 'delivery': QuestType.DELIVERY,
                'exploration': QuestType.EXPLORATION, 'puzzle': QuestType.PUZZLE,
            }
            quest_type = type_map.get(quest_type_str, QuestType.FETCH)

            # Create quest giver NPC if provided
            giver_id = None
            npc_name = data.get('npc_name')
            if npc_name and isinstance(npc_name, str):
                npc_role = NPCRole(self._coerce_enum(data.get('npc_role', 'quest_giver'), NPCRole))
                npc = NPC(
                    name=npc_name,
                    description=str(data.get('npc_description', 'A mysterious figure.')),
                    role=npc_role,
                    current_location_id=loc.id,
                    home_location_id=loc.id,
                )
                self._create_services_for_npc(npc)
                self.game_state.characters[npc.id] = npc
                giver_id = npc.id

            # EXPLORATION → generate a dungeon
            if quest_type == QuestType.EXPLORATION:
                available_locs = [
                    l for l in self.game_state.locations.values()
                    if l.id != loc.id and l.location_type == LocationType.LOCATION
                    and l.parent_id == self.game_state.session.current_region_id
                ]
                if available_locs:
                    entrance_loc = random.choice(available_locs)
                    dungeon_theme = str(data.get('name', 'Ancient Dungeon'))
                    depth = random.randint(2, 3)
                    deepest_id = await self.generate_dungeon_levels(entrance_loc, depth, dungeon_theme, model_name)
                    if deepest_id:
                        entrance_feature = NotableFeature(
                            name=f"Hidden Passage",
                            detailed_description=(
                                f"Pushing aside debris, you discover a dark opening leading underground. "
                                f"Cold air rushes up from below. This must be the entrance to {dungeon_theme}."
                            ),
                            metadata={"dungeon_entrance": True, "hidden_connection_direction": "down"},
                        )
                        entrance_loc.notable_features.append(entrance_feature)
                        objective = QuestObjective(
                            description=str(data.get('objective_description', f"Explore the depths of {dungeon_theme}.")),
                            objective_type="explore",
                            target=str(deepest_id),
                        )
                        quest = Quest(
                            name=dungeon_theme,
                            description=str(data.get('description', 'A hidden dungeon awaits exploration.')),
                            quest_type=QuestType.EXPLORATION,
                            giver_id=giver_id,
                            objectives=[objective],
                            status=QuestStatus.ACTIVE,
                            location_hint=f"Search for a hidden passage near {entrance_loc.name}.",
                            rewards=QuestReward(
                                experience=int(data.get('reward_xp', random.randint(80, 200))),
                                currency={"gold": int(data.get('reward_gold', random.randint(15, 40)))},
                            ),
                            is_dynamic=True,
                            start_location_id=loc.id,
                        )
                        self.game_state.quests[quest.id] = quest
                        player.active_quests.append(quest.id)
                        self.game_state.session.actions_since_last_quest = 0
                        return quest

            # Create quest item for types that need one
            target_item_id = None
            item_placement_loc = None
            item_placement_feature = None
            if quest_type in (QuestType.FETCH, QuestType.DELIVERY, QuestType.PUZZLE):
                item_name = str(data.get('item_name', 'Quest Object'))
                quest_item = Item(
                    name=item_name,
                    description=str(data.get('item_description', 'An item of importance.')),
                    item_type=ItemType.QUEST_ITEM,
                    value=random.randint(10, 50),
                )
                self.game_state.items[quest_item.id] = quest_item
                target_item_id = quest_item.id

                # Place item in a nearby location within the same region
                available = [
                    l for l in self.game_state.locations.values()
                    if l.id != loc.id and l.location_type == LocationType.LOCATION
                    and l.parent_id == self.game_state.session.current_region_id
                ]
                if available:
                    target_loc = random.choice(available)
                    item_placement_loc = target_loc.name
                    if target_loc.notable_features and random.random() < 0.5:
                        feature = random.choice(target_loc.notable_features)
                        feature.contained_items.append(quest_item.id)
                        item_placement_feature = feature.name
                    else:
                        target_loc.items.append(quest_item.id)

            # Create hostile NPC for kill quests
            kill_target_id = None
            if quest_type == QuestType.KILL:
                enemy_name = str(data.get('item_name', 'Hostile Creature'))
                enemy = NPC(
                    name=enemy_name,
                    description=str(data.get('item_description', 'A dangerous foe.')),
                    role=NPCRole.COMMONER,
                    current_location_id=loc.id,
                    home_location_id=loc.id,
                    mood=0.0,
                    goal=NPCGoal.ATTACK_PLAYER,
                    level=max(1, player.level + random.randint(-1, 1)),
                )
                enemy.stats.health = 30 + (enemy.level * 10)
                enemy.stats.max_health = enemy.stats.health
                enemy.stats.strength = 8 + enemy.level
                enemy.base_stats = enemy.stats.model_copy()
                self.game_state.characters[enemy.id] = enemy
                kill_target_id = enemy.id

            # Build objective
            obj_desc = str(data.get('objective_description', 'Complete this task.'))
            obj_type_map = {
                QuestType.FETCH: "fetch", QuestType.KILL: "kill",
                QuestType.ESCORT: "escort", QuestType.DELIVERY: "delivery",
                QuestType.EXPLORATION: "explore", QuestType.PUZZLE: "puzzle",
            }
            objective = QuestObjective(
                description=obj_desc,
                objective_type=obj_type_map.get(quest_type, "fetch"),
                target=str(kill_target_id or target_item_id or loc.id),
            )

            quest = Quest(
                name=str(data.get('name', 'A Dynamic Quest')),
                description=str(data.get('description', 'Something needs doing.')),
                quest_type=quest_type,
                giver_id=giver_id,
                objectives=[objective],
                status=QuestStatus.ACTIVE,
                target_item_id=target_item_id,
                location_hint=self._build_location_hint(item_placement_loc, item_placement_feature) if target_item_id else None,
                rewards=QuestReward(
                    experience=int(data.get('reward_xp', random.randint(50, 150))),
                    currency={"gold": int(data.get('reward_gold', random.randint(5, 25)))},
                ),
                is_dynamic=True,
                start_location_id=loc.id,
            )

            self.game_state.quests[quest.id] = quest
            player.active_quests.append(quest.id)
            self.game_state.session.actions_since_last_quest = 0
            return quest

        except Exception as e:
            llm_logger.error(f"Error generating dynamic quest: {e}")
            return None

    async def _check_dynamic_quest_trigger(self, model_name: str) -> None:
        """Check if a dynamic quest should spawn after movement."""
        if not self.game_state:
            return

        player = self.game_state.session.player_character
        session = self.game_state.session

        # Don't spawn if player already has too many active quests
        active_side_quests = sum(
            1 for qid in player.active_quests
            if qid in self.game_state.quests
            and self.game_state.quests[qid].quest_type != QuestType.MAIN_STORY
        )
        if active_side_quests >= 4:
            return

        # Trigger 1: Random on-enter (~12% chance)
        on_enter = random.random() < 0.12
        # Trigger 2: Forced after 20 actions without a new quest
        idle_forced = (session.actions_since_last_quest >= 20
                       and active_side_quests == 0)

        if on_enter or idle_forced:
            reason = "idle_exploration" if idle_forced else "chance_encounter"
            quest = await self.generate_dynamic_quest(model_name, trigger_reason=reason)
            if quest:
                loc = self.get_current_location()
                if quest.giver_id:
                    giver = self.game_state.characters.get(quest.giver_id)
                    giver_name = giver.name if giver else "A stranger"
                    self.pending_messages.append(
                        f"QUEST ENCOUNTER: {giver_name} approaches you. \"{quest.description}\""
                    )
                else:
                    # No quest giver — place a discoverable note/scroll in the location
                    triggers = [
                        ("Weathered Scroll", "A worn scroll left behind by a previous traveler."),
                        ("Torn Notice", "A notice pinned to a nearby surface, fluttering in the wind."),
                        ("Mysterious Letter", "A sealed letter someone dropped on the ground."),
                    ]
                    note_name, note_desc = random.choice(triggers)
                    note = Item(
                        name=note_name,
                        description=f"{note_desc} It reads: \"{quest.description}\"",
                        item_type=ItemType.BOOK,
                        value=0,
                        lore_text=quest.location_hint or "Search the area.",
                    )
                    self.game_state.items[note.id] = note
                    loc.items.append(note.id)
                    self.pending_messages.append(
                        f"DISCOVERY: You notice a {note_name.lower()} lying on the ground nearby."
                    )

    async def handle_service_request(self, npc: NPC, service_name: str, model_name: str) -> str:
        if not npc.services_offered: return f"{npc.name} doesn't offer any services."
        service = next((s for s in npc.services_offered if service_name.lower() in s.name.lower()), None)
        if not service: return f"{npc.name} doesn't offer '{service_name}'."
        
        player = self.game_state.session.player_character
        if service.service_type == ServiceType.BUY_SELL:
            return f"{npc.name}: \"{service.description} Use 'buy <item>' or 'sell <item>' to trade.\""
        elif service.service_type == ServiceType.REST:
            cost = sum(service.cost.values())
            if player.currency.get("gold", 0) < cost: return f"{npc.name}: \"You need {cost} gold.\""
            player.currency["gold"] -= cost
            player.stats.health, player.stats.mana, player.stats.stamina = player.stats.max_health, player.stats.max_mana, player.stats.max_stamina
            await self.advance_time(480, model_name)
            return f"{npc.name}: \"Sleep well.\" (Restored and time advanced)"
        elif service.service_type == ServiceType.HEAL:
            cost = service.cost.get("gold", 1)
            if player.currency.get("gold", 0) < cost: return f"{npc.name}: \"You need {cost} gold.\""
            player.currency["gold"] -= cost
            player.stats.health = min(player.stats.max_health, player.stats.health + 30)
            return f"{npc.name}: \"Healed.\""
        return f"{npc.name}: \"{service.description}\""

    async def combine_items(self, item1_id: UUID, item2_id: UUID, model_name: str) -> Tuple[bool, str, Optional[Item]]:
        item1 = self.game_state.items.get(item1_id)
        item2 = self.game_state.items.get(item2_id)
        if not item1 or not item2: return False, "Item not found.", None

        # 1. Check predefined combinations
        result_name = item1.combinations.get(item2.name) or item2.combinations.get(item1.name)
        
        # 2. Crafting logic for materials
        is_material_crafting = item1.item_type == ItemType.MATERIAL and item2.item_type == ItemType.MATERIAL
        
        if result_name or is_material_crafting:
            prompt = f"Combine '{item1.name}' ({item1.description}) and '{item2.name}' ({item2.description}). "
            if is_material_crafting:
                prompt += "This is a CRAFTING attempt with RAW MATERIALS. Suggest a logical outcome (weapon, tool, or refined item). "
            prompt += "JSON with: name, description, item_type (weapon, armor, consumable, material, quest_item), value (int), rarity (common, rare, epic, legendary)."
            
            try:
                data = await self._generate_and_validate(prompt, model_name)
                # Create the item
                new_item = Item(
                    name=data.get('name', 'Crafted Item'),
                    description=data.get('description', 'Something you made.'),
                    item_type=ItemType(self._coerce_enum(data.get('item_type', 'material'), ItemType)),
                    value=data.get('value', 10),
                    rarity=ItemRarity(self._coerce_enum(data.get('rarity', 'common'), ItemRarity))
                )
                self.game_state.items[new_item.id] = new_item
                
                # Consume materials
                inventory = self.game_state.session.player_character.inventory
                if item1.id in inventory: inventory.remove(item1.id)
                if item2.id in inventory: inventory.remove(item2.id)
                
                msg = f"You successfully crafted {new_item.name} from {item1.name} and {item2.name}!"
                return True, msg, new_item
            except:
                return False, f"You tried to combine {item1.name} and {item2.name}, but nothing useful came of it.", None

        return False, "These items don't seem like they would work together.", None

    async def use_item_on_feature(self, item_id: UUID, feature_id: UUID, model_name: str) -> Tuple[bool, str]:
        """Attempt to use an item on a notable feature to trigger an effect or puzzle solution"""
        item = self.game_state.items.get(item_id)
        loc = self.get_current_location()
        feature = next((f for f in loc.notable_features if f.id == feature_id), None)
        
        if not item or not feature: return False, "Target not found."
        
        prompt = f"The player is using '{item.name}' on '{feature.name}' ({feature.detailed_description or feature.name}). If this solves a puzzle or triggers a meaningful change in the environment, return JSON with: success (bool), outcome_description, new_exit_direction (optional), new_exit_description (optional). Example: Using a 'Makeshift Bridge' on a 'Deep Chasm' could unlock the 'EAST' exit."
        
        try:
            data = await self._generate_and_validate(prompt, model_name)
            if data.get('success'):
                msg = data.get('outcome_description', 'Something happened.')
                if data.get('new_exit_direction'):
                    direction = Direction(self._coerce_enum(data.get('new_exit_direction', 'north'), Direction))
                    # Check if connection already exists
                    if not any(c.direction == direction for c in loc.connections):
                        # Create a dummy target location or discover a hidden one?
                        # For simplicity, let's just say it unlocks a hidden connection if it existed, 
                        # or we create a new "Secret Path" location.
                        target_loc = GeneralLocation(
                            name="Secret Path",
                            description="A path revealed by your ingenuity.",
                            short_description="A secret area.",
                            parent_id=loc.parent_id,
                            coordinates=Coordinates(x=loc.coordinates.x, y=loc.coordinates.y, z=loc.coordinates.z) # Same coords but different ID
                        )
                        self.game_state.locations[target_loc.id] = target_loc
                        loc.connections.append(LocationConnection(
                            target_location_id=target_loc.id,
                            direction=direction,
                            description=data.get('new_exit_description', f"A path leading {direction.value}.")
                        )
                        )
                        msg += f" (Unlocked exit to the {direction.value})"
                return True, msg
            else:
                return False, data.get('outcome_description', "Nothing happens.")
        except Exception as e:
            llm_logger.error(f"Error using item on feature: {e}")
            return False, "Nothing happens."

    def buy_item_from_npc(self, npc: NPC, item_name: str) -> str:
        item = next((self.game_state.items.get(iid) for iid in npc.shop_inventory if iid in self.game_state.items and item_name.lower() in self.game_state.items[iid].name.lower()), None)
        if not item: return f"{npc.name} doesn't have that."
        player = self.game_state.session.player_character
        price = int(item.value * npc.prices_modifier)
        if player.currency.get("gold", 0) < price: return f"Need {price} gold."
        player.currency["gold"] -= price
        player.inventory.append(item.id)
        npc.shop_inventory.remove(item.id)
        return f"Bought {item.name}."

    def sell_item_to_npc(self, npc: NPC, item_name: str) -> str:
        item = self.find_item_in_inventory(item_name)
        if not item: return "You don't have it."
        if item.item_type == ItemType.QUEST_ITEM: return "Can't sell quest items."
        
        player = self.game_state.session.player_character
        region = self.game_state.session.world.regions.get(self.game_state.session.current_region_id)
        
        # Determine basic price
        price = max(1, int(item.value * 0.5 * npc.prices_modifier))
        
        # Economy: Supply and Demand
        if region:
            supply = region.market_history.get(item.name, 0)
            # Faction discount/bonus
            faction_mod = 1.0
            if npc.faction:
                standing = player.faction_standings.get(npc.faction, 0)
                faction_mod = 1.0 + (standing / 100.0)
            
            # Mood modifier
            mood_mod = 0.5 + npc.mood
            
            total_mod = npc.prices_modifier * faction_mod * mood_mod
            price = max(1, int(item.value * 0.5 * total_mod))

            if supply > 5:
                penalty = min(0.9, (supply - 5) * 0.1)
                price = max(1, int(price * (1.0 - penalty)))
            
            region.market_history[item.name] = supply + 1
        
        player.inventory.remove(item.id)
        player.currency["gold"] = player.currency.get("gold", 0) + price
        npc.shop_inventory.append(item.id)
        
        msg = f"Sold {item.name} for {price}g."
        if region and region.market_history.get(item.name, 0) > 6:
            msg += " (Price lowered due to high supply)"
        return msg

    def find_item_in_inventory(self, item_name: str) -> Optional[Item]:
        if not self.game_state: return None
        for item_id in self.game_state.session.player_character.inventory:
            item = self.game_state.items.get(item_id)
            if item and item_name.lower() in item.name.lower(): return item
        return None
    
    def pickup_item(self, item_name: str) -> str:
        """Pick up an item from the current location and add it to inventory"""
        item = self.find_item_in_location(item_name)
        if not item:
            return f"There is no '{item_name}' here."
        
        player = self.game_state.session.player_character
        location = self.get_current_location()
        
        # Add to inventory
        self.add_item_to_inventory(player, item, 1)
        
        # Remove from location
        if item.id in location.items:
            location.items.remove(item.id)
        
        return f"You picked up the {item.name}."
    
    def find_item_in_location(self, item_name: str) -> Optional[Item]:
        if not self.game_state: return None
        for item_id in self.get_current_location().items:
            item = self.game_state.items.get(item_id)
            if item and item_name.lower() in item.name.lower(): return item
        return None

    def equip_item(self, item_name: str) -> str:
        item = self.find_item_in_inventory(item_name)
        if not item or not item.equipment_slot: return "Can't equip."
        player = self.game_state.session.player_character
        if item.equipment_slot in player.equipped_items:
            old_id = player.equipped_items[item.equipment_slot]
            old = self.game_state.items.get(old_id)
            msg = f"Unequipped {old.name}. " if old else ""
        else: msg = ""
        player.equipped_items[item.equipment_slot] = item.id
        self.apply_equipment_effects(player)
        return msg + f"Equipped {item.name}."

    def unequip_item(self, item_name: str) -> str:
        player = self.game_state.session.player_character
        item = next((self.game_state.items.get(iid) for iid in player.equipped_items.values() if iid in self.game_state.items and item_name.lower() in self.game_state.items[iid].name.lower()), None)
        if not item: return "Not equipped."
        slot = next(s for s, iid in player.equipped_items.items() if iid == item.id)
        del player.equipped_items[slot]
        self.apply_equipment_effects(player)
        return f"Unequipped {item.name}."

    async def save_game(self, filepath: Path) -> bool:
        if not self.game_state: return False
        try:
            # GameState is a BaseModel, model_dump(mode='json') handles everything
            # as long as sub-models like WorldGrid are also BaseModels.
            game_dict = self.game_state.model_dump(mode='json')
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(game_dict, f, indent=2, default=str)
            return True
        except Exception as e:
            llm_logger.error(f"Save error: {e}")
            import traceback
            llm_logger.error(traceback.format_exc())
            return False

    async def load_game(self, filepath: Path) -> bool:
        try:
            with open(filepath, 'r', encoding='utf-8') as f: game_dict = json.load(f)
            # Migration: old saves used 'world_grid', new uses 'region_grids'
            session_data = game_dict['session']
            if 'world_grid' in session_data and 'region_grids' not in session_data:
                starting_id = session_data.get('current_region_id') or next(iter(session_data.get('world', {}).get('regions', {})), None)
                if starting_id:
                    session_data['region_grids'] = {starting_id: session_data.pop('world_grid')}
                else:
                    session_data.pop('world_grid', None)
            grids_data = session_data.get('region_grids', {})
            for region_id_str, grid_data in grids_data.items():
                w, h = grid_data['width'], grid_data['height']
                raw_grid = [[UUID(cell) if cell else None for cell in row] for row in grid_data['grid']]
                grids_data[region_id_str] = WorldGrid(width=w, height=h, grid=raw_grid)
            game_dict['session']['region_grids'] = grids_data
            self.game_state = CompleteGameState.model_validate(game_dict)
            return True
        except Exception as e:
            llm_logger.error(f"Load error: {e}")
            return False
