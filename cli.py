# AI Text Adventure Engine - Enhanced Version
# Added: Services/shops, quest completion tracking, environmental descriptions, fetch quests

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Type, Tuple
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ValidationError
import json
import asyncio
import httpx
from pathlib import Path
import random
import logging
import sys

# Import settings from config.py
from config import settings

# ============================================================================
# Logging Configuration
# ============================================================================
llm_logger = logging.getLogger("llm_responses")
llm_logger.setLevel(logging.INFO)
llm_logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if settings.log_file:
    file_handler = logging.FileHandler(settings.log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    llm_logger.addHandler(file_handler)
else:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    llm_logger.addHandler(stream_handler)
logging.basicConfig(level=settings.log_level.upper(), format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# Base Models and Enums
# ============================================================================
class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"
    UP = "up"
    DOWN = "down"
    IN = "in"
    OUT = "out"

class Coordinates(BaseModel):
    x: int = 0
    y: int = 0
    z: int = 0

class LocationConnection(BaseModel):
    target_location_id: UUID
    direction: Direction
    description: str
    is_visible: bool = True
    is_passable: bool = True
    requirements: List[str] = []
    travel_time: int = 1

# ============================================================================
# Location Models
# ============================================================================
class LocationType(str, Enum):
    WORLD = "world"
    REGION = "region"
    CITY = "city"
    DISTRICT = "district"
    LOCATION = "location"
    BUILDING = "building"
    ROOM = "room"

class NotableFeature(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    detailed_description: Optional[str] = None
    contained_items: List[UUID] = Field(default_factory=list)

class BaseLocation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    short_description: str
    location_type: LocationType
    coordinates: Coordinates = Field(default_factory=Coordinates)
    parent_id: Optional[UUID] = None
    connections: List[LocationConnection] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    is_discovered: bool = False
    visit_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    last_visited: Optional[datetime] = None
    atmosphere: str = ""
    notable_features: List[NotableFeature] = Field(default_factory=list)
    items: List[UUID] = Field(default_factory=list)
    ambient_sounds: List[str] = Field(default_factory=list)
    ambient_smells: List[str] = Field(default_factory=list)
    light_level: int = 100
    temperature: str = "moderate"
    weather: Optional[str] = None

class World(BaseLocation):
    location_type: LocationType = LocationType.WORLD
    theme: str = "medieval_fantasy"
    lore_summary: str = ""
    key_npcs: List[UUID] = Field(default_factory=list)
    main_questlines: List[UUID] = Field(default_factory=list)
    world_events: List[UUID] = Field(default_factory=list)
    time_period: str = "medieval"
    magic_system: Optional[str] = None
    technology_level: str = "medieval"

class GeneralLocationType(str, Enum):
    FOREST = "forest"
    MEADOW = "meadow"
    MOUNTAIN = "mountain"
    CAVE = "cave"
    RIVER = "river"
    CROSSROADS = "crossroads"
    CLEARING = "clearing"
    RUINS = "ruins"
    GRAVEYARD = "graveyard"
    BRIDGE = "bridge"
    
class GeneralLocation(BaseLocation):
    location_type: LocationType = LocationType.LOCATION
    general_type: GeneralLocationType = GeneralLocationType.CLEARING
    danger_level: int = 0
    resources: List[str] = Field(default_factory=list)
    hidden_items: List[UUID] = Field(default_factory=list)
    encounters: List[UUID] = Field(default_factory=list)

# ============================================================================
# Character Models
# ============================================================================
class CharacterStats(BaseModel):
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10
    health: int = 100
    max_health: int = 100
    mana: int = 50
    max_mana: int = 50
    stamina: int = 100
    max_stamina: int = 100
    armor_class: int = 10
    attack_bonus: int = 0
    damage_bonus: int = 0

class CharacterClass(str, Enum):
    WARRIOR = "warrior"
    MAGE = "mage"
    ROGUE = "rogue"
    CLERIC = "cleric"
    RANGER = "ranger"
    BARD = "bard"
    COMMONER = "commoner"

class CharacterType(str, Enum):
    PLAYER = "player"
    NPC = "npc"
    MONSTER = "monster"
    COMPANION = "companion"

class BaseCharacter(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    character_type: CharacterType
    race: str = "human"
    character_class: CharacterClass = CharacterClass.COMMONER
    level: int = 1
    experience: int = 0
    description: str = ""
    age: int = 25
    height: str = "average"
    build: str = "medium"
    hair_color: str = "brown"
    eye_color: str = "brown"
    distinctive_features: List[str] = Field(default_factory=list)
    stats: CharacterStats = Field(default_factory=CharacterStats)
    inventory: List[UUID] = Field(default_factory=list)
    equipped_items: Dict[str, UUID] = Field(default_factory=dict)
    currency: Dict[str, int] = Field(default_factory=lambda: {"gold": 0, "silver": 0, "copper": 0})
    current_location_id: UUID
    previous_location_id: Optional[UUID] = None
    personality_traits: List[str] = Field(default_factory=list)
    alignment: str = "neutral"
    motivation: str = ""
    fears: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    relationships: Dict[UUID, int] = Field(default_factory=dict)
    faction_standings: Dict[str, int] = Field(default_factory=dict)
    conditions: List[str] = Field(default_factory=list)
    temporary_effects: List[Dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: Optional[datetime] = None

class PlayerCharacter(BaseCharacter):
    character_type: CharacterType = CharacterType.PLAYER
    player_id: UUID
    play_time: int = 0
    deaths: int = 0
    quests_completed: int = 0
    locations_discovered: int = 0
    completed_quests: List[UUID] = Field(default_factory=list)
    active_quests: List[UUID] = Field(default_factory=list)
    failed_quests: List[UUID] = Field(default_factory=list)
    discovered_locations: Set[UUID] = Field(default_factory=set)
    known_npcs: Set[UUID] = Field(default_factory=set)
    major_decisions: List[Dict] = Field(default_factory=list)
    reputation_modifiers: Dict[str, int] = Field(default_factory=dict)

class NPCRole(str, Enum):
    SHOPKEEPER = "shopkeeper"
    INNKEEPER = "innkeeper"
    GUARD = "guard"
    NOBLE = "noble"
    COMMONER = "commoner"
    QUEST_GIVER = "quest_giver"
    COMPANION = "companion"
    ANTAGONIST = "antagonist"
    MERCHANT = "merchant"
    CRAFTSMAN = "craftsman"

class ServiceType(str, Enum):
    BUY_SELL = "buy_sell"
    REPAIR = "repair"
    ENCHANT = "enchant"
    HEAL = "heal"
    REST = "rest"
    TRAIN = "train"
    INFORMATION = "information"

class Service(BaseModel):
    service_type: ServiceType
    name: str
    description: str
    cost: Dict[str, int] = Field(default_factory=dict)  # {"gold": 50}
    requirements: List[str] = Field(default_factory=list)

class NPC(BaseCharacter):
    character_type: CharacterType = CharacterType.NPC
    role: NPCRole = NPCRole.COMMONER
    behavior_patterns: List[str] = Field(default_factory=list)
    dialogue_style: str = "formal"
    speech_patterns: List[str] = Field(default_factory=list)
    home_location_id: UUID
    work_location_id: Optional[UUID] = None
    movement_pattern: str = "stationary"
    movement_schedule: Dict[str, UUID] = Field(default_factory=dict)
    shop_inventory: List[UUID] = Field(default_factory=list)
    services_offered: List[Service] = Field(default_factory=list)
    prices_modifier: float = 1.0
    available_quests: List[UUID] = Field(default_factory=list)
    given_quests: List[UUID] = Field(default_factory=list)
    quest_cooldowns: Dict[UUID, datetime] = Field(default_factory=dict)
    known_information: List[str] = Field(default_factory=list)
    rumors: List[str] = Field(default_factory=list)
    secrets: List[str] = Field(default_factory=list)

# ============================================================================
# Item and Quest Models
# ============================================================================
class ItemType(str, Enum):
    WEAPON = "weapon"
    ARMOR = "armor"
    CONSUMABLE = "consumable"
    TOOL = "tool"
    TREASURE = "treasure"
    QUEST_ITEM = "quest_item"
    BOOK = "book"
    KEY = "key"
    CONTAINER = "container"
    MATERIAL = "material"

class ItemRarity(str, Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"

class Item(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    item_type: ItemType
    rarity: ItemRarity = ItemRarity.COMMON
    weight: float = 0.0
    value: int = 0
    durability: Optional[int] = None
    max_durability: Optional[int] = None
    stackable: bool = False
    stack_size: int = 1
    equipment_slot: Optional[str] = None
    stat_modifiers: Dict[str, int] = Field(default_factory=dict)
    magical: bool = False
    cursed: bool = False
    enchantments: List[str] = Field(default_factory=list)
    special_abilities: List[str] = Field(default_factory=list)
    consumable: bool = False
    use_effects: List[str] = Field(default_factory=list)
    charges: Optional[int] = None
    max_charges: Optional[int] = None
    lore_text: str = ""
    creator: Optional[str] = None
    age: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    self_use_effect_description: Optional[str] = None
    interactions: Dict[UUID, str] = Field(default_factory=dict)
    contained_items: List[UUID] = Field(default_factory=list)
    current_stack_size: int = 1

class QuestType(str, Enum):
    MAIN_STORY = "main_story"
    SIDE_QUEST = "side_quest"
    FETCH = "fetch"
    KILL = "kill"
    ESCORT = "escort"
    DELIVERY = "delivery"
    EXPLORATION = "exploration"
    PUZZLE = "puzzle"

class QuestStatus(str, Enum):
    AVAILABLE = "available"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TURNED_IN = "turned_in"

class QuestObjective(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    description: str
    objective_type: str
    target: str
    current_progress: int = 0
    required_progress: int = 1
    completed: bool = False
    optional: bool = False

class QuestReward(BaseModel):
    experience: int = 0
    currency: Dict[str, int] = Field(default_factory=dict)
    items: List[UUID] = Field(default_factory=list)
    reputation_changes: Dict[str, int] = Field(default_factory=dict)

class Quest(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    quest_type: QuestType
    giver_id: Optional[UUID] = None
    start_location_id: Optional[UUID] = None
    level_requirement: int = 1
    prerequisite_quests: List[UUID] = Field(default_factory=list)
    required_items: List[UUID] = Field(default_factory=list)
    required_skills: Dict[str, int] = Field(default_factory=dict)
    objectives: List[QuestObjective] = Field(default_factory=list)
    status: QuestStatus = QuestStatus.AVAILABLE
    rewards: QuestReward = Field(default_factory=QuestReward)
    auto_complete: bool = False
    turn_in_location_id: Optional[UUID] = None
    turn_in_npc_id: Optional[UUID] = None
    time_limit: Optional[int] = None
    repeatable: bool = False
    cooldown_minutes: int = 0
    story_impact: List[str] = Field(default_factory=list)
    branching_choices: List[Dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # New fields for fetch quests
    target_item_id: Optional[UUID] = None  # The item to fetch
    location_hint: Optional[str] = None    # Vague hint about where to find it

# ============================================================================
# Game Time, Grid, and State Models
# ============================================================================
class TimeOfDay(str, Enum):
    DAWN = "dawn"
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    MIDNIGHT = "midnight"

class GameTime(BaseModel):
    day: int = 1
    hour: int = 8
    minute: int = 0
    time_of_day: TimeOfDay = TimeOfDay.MORNING
    season: str = "spring"
    year: int = 1
    
    def advance_time(self, minutes: int) -> None:
        self.minute += minutes
        while self.minute >= 60:
            self.hour += 1; self.minute -= 60
        while self.hour >= 24:
            self.day += 1; self.hour -= 24
        
        if 5 <= self.hour < 7: self.time_of_day = TimeOfDay.DAWN
        elif 7 <= self.hour < 12: self.time_of_day = TimeOfDay.MORNING
        elif 12 <= self.hour < 14: self.time_of_day = TimeOfDay.MIDDAY
        elif 14 <= self.hour < 18: self.time_of_day = TimeOfDay.AFTERNOON
        elif 18 <= self.hour < 21: self.time_of_day = TimeOfDay.EVENING
        elif 21 <= self.hour < 24: self.time_of_day = TimeOfDay.NIGHT
        else: self.time_of_day = TimeOfDay.MIDNIGHT

class WorldGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid: List[List[Optional[UUID]]] = [[None for _ in range(width)] for _ in range(height)]

    def get_location_id(self, x: int, y: int) -> Optional[UUID]:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def set_location_id(self, x: int, y: int, location_id: UUID):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = location_id

class GameSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    session_name: str
    player_id: UUID
    world: World
    game_time: GameTime = Field(default_factory=GameTime)
    player_character: Optional[PlayerCharacter] = None
    world_grid: WorldGrid = Field(default_factory=lambda: WorldGrid(10, 10))
    created_at: datetime = Field(default_factory=datetime.now)
    last_played: datetime = Field(default_factory=datetime.now)
    class Config: arbitrary_types_allowed = True

class CompleteGameState(BaseModel):
    session: GameSession
    locations: Dict[UUID, BaseLocation] = Field(default_factory=dict)
    characters: Dict[UUID, BaseCharacter] = Field(default_factory=dict)
    items: Dict[UUID, Item] = Field(default_factory=dict)
    quests: Dict[UUID, Quest] = Field(default_factory=dict)
    @property
    def character_locations(self) -> Dict[UUID, UUID]:
        return {char_id: char.current_location_id for char_id, char in self.characters.items()}
    class Config: arbitrary_types_allowed = True

# ============================================================================
# Core Game Engine
# ============================================================================
class GameEngine:
    """Core game engine that manages game state and interactions"""
    def __init__(self):
        self.ollama_url = settings.ollama_url
        self.ollama_model = settings.ollama_model
        self.ollama_timeout = settings.ollama_timeout
        self.game_state: Optional[CompleteGameState] = None
        self.client = httpx.AsyncClient()
        self.in_combat = False
        self.combat_opponents: List[BaseCharacter] = []

    def calculate_equipment_bonuses(self, character: BaseCharacter) -> CharacterStats:
        """Calculate total bonuses from equipped items"""
        bonuses = CharacterStats()  # Start with zero bonuses
        
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
        """Apply equipment bonuses to character stats"""
        if not isinstance(character, (PlayerCharacter, NPC)):
            return
            
        # Get base stats (you might want to store these separately)
        base_stats = character.stats
        bonuses = self.calculate_equipment_bonuses(character)
        
        # Apply bonuses to current stats
        character.stats.strength = max(1, base_stats.strength + bonuses.strength)
        character.stats.dexterity = max(1, base_stats.dexterity + bonuses.dexterity)
        character.stats.constitution = max(1, base_stats.constitution + bonuses.constitution)
        character.stats.intelligence = max(1, base_stats.intelligence + bonuses.intelligence)
        character.stats.wisdom = max(1, base_stats.wisdom + bonuses.wisdom)
        character.stats.charisma = max(1, base_stats.charisma + bonuses.charisma)
        character.stats.armor_class = max(1, base_stats.armor_class + bonuses.armor_class)
        character.stats.attack_bonus = base_stats.attack_bonus + bonuses.attack_bonus
        character.stats.damage_bonus = base_stats.damage_bonus + bonuses.damage_bonus
        
        # Update max health/mana based on constitution/intelligence bonuses
        character.stats.max_health = max(1, base_stats.max_health + (bonuses.constitution * 5))
        character.stats.max_mana = max(1, base_stats.max_mana + (bonuses.intelligence * 3))
    
    def add_item_to_inventory(self, character: BaseCharacter, item: Item, quantity: int = 1) -> str:
        """Add an item to character's inventory, handling stacking"""
        if item.stackable:
            # Look for existing stack of the same item
            for existing_item_id in character.inventory:
                existing_item = self.game_state.items.get(existing_item_id)
                if (existing_item and 
                    existing_item.name == item.name and 
                    existing_item.current_stack_size < existing_item.stack_size):
                    
                    # Add to existing stack
                    space_available = existing_item.stack_size - existing_item.current_stack_size
                    items_to_add = min(quantity, space_available)
                    existing_item.current_stack_size += items_to_add
                    quantity -= items_to_add
                    
                    if quantity == 0:
                        return f"Added {items_to_add} {item.name}(s) to existing stack."
            
            # If we still have items to add, create new stacks
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
            # Non-stackable items
            for _ in range(quantity):
                new_item = Item.model_validate(item.model_dump())
                new_item.id = uuid4()
                self.game_state.items[new_item.id] = new_item
                character.inventory.append(new_item.id)
            
            return f"Added {quantity} {item.name}(s) to inventory."
    
    def remove_item_from_inventory(self, character: BaseCharacter, item_name: str, quantity: int = 1) -> tuple[bool, str]:
        """Remove items from inventory, handling stacks"""
        items_to_remove = []
        remaining_quantity = quantity
        
        # Find items to remove
        for item_id in character.inventory:
            if remaining_quantity <= 0:
                break
                
            item = self.game_state.items.get(item_id)
            if item and item_name.lower() in item.name.lower():
                if item.stackable and item.current_stack_size > 1:
                    # Remove from stack
                    items_from_stack = min(remaining_quantity, item.current_stack_size)
                    item.current_stack_size -= items_from_stack
                    remaining_quantity -= items_from_stack
                    
                    if item.current_stack_size <= 0:
                        items_to_remove.append(item_id)
                else:
                    # Remove entire item
                    items_to_remove.append(item_id)
                    remaining_quantity -= 1
        
        # Actually remove the items
        for item_id in items_to_remove:
            character.inventory.remove(item_id)
            if item_id in self.game_state.items:
                del self.game_state.items[item_id]
        
        removed_count = quantity - remaining_quantity
        if removed_count > 0:
            return True, f"Removed {removed_count} {item_name}(s) from inventory."
        else:
            return False, f"You don't have enough {item_name}(s) to remove."
    
    async def get_available_models(self) -> List[str]:
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return sorted([model['name'] for model in data.get('models', [])])
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError): return []

    async def _generate_and_validate(self, prompt: str, model_name: str) -> Any:
        max_retries = 3
        last_error = None
        raw_response = ""
        for attempt in range(max_retries):
            try:
                raw_response = await self.generate_ai_response(prompt, is_content_generation=True, model_name=model_name)
                llm_logger.info(f"Received raw response for prompt '{prompt[:60]}...'. Attempt {attempt + 1}:\n---\n{raw_response}\n---")
                if raw_response.strip().startswith("```json"):
                    raw_response = raw_response.strip()[7:-3].strip()
                return json.loads(raw_response)
            except (httpx.RequestError, json.JSONDecodeError, ValidationError) as e:
                last_error = e
                llm_logger.error(f"Generation attempt {attempt + 1}/{max_retries} failed for prompt '{prompt[:60]}...'. Error: {e}\n--- FAILED RAW RESPONSE ---\n{raw_response}\n---")
                print(f"   - Generation attempt {attempt + 1}/{max_retries} failed. Retrying...")
        raise ConnectionError("Failed to generate content after multiple retries.") from last_error

    def _create_services_for_npc(self, npc: NPC) -> None:
        """Create appropriate services based on NPC role"""
        if npc.role == NPCRole.SHOPKEEPER:
            npc.services_offered.append(Service(
                service_type=ServiceType.BUY_SELL,
                name="General Goods",
                description="I buy and sell various items.",
                cost={"gold": 0}  # Cost varies by item
            ))
        elif npc.role == NPCRole.MERCHANT:
            npc.services_offered.append(Service(
                service_type=ServiceType.BUY_SELL,
                name="Trade Goods",
                description="I deal in fine wares and exotic items.",
                cost={"gold": 0}
            ))
        elif npc.role == NPCRole.INNKEEPER:
            npc.services_offered.extend([
                Service(
                    service_type=ServiceType.REST,
                    name="Room for the Night",
                    description="A warm bed and a hot meal.",
                    cost={"gold": 2, "silver": 5}
                ),
                Service(
                    service_type=ServiceType.HEAL,
                    name="Herbal Remedies",
                    description="Basic healing herbs and tonics.",
                    cost={"gold": 1}
                )
            ])
        elif npc.role == NPCRole.CRAFTSMAN:
            npc.services_offered.append(Service(
                service_type=ServiceType.REPAIR,
                name="Item Repair",
                description="I can mend your broken equipment.",
                cost={"gold": 5}
            ))

    async def create_new_game(self, player_name: str, player_id: UUID, session_name: str, model_name: str) -> CompleteGameState:
        try:
            print("\n1/3: Conceptualizing a new world...")
            world_prompt = "Generate the high-level details for a new fantasy world. Respond with a single JSON object with keys: name, description, short_description, theme, and lore_summary."
            world_data = await self._generate_and_validate(world_prompt, model_name)
            world = World.model_validate(world_data)
            print(f"   ✓ World created: {world.name}")

            print("2/3: Generating the starting region cell by cell...")
            session = GameSession(session_name=session_name, player_id=player_id, world=world)
            game_state = CompleteGameState(session=session)
            
            start_x, start_y = 1, 1
            start_location_id = None
            allowed_loc_types = [t.value for t in GeneralLocationType]

            for y in range(3):
                for x in range(3):
                    print(f"   - Generating location at ({x}, {y})...", end="", flush=True)
                    is_start_location = (x == start_x and y == start_y)
                    
                    base_prompt = f"Generate a single location within the world of {world.name} at coordinates ({x},{y}). Respond with a single JSON object with keys: name, description, short_description, general_type, atmosphere, notable_features (list of strings), ambient_sounds (list of strings), and ambient_smells (list of strings). The 'general_type' MUST be one of the following values: {allowed_loc_types}."
                    
                    if is_start_location:
                        prompt = f"""Generate the starting location for an adventure in the world of {world.name} at coordinates ({x},{y}). It should be a relatively safe place like a quiet crossroads. Respond with a single JSON object with keys: name, description, short_description, general_type, atmosphere, notable_features, ambient_sounds, and ambient_smells. The 'general_type' MUST be one of the following values: {allowed_loc_types}. Also, include a nested JSON object with the key 'npc'. The NPC should be a 'quest_giver' and needs a name, description, race, and personality_traits."""
                    elif random.random() < 0.4:
                        prompt = base_prompt + " Also, include a key called 'puzzle', a JSON object with three keys: 'tool' (an item object like a key), 'container' (an item object like a chest), and 'reward' (an item object like a potion). Also provide a top-level key 'interaction_description' with a string describing the result of using the tool on the container."
                    else:
                        prompt = base_prompt
                    
                    loc_data = await self._generate_and_validate(prompt, model_name)
                    
                    feature_names = loc_data.get("notable_features", [])
                    loc_data["notable_features"] = [{"name": name} for name in feature_names]
                    
                    npc_info = loc_data.pop('npc', None)
                    puzzle_data = loc_data.pop('puzzle', None)
                    interaction_desc = loc_data.pop('interaction_description', None)

                    loc = GeneralLocation.model_validate(loc_data)
                    loc.coordinates = Coordinates(x=x, y=y)
                    
                    if puzzle_data and interaction_desc:
                        tool = Item.model_validate(puzzle_data['tool'])
                        container = Item.model_validate(puzzle_data['container'])
                        reward = Item.model_validate(puzzle_data['reward'])
                        
                        tool.interactions[container.id] = interaction_desc
                        container.contained_items.append(reward.id)
                        
                        game_state.items[tool.id] = tool
                        game_state.items[container.id] = container
                        game_state.items[reward.id] = reward
                        
                        loc.items.extend([tool.id, container.id])
                        print(" (puzzle added)", end="")

                    game_state.locations[loc.id] = loc
                    game_state.session.world_grid.set_location_id(x, y, loc.id)
                    
                    if npc_info:
                        npc = NPC.model_validate(npc_info)
                        npc.role = NPCRole.QUEST_GIVER
                        npc.current_location_id = loc.id
                        npc.home_location_id = loc.id
                        self._create_services_for_npc(npc)
                        game_state.characters[npc.id] = npc
                    
                    if is_start_location:
                        start_location_id = loc.id
                    
                    print(" ✓")

            print("3/3: Packing the adventurer's bag...")
            item_prompt = "Generate a simple starting container item for a new adventurer. Respond with a single JSON object with keys: name, description, item_type ('container'), value (integer), and weight (float)."
            item_data = await self._generate_and_validate(item_prompt, model_name)
            start_item = Item.model_validate(item_data)
            game_state.items[start_item.id] = start_item
            print(f"   ✓ Item crafted: {start_item.name}")
            
            print("\nBringing it all together...")
            print("   Creating connections between locations...")
            all_locations = list(game_state.locations.values())
            total_connections = 0
            for loc in all_locations:
                if loc.location_type != LocationType.LOCATION: continue
                x, y = loc.coordinates.x, loc.coordinates.y
                potential_neighbors = { Direction.NORTH: (x, y - 1), Direction.SOUTH: (x, y + 1), Direction.EAST: (x + 1, y), Direction.WEST: (x - 1, y) }
                for direction, (nx, ny) in potential_neighbors.items():
                    neighbor_id = game_state.session.world_grid.get_location_id(nx, ny)
                    if neighbor_id:
                        connection = LocationConnection(target_location_id=neighbor_id, direction=direction, description=f"A path leads {direction.value}.")
                        loc.connections.append(connection)
                        total_connections += 1
            print(f"   ✓ {total_connections} connections established.")
            
            player_char = PlayerCharacter(name=player_name, player_id=player_id, current_location_id=start_location_id, description=f"{player_name}, a new adventurer ready to explore {world.name}.", currency={"gold": 50, "silver": 25}, inventory=[start_item.id])
            game_state.session.player_character = player_char
            game_state.characters[player_char.id] = player_char
            
            self.game_state = game_state
            return self.game_state

        except Exception as e:
            print(f"\n❌ A mystical rift prevented world generation.")
            print(f"   Final error: {e}")
            logging.exception("World generation failed")
            raise ConnectionError("Failed to generate a world using the LLM.") from e

    def get_current_location(self) -> BaseLocation:
        if not self.game_state or not self.game_state.session.player_character: raise ValueError("Game state or player character not initialized")
        player_loc_id = self.game_state.session.player_character.current_location_id
        return self.game_state.locations[player_loc_id]

    def move_player(self, direction: Direction) -> tuple[bool, str]:
        if not self.game_state: return False, "No active game session"
        current_loc = self.get_current_location()
        target_connection = next((conn for conn in current_loc.connections if conn.direction == direction), None)
        if not target_connection: return False, f"You can't go {direction.value} from here."
        if not target_connection.is_visible: return False, "You see no path in that direction."
        if not target_connection.is_passable: return False, "The way is blocked."
        target_location = self.game_state.locations.get(target_connection.target_location_id)
        if not target_location:
            logging.error(f"Dangling connection from {current_loc.id} to {target_connection.target_location_id}")
            return False, "The path leads to a void. An error has occurred."
        pc = self.game_state.session.player_character
        pc.previous_location_id = pc.current_location_id
        pc.current_location_id = target_location.id
        target_location.visit_count += 1
        target_location.last_visited = datetime.now()
        self.game_state.session.game_time.advance_time(target_connection.travel_time)
        
        # Check for quest completion when moving
        self.check_quest_progress()
        
        return True, f"You travel {direction.value}..."

    def check_quest_progress(self) -> List[str]:
        """Check and update quest progress, return list of completion messages"""
        if not self.game_state:
            return []
        
        player = self.game_state.session.player_character
        completion_messages = []
        
        for quest_id in list(player.active_quests):
            quest = self.game_state.quests.get(quest_id)
            if not quest or quest.status != QuestStatus.ACTIVE:
                continue
                
            # Check fetch quest completion
            if quest.quest_type == QuestType.FETCH and quest.target_item_id:
                if quest.target_item_id in player.inventory:
                    # Mark objective as complete
                    for objective in quest.objectives:
                        if not objective.completed and objective.objective_type == "fetch":
                            objective.completed = True
                            objective.current_progress = objective.required_progress
                    
                    # Check if all objectives are complete
                    if all(obj.completed for obj in quest.objectives):
                        quest.status = QuestStatus.COMPLETED
                        completion_messages.append(f"Quest objective completed: {quest.objectives[0].description}")
        
        return completion_messages

    def complete_quest(self, quest_id: UUID) -> str:
        """Complete a quest and give rewards"""
        if not self.game_state:
            return "No active game session."
            
        quest = self.game_state.quests.get(quest_id)
        if not quest:
            return "Quest not found."
            
        if quest.status != QuestStatus.COMPLETED:
            return "Quest is not ready to be turned in."
            
        player = self.game_state.session.player_character
        
        # Remove from active quests, add to completed
        if quest_id in player.active_quests:
            player.active_quests.remove(quest_id)
        player.completed_quests.append(quest_id)
        player.quests_completed += 1
        
        # Give rewards
        quest.status = QuestStatus.TURNED_IN
        quest.completed_at = datetime.now()
        
        reward_messages = []
        
        # Experience
        if quest.rewards.experience > 0:
            player.experience += quest.rewards.experience
            reward_messages.append(f"{quest.rewards.experience} experience")
            
        # Currency
        for currency, amount in quest.rewards.currency.items():
            player.currency[currency] = player.currency.get(currency, 0) + amount
            reward_messages.append(f"{amount} {currency}")
            
        # Items
        for item_id in quest.rewards.items:
            if item_id in self.game_state.items:
                player.inventory.append(item_id)
                item = self.game_state.items[item_id]
                reward_messages.append(f"{item.name}")
        
        # Remove quest item from inventory if it was a fetch quest
        if quest.quest_type == QuestType.FETCH and quest.target_item_id:
            if quest.target_item_id in player.inventory:
                player.inventory.remove(quest.target_item_id)
        
        if reward_messages:
            return f"Quest '{quest.name}' completed! You receive: {', '.join(reward_messages)}."
        else:
            return f"Quest '{quest.name}' completed!"

    async def generate_ai_response(self, prompt: str, context: str = "", is_content_generation: bool = False, model_name: Optional[str] = None) -> str:
        try:
            active_model = model_name or self.ollama_model
            if is_content_generation:
                json_payload = {"model": active_model, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0.9}}
            else:
                full_prompt = f"{context}\n\n{prompt}"
                json_payload = {"model": active_model, "prompt": full_prompt, "stream": False, "options": {"temperature": 0.8, "max_tokens": 300}}
            response = await self.client.post(f"{self.ollama_url}/api/generate", json=json_payload, timeout=self.ollama_timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "The AI is silent.")
        except httpx.RequestError as e: return f"The magical energies are disrupted by a network storm: {str(e)}"
        except httpx.HTTPStatusError as e: return f"The mystical forces are responding with an error: {e.response.status_code} - {e.response.text}"
        except Exception as e: return f"The magical energies are disrupted: {str(e)}"
    
    def build_context_for_ai(self) -> str:
        if not self.game_state: return ""
        loc = self.get_current_location()
        p = self.game_state.session.player_character
        npcs = [char for char in self.game_state.characters.values() if char.current_location_id == loc.id and char.id != p.id]
        
        # Enhanced context with environmental details
        context_parts = [
            f"WORLD: {self.game_state.session.world.name}",
            f"TIME: {self.game_state.session.game_time.time_of_day.value}",
            f"LOCATION: {loc.name} - {loc.description}",
        ]
        
        # Add environmental details
        if loc.atmosphere:
            context_parts.append(f"ATMOSPHERE: {loc.atmosphere}")
        if loc.ambient_sounds:
            context_parts.append(f"SOUNDS: {', '.join(loc.ambient_sounds)}")
        if loc.ambient_smells:
            context_parts.append(f"SMELLS: {', '.join(loc.ambient_smells)}")
        if loc.weather:
            context_parts.append(f"WEATHER: {loc.weather}")
        
        context_parts.extend([
            f"NPCS HERE: {', '.join([n.name for n in npcs]) if npcs else 'None'}",
            f"PLAYER: {p.name}"
        ])
        
        return "\n".join(context_parts)

    async def generate_rumor(self, npc: NPC, model_name: str) -> str:
        context = self.build_context_for_ai()
        prompt = f"The player is asking {npc.name} ({npc.description}) for a rumor. Based on the context, generate a short, intriguing rumor this NPC might know. Respond with just the rumor itself, as a single sentence."
        rumor = await self.generate_ai_response(prompt, context, model_name=model_name)
        npc.rumors.append(rumor)
        return rumor

    async def generate_fetch_quest(self, npc: NPC, model_name: str) -> Optional[Quest]:
        """Generate a fetch quest with item placement and location hint"""
        context = self.build_context_for_ai()
        
        # Generate quest details
        quest_prompt = f"""The player is asking {npc.name} ({npc.description}) for a quest. Generate a fetch quest where the player must find and return a specific item. Respond with a single JSON object with keys:
        - name: quest name
        - description: quest description explaining what item is needed and why
        - item_name: name of the item to fetch
        - item_description: description of the item
        - location_hint: a vague hint about where the item might be found (like "I remember losing it in the forest" or "It was last seen near some old ruins")"""
        
        try:
            quest_data = await self._generate_and_validate(quest_prompt, model_name)
            
            # Create the quest item
            quest_item = Item(
                name=quest_data['item_name'],
                description=quest_data['item_description'],
                item_type=ItemType.QUEST_ITEM,
                value=random.randint(10, 50),
                weight=random.uniform(0.1, 2.0)
            )
            
            # Place the item in a random location (not the current one)
            available_locations = [
                loc for loc in self.game_state.locations.values() 
                if loc.id != npc.current_location_id and loc.location_type == LocationType.LOCATION
            ]
            
            if not available_locations:
                return None
                
            target_location = random.choice(available_locations)
            
            # Decide whether to place item in plain sight or hidden in a feature
            if target_location.notable_features and random.random() < 0.6:
                # Hide in a random notable feature
                feature = random.choice(target_location.notable_features)
                feature.contained_items.append(quest_item.id)
            else:
                # Place in plain sight
                target_location.items.append(quest_item.id)
            
            # Store the item in game state
            self.game_state.items[quest_item.id] = quest_item
            
            # Create the quest
            objective = QuestObjective(
                description=f"Find and return the {quest_item.name} to {npc.name}",
                objective_type="fetch",
                target=str(quest_item.id)
            )
            
            quest = Quest(
                name=quest_data['name'],
                description=quest_data['description'],
                quest_type=QuestType.FETCH,
                giver_id=npc.id,
                objectives=[objective],
                status=QuestStatus.ACTIVE,
                target_item_id=quest_item.id,
                location_hint=quest_data['location_hint'],
                rewards=QuestReward(
                    experience=random.randint(50, 150),
                    currency={"gold": random.randint(5, 25)}
                )
            )
            
            # Add quest to game state and player
            self.game_state.quests[quest.id] = quest
            npc.available_quests.append(quest.id)
            npc.given_quests.append(quest.id)
            self.game_state.session.player_character.active_quests.append(quest.id)
            
            return quest
            
        except Exception as e:
            print(f"Error generating fetch quest: {e}")
            return None

    async def generate_quest(self, npc: NPC, model_name: str) -> Optional[Quest]:
        """Generate a quest - currently only fetch quests are implemented"""
        return await self.generate_fetch_quest(npc, model_name)

    def handle_service_request(self, npc: NPC, service_name: str) -> str:
        """Handle a service request from an NPC"""
        if not npc.services_offered:
            return f"{npc.name} doesn't offer any services."
            
        # Find the requested service
        service = next((s for s in npc.services_offered if service_name.lower() in s.name.lower()), None)
        if not service:
            available_services = [s.name for s in npc.services_offered]
            return f"{npc.name} doesn't offer '{service_name}'. Available services: {', '.join(available_services)}"
        
        player = self.game_state.session.player_character
        
        if service.service_type == ServiceType.BUY_SELL:
            return f"{npc.name}: \"{service.description} Use 'buy <item>' or 'sell <item>' to trade with me.\""
        
        elif service.service_type == ServiceType.REST:
            # Check if player can afford it
            total_cost = sum(service.cost.values())  # Simple cost calculation
            player_gold = player.currency.get("gold", 0)
            
            if player_gold < total_cost:
                return f"{npc.name}: \"You need {total_cost} gold for that service.\""
                
            # Deduct cost and provide service
            player.currency["gold"] -= total_cost
            player.stats.health = player.stats.max_health
            player.stats.mana = player.stats.max_mana
            player.stats.stamina = player.stats.max_stamina
            self.game_state.session.game_time.advance_time(480)  # 8 hours
            
            return f"{npc.name}: \"Sleep well, traveler. You feel fully rested.\" (Health, mana, and stamina restored)"
        
        elif service.service_type == ServiceType.HEAL:
            player_gold = player.currency.get("gold", 0)
            cost = service.cost.get("gold", 1)
            
            if player_gold < cost:
                return f"{npc.name}: \"You need {cost} gold for healing.\""
                
            if player.stats.health >= player.stats.max_health:
                return f"{npc.name}: \"You look healthy already!\""
                
            player.currency["gold"] -= cost
            heal_amount = min(30, player.stats.max_health - player.stats.health)
            player.stats.health += heal_amount
            
            return f"{npc.name}: \"There you go.\" (Healed {heal_amount} health)"
        
        elif service.service_type == ServiceType.REPAIR:
            return f"{npc.name}: \"{service.description} Use 'repair <item>' to fix your equipment.\""
        
        else:
            return f"{npc.name}: \"{service.description}\""

    def buy_item_from_npc(self, npc: NPC, item_name: str) -> str:
        """Handle buying an item from an NPC"""
        if not npc.shop_inventory:
            return f"{npc.name} has nothing for sale."
            
        # Find item in NPC's inventory
        item = None
        for item_id in npc.shop_inventory:
            shop_item = self.game_state.items.get(item_id)
            if shop_item and item_name.lower() in shop_item.name.lower():
                item = shop_item
                break
                
        if not item:
            available_items = []
            for item_id in npc.shop_inventory:
                shop_item = self.game_state.items.get(item_id)
                if shop_item:
                    available_items.append(f"{shop_item.name} ({shop_item.value}g)")
            
            if available_items:
                return f"{npc.name} doesn't have '{item_name}'. Available: {', '.join(available_items)}"
            else:
                return f"{npc.name} has nothing for sale right now."
        
        player = self.game_state.session.player_character
        modified_price = int(item.value * npc.prices_modifier)
        
        if player.currency.get("gold", 0) < modified_price:
            return f"You need {modified_price} gold to buy the {item.name}."
            
        # Complete the transaction
        player.currency["gold"] -= modified_price
        player.inventory.append(item.id)
        npc.shop_inventory.remove(item.id)
        
        return f"You buy the {item.name} for {modified_price} gold."

    def sell_item_to_npc(self, npc: NPC, item_name: str) -> str:
        """Handle selling an item to an NPC"""
        player = self.game_state.session.player_character
        
        # Find item in player inventory
        item = self.find_item_in_inventory(item_name)
        if not item:
            return f"You don't have a '{item_name}' to sell."
            
        if item.item_type == ItemType.QUEST_ITEM:
            return f"You can't sell quest items."
            
        # Calculate sell price (typically 50% of value)
        sell_price = max(1, int(item.value * 0.5 * npc.prices_modifier))
        
        # Complete the transaction
        player.inventory.remove(item.id)
        player.currency["gold"] = player.currency.get("gold", 0) + sell_price
        npc.shop_inventory.append(item.id)
        
        return f"You sell the {item.name} for {sell_price} gold."

    def find_item_in_inventory(self, item_name: str) -> Optional[Item]:
        if not self.game_state: return None
        for item_id in self.game_state.session.player_character.inventory:
            item = self.game_state.items.get(item_id)
            if item and item_name.lower() in item.name.lower():
                return item
        return None
    
    def find_item_in_location(self, item_name: str) -> Optional[Item]:
        if not self.game_state: return None
        loc = self.get_current_location()
        for item_id in loc.items:
            item = self.game_state.items.get(item_id)
            if item and item_name.lower() in item.name.lower():
                return item
        return None

    def equip_item(self, item_name: str) -> str:
        item = self.find_item_in_inventory(item_name)
        if not item: 
            return f"You don't have a '{item_name}'."
        if not item.equipment_slot: 
            return f"You can't equip the {item.name}."
        
        player = self.game_state.session.player_character
        message = ""
        
        if item.equipment_slot in player.equipped_items:
            old_item_id = player.equipped_items[item.equipment_slot]
            if old_item := self.game_state.items.get(old_item_id):
                message += f"You unequip the {old_item.name}. "
        
        player.equipped_items[item.equipment_slot] = item.id
        message += f"You equip the {item.name}."
        
        # Apply equipment effects after equipping
        self.apply_equipment_effects(player)
        
        return message

    def unequip_item(self, item_name: str) -> str:
        item = self.find_item_in_inventory(item_name)
        if not item or item.id not in self.game_state.session.player_character.equipped_items.values(): 
            return f"You don't have a '{item_name}' equipped."
        
        player = self.game_state.session.player_character
        slot_to_remove = next((slot for slot, eq_id in player.equipped_items.items() if eq_id == item.id), None)
        
        if slot_to_remove:
            del player.equipped_items[slot_to_remove]
            
            # Reapply equipment effects after unequipping
            self.apply_equipment_effects(player)
            
            return f"You unequip the {item.name}."
        return f"Error unequipping {item.name}."
    
    async def save_game(self, filepath: Path) -> bool:
        if not self.game_state: 
            return False
        try:
            # Handle the WorldGrid serialization BEFORE model_dump()
            grid_obj = self.game_state.session.world_grid
            
            # Convert UUID objects in the grid to strings for JSON serialization
            serializable_grid = []
            for row in grid_obj.grid:
                serializable_row = []
                for cell in row:
                    if cell is None:
                        serializable_row.append(None)
                    else:
                        serializable_row.append(str(cell))  # Convert UUID to string
                serializable_grid.append(serializable_row)
            
            # Temporarily replace the WorldGrid object with serializable data
            original_grid = self.game_state.session.world_grid
            self.game_state.session.world_grid = {
                'width': grid_obj.width, 
                'height': grid_obj.height, 
                'grid': serializable_grid
            }
            
            # Now convert the game state to a dictionary
            game_dict = self.game_state.model_dump(mode='json')
            
            # Restore the original WorldGrid object
            self.game_state.session.world_grid = original_grid
            
            # Ensure the directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(game_dict, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving game: {e}")
            # Make sure to restore the original grid even if there's an error
            if 'original_grid' in locals():
                self.game_state.session.world_grid = original_grid
            return False
    
    async def load_game(self, filepath: Path) -> bool:
        try:
            if not filepath.exists(): 
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                game_dict = json.load(f)
            
            # Handle WorldGrid deserialization
            grid_data = game_dict['session']['world_grid']
            world_grid_obj = WorldGrid(width=grid_data['width'], height=grid_data['height'])
            
            # Convert string UUIDs back to UUID objects
            for y, row in enumerate(grid_data['grid']):
                for x, cell in enumerate(row):
                    if cell is not None:
                        world_grid_obj.grid[y][x] = UUID(cell)  # Convert string back to UUID
                    else:
                        world_grid_obj.grid[y][x] = None
            
            # Replace the serialized grid data with the WorldGrid object
            game_dict['session']['world_grid'] = world_grid_obj
            
            # Validate and create the game state
            self.game_state = CompleteGameState.model_validate(game_dict)
            return True
            
        except Exception as e:
            print(f"Error loading game: {e}")
            return False

# ============================================================================
# Command Line Interface
# ============================================================================
class GameCLI:
    """Command line interface for the game"""
    def __init__(self):
        self.engine = GameEngine()
        self.running = False
        self.selected_model = ""
    
    def display_header(self):
        print("\n" + "="*60 + "\n    AI-POWERED TEXT ADVENTURE ENGINE\n" + "="*60)
    
    def display_location(self):
        if not self.engine.game_state: return
        location = self.engine.get_current_location()
        print(f"\n🏛 {location.name.upper()} ({location.coordinates.x}, {location.coordinates.y})")
        print(f"{location.description}")
        
        # Enhanced environmental descriptions
        env_details = []
        if location.atmosphere:
            env_details.append(f"The atmosphere is {location.atmosphere}")
        if location.ambient_sounds:
            env_details.append(f"You hear {', '.join(location.ambient_sounds)}")
        if location.ambient_smells:
            env_details.append(f"You smell {', '.join(location.ambient_smells)}")
        if location.weather:
            env_details.append(f"The weather is {location.weather}")
        if location.temperature != "moderate":
            env_details.append(f"It feels {location.temperature}")
            
        if env_details:
            print(" ".join([f"{detail}." for detail in env_details]))
        
        if location.notable_features: 
            print(f"You notice: {', '.join([f.name for f in location.notable_features])}.")
        if location.items:
            item_names = [self.engine.game_state.items[item_id].name for item_id in location.items if item_id in self.engine.game_state.items]
            if item_names:
                print(f"On the ground you see: {', '.join(item_names)}.")
        npcs = [char for char in self.engine.game_state.characters.values() if char.current_location_id == location.id and char.character_type != CharacterType.PLAYER]
        if npcs: 
            print(f"People here: {', '.join([npc.name for npc in npcs])}.")
        visible_exits = sorted([conn.direction.value.title() for conn in location.connections if conn.is_visible])
        if visible_exits:
            print(f"Exits: {', '.join(visible_exits)}")
        else:
            print("There are no obvious exits.")
    
    def display_character_status(self):
        if not self.engine.game_state: return
        player = self.engine.game_state.session.player_character
        stats = player.stats
        print(f"\n💤 {player.name} (Lvl {player.level} {player.character_class.value.title()})")
        print(f"   Health: {stats.health}/{stats.max_health} | Gold: {player.currency.get('gold', 0)} | Silver: {player.currency.get('silver', 0)}")
        
        # Show active quests with more detail
        active_quests = [self.engine.game_state.quests[qid] for qid in player.active_quests if qid in self.engine.game_state.quests]
        if active_quests:
            print("   Active Quests:")
            for quest in active_quests:
                status_icons = {"active": "⏳", "completed": "✅"}
                icon = status_icons.get(quest.status.value, "❓")
                print(f"     {icon} {quest.name}: {quest.objectives[0].description}")
                if quest.location_hint and quest.status == QuestStatus.ACTIVE:
                    print(f"        Hint: \"{quest.location_hint}\"")
        
        # Show completed quest count
        if player.quests_completed > 0:
            print(f"   Quests completed: {player.quests_completed}")

    def display_time(self):
        if not self.engine.game_state:
            print("No active game session.")
            return
        gt = self.engine.game_state.session.game_time
        time_str = f"{gt.hour:02d}:{gt.minute:02d}"
        print(f"\n⏳ It is {gt.time_of_day.value} on day {gt.day} of the {gt.season.title()}. The time is {time_str}.")

    async def display_map(self):
        if not self.engine.game_state:
            print("No active game to map.")
            return

        player = self.engine.game_state.session.player_character
        all_locs = self.engine.game_state.locations
        
        discovered_locs = [loc for loc in all_locs.values() if loc.visit_count > 0]
        if not discovered_locs:
            print("You haven't explored anywhere yet.")
            return

        min_x = min(loc.coordinates.x for loc in discovered_locs)
        max_x = max(loc.coordinates.x for loc in discovered_locs)
        min_y = min(loc.coordinates.y for loc in discovered_locs)
        max_y = max(loc.coordinates.y for loc in discovered_locs)

        BOX_WIDTH = 12
        BOX_HEIGHT = 3
        H_SPACING = 3
        
        canvas_width = (max_x - min_x + 1) * (BOX_WIDTH + H_SPACING)
        canvas_height = (max_y - min_y + 1) * (BOX_HEIGHT + 1)
        canvas = [[' ' for _ in range(canvas_width)] for _ in range(canvas_height)]

        def draw_box(x, y, loc_name, is_current):
            display_name = loc_name[:BOX_WIDTH - 2].center(BOX_WIDTH - 2)
            
            lines = [f"|{display_name}|"]
            if is_current:
                lines[0] = f"|*{display_name[1:-1]}*|"

            for i, line in enumerate(lines):
                row = y + i + (BOX_HEIGHT // 2)
                for j, char in enumerate(line):
                    if 0 <= row < canvas_height and 0 <= x + j < canvas_width:
                        canvas[row][x+j] = char
        
        loc_map = { (loc.coordinates.x, loc.coordinates.y): loc for loc in discovered_locs }
        for loc in discovered_locs:
            canvas_x = (loc.coordinates.x - min_x) * (BOX_WIDTH + H_SPACING)
            canvas_y = (loc.coordinates.y - min_y) * (BOX_HEIGHT + 1)
            is_current = (loc.id == player.current_location_id)
            draw_box(canvas_x, canvas_y, loc.name.upper(), is_current)
            
            start_y_conn = canvas_y + BOX_HEIGHT // 2
            start_x_conn = canvas_x + BOX_WIDTH // 2

            for conn in loc.connections:
                target_loc = all_locs.get(conn.target_location_id)
                if target_loc and target_loc.visit_count > 0:
                    if conn.direction == Direction.EAST:
                        for i in range(H_SPACING):
                            canvas[start_y_conn][start_x_conn + 1 + i] = '-'
                    elif conn.direction == Direction.SOUTH:
                         for i in range(BOX_HEIGHT // 2 + 2):
                            canvas[start_y_conn + 1 + i][start_x_conn] = '|'

        print("\n--- Your Map ---")
        for row in canvas:
            print("".join(row).rstrip())
        print("----------------")

    def display_help(self):
        commands = {
            "go <dir>": "Move using one of the available exits.",
            "look, l": "Examine your surroundings again.",
            "map": "Display a map of explored areas.",
            "time": "Check the current in-game time and date.",
            "examine <item/feature>": "Look closely at an item or feature.",
            "pick up/take/grab <item>": "Pick up an item from the ground.",
            "drop <item>": "Drop an item from your inventory.",
            "use <item> [with <target>]": "Use an item, optionally on a target.",
            "inventory, i": "Show your inventory.",
            "status": "Show character status and quests.",
            "talk <name>": "Chitchat with an NPC.",
            "ask <npc> about <topic>": "Topics: 'rumor', 'quest', 'services', or service names.",
            "buy <item> [from <npc>]": "Buy an item from an NPC.",
            "sell <item> [to <npc>]": "Sell an item to an NPC.",
            "complete quest <npc>": "Turn in a completed quest to an NPC.",
            "attack <target>": "Attack an NPC.",
            "equip/wield <item>": "Equip armor or wield a weapon.",
            "unequip <item>": "Unequip an item.",
            "help": "Show this help message.",
            "quit": "Exit the game."
        }
        print("\n📋 Available Commands:")
        for command, description in commands.items():
            print(f"  {command:<30} - {description}")
    
    async def process_command(self, command: str) -> bool:
        command = command.strip().lower()
        parts = command.split()
        if not parts: return True
        cmd, args = parts[0], " ".join(parts[1:])
        
        if cmd in ["quit", "exit", "q"]: return False
        elif cmd in ["help", "h"]: self.display_help()
        elif cmd in ["look", "l"]: self.display_location()
        elif cmd == "map": await self.display_map()
        elif cmd == "time": self.display_time()
        elif cmd in ["status", "stat"]: self.display_character_status()
        elif cmd in ["inventory", "inv", "i"]: self.show_inventory()
        elif cmd == "go": self.handle_movement(args)
        elif cmd in [d.value.lower() for d in Direction]: self.handle_movement(cmd)
        elif cmd in ["save"]: await self.save_game()
        elif cmd in ["load"]: await self.load_game()
        elif cmd == "talk": await self.handle_talk(args)
        elif cmd == "ask": await self.handle_ask(args)
        elif cmd == "examine": await self.handle_examine(args)
        elif cmd in ["equip", "wield"]: print(self.engine.equip_item(args))
        elif cmd == "unequip": print(self.engine.unequip_item(args))
        elif cmd in ["pick", "take", "grab"]: await self.handle_pickup(args)
        elif cmd == "drop": await self.handle_drop(args)
        elif cmd == "use": await self.handle_use(args)
        elif cmd == "attack": await self.handle_attack(args)
        elif cmd == "complete": await self.handle_complete_quest(args)
        else: await self.handle_ai_command(command)
        return True
    
    async def handle_complete_quest(self, args: str):
        """Handle quest completion command"""
        if not args:
            print("Complete quest with whom? (e.g., 'complete quest with merchant' or 'complete quest merchant')")
            return
        
        # Handle both "complete quest with npc" and "complete quest npc" formats
        npc_name = args.replace("quest with", "").replace("with", "").strip()
        
        if not self.engine.game_state:
            print("No active game session.")
            return
        
        # Find the NPC in the current location
        loc = self.engine.get_current_location()
        target_npc = next((char for char in self.engine.game_state.characters.values() 
                        if char.current_location_id == loc.id and 
                        npc_name.lower() in char.name.lower() and
                        isinstance(char, NPC)), None)
        
        if not target_npc:
            print(f"There's no one named '{npc_name}' here.")
            return
        
        player = self.engine.game_state.session.player_character
        
        # Find completed quests that were given by this NPC
        completable_quests = []
        for quest_id in player.active_quests:
            quest = self.engine.game_state.quests.get(quest_id)
            if (quest and 
                quest.status == QuestStatus.COMPLETED and 
                quest.giver_id == target_npc.id):
                completable_quests.append(quest)
        
        if not completable_quests:
            # Check if there are any active quests from this NPC that aren't completed yet
            active_from_npc = []
            for quest_id in player.active_quests:
                quest = self.engine.game_state.quests.get(quest_id)
                if quest and quest.giver_id == target_npc.id:
                    active_from_npc.append(quest)
            
            if active_from_npc:
                incomplete_quest = active_from_npc[0]  # Show the first one
                print(f"{target_npc.name} looks at you expectantly.")
                print(f"\"You still need to: {incomplete_quest.objectives[0].description}\"")
                if hasattr(incomplete_quest, 'location_hint') and incomplete_quest.location_hint:
                    print(f"\"Remember: {incomplete_quest.location_hint}\"")
            else:
                print(f"{target_npc.name} smiles. \"I have no quests for you to complete right now.\"")
            return
        
        # Complete the first available quest
        quest_to_complete = completable_quests[0]
        result_message = self.engine.complete_quest(quest_to_complete.id)
        
        print(f"\nYou approach {target_npc.name} to complete your quest.")
        print(f"🎭 {target_npc.name}: \"Excellent work! Thank you for your service.\"")
        print(f"✅ {result_message}")
        
        # If there are multiple completed quests, mention them
        if len(completable_quests) > 1:
            print(f"\n(You have {len(completable_quests)-1} other completed quest(s) with {target_npc.name}. Use the command again to turn them in.)")

    def handle_movement(self, direction_str: str):
        if not direction_str: print("Go where? (e.g., 'go north')"); return
        try:
            direction = Direction(direction_str.lower())
            success, message = self.engine.move_player(direction)
            print(f"\n{message}")
            if success: self.display_location()
        except ValueError: print(f"'{direction_str}' is not a valid direction.")
    
    def show_inventory(self):
        if not self.engine.game_state: 
            return
        player = self.engine.game_state.session.player_character
        print(f"\n🎒 {player.name}'s Inventory:")
        if not player.inventory: 
            print("  (empty)")
            return
            
        for item_id in player.inventory:
            item = self.engine.game_state.items.get(item_id)
            if item:
                equipped_str = " (equipped)" if item.id in player.equipped_items.values() else ""
                
                # Show stack size for stackable items
                stack_str = ""
                if item.stackable and item.current_stack_size > 1:
                    stack_str = f" x{item.current_stack_size}"
                
                # Show stat bonuses for equipment
                bonus_str = ""
                if item.stat_modifiers:
                    bonuses = []
                    for stat, bonus in item.stat_modifiers.items():
                        if bonus != 0:
                            bonuses.append(f"{stat}: {bonus:+d}")
                    if bonuses:
                        bonus_str = f" [{', '.join(bonuses)}]"
                
                print(f"  • {item.name}{stack_str}{equipped_str}{bonus_str} (w: {item.weight}) - {item.description}")

    # MODIFIED: Upgraded to search for hidden items upon examining.
    async def handle_examine(self, target_name: str):
        if not target_name:
            print("Examine what?")
            return

        location = self.engine.get_current_location()

        # Helper function to reveal contained items
        def reveal_items(target, target_type_name):
            if not target.contained_items:
                return
            
            revealed_item_names = []
            for item_id in list(target.contained_items):
                item = self.engine.game_state.items.get(item_id)
                if item:
                    revealed_item_names.append(item.name)
                    location.items.append(item_id) # Move item to location
            
            if revealed_item_names:
                print(f"Searching the {target_type_name}, you find: {', '.join(revealed_item_names)}.")
                target.contained_items.clear() # Empty the container

        # 1. Check inventory
        item_in_inventory = self.engine.find_item_in_inventory(target_name)
        if item_in_inventory:
            print(f"\nYou examine the {item_in_inventory.name} in your inventory:\n  Name: {item_in_inventory.name} ({item_in_inventory.rarity.value})\n  Type: {item_in_inventory.item_type.value}\n  Description: {item_in_inventory.description}")
            if item_in_inventory.stat_modifiers: print(f"  Modifiers: {item_in_inventory.stat_modifiers}")
            reveal_items(item_in_inventory, f"the {item_in_inventory.name}")
            return
        
        # 2. Check items on the ground
        item_on_ground = self.engine.find_item_in_location(target_name)
        if item_on_ground:
            print(f"\nYou examine the {item_on_ground.name} on the ground:\n  Name: {item_on_ground.name} ({item_on_ground.rarity.value})\n  Type: {item_on_ground.item_type.value}\n  Description: {item_on_ground.description}")
            reveal_items(item_on_ground, f"the {item_on_ground.name}")
            return

        # 3. Check notable features
        feature_to_examine = next((f for f in location.notable_features if target_name.lower() in f.name.lower()), None)
        if feature_to_examine:
            if feature_to_examine.detailed_description:
                print(f"\nYou look again at the {feature_to_examine.name.lower()}.")
                print(f"🎭 {feature_to_examine.detailed_description}")
            else:
                print(f"\nYou take a closer look at the {feature_to_examine.name.lower()}...")
                response = await self.handle_ai_command(f"Describe the {feature_to_examine.name} in vivid, sensory detail. Respond with exactly one paragraph of 2-4 sentences.", print_response=False)
                if response:
                    feature_to_examine.detailed_description = response
                    print(f"🎭 {response}")
            
            reveal_items(feature_to_examine, f"the {feature_to_examine.name.lower()}")
            return

        print(f"You see no '{target_name}' here to examine.")

    async def handle_talk(self, npc_name: str):
        if not npc_name: print("Talk to whom?"); return
        await self.handle_ai_command(f"chitchat with {npc_name}")

    async def handle_ask(self, args: str):
        if " about " not in args:
            print("What do you want to ask about? (e.g., 'ask <npc> about rumor')"); return
        npc_name, topic = [p.strip() for p in args.split(" about ", 1)]
        if not self.engine.game_state: return
        
        loc = self.engine.get_current_location()
        target_npc = next((char for char in self.engine.game_state.characters.values() if char.current_location_id == loc.id and npc_name.lower() in char.name.lower()), None)
        if not target_npc: print(f"There's no one named '{npc_name}' here."); return

        print(f"\nYou ask {target_npc.name} about {topic}...")
        
        if topic == 'rumor':
            print("🤔 Thinking...", end="", flush=True)
            rumor = await self.engine.generate_rumor(target_npc, self.selected_model)
            print("\r" + " " * 20 + "\r", end="")
            print(f"🎭 {target_npc.name} leans in conspiratorially. \"{rumor}\"")
        elif topic == 'quest':
            print("🤔 Thinking...", end="", flush=True)
            quest = await self.engine.generate_quest(target_npc, self.selected_model)
            print("\r" + " " * 20 + "\r", end="")
            if quest:
                print(f"🎭 {target_npc.name} looks you over. \"Yes, perhaps you can help. {quest.description}\"")
                print(f"New quest added: {quest.name}")
            else:
                print(f"🎭 {target_npc.name} shakes their head. \"I have nothing for you at the moment.\"")
        else:
            await self.handle_ai_command(f"ask {npc_name} about {topic}")
    
    async def handle_attack(self, target_name: str):
        if not self.engine.game_state: return
        loc = self.engine.get_current_location()
        target = next((char for char in self.engine.game_state.characters.values() if char.current_location_id == loc.id and target_name.lower() in char.name.lower()), None)
        if not target: print(f"There is no '{target_name}' here to attack."); return

        print(f"You attack {target.name}!"); self.engine.in_combat = True
        player = self.engine.game_state.session.player_character
        player_damage = random.randint(1, 6) + player.stats.damage_bonus
        target.stats.health -= player_damage
        print(f"You hit {target.name} for {player_damage} damage. ({target.stats.health}/{target.stats.max_health} HP)")
        if target.stats.health <= 0:
            print(f"You have defeated {target.name}!"); self.engine.in_combat = False
            del self.engine.game_state.characters[target.id]
            return

        npc_damage = random.randint(1, 4) + target.stats.damage_bonus
        player.stats.health -= npc_damage
        print(f"{target.name} hits you for {npc_damage} damage. ({player.stats.health}/{player.stats.max_health} HP)")
        if player.stats.health <= 0:
            print(f"You have been defeated by {target.name}! Game over."); self.running = False; return
        
        self.engine.in_combat = False; print("The combatants break apart, eyeing each other warily.")

    async def handle_ai_command(self, command: str, print_response: bool = True) -> Optional[str]:
        if not self.engine.game_state: 
            print("No active game session.")
            return None
        context = self.engine.build_context_for_ai()
        print("\n🤔 Thinking...", end="", flush=True)
        response = await self.engine.generate_ai_response(command, context, model_name=self.selected_model)
        print("\r" + " " * 20 + "\r", end="")
        if print_response:
            print(f"🎭 {response}")
        return response
    
    async def save_game(self):
        if not self.engine.game_state: print("No active game to save."); return
        filepath = Path("saves") / f"{self.engine.game_state.session.session_name}.json"
        if await self.engine.save_game(filepath): print(f"Game saved as '{filepath.name}'")
        else: print("Failed to save game.")
    
    async def load_game(self):
        saves_dir = Path("saves")
        if not saves_dir.exists() or not any(saves_dir.glob("*.json")): print("No saved games found."); return
        save_files = list(saves_dir.glob("*.json"))
        print("\nSaved games:")
        for i, save_file in enumerate(save_files): print(f"  {i+1}. {save_file.stem}")
        try:
            choice = input("Enter number to load (or press Enter to cancel): ").strip()
            if not choice: return
            filepath = save_files[int(choice) - 1]
            if await self.engine.load_game(filepath):
                print(f"\nGame '{filepath.stem}' loaded successfully!")
                self.display_location()
            else: 
                print("Failed to load game.")
        except (ValueError, IndexError): print("Invalid selection.")
    
    async def start_new_game(self) -> bool:
        print("\n🌟 Starting New Adventure")
        print("Checking Ollama connection...")
        models = await self.engine.get_available_models()
        if not models:
            print("\n❌ Could not connect or find any Ollama models."); print("   Ensure Ollama is running and has models."); return False
        
        self.selected_model = ""
        if settings.ollama_model in models:
            print(f"✓ Default model '{settings.ollama_model}' found."); self.selected_model = settings.ollama_model
        else:
            print(f"⚠️ Default model '{settings.ollama_model}' not found. Please select an available model:")
            for i, model_name in enumerate(models): print(f"  {i+1}. {model_name}")
            while not self.selected_model:
                try:
                    choice = input(f"Enter number (1-{len(models)}): ").strip()
                    self.selected_model = models[int(choice) - 1]
                except (ValueError, IndexError): print("Invalid selection.")
        
        player_name = input("Enter your character's name: ").strip() or "Adventurer"
        session_name = input("Enter a name for this adventure: ").strip() or f"{player_name}'s Adventure"
        
        try:
            await self.engine.create_new_game(player_name, uuid4(), session_name, self.selected_model)
            print("\n✨ World created successfully! Your adventure begins...")
            self.display_location()
            return True
        except ConnectionError: return False
    
    async def show_main_menu(self) -> bool:
        print("\n🏰 Welcome, Adventurer!")
        while True:
            print("\n--- Main Menu ---\n1. Start New Game\n2. Load Game\n3. Exit")
            choice = input("Enter your choice (1-3): ").strip()
            if choice == "1":
                if await self.start_new_game(): return True 
            elif choice == "2":
                await self.load_game()
                if self.engine.game_state: return True
            elif choice == "3": return False
            else: print("Invalid choice.")
    
    async def handle_pickup(self, item_name: str):
        if not item_name:
            print("Pick up what?")
            return
        
        loc = self.engine.get_current_location()
        player = self.engine.game_state.session.player_character
        item_to_get = self.engine.find_item_in_location(item_name)
        
        if item_to_get:
            loc.items.remove(item_to_get.id)
            # Remove from game state temporarily
            del self.engine.game_state.items[item_to_get.id]
            
            # Use the new stacking system
            message = self.engine.add_item_to_inventory(player, item_to_get, item_to_get.current_stack_size)
            print(message)
        else:
            print(f"You don't see a '{item_name}' here.")

    async def handle_drop(self, item_name: str):
        if not item_name:
            print("Drop what?")
            return
            
        loc = self.engine.get_current_location()
        player = self.engine.game_state.session.player_character
        item_to_drop = self.engine.find_item_in_inventory(item_name)
        
        if item_to_drop:
            # Create a copy to drop
            dropped_item = Item.model_validate(item_to_drop.model_dump())
            dropped_item.id = uuid4()
            
            # Remove from inventory using new system
            success, message = self.engine.remove_item_from_inventory(player, item_name, 1)
            
            if success:
                # Add to location
                self.engine.game_state.items[dropped_item.id] = dropped_item
                loc.items.append(dropped_item.id)
                print(f"You drop the {dropped_item.name}.")
            else:
                print(message)
        else:
            print(f"You don't have a '{item_name}' in your inventory.")

    async def handle_use(self, args: str):
        if not args:
            print("Use what? (e.g., 'use key' or 'use key with chest')")
            return

        player = self.engine.game_state.session.player_character

        if " with " in args:
            tool_name, target_name = [p.strip() for p in args.split(" with ", 1)]
            
            tool_item = self.engine.find_item_in_inventory(tool_name)
            if not tool_item:
                print(f"You don't have a '{tool_name}'.")
                return
            
            location = self.engine.get_current_location()
            target_item = self.engine.find_item_in_location(target_name)
            target_feature = next((f for f in location.notable_features if target_name.lower() in f.name.lower()), None)
            target = target_item or target_feature
            
            if not target:
                print(f"You don't see a '{target_name}' here to use that on.")
                return

            if target.id in tool_item.interactions:
                print(tool_item.interactions[target.id])

                if target.contained_items:
                    revealed_items = []
                    for item_id in list(target.contained_items):
                        target.contained_items.remove(item_id)
                        location.items.append(item_id)
                        item = self.engine.game_state.items.get(item_id)
                        if item:
                            revealed_items.append(item.name)
                    
                    if revealed_items:
                        print(f"Inside, you find: {', '.join(revealed_items)}.")

                if tool_item.consumable:
                    if tool_item.charges is not None:
                        tool_item.charges -= 1
                        if tool_item.charges <= 0:
                            player.inventory.remove(tool_item.id)
                            del self.engine.game_state.items[tool_item.id]
                            print(f"(The {tool_item.name} is used up.)")
            else:
                print("That doesn't seem to do anything.")
        else:
            item_to_use = self.engine.find_item_in_inventory(args)
            if not item_to_use:
                print(f"You don't have a '{args}'.")
                return
            
            if item_to_use.self_use_effect_description:
                print(item_to_use.self_use_effect_description)
                
                if item_to_use.consumable:
                    if item_to_use.charges is not None:
                        item_to_use.charges -= 1
                        if item_to_use.charges <= 0:
                            player.inventory.remove(item_to_use.id)
                            del self.engine.game_state.items[item_to_use.id]
                            print(f"(You have used the {item_to_use.name}.)")
            else:
                print(f"You can't seem to use the {item_to_use.name} by itself.")
    
    async def main_loop(self):
        self.display_header()
        if not await self.show_main_menu():
            print("\nThanks for playing!"); return
        
        self.running = True
        print("\nType 'help' for available commands, 'quit' to exit.")
        while self.running:
            try:
                command = input("\n> ").strip()
                if command:
                    if not await self.process_command(command): self.running = False
            except KeyboardInterrupt:
                print("\n\nGoodbye!"); self.running = False
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                logging.exception("An unexpected error occurred in the main loop")
        
        if self.engine.game_state:
            print("\nAuto-saving..."); await self.save_game()
        
        print("\nThanks for playing!")

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    cli = GameCLI()
    try:
        asyncio.run(cli.main_loop())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            pass
        else:
            raise
