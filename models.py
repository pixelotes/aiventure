from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Type, Tuple
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict

# ============================================================================
# Base Models and Enums
# ============================================================================
class EventScope(str, Enum):
    GLOBAL = "global"
    REGIONAL = "regional"
    LOCAL = "local"

class GlobalEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    scope: EventScope = EventScope.GLOBAL
    target_ids: List[UUID] = Field(default_factory=list) # Regions or Locations affected
    start_time: datetime = Field(default_factory=datetime.now)
    duration_minutes: int = 1440 # 24 hours default
    modifiers: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

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
    is_interactive: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    history: List[str] = Field(default_factory=list)
    current_state_tag: Optional[str] = None # e.g., "ruined", "prosperous"

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
    regions: Dict[UUID, Region] = Field(default_factory=dict)
    starting_region_id: Optional[UUID] = None

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
    CITY_CENTER = "city_center"
    MARKET_DISTRICT = "market_district"
    RESIDENTIAL_DISTRICT = "residential_district"
    INDUSTRIAL_DISTRICT = "industrial_district"
    PLAZA = "plaza"

class Region(BaseLocation):
    location_type: LocationType = LocationType.REGION
    width: int = 1
    height: int = 1
    is_generated: bool = False
    region_type: str = "wilderness" # e.g. "city", "forest", "mountain"
    connections_to_regions: Dict[Direction, UUID] = Field(default_factory=dict)
    active_events: List[str] = Field(default_factory=list)
    event_modifiers: Dict[str, float] = Field(default_factory=dict)
    market_history: Dict[str, int] = Field(default_factory=dict) # item_name -> total_sold_by_player
    
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
    background_lore: str = ""
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
    interaction_summary: str = ""
    mood: float = 0.5  # 0.0 (hostile/sad) to 1.0 (friendly/happy)
    faction: Optional[str] = None

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
    combinations: Dict[str, str] = Field(default_factory=dict) # target_item_name -> result_item_name

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

class WorldGrid(BaseModel):
    width: int
    height: int
    grid: List[List[Optional[UUID]]]

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
    current_region_id: Optional[UUID] = None
    region_grids: Dict[UUID, WorldGrid] = Field(default_factory=dict)
    active_global_events: List[GlobalEvent] = Field(default_factory=list)
    major_decision_history: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_played: datetime = Field(default_factory=datetime.now)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def current_grid(self) -> Optional[WorldGrid]:
        if self.current_region_id:
            return self.region_grids.get(self.current_region_id)
        return None

class CompleteGameState(BaseModel):
    session: GameSession
    locations: Dict[UUID, BaseLocation] = Field(default_factory=dict)
    characters: Dict[UUID, BaseCharacter] = Field(default_factory=dict)
    items: Dict[UUID, Item] = Field(default_factory=dict)
    quests: Dict[UUID, Quest] = Field(default_factory=dict)
    @property
    def character_locations(self) -> Dict[UUID, UUID]:
        return {char_id: char.current_location_id for char_id, char in self.characters.items()}
    model_config = ConfigDict(arbitrary_types_allowed=True)
