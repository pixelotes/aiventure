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
    CharacterClass, CharacterType, BaseCharacter, PlayerCharacter, NPCRole, 
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
        """Apply equipment bonuses to character stats"""
        if not isinstance(character, (PlayerCharacter, NPC)):
            return
            
        base_stats = character.stats
        bonuses = self.calculate_equipment_bonuses(character)
        
        character.stats.strength = max(1, base_stats.strength + bonuses.strength)
        character.stats.dexterity = max(1, base_stats.dexterity + bonuses.dexterity)
        character.stats.constitution = max(1, base_stats.constitution + bonuses.constitution)
        character.stats.intelligence = max(1, base_stats.intelligence + bonuses.intelligence)
        character.stats.wisdom = max(1, base_stats.wisdom + bonuses.wisdom)
        character.stats.charisma = max(1, base_stats.charisma + bonuses.charisma)
        character.stats.armor_class = max(1, base_stats.armor_class + bonuses.armor_class)
        character.stats.attack_bonus = base_stats.attack_bonus + bonuses.attack_bonus
        character.stats.damage_bonus = base_stats.damage_bonus + bonuses.damage_bonus
        
        character.stats.max_health = max(1, base_stats.max_health + (bonuses.constitution * 5))
        character.stats.max_mana = max(1, base_stats.max_mana + (bonuses.intelligence * 3))
    
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

    def _sanitize_location_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize AI-generated location data to prevent Pydantic validation errors"""
        if not isinstance(data, dict):
            return data
            
        # Handle fields that should be strings but might be returned as lists
        string_fields = ['general_type', 'atmosphere', 'temperature', 'weather', 'name', 'description', 'short_description']
        for field in string_fields:
            if field in data and isinstance(data[field], list):
                if data[field]:
                    data[field] = str(data[field][0])
                else:
                    data[field] = ""
            elif field in data and data[field] is None:
                data[field] = ""
        
        # Ensure coordinates is not in the data as it's handled separately
        if 'coordinates' in data:
            data.pop('coordinates')
            
        return data

    def _create_services_for_npc(self, npc: NPC) -> None:
        if npc.role == NPCRole.SHOPKEEPER:
            npc.services_offered.append(Service(service_type=ServiceType.BUY_SELL, name="General Goods", description="I buy and sell various items.", cost={"gold": 0}))
        elif npc.role == NPCRole.MERCHANT:
            npc.services_offered.append(Service(service_type=ServiceType.BUY_SELL, name="Trade Goods", description="I deal in fine wares and exotic items.", cost={"gold": 0}))
        elif npc.role == NPCRole.INNKEEPER:
            npc.services_offered.extend([
                Service(service_type=ServiceType.REST, name="Room for the Night", description="A warm bed and a hot meal.", cost={"gold": 2, "silver": 5}),
                Service(service_type=ServiceType.HEAL, name="Herbal Remedies", description="Basic healing herbs and tonics.", cost={"gold": 1})
            ])
        elif npc.role == NPCRole.CRAFTSMAN:
            npc.services_offered.append(Service(service_type=ServiceType.REPAIR, name="Item Repair", description="I can mend your broken equipment.", cost={"gold": 5}))

    async def create_new_game(self, player_name: str, player_id: UUID, session_name: str, model_name: str) -> CompleteGameState:
        try:
            print("\n1/3: Conceptualizing the world and its main story...")
            world_prompt = (
                "Generate the high-level details for a new fantasy world and a main quest. "
                "Also generate a short, evocative background for the hero. "
                "JSON with keys: name (str), description (str), theme (str), lore_summary (str), "
                "quest_name (str), quest_description (str), player_background (str), "
                "starter_region_type (str), goal_region_type (str)."
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
            r_start = Region(name=f"The {world_data['starter_region_type']}", description="Where your journey begins.", region_type=world_data['starter_region_type'], location_type=LocationType.REGION, short_description="Initial region", tags=["starter"])
            r_mid = Region(name="The Forbidden Boundary", description="A heavily guarded or dangerous zone.", region_type="wilderness", location_type=LocationType.REGION, short_description="A gated passage", tags=["boundary"])
            r_end = Region(name=f"The {world_data['goal_region_type']}", description="The place of your destiny.", region_type=world_data['goal_region_type'], location_type=LocationType.REGION, short_description="Goal region", tags=["goal"])
            
            # Link regions
            r_start.connections_to_regions[Direction.EAST] = r_mid.id
            r_mid.connections_to_regions[Direction.WEST] = r_start.id
            r_mid.connections_to_regions[Direction.EAST] = r_end.id
            r_end.connections_to_regions[Direction.WEST] = r_mid.id
            
            world.regions = {r.id: r for r in [r_start, r_mid, r_end]}
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
                description=f"{player_name} in {world.name}.", 
                background_lore=world_data.get('player_background', f"{player_name} is a traveler seeking adventure."),
                currency={"gold": 50}, 
                inventory=[]
            )
            game_state.characters[player_char.id] = player_char
            session.player_character = player_char
            
            # Setup Main Quest
            quest = Quest(
                name=world_data['quest_name'],
                description=world_data['quest_description'],
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
        
        width = random.randint(2, 3) # Keeping it small for now
        height = random.randint(2, 3)
        region.width, region.height = width, height
        grid = WorldGrid(width=width, height=height, grid=[[None]*width for _ in range(height)])
        self.game_state.session.region_grids[region_id] = grid
        
        allowed_loc_types = [t.value for t in GeneralLocationType]
        is_city = "city" in region.region_type.lower()
        
        for y in range(height):
            for x in range(width):
                print(f"   - Building {region.name} cell ({x}, {y})...", end="", flush=True)
                
                # Contextual prompt
                if is_city:
                    prompt = f"Generate a city district in '{region.name}'. JSON with: name, description, short_description, general_type (one of: city_center, market_district, residential_district, plaza), atmosphere, buildings (list of: name, type, description), and optional 'npc' object (name, description, race, role)."
                else:
                    prompt = f"Generate a location in {region.region_type} area of {self.game_state.session.world.name} at ({x},{y}). JSON with: name, description, short_description, general_type (one of: {allowed_loc_types}), atmosphere, notable_features (list name/desc), and optional 'npc' object."

                loc_data = await self._generate_and_validate(prompt, model_name)
                loc_data = self._sanitize_location_data(loc_data)
                
                buildings = loc_data.pop('buildings', [])
                notable = loc_data.pop('notable_features', [])
                npc_info = loc_data.pop('npc', None) if is_city else (loc_data.pop('npc', None) if random.random() < 0.2 else None)
                
                loc = GeneralLocation.model_validate(loc_data)
                loc.coordinates = Coordinates(x=x, y=y)
                loc.parent_id = region_id
                
                # Add buildings or features
                for b in buildings:
                    if isinstance(b, dict):
                        b_name = b.get('name', 'A building')
                        b_type = b.get('type', 'building')
                        b_desc = b.get('description', 'A generic building.')
                        loc.notable_features.append(NotableFeature(name=b_name, detailed_description=f"A {b_type}: {b_desc}"))
                    elif isinstance(b, str):
                        loc.notable_features.append(NotableFeature(name=b))

                for n in notable:
                    if isinstance(n, dict):
                        loc.notable_features.append(NotableFeature(name=n.get('name', 'Feature'), detailed_description=n.get('description')))
                    elif isinstance(n, str):
                        loc.notable_features.append(NotableFeature(name=n))
                
                self.game_state.locations[loc.id] = loc
                grid.set_location_id(x, y, loc.id)
                
                if npc_info:
                    npc_info.update({'id': str(uuid4()), 'current_location_id': loc.id, 'home_location_id': loc.id})
                    npc = NPC.model_validate(npc_info)
                    self._create_services_for_npc(npc)
                    self.game_state.characters[npc.id] = npc
                
                # Chance to generate sub-levels (dungeons/buildings)
                if any(tag in loc.name.lower() or tag in loc.description.lower() for tag in ["cave", "tower", "cellar", "ruins", "dungeon"]):
                    if random.random() < 0.7:
                        await self.generate_sub_levels(loc, depth=random.randint(1, 2), model_name=model_name)

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
        # Loot items
        for item_id in list(target.inventory):
            if item_id in self.game_state.items:
                self.get_current_location().items.append(item_id)

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
        return True, f"Moving {direction.value}... (Consumed {stamina_cost} stamina)"

    def check_quest_progress(self) -> List[str]:
        if not self.game_state: return []
        player = self.game_state.session.player_character
        msgs = []
        for qid in list(player.active_quests):
            q = self.game_state.quests.get(qid)
            if q and q.status == QuestStatus.ACTIVE and q.quest_type == QuestType.FETCH and q.target_item_id in player.inventory:
                for obj in q.objectives:
                    if obj.objective_type == "fetch":
                        obj.completed, obj.current_progress = True, obj.required_progress
                if all(o.completed for o in q.objectives):
                    q.status = QuestStatus.COMPLETED
                    msgs.append(f"Quest ready: {q.name}")
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
        
        if q.rewards:
            player.stats.experience = getattr(player.stats, 'experience', 0) + q.rewards.experience
            player.experience = getattr(player, 'experience', 0) + q.rewards.experience
            
            for cur, amt in q.rewards.currency.items():
                player.currency[cur] = player.currency.get(cur, 0) + amt
        return f"Completed {q.name}!"

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
        
        # Update world if time period changed or enough time passed
        await self.update_world_state(model_name)
        
    async def update_world_state(self, model_name: str) -> None:
        """Update NPCs, regional events, and weather"""
        if not self.game_state: return
        
        # 0. Plot Heartbeat (World Events)
        await self._run_plot_heartbeat(model_name)
        
        # 1. Update NPC positions based on schedule
        self._update_npc_positions()
        
        # 2. Update Regional Events
        await self._update_regional_events(model_name)
        
        # 3. Update Local Weather (Randomly)
        for loc in self.game_state.locations.values():
            if random.random() < 0.05: # 5% chance of weather change per update
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
                        name=data['name'],
                        description=data['description'],
                        scope=EventScope(data.get('scope', 'global').lower()),
                        duration_minutes=data.get('duration_days', 1) * 1440
                    )
                    self.game_state.session.active_global_events.append(event)
                    llm_logger.info(f"PLOT EVENT TRIGGERED: {event.name}")
            except Exception as e:
                llm_logger.error(f"Error in plot heartbeat: {e}")

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
            loc.name = data['new_name']
            loc.description = data['new_description']
            loc.current_state_tag = data['state_tag']
            loc.history.append(f"Once known as {old_name}, it was changed by: {action_description}")
            
            self.game_state.session.major_decision_history.append(f"Transformed {old_name} into {loc.name} through action: {action_description}")
            
            return f"\n⚠️  {Colors.BOLD}WORLD CHANGE{Colors.ENDC}\n{old_name} has become {Colors.CYAN}{loc.name}{Colors.ENDC}.\n{loc.description}"
        except Exception as e:
            llm_logger.error(f"Error applying persistent change: {e}")
            return "The world feels different, but the change is hard to describe."

    def _update_npc_positions(self) -> None:
        """Move NPCs based on the time of day"""
        gt = self.game_state.session.game_time
        # Daytime: 7:00 to 19:00
        is_work_time = 7 <= gt.hour < 19
        
        for char in self.game_state.characters.values():
            if isinstance(char, NPC):
                target_loc = char.work_location_id if (is_work_time and char.work_location_id) else char.home_location_id
                if target_loc and char.current_location_id != target_loc:
                    char.previous_location_id = char.current_location_id
                    char.current_location_id = target_loc
                    # llm_logger.info(f"NPC {char.name} moved to {'work' if is_work_time else 'home'}")

    async def _update_regional_events(self, model_name: str) -> None:
        """Randomly trigger or end regional events"""
        for region_id, region in self.game_state.session.world.regions.items():
            # Chance to end existing events
            if region.active_events and random.random() < 0.2:
                region.active_events.pop(0)
                region.event_modifiers.clear()
            
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
                # print(f"\n📢 EVENT in {region.name}: {event_name} - {desc}")

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
            f"PLAYER: {p.name}"
        ])

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
            npc.interaction_summary = data['summary'][:500]
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

    async def generate_quest(self, npc: NPC, model_name: str) -> Optional[Quest]:
        return await self.generate_fetch_quest(npc, model_name)

    async def generate_fetch_quest(self, npc: NPC, model_name: str) -> Optional[Quest]:
        context = self.build_context_for_ai()
        quest_prompt = f"The player is asking {npc.name} ({npc.description}) for a quest. Generate a fetch quest. JSON: name, description, item_name, item_description, location_hint."
        
        try:
            quest_data = await self._generate_and_validate(quest_prompt, model_name)
            quest_item = Item(name=quest_data['item_name'], description=quest_data['item_description'], item_type=ItemType.QUEST_ITEM, value=random.randint(10, 50), weight=random.uniform(0.1, 2.0))
            
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
                name=quest_data['name'], description=quest_data['description'], quest_type=QuestType.FETCH,
                giver_id=npc.id, objectives=[objective], status=QuestStatus.ACTIVE,
                target_item_id=quest_item.id, location_hint=quest_data['location_hint'],
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
                    name=data['name'],
                    description=data['description'],
                    item_type=ItemType(data['item_type'].lower()),
                    value=data['value'],
                    rarity=ItemRarity(data['rarity'].lower())
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
                msg = data['outcome_description']
                if data.get('new_exit_direction'):
                    direction = Direction(data['new_exit_direction'].lower())
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
            grid_data = game_dict['session']['world_grid']
            grid_obj = WorldGrid(grid_data['width'], grid_data['height'])
            for y, row in enumerate(grid_data['grid']):
                for x, cell in enumerate(row): grid_obj.grid[y][x] = UUID(cell) if cell else None
            game_dict['session']['world_grid'] = grid_obj
            self.game_state = CompleteGameState.model_validate(game_dict)
            return True
        except Exception as e:
            llm_logger.error(f"Load error: {e}")
            return False
