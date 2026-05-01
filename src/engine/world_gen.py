from __future__ import annotations
from datetime import datetime
from typing import Any, Optional, Tuple
from uuid import UUID, uuid4
import random

from models import (
    Direction, Coordinates, LocationConnection, LocationType, NotableFeature,
    BaseLocation, Region, World, GeneralLocationType, GeneralLocation,
    CharacterClass, NPCRole, NPCGoal, NPC, ItemType, ItemRarity, Item,
    QuestType, QuestStatus, QuestObjective, Quest, GameSession, CompleteGameState,
    WorldGrid,
)

llm_logger = __import__('logging').getLogger("llm_responses")


class WorldGenMixin:

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
            start_location_id = start_grid.get_location_id(0, 0)  # Top-left corner for now

            from models import PlayerCharacter
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
            from models import QuestReward
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
        if not region:
            return

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
                            target_location_id=target_region_id,  # Actually points to region ID, we'll handle this in move_player
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
                sub_loc.parent_id = parent_location.parent_id  # Same region
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
