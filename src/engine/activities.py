from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4
import random

from models import (
    Direction, LocationConnection, LocationType, NotableFeature,
    BaseLocation, GeneralLocation, Coordinates,
    NPC, NPCRole, NPCGoal, ItemType, ItemRarity, Item,
)

llm_logger = __import__('logging').getLogger("llm_responses")


class ActivitiesMixin:

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

    AI_EFFECT_CAPS = {
        "max_time_minutes": 480,
        "max_gold_give": 50,
        "max_item_value": 30,
        "max_effects": 5,
    }

    async def move_player(self, direction: Direction, model_name: str, confirmed: bool = False) -> Tuple[bool, str]:
        if not self.game_state:
            return False, "No session"
        current_loc = self.get_current_location()
        conn = next((c for c in current_loc.connections if c.direction == direction), None)
        if not conn or not conn.is_visible or not conn.is_passable:
            return False, "Blocked"

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
            if direction == Direction.NORTH:
                entry_y = target_grid.height - 1
            elif direction == Direction.SOUTH:
                entry_y = 0
            elif direction == Direction.WEST:
                entry_x = target_grid.width - 1
            elif direction == Direction.EAST:
                entry_x = 0

            target_id = target_grid.get_location_id(entry_x, entry_y)
            target = self.game_state.locations.get(target_id)
            travel_time = 60  # 1 hour for region travel
        else:
            target = self.game_state.locations.get(conn.target_location_id)
            travel_time = conn.travel_time

        if not target:
            return False, "Void error"

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

    def _get_region_modifiers(self) -> Dict[str, float]:
        """Return event_modifiers for the current region, or empty dict."""
        session = self.game_state.session
        region = session.world.regions.get(session.current_region_id)
        return region.event_modifiers if region else {}

    def hunt(self) -> Tuple[bool, str]:
        """Hunt for game at the current location. Deterministic, no AI."""
        player = self.game_state.session.player_character
        loc = self.get_current_location()
        terrain = getattr(loc, 'general_type', None)
        terrain_key = terrain.value if terrain else None

        if terrain_key not in self.HUNT_TABLE:
            return False, "There's nothing to hunt here."

        mods = self._get_region_modifiers()
        stamina_cost = int(self.HUNT_STAMINA_COST * mods.get("stamina_cost", 1.0))
        if player.stats.stamina < stamina_cost:
            return False, "You're too tired to hunt."

        player.stats.stamina -= stamina_cost
        self.game_state.session.game_time.advance_time(self.HUNT_TIME_MINUTES)

        base_chance, loot_table = self.HUNT_TABLE[terrain_key]
        chance = min(0.95, base_chance * mods.get("hunt_mod", 1.0))
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

        mods = self._get_region_modifiers()
        stamina_cost = int(self.FORAGE_STAMINA_COST * mods.get("stamina_cost", 1.0))
        if player.stats.stamina < stamina_cost:
            return False, "You're too tired to forage."

        player.stats.stamina -= stamina_cost
        self.game_state.session.game_time.advance_time(self.FORAGE_TIME_MINUTES)

        base_chance, loot_table = self.FORAGE_TABLE[terrain_key]
        chance = min(0.95, base_chance * mods.get("forage_mod", 1.0))
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

    async def use_item_on_feature(self, item_id: UUID, feature_id: UUID, model_name: str) -> Tuple[bool, str]:
        """Attempt to use an item on a notable feature to trigger an effect or puzzle solution"""
        item = self.game_state.items.get(item_id)
        loc = self.get_current_location()
        feature = next((f for f in loc.notable_features if f.id == feature_id), None)

        if not item or not feature:
            return False, "Target not found."

        prompt = f"The player is using '{item.name}' on '{feature.name}' ({feature.detailed_description or feature.name}). If this solves a puzzle or triggers a meaningful change in the environment, return JSON with: success (bool), outcome_description, new_exit_direction (optional), new_exit_description (optional). Example: Using a 'Makeshift Bridge' on a 'Deep Chasm' could unlock the 'EAST' exit."

        try:
            data = await self._generate_and_validate(prompt, model_name)
            if data.get('success'):
                msg = data.get('outcome_description', 'Something happened.')
                if data.get('new_exit_direction'):
                    direction = Direction(self._coerce_enum(data.get('new_exit_direction', 'north'), Direction))
                    # Check if connection already exists
                    if not any(c.direction == direction for c in loc.connections):
                        target_loc = GeneralLocation(
                            name="Secret Path",
                            description="A path revealed by your ingenuity.",
                            short_description="A secret area.",
                            parent_id=loc.parent_id,
                            coordinates=Coordinates(x=loc.coordinates.x, y=loc.coordinates.y, z=loc.coordinates.z)
                        )
                        self.game_state.locations[target_loc.id] = target_loc
                        loc.connections.append(LocationConnection(
                            target_location_id=target_loc.id,
                            direction=direction,
                            description=data.get('new_exit_description', f"A path leading {direction.value}.")
                        ))
                        msg += f" (Unlocked exit to the {direction.value})"
                return True, msg
            else:
                return False, data.get('outcome_description', "Nothing happens.")
        except Exception as e:
            llm_logger.error(f"Error using item on feature: {e}")
            return False, "Nothing happens."

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
            from uuid import UUID
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
