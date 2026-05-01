from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from uuid import UUID
import random

from models import (
    NPC, ItemType, Item, QuestType, QuestStatus, QuestObjective, QuestReward, Quest,
    LocationType, NPCRole, NPCGoal,
)

llm_logger = __import__('logging').getLogger("llm_responses")


class QuestMixin:

    def check_quest_progress(self) -> List[str]:
        if not self.game_state:
            return []
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
        if not self.game_state:
            return "No session"
        q = self.game_state.quests.get(quest_id)
        if not q or q.status != QuestStatus.COMPLETED:
            return "Not ready"

        player = self.game_state.session.player_character
        if quest_id in player.active_quests:
            player.active_quests.remove(quest_id)
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

    async def generate_quest(self, npc: NPC, model_name: str) -> Optional[Quest]:
        return await self.generate_fetch_quest(npc, model_name)

    async def generate_fetch_quest(self, npc: NPC, model_name: str) -> Optional[Quest]:
        context = self.build_context_for_ai()
        quest_prompt = f"The player is asking {npc.name} ({npc.description}) for a quest. Generate a fetch quest. JSON: name, description, item_name, item_description, location_hint."

        try:
            quest_data = await self._generate_and_validate(quest_prompt, model_name)
            quest_item = Item(
                name=quest_data.get('item_name', 'Mysterious Object'),
                description=quest_data.get('item_description', 'An item of unknown origin.'),
                item_type=ItemType.QUEST_ITEM,
                value=random.randint(10, 50),
                weight=random.uniform(0.1, 2.0)
            )

            available_locations = [loc for loc in self.game_state.locations.values() if loc.id != npc.current_location_id and loc.location_type == LocationType.LOCATION]
            if not available_locations:
                return None
            target_location = random.choice(available_locations)

            if target_location.notable_features and random.random() < 0.6:
                random.choice(target_location.notable_features).contained_items.append(quest_item.id)
            else:
                target_location.items.append(quest_item.id)

            self.game_state.items[quest_item.id] = quest_item
            objective = QuestObjective(description=f"Find and return the {quest_item.name} to {npc.name}", objective_type="fetch", target=str(quest_item.id))

            quest = Quest(
                name=quest_data.get('name', 'A Task'),
                description=quest_data.get('description', 'Complete this task.'),
                quest_type=QuestType.FETCH,
                giver_id=npc.id,
                objectives=[objective],
                status=QuestStatus.ACTIVE,
                target_item_id=quest_item.id,
                location_hint=quest_data.get('location_hint', 'Search the area.'),
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
                        from models import NotableFeature
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
