from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import random

from models import (
    NPC, NPCRole, NPCGoal, NotableFeature, ItemType, Item,
    EventScope, GlobalEvent, Direction,
)
from utils import Colors

llm_logger = __import__('logging').getLogger("llm_responses")


class VirtualDMMixin:

    DM_ACTION_INTERVAL = 15  # Run DM check every N player actions

    _REGIONAL_EVENTS_BY_TERRAIN: Dict[str, List[Tuple[str, str, Dict]]] = {
        "desert": [
            ("Sandstorm", "A wall of sand swallows the horizon. Visibility drops to nothing.", {"stamina_cost": 2.0, "visibility": 0}),
            ("Scorching Heat", "The sun beats down mercilessly. Every step costs more.", {"stamina_cost": 1.5}),
            ("Oasis Discovered", "Word spreads of a fresh water source nearby. Travelers converge.", {"stamina_mod": -0.5}),
            ("Dust Wraiths", "Locals warn of spirits that walk in the dunes at night.", {"danger_mod": 1.5}),
        ],
        "forest": [
            ("Storm", "Thunder rolls through the canopy. Rain turns the paths to mud.", {"stamina_cost": 2.0, "visibility": 0}),
            ("Migration Season", "Herds move through. Hunting is easier but paths are blocked.", {"hunt_mod": 1.5, "stamina_cost": 1.2}),
            ("Blight Spreading", "Dark patches appear on the trees. Animals are fleeing.", {"forage_mod": 0.5, "danger_mod": 1.3}),
            ("Mushroom Bloom", "After the rain, rare fungi cover the forest floor.", {"forage_mod": 2.0}),
            ("Wild Hunt", "Something large is being chased through the woods.", {"danger_mod": 1.8}),
        ],
        "mountain": [
            ("Storm", "A blizzard descends from the peaks. Travel is treacherous.", {"stamina_cost": 2.5, "visibility": 0}),
            ("Rockslide", "A path has been blocked by fallen stone. Detours are necessary.", {"stamina_cost": 1.5}),
            ("Eagle Nesting", "The mountain eagles are aggressive near their nests.", {"danger_mod": 1.4}),
            ("Clear Skies", "Rare calm weather makes the peaks visible from far away.", {"stamina_mod": -0.3}),
        ],
        "swamp": [
            ("Fog", "A thick fog rises from the water. Everything looks the same.", {"visibility": 0, "stamina_cost": 1.3}),
            ("Plague Rumors", "Locals whisper of a sickness spreading through the wetlands.", {"danger_mod": 1.5}),
            ("Flooding", "Heavy rains have raised the water level. Some paths are submerged.", {"stamina_cost": 2.0}),
            ("Fireflies", "Millions of fireflies light the swamp. Beautiful and disorienting.", {"visibility": 0}),
        ],
        "coast": [
            ("Storm", "Waves crash violently against the shore. Ships stay in harbor.", {"stamina_cost": 1.5, "visibility": 0}),
            ("High Tide", "The tide is unusually high. Some shore paths are cut off.", {"stamina_cost": 1.2}),
            ("Shipwreck Salvage", "Locals are picking through wreckage washed ashore.", {"price_mod": 0.85}),
            ("Sea Mist", "A thick mist rolls in from the sea. Sounds carry strangely.", {"visibility": 0}),
        ],
        "city": [
            ("Market Day", "Merchants from distant towns set up stalls. Prices drop.", {"price_mod": 0.75}),
            ("Festival", "Music, food, and laughter fill the streets.", {"stamina_mod": -0.5, "price_mod": 1.1}),
            ("Guard Inspection", "Guards are checking everyone's documents and packs.", {"suspicion": 1.0}),
            ("Thieves Active", "Pickpockets and cutpurses are working the crowds.", {"danger_mod": 1.2}),
            ("Public Execution", "A crowd gathers in the square. The mood is grim.", {"suspicion": 0.8}),
            ("Trade Caravan Arrived", "A large caravan has brought exotic goods. Prices shift.", {"price_mod": 0.9}),
            ("Curfew", "After an incident, guards enforce a dusk-to-dawn curfew.", {"suspicion": 1.5}),
        ],
        "ruins": [
            ("Grave Robbers", "Other scavengers are working the ruins. Competition is fierce.", {"danger_mod": 1.3}),
            ("Unstable Ground", "Recent tremors have weakened the structure. Collapses are possible.", {"danger_mod": 1.4, "stamina_cost": 1.3}),
            ("Strange Lights", "Faint lights flicker from deep within the ruins at night.", {"danger_mod": 1.2}),
            ("Archaeological Expedition", "Scholars have arrived to study the site. They pay for finds.", {"price_mod": 1.2}),
        ],
    }

    _REGIONAL_EVENTS_GENERIC: List[Tuple[str, str, Dict]] = [
        ("Bandit Activity", "Travelers are warned to move in groups. The roads are dangerous.", {"danger_mod": 1.5}),
        ("Wandering Merchant", "An unusual merchant has set up camp nearby with rare wares.", {"price_mod": 0.95}),
        ("Sickness Passing Through", "People are falling ill. Healers are in demand.", {"stamina_cost": 1.1}),
        ("Refugee Influx", "Displaced people are flooding in from somewhere to the east.", {"price_mod": 1.15}),
        ("Bounty Posted", "A bounty board has new postings. Dangerous work, good pay.", {"danger_mod": 1.2}),
        ("Traveling Performers", "A troupe of entertainers has arrived. Spirits are lifted.", {"stamina_mod": -0.4}),
        ("Rumor Mill", "Something has people talking. Information flows freely tonight.", {}),
        ("Eerie Quiet", "Animals have gone silent and locals stay indoors. Something is wrong.", {"danger_mod": 1.3}),
    ]

    async def advance_time(self, minutes: int, model_name: str) -> None:
        """Advance game time and update the world state"""
        if not self.game_state:
            return
        self.game_state.session.game_time.advance_time(minutes)

        # Apply passive recovery bonuses
        from models import CharacterClass
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
        if not self.game_state:
            return

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
        if not self.game_state:
            return

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

    async def apply_persistent_change(self, location_id, action_description: str, model_name: str) -> str:
        """Permanently change a location based on high-impact player action"""
        loc = self.game_state.locations.get(location_id)
        if not loc:
            return "Location not found."

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

    def _get_reactive_events(self, region) -> List[Tuple[str, str, Dict]]:
        """Return events that make sense given what the player has done."""
        if not self.game_state:
            return []
        session = self.game_state.session
        history = " ".join(session.major_decision_history[-10:]).lower()
        completed = [self.game_state.quests[qid].name.lower()
                     for qid in session.player_character.completed_quests
                     if qid in self.game_state.quests]
        completed_text = " ".join(completed)
        reactive = []
        if any(w in history or w in completed_text for w in ("bandit", "kill", "defeat", "cleared")):
            reactive.append(("Grateful Survivors", "Word of your deeds has reached common folk. Some approach with thanks.", {"price_mod": 0.85}))
        if any(w in history or w in completed_text for w in ("antagonist", "dark", "threat")):
            reactive.append(("Tension in the Air", "People sense something is coming. Guards patrol more frequently.", {"suspicion": 0.9, "danger_mod": 1.2}))
        if any(w in history for w in ("transformed", "liberated", "ruined")):
            reactive.append(("Pilgrims Arriving", "Travelers come to witness the changed place you left behind.", {"price_mod": 1.05}))
        if len(completed) >= 3:
            reactive.append(("Your Reputation Precedes You", "People recognize your name. Some are curious; some are wary.", {}))
        return reactive

    async def _update_regional_events(self, model_name: str) -> None:
        """Randomly trigger or end regional events."""
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
                rtype = region.region_type.lower()

                # Terrain-specific pool
                terrain_events: List[Tuple[str, str, Dict]] = []
                for key, events in self._REGIONAL_EVENTS_BY_TERRAIN.items():
                    if key in rtype:
                        terrain_events = events
                        break

                # Reactive events (player-history-driven), checked first
                reactive = self._get_reactive_events(region)

                # Build weighted pool: reactive (weight 3), terrain (weight 2), generic (weight 1)
                pool: List[Tuple[str, str, Dict]] = (
                    reactive * 3 + terrain_events * 2 + self._REGIONAL_EVENTS_GENERIC
                )

                event_name, desc, mods = random.choice(pool)
                region.active_events.append(event_name)
                region.event_modifiers.update(mods)
                if region_id == current_region_id:
                    self.pending_messages.append(f"EVENT in {region.name}: {event_name} — {desc}")
