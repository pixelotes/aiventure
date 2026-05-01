"""
Automated playtest script for AIVenture.
Uses a mock AI provider to run a full game session without requiring a real LLM server.
Reports bugs, crashes, and gameplay observations.
"""
import asyncio
import sys
import traceback
import random
import re
import json
from uuid import uuid4, UUID
from pathlib import Path
from typing import List, Optional, Any, Dict

from ai_provider import AIProvider
from engine import GameEngine
from models import (
    Direction, CharacterType, QuestStatus, ItemType, NPC, NPCGoal,
    LocationType, GeneralLocation, GeneralLocationType, NotableFeature, Item, NPCRole
)

# ============================================================================
# Mock AI Provider
# ============================================================================
class MockAIProvider(AIProvider):
    """Returns pre-crafted responses based on prompt patterns for testing."""

    def __init__(self):
        self.url = "http://mock"
        self.model = "mock-model"
        self.timeout = 10
        self.client = None
        self.call_log: List[Dict[str, str]] = []
        self._location_count = 0

    async def get_available_models(self) -> List[str]:
        return ["mock-model"]

    async def generate_response(self, prompt: str, context: str = "", is_content_generation: bool = False, model_name: Optional[str] = None) -> str:
        self.call_log.append({"prompt": prompt[:200], "is_json": is_content_generation})

        if is_content_generation:
            return self._generate_json_response(prompt)
        else:
            return self._generate_narrative_response(prompt)

    def _generate_json_response(self, prompt: str) -> str:
        p = prompt.lower()

        # World generation
        if "fantasy world" in p or "high-level details" in p or "main quest" in p:
            return '''{
                "name": "Eldoria",
                "description": "A vast land of ancient forests and crumbling ruins where magic still lingers.",
                "theme": "dark_fantasy",
                "lore_summary": "Once united under the Crystal Crown, Eldoria shattered into warring territories after the Great Sundering.",
                "quest_name": "The Shattered Crown",
                "quest_description": "Reassemble the Crystal Crown to restore peace to Eldoria.",
                "player_background": "You are a former blacksmith's apprentice who discovered a fragment of the Crystal Crown in the forge's ashes.",
                "starter_region_type": "Whispering Woods City",
                "goal_region_type": "Crystal Citadel",
                "optional_regions": [
                    {"name": "Port of Silvermist", "region_type": "port_city", "description": "A bustling harbor town where merchants trade exotic wares.", "connected_to": "start"},
                    {"name": "Ruins of Thelkara", "region_type": "ancient_ruins", "description": "Crumbling towers hiding forgotten knowledge and cursed treasures.", "connected_to": "boundary"},
                    {"name": "Greenhollow Countryside", "region_type": "countryside", "description": "Rolling hills with farmsteads and rumored cave systems beneath.", "connected_to": "goal"}
                ]
            }'''

        # Location generation (city)
        if "city district" in p:
            return '''{
                "name": "Market Square",
                "description": "A bustling square lined with colorful merchant stalls and the smell of fresh bread.",
                "short_description": "A busy market area.",
                "general_type": "market_district",
                "atmosphere": "lively and crowded",
                "buildings": [
                    {"name": "The Golden Goblet", "type": "tavern", "description": "A popular drinking establishment."},
                    {"name": "Ironhand Smithy", "type": "smithy", "description": "A forge run by a skilled dwarf."}
                ],
                "npc": {
                    "name": "Greta the Merchant",
                    "description": "A stout woman with a keen eye for valuables.",
                    "race": "human",
                    "role": "shopkeeper",
                    "dialogue_style": "friendly"
                }
            }'''

        # Location generation (wilderness)
        if "generate a location" in p or "location in" in p:
            loc_types = ["forest", "meadow", "clearing", "river", "ruins"]
            loc_type = random.choice(loc_types)
            names = {
                "forest": "Darkwood Thicket",
                "meadow": "Sunlit Meadow",
                "clearing": "Moonstone Clearing",
                "river": "Silver River Crossing",
                "ruins": "Ancient Ruins of Thel"
            }
            self._location_count += 1
            # Guarantee NPC on first location (shopkeeper), second (quest_giver)
            npc_block = ""
            if self._location_count == 1:
                npc_block = ''',
                "npc": {
                    "name": "Greta the Merchant",
                    "description": "A stout woman with a keen eye for valuables.",
                    "race": "human",
                    "role": "shopkeeper",
                    "dialogue_style": "friendly"
                }'''
            elif self._location_count == 2:
                npc_block = ''',
                "npc": {
                    "name": "Old Hermit Finn",
                    "description": "A weathered old man who speaks in riddles.",
                    "race": "human",
                    "role": "quest_giver",
                    "dialogue_style": "cryptic"
                }'''
            return f'''{{
                "name": "{names.get(loc_type, 'Mysterious Glade')}",
                "description": "A {loc_type} area with tall ancient trees and the sound of distant water.",
                "short_description": "A quiet {loc_type}.",
                "general_type": "{loc_type}",
                "atmosphere": "serene yet watchful",
                "notable_features": [
                    {{"name": "Mossy Boulder", "description": "A large boulder covered in luminescent moss."}},
                    {{"name": "Old Signpost", "description": "A weathered wooden signpost pointing in several directions."}}
                ]{npc_block}
            }}'''

        # Dungeon level generation (dynamic quest dungeon)
        if "dungeon" in p and "levels" in p:
            depth_match = re.search(r'(\d+)-level', p)
            depth = int(depth_match.group(1)) if depth_match else 2
            levels = []
            for i in range(depth):
                is_final = (i == depth - 1)
                levels.append({
                    "name": f"Dungeon Level {i+1}",
                    "description": f"A dark {'treasure chamber' if is_final else 'corridor'} deep underground.",
                    "short_description": f"Level {i+1}",
                    "atmosphere": "dark and foreboding",
                    "notable_features": [{"name": f"Stone Alcove {i+1}", "description": "A carved niche in the wall."}],
                    "enemy": {
                        "name": f"Dungeon Creature {i+1}",
                        "description": "A hostile inhabitant of the depths.",
                    } if not is_final else None,
                    "is_final": is_final,
                })
            return json.dumps({"levels": levels})

        # Building interior generation
        if "interior" in p and ("floor" in p or "building" in p):
            floor_match = re.search(r'has (\d+) floor', p)
            num_floors = int(floor_match.group(1)) if floor_match else 1
            floors = []
            for i in range(num_floors):
                npcs = []
                if i == 0:
                    npcs = [
                        {"name": "Bran the Innkeeper", "description": "A burly man with a warm smile.", "race": "human", "role": "innkeeper"},
                        {"name": "Elda", "description": "A young barmaid.", "race": "human", "role": "commoner"},
                    ]
                else:
                    npcs = [{"name": "Weary Traveler", "description": "A hooded figure resting.", "race": "human", "role": "commoner"}]
                floors.append({
                    "name": f"{'Main Hall' if i == 0 else f'Upper Floor {i}'}",
                    "description": f"{'A warm room with a roaring fireplace.' if i == 0 else 'A quiet corridor with guest rooms.'}",
                    "short_description": f"{'A cozy hall.' if i == 0 else 'Guest rooms.'}",
                    "atmosphere": "warm and inviting" if i == 0 else "quiet",
                    "notable_features": [{"name": f"{'Fireplace' if i == 0 else 'Locked Door'}", "description": "An interesting feature."}],
                    "npcs": npcs,
                })
            return json.dumps({"floors": floors})

        # Sub-level generation
        if "sub-level" in p:
            return '''{
                "name": "Hidden Grotto",
                "description": "A damp underground chamber with glowing crystals embedded in the walls.",
                "short_description": "An underground grotto.",
                "atmosphere": "mysterious and damp",
                "notable_features": [
                    {"name": "Crystal Formation", "description": "Clusters of glowing blue crystals."}
                ]
            }'''

        # NPC memory update
        if "summarize the relationship" in p or "summarize:" in p:
            return '{"summary": "The player greeted the NPC warmly. They seem trustworthy.", "sentiment": 0.3}'

        # Rumor generation
        if "rumor" in p:
            return "I heard strange lights appear near the old ruins at midnight."

        # Fetch quest generation
        if "fetch quest" in p:
            return '''{
                "name": "The Lost Amulet",
                "description": "Find the ancient Amulet of Seeing that was lost in the wilderness.",
                "item_name": "Amulet of Seeing",
                "item_description": "A tarnished silver amulet with a cracked gemstone eye.",
                "location_hint": "Look near the old ruins to the east."
            }'''

        # Dynamic quest generation
        if "quest encounter" in p or ("quest type" in p and "fetch" in p.lower()):
            quest_types = ["fetch", "kill", "exploration"]
            qt = random.choice(quest_types)
            return f'''{{
                "quest_type": "{qt}",
                "name": "The Wanderer's Request",
                "description": "A traveling stranger needs help with a dangerous task.",
                "npc_name": "Traveling Stranger",
                "npc_description": "A weary traveler with a haunted look in their eyes.",
                "npc_role": "quest_giver",
                "objective_description": "Help the stranger with their request.",
                "item_name": "Strange Relic",
                "item_description": "A small carved stone pulsing with faint light.",
                "target_location_hint": "Search the nearby area.",
                "reward_gold": 15,
                "reward_xp": 100
            }}'''

        # DM command (process_ai_command) — must be checked before Virtual DM
        if "freeform dm command" in p:
            return json.dumps({
                "narrative": "You rest by the warm fire, feeling strength return to your limbs.",
                "effects": [
                    {"type": "heal", "amount": 30},
                    {"type": "advance_time", "minutes": 60},
                    {"type": "give_gold", "amount": 10},
                    {"type": "spawn_item", "name": "Warm Bread", "description": "Fresh from the fire.", "item_type": "consumable", "value": 3},
                    {"type": "add_feature", "name": "Hidden Alcove", "description": "A small nook behind the fireplace."}
                ]
            })

        # Virtual DM
        if "dungeon master" in p.lower() or "advance the narrative" in p.lower():
            actions = ["premonition", "noop", "discovery"]
            act = random.choice(actions)
            return f'''{{
                "action": "{act}",
                "description": "A cold wind picks up and you sense something watching from the shadows.",
                "dm_note": "Building atmosphere and tension for the player."
            }}'''

        # Plot heartbeat
        if "plot shift" in p or "world event" in p:
            return '{"trigger": false}'

        # Persistent change
        if "permanently changed" in p:
            return '''{
                "new_name": "Scorched Clearing",
                "new_description": "The once-green clearing is now blackened and smoldering.",
                "state_tag": "ruined"
            }'''

        # Item combination
        if "combine" in p:
            return '''{
                "name": "Reinforced Staff",
                "description": "A sturdy staff reinforced with metal bands.",
                "item_type": "weapon",
                "value": 25,
                "rarity": "uncommon"
            }'''

        # Cooking at campfire
        if "cooking at a campfire" in p or ("cook" in p and "ingredients" in p):
            return '''{
                "name": "Hearty Mushroom Stew",
                "description": "A bubbling stew of forest mushrooms and wild berries, fragrant with herbs.",
                "effects": [
                    {"stat": "strength", "bonus": 2, "duration_minutes": 60},
                    {"stat": "constitution", "bonus": 1, "duration_minutes": 60}
                ],
                "heal_amount": 20
            }'''

        # Item on feature
        if "using" in p and "on" in p:
            return '{"success": false, "outcome_description": "Nothing happens when you try that."}'

        # Item lore
        if "lore" in p:
            return "This artifact dates back to the Age of Wonders, when such items were common among scholars."

        # Default fallback
        return '{"result": "ok"}'

    def _generate_narrative_response(self, prompt: str) -> str:
        return "The wind whispers through the trees as you contemplate your next move."


# ============================================================================
# Playtest Runner
# ============================================================================
class PlaytestReport:
    def __init__(self):
        self.bugs: List[Dict[str, str]] = []
        self.crashes: List[Dict[str, str]] = []
        self.warnings: List[str] = []
        self.actions_taken: List[str] = []
        self.observations: List[str] = []

    def bug(self, title: str, detail: str):
        self.bugs.append({"title": title, "detail": detail})

    def crash(self, title: str, detail: str):
        self.crashes.append({"title": title, "detail": detail})

    def warn(self, msg: str):
        self.warnings.append(msg)

    def action(self, msg: str):
        self.actions_taken.append(msg)

    def observe(self, msg: str):
        self.observations.append(msg)

    def print_report(self):
        sep = "=" * 70
        print(f"\n{sep}")
        print("  PLAYTEST REPORT")
        print(sep)

        # Crashes
        print(f"\n  CRASHES ({len(self.crashes)}):")
        if self.crashes:
            for i, c in enumerate(self.crashes, 1):
                print(f"    {i}. [{c['title']}]")
                for line in c['detail'].split('\n'):
                    print(f"       {line}")
        else:
            print("    None!")

        # Bugs
        print(f"\n  BUGS ({len(self.bugs)}):")
        if self.bugs:
            for i, b in enumerate(self.bugs, 1):
                print(f"    {i}. [{b['title']}]")
                for line in b['detail'].split('\n'):
                    print(f"       {line}")
        else:
            print("    None!")

        # Warnings
        print(f"\n  WARNINGS ({len(self.warnings)}):")
        if self.warnings:
            for i, w in enumerate(self.warnings, 1):
                print(f"    {i}. {w}")
        else:
            print("    None!")

        # Summary
        print(f"\n  ACTIONS LOG ({len(self.actions_taken)} actions):")
        for a in self.actions_taken:
            print(f"    - {a}")

        # Observations
        print(f"\n  OBSERVATIONS ({len(self.observations)}):")
        for o in self.observations:
            print(f"    - {o}")

        print(f"\n{sep}")
        total_issues = len(self.crashes) + len(self.bugs) + len(self.warnings)
        print(f"  TOTAL ISSUES: {total_issues} ({len(self.crashes)} crashes, {len(self.bugs)} bugs, {len(self.warnings)} warnings)")
        print(sep)


async def run_playtest():
    report = PlaytestReport()
    mock_ai = MockAIProvider()
    engine = GameEngine(mock_ai)
    model = "mock-model"

    # ====================================================================
    # PHASE 1: Create a new game
    # ====================================================================
    print("\n--- PHASE 1: World Creation ---")
    try:
        report.action("Creating new game...")
        game_state = await engine.create_new_game("TestHero", uuid4(), "PlaytestSession", model)
        report.action("World created successfully!")

        # Verify world structure
        world = game_state.session.world
        report.observe(f"World name: {world.name}")
        report.observe(f"Regions: {len(world.regions)}")
        report.observe(f"Total locations: {len(game_state.locations)}")
        report.observe(f"Total characters: {len(game_state.characters)}")
        report.observe(f"Total items: {len(game_state.items)}")

        player = game_state.session.player_character
        if not player:
            report.crash("No player character", "player_character is None after game creation")
            return report

        report.observe(f"Player: {player.name} at location {player.current_location_id}")
        report.observe(f"Player gold: {player.currency}")
        report.observe(f"Active quests: {len(player.active_quests)}")

        # Check starting location has the Pass Permit
        start_loc = engine.get_current_location()
        items_in_loc = [game_state.items[iid].name for iid in start_loc.items if iid in game_state.items]
        if "Pass Permit" in items_in_loc:
            report.observe("Pass Permit found in starting location - OK")
        else:
            report.bug("Missing Pass Permit", "Pass Permit not found in starting location items")

    except Exception as e:
        report.crash("Game creation failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return report

    # ====================================================================
    # PHASE 2: Look around and explore
    # ====================================================================
    print("\n--- PHASE 2: Exploration ---")
    try:
        # Get location info
        loc = engine.get_current_location()
        report.action(f"Looking at: {loc.name} ({loc.coordinates.x}, {loc.coordinates.y})")
        report.observe(f"Location type: {loc.location_type}")
        report.observe(f"Features: {[f.name for f in loc.notable_features]}")
        report.observe(f"Connections: {[(c.direction.value, c.is_visible) for c in loc.connections]}")

        loc.visit_count += 1  # Simulate first visit

    except Exception as e:
        report.crash("Look failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 3: Pick up items
    # ====================================================================
    print("\n--- PHASE 3: Item Pickup ---")
    try:
        # Try picking up the Pass Permit
        result = engine.pickup_item("Pass Permit")
        report.action(f"Pick up Pass Permit: {result}")

        if player.inventory:
            report.observe(f"Inventory after pickup: {[game_state.items[iid].name for iid in player.inventory if iid in game_state.items]}")
        else:
            report.bug("Pickup failed", "Inventory empty after picking up Pass Permit")

        # Try picking up non-existent item
        result = engine.pickup_item("Nonexistent Item")
        report.action(f"Pick up nonexistent: {result}")

    except Exception as e:
        report.crash("Item pickup failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 4: Movement
    # ====================================================================
    print("\n--- PHASE 4: Movement ---")
    try:
        # Try all directions from starting position
        for direction in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            conn = next((c for c in engine.get_current_location().connections if c.direction == direction), None)
            if conn:
                success, msg = await engine.move_player(direction, model)
                if "TRAVEL_CONFIRM" in msg:
                    report.action(f"Move {direction.value}: Region travel prompt - {msg}")
                    # Try confirmed travel
                    success, msg = await engine.move_player(direction, model, confirmed=True)
                    report.action(f"Move {direction.value} (confirmed): success={success}, msg={msg}")
                else:
                    report.action(f"Move {direction.value}: success={success}, msg={msg}")
                    if success:
                        new_loc = engine.get_current_location()
                        report.observe(f"Now at: {new_loc.name} ({new_loc.coordinates.x}, {new_loc.coordinates.y})")
                        break
            else:
                report.action(f"Move {direction.value}: No connection")
    except Exception as e:
        report.crash("Movement failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Try to move back
    try:
        for direction in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            conn = next((c for c in engine.get_current_location().connections if c.direction == direction), None)
            if conn and conn.target_location_id in game_state.locations:
                success, msg = await engine.move_player(direction, model)
                if success:
                    report.action(f"Move back {direction.value}: success")
                    break
    except Exception as e:
        report.crash("Move back failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 5: NPC Interaction
    # ====================================================================
    print("\n--- PHASE 5: NPC Interaction ---")
    try:
        npcs = [c for c in game_state.characters.values() if isinstance(c, NPC)]
        report.observe(f"Total NPCs in world: {len(npcs)}")

        if npcs:
            npc = npcs[0]
            report.action(f"Found NPC: {npc.name} (role: {npc.role}, location: {npc.current_location_id})")

            # Move player to NPC location for interaction
            old_loc = player.current_location_id
            player.current_location_id = npc.current_location_id

            # Test AI context building
            context = engine.build_context_for_ai(target_npc=npc)
            report.observe(f"AI context length: {len(context)} chars")

            # Test rumor generation
            try:
                rumor = await engine.generate_rumor(npc, model)
                report.action(f"Rumor from {npc.name}: {rumor[:80]}...")
            except Exception as e:
                report.bug("Rumor generation failed", f"{type(e).__name__}: {e}")

            # Test quest generation
            try:
                quest = await engine.generate_quest(npc, model)
                if quest:
                    report.action(f"Quest from {npc.name}: {quest.name}")
                    report.observe(f"Quest target item: {quest.target_item_id}")
                else:
                    report.warn(f"Quest generation returned None for {npc.name}")
            except Exception as e:
                report.bug("Quest generation failed", f"{type(e).__name__}: {e}")

            # Test NPC memory update
            try:
                await engine.update_npc_memory(npc, "Hello there!", "Welcome, traveler.", model)
                report.action(f"NPC memory updated: {npc.interaction_summary[:80] if npc.interaction_summary else 'empty'}")
            except Exception as e:
                report.bug("NPC memory update failed", f"{type(e).__name__}: {e}")

            # Test services
            if npc.services_offered:
                report.observe(f"NPC services: {[s.name for s in npc.services_offered]}")

            # Restore player position
            player.current_location_id = old_loc
        else:
            report.warn("No NPCs generated in the world - cannot test NPC interactions")
    except Exception as e:
        report.crash("NPC interaction failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 6: Combat
    # ====================================================================
    print("\n--- PHASE 6: Combat ---")
    try:
        # Find any NPC to fight (or create a test dummy)
        npcs = [c for c in game_state.characters.values() if isinstance(c, NPC)]
        if npcs:
            target = npcs[-1]  # Use last NPC
            report.action(f"Initiating combat with {target.name}")

            # Attack
            result = await engine.execute_combat_turn(target.id, "attack")
            report.action(f"Attack: player={result['player_msg']}, enemy={result['enemy_msg']}")

            if not result['victory']:
                # Parry
                result = await engine.execute_combat_turn(target.id, "parry")
                report.action(f"Parry: player={result['player_msg']}, enemy={result['enemy_msg']}")

            if not result['victory'] and not result['fled']:
                # Flee
                result = await engine.execute_combat_turn(target.id, "flee")
                report.action(f"Flee: player={result['player_msg']}, fled={result['fled']}")

            report.observe(f"Player health after combat: {player.stats.health}/{player.stats.max_health}")
        else:
            report.warn("No NPCs to test combat with")
    except Exception as e:
        report.crash("Combat failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 7: Time and World State
    # ====================================================================
    print("\n--- PHASE 7: Time & World State ---")
    try:
        gt = game_state.session.game_time
        report.observe(f"Time before advance: Day {gt.day}, {gt.hour:02d}:{gt.minute:02d} ({gt.time_of_day.value})")

        await engine.advance_time(120, model)  # Advance 2 hours
        report.action("Advanced time by 120 minutes")
        report.observe(f"Time after advance: Day {gt.day}, {gt.hour:02d}:{gt.minute:02d} ({gt.time_of_day.value})")

        # Check stamina recovery
        report.observe(f"Stamina after time: {player.stats.stamina}/{player.stats.max_stamina}")

    except Exception as e:
        report.crash("Time advance failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 8: Quest Progress
    # ====================================================================
    print("\n--- PHASE 8: Quest System ---")
    try:
        active = [game_state.quests[qid] for qid in player.active_quests if qid in game_state.quests]
        report.observe(f"Active quests: {len(active)}")
        for q in active:
            report.observe(f"  Quest: {q.name} ({q.quest_type.value}) - Status: {q.status.value}")
            for obj in q.objectives:
                report.observe(f"    Objective: {obj.description} - Completed: {obj.completed}")

        # Test quest progress check
        msgs = engine.check_quest_progress()
        report.action(f"Quest progress check: {msgs}")

    except Exception as e:
        report.crash("Quest system failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 9: Save/Load
    # ====================================================================
    print("\n--- PHASE 9: Save/Load ---")
    save_path = Path("/tmp/aiventure_playtest_save.json")

    # Test Save
    try:
        saved = await engine.save_game(save_path)
        if saved:
            report.action(f"Game saved to {save_path}")
            report.observe(f"Save file size: {save_path.stat().st_size} bytes")
        else:
            report.bug("Save returned False", "engine.save_game returned False without exception")
    except Exception as e:
        report.crash("Save failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test Load
    try:
        loaded = await engine.load_game(save_path)
        if loaded:
            report.action("Game loaded successfully")
        else:
            report.bug("Load returned False", "engine.load_game returned False without exception")
    except Exception as e:
        report.crash("Load failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Cleanup
    if save_path.exists():
        save_path.unlink()

    # ====================================================================
    # PHASE 10: Edge Cases & Validation
    # ====================================================================
    print("\n--- PHASE 10: Edge Cases ---")

    # Test equip with no equippable items
    try:
        result = engine.equip_item("nonexistent")
        report.action(f"Equip nonexistent: {result}")
    except Exception as e:
        report.crash("Equip edge case failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test unequip with nothing equipped
    try:
        result = engine.unequip_item("nothing")
        report.action(f"Unequip nothing: {result}")
    except Exception as e:
        report.crash("Unequip edge case failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test finding items
    try:
        result = engine.find_item_in_inventory("nonexistent")
        report.action(f"Find nonexistent in inventory: {result}")

        result = engine.find_item_in_location("nonexistent")
        report.action(f"Find nonexistent in location: {result}")
    except Exception as e:
        report.crash("Item search edge case failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # Test move to blocked direction
    try:
        success, msg = await engine.move_player(Direction.UP, model)
        report.action(f"Move UP (should fail): success={success}, msg={msg}")
    except Exception as e:
        report.crash("Blocked move failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 11: Building Entry
    # ====================================================================
    print("\n--- PHASE 11: Building Entry ---")
    try:
        found_building = False
        for loc_id, loc in game_state.locations.items():
            enterable = [f for f in loc.notable_features if f.metadata.get("enterable")]
            if enterable:
                player.current_location_id = loc_id
                feature = enterable[0]
                report.action(f"Entering building: {feature.name}")
                success, msg = await engine.enter_building(feature.name, model)
                report.action(f"Enter result: success={success}, msg={msg}")
                if success:
                    interior = engine.get_current_location()
                    report.observe(f"Inside: {interior.name} (type: {interior.location_type})")
                    out_conn = next((c for c in interior.connections if c.direction == Direction.OUT), None)
                    if out_conn:
                        report.observe("OUT connection exists — OK")
                    else:
                        report.bug("Missing OUT connection", "Building interior has no OUT exit")
                    npcs_inside = [c for c in engine.game_state.characters.values() if c.current_location_id == interior.id and getattr(c, 'character_type', None) != CharacterType.PLAYER]
                    report.observe(f"NPCs inside: {[n.name for n in npcs_inside]}")
                    # Test going out
                    if out_conn:
                        success2, msg2 = await engine.move_player(Direction.OUT, model)
                        report.action(f"Go out: success={success2}, msg={msg2}")
                found_building = True
                break
        if not found_building:
            report.warn("No enterable buildings found to test")
    except Exception as e:
        report.crash("Building entry failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 13: Cooking System
    # ====================================================================
    print("\n--- PHASE 13: Cooking System ---")
    try:
        # Refresh references after save/load in Phase 9
        game_state = engine.game_state
        player = game_state.session.player_character

        # Use any location and place a campfire on it
        cook_loc = engine.get_current_location()
        player.current_location_id = cook_loc.id

        # Place a campfire if none exists
        has_campfire = any(f.metadata.get("campfire") for f in cook_loc.notable_features)
        if not has_campfire:
            cook_loc.notable_features.append(NotableFeature(
                name="Test Campfire",
                detailed_description="A ring of stones with charred wood.",
                metadata={"campfire": True},
            ))
            report.action("Placed test campfire at location")
        else:
            report.action("Campfire already exists at location")

        # Create food ingredients in player inventory
        food_ids = []
        for food_name in ["Wild Berries", "Forest Mushrooms"]:
            item = engine._create_catalog_item(food_name)
            if item:
                player.inventory.append(item.id)
                food_ids.append(item.id)
                report.action(f"Added {food_name} to inventory")
            else:
                report.bug("Missing food item", f"Could not create {food_name} from catalog")

        if len(food_ids) >= 2:
            # Cook the items
            success, msg, meal = await engine.cook_items(food_ids, model)
            report.action(f"Cook result: success={success}, msg={msg}")
            if success and meal:
                report.observe(f"Meal created: {meal.name}, effects: {meal.use_effects}")

                # Record stats before eating
                old_health = player.stats.health
                old_strength = player.stats.strength

                # Eat the meal
                effect_desc = engine.apply_item_effects(player, meal)
                report.action(f"Ate {meal.name}: {effect_desc}")

                # Verify heal applied
                if player.stats.health > old_health or old_health == player.stats.max_health:
                    report.observe(f"Health after eating: {player.stats.health}/{player.stats.max_health} — OK")
                else:
                    report.bug("Heal not applied", f"Health unchanged: {old_health} → {player.stats.health}")

                # Verify buff applied
                if player.temporary_effects:
                    report.observe(f"Active temporary effects: {len(player.temporary_effects)}")
                    for te in player.temporary_effects:
                        report.observe(f"  Buff: +{te.get('bonus')} {te.get('stat')} ({te.get('remaining_minutes')}min from {te.get('source')})")

                    # Verify stat increased
                    if player.stats.strength > old_strength:
                        report.observe(f"Strength buffed: {old_strength} → {player.stats.strength} — OK")

                    # Test effect expiry
                    msgs = engine.expire_temporary_effects(player, 9999)
                    report.action(f"Expired all effects: {msgs}")
                    if len(player.temporary_effects) == 0:
                        report.observe("All temporary effects expired — OK")
                    else:
                        report.bug("Effects not expired", f"Still have {len(player.temporary_effects)} effects after 9999 min")
                else:
                    report.warn("No temporary effects after eating meal — cooking buffs may not have applied")
            elif not success:
                report.bug("Cooking failed", f"cook_items returned False: {msg}")
        else:
            report.warn("Could not create enough food ingredients for cooking test")
    except Exception as e:
        report.crash("Cooking system failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 14: Environmental Puzzles
    # ====================================================================
    print("\n--- PHASE 14: Environmental Puzzles ---")
    try:
        # Use fresh references
        game_state = engine.game_state
        player = game_state.session.player_character
        puzzle_loc = engine.get_current_location()
        player.current_location_id = puzzle_loc.id

        # Create a puzzle
        puzzle_feature = engine.create_puzzle_feature(puzzle_loc)
        report.action(f"Created puzzle: {puzzle_feature.name} (type: {puzzle_feature.metadata.get('puzzle_type')})")
        report.observe(f"Puzzle metadata: solved={puzzle_feature.metadata.get('solved')}, "
                      f"accepted={puzzle_feature.metadata.get('accepted_item_types')}, "
                      f"hint={puzzle_feature.metadata.get('solution_hint')}")

        # Verify reward was created
        reward_id_str = puzzle_feature.metadata.get("reward_item_id")
        if reward_id_str:
            reward_item = game_state.items.get(UUID(reward_id_str))
            if reward_item:
                report.observe(f"Puzzle reward: {reward_item.name} (rarity: {reward_item.rarity.value})")
            else:
                report.bug("Missing puzzle reward", f"Reward item {reward_id_str} not in game_state.items")
        else:
            report.bug("No reward_item_id", "Puzzle metadata missing reward_item_id")

        # Try solving with wrong item type (BOOK is never in any puzzle's accepted types)
        wrong_item = Item(name="Test Old Book", description="A dusty tome.", item_type=ItemType.BOOK, value=1)
        game_state.items[wrong_item.id] = wrong_item
        player.inventory.append(wrong_item.id)
        success_wrong, msg_wrong = engine.attempt_solve_puzzle(puzzle_feature, wrong_item)
        report.action(f"Wrong item: success={success_wrong}, msg={msg_wrong[:80]}")
        if not success_wrong:
            report.observe("Wrong item type correctly rejected — OK")
        else:
            report.bug("Puzzle accepted wrong item", f"Book item should not solve puzzle")

        # Solve with correct item type
        accepted_types = puzzle_feature.metadata.get("accepted_item_types", [])
        solve_item = None
        if "material" in accepted_types:
            solve_item = Item(name="Test Stone", description="A stone.", item_type=ItemType.MATERIAL, value=1)
        elif "consumable" in accepted_types:
            solve_item = Item(name="Test Berry", description="A berry.", item_type=ItemType.CONSUMABLE, consumable=True, value=1)
        elif "tool" in accepted_types:
            solve_item = Item(name="Test Rope", description="A rope.", item_type=ItemType.TOOL, value=1)
        elif "weapon" in accepted_types:
            solve_item = Item(name="Test Blade", description="A blade.", item_type=ItemType.WEAPON, value=5)

        if solve_item:
            game_state.items[solve_item.id] = solve_item
            player.inventory.append(solve_item.id)
            old_gold = player.currency.get("gold", 0)
            old_xp = player.experience

            success_right, msg_right = engine.attempt_solve_puzzle(puzzle_feature, solve_item)
            report.action(f"Correct item: success={success_right}, msg={msg_right[:120]}")

            if success_right:
                report.observe("Puzzle solved successfully — OK")
                if puzzle_feature.metadata.get("solved"):
                    report.observe("Puzzle marked as solved — OK")
                else:
                    report.bug("Puzzle not marked solved", "metadata['solved'] still False after success")

                new_gold = player.currency.get("gold", 0)
                if new_gold > old_gold:
                    report.observe(f"Gold reward: {old_gold} → {new_gold} — OK")

                if player.experience > old_xp:
                    report.observe(f"XP reward: {old_xp} → {player.experience} — OK")

                # Check reward dropped on ground
                reward_in_loc = any(iid == UUID(reward_id_str) for iid in puzzle_loc.items) if reward_id_str else False
                if reward_in_loc:
                    report.observe("Reward item on ground — OK")
                else:
                    report.warn("Reward item not found on location ground (may have been placed elsewhere)")

                # Try solving again — should fail
                success_again, msg_again = engine.attempt_solve_puzzle(puzzle_feature, solve_item)
                if not success_again:
                    report.observe("Cannot re-solve — OK")
                else:
                    report.bug("Puzzle re-solvable", "Solved puzzle accepted another solution")
            else:
                report.bug("Correct item rejected", f"Item type {solve_item.item_type.value} should be in {accepted_types}")
        else:
            report.warn("Could not create matching item for puzzle test")

    except Exception as e:
        report.crash("Puzzle system failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 15: NPC Tick System
    # ====================================================================
    print("\n--- PHASE 15: NPC Tick System ---")
    try:
        game_state = engine.game_state
        player = game_state.session.player_character
        tick_loc = engine.get_current_location()
        player.current_location_id = tick_loc.id

        # Find a different location for the hunter to start at
        other_loc_id = None
        for lid in game_state.locations:
            if lid != tick_loc.id:
                other_loc_id = lid
                break

        # -- Test 1: Hunter tracks and attacks --
        hunter = NPC(
            name="Test Assassin",
            description="A shadowy figure.",
            role=NPCRole.COMMONER,
            current_location_id=other_loc_id or tick_loc.id,
            home_location_id=other_loc_id or tick_loc.id,
            mood=0.0,
            goal=NPCGoal.ATTACK_PLAYER,
            max_ticks=20,
            level=1,
        )
        hunter.base_stats = hunter.stats.model_copy()
        game_state.characters[hunter.id] = hunter
        report.action(f"Spawned hunter at different location (goal=ATTACK_PLAYER)")

        engine._tick_npcs()
        if hunter.current_location_id == player.current_location_id:
            report.observe("Hunter tracked player — OK")
        else:
            report.bug("Hunter didn't track", "Hunter should move to player location on tick")

        # Tick again — should initiate combat
        engine.in_combat = False
        engine.combat_opponents = []
        engine._tick_npcs()
        if engine.in_combat and engine.combat_opponents and engine.combat_opponents[0].id == hunter.id:
            report.observe("Hunter initiated combat — OK")
        else:
            report.bug("Hunter didn't attack", f"in_combat={engine.in_combat}, opponents={engine.combat_opponents}")

        # Clean up
        engine.in_combat = False
        engine.combat_opponents = []
        if hunter.id in game_state.characters:
            del game_state.characters[hunter.id]
        engine.pending_messages.clear()

        # -- Test 2: Messenger delivers and despawns --
        messenger = NPC(
            name="Mysterious Courier",
            description="A cloaked messenger.",
            role=NPCRole.COMMONER,
            current_location_id=other_loc_id or tick_loc.id,
            home_location_id=other_loc_id or tick_loc.id,
            mood=0.5,
            goal=NPCGoal.DELIVER_MESSAGE,
            goal_data={"message": "The dark lord sends his regards."},
            is_transient=True,
            level=1,
        )
        messenger.base_stats = messenger.stats.model_copy()
        game_state.characters[messenger.id] = messenger
        messenger_id = messenger.id
        report.action("Spawned transient messenger")

        # Tick 1: messenger moves to player
        engine._tick_npcs()
        if messenger.current_location_id == player.current_location_id:
            report.observe("Messenger approached player — OK")
        else:
            report.bug("Messenger didn't move", "Should approach player location")

        # Tick 2: delivers message
        engine._tick_npcs()
        msg_delivered = any("dark lord" in m for m in engine.pending_messages)
        if msg_delivered:
            report.observe("Message delivered — OK")
        else:
            report.bug("Message not delivered", f"pending_messages: {engine.pending_messages}")

        # Tick 3: despawns
        engine._tick_npcs()
        if messenger_id not in game_state.characters:
            report.observe("Transient messenger despawned — OK")
        else:
            report.bug("Messenger not despawned", "Transient NPC should despawn after delivering message")

        engine.pending_messages.clear()

        # -- Test 3: Follower moves with player --
        follower = NPC(
            name="Loyal Squire",
            description="A young follower.",
            role=NPCRole.COMPANION,
            current_location_id=tick_loc.id,
            home_location_id=tick_loc.id,
            mood=0.9,
            goal=NPCGoal.FOLLOW_PLAYER,
            level=1,
        )
        follower.base_stats = follower.stats.model_copy()
        game_state.characters[follower.id] = follower

        # Move player to different location
        if other_loc_id:
            player.current_location_id = other_loc_id
            engine._tick_npcs()
            if follower.current_location_id == other_loc_id:
                report.observe("Follower moved with player — OK")
            else:
                report.bug("Follower didn't follow", "Should match player location")
            player.current_location_id = tick_loc.id

        if follower.id in game_state.characters:
            del game_state.characters[follower.id]
        engine.pending_messages.clear()

    except Exception as e:
        report.crash("NPC tick system failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 16: Corpse & XP System
    # ====================================================================
    print("\n--- PHASE 16: Corpse & XP ---")
    try:
        game_state = engine.game_state
        player = game_state.session.player_character
        corpse_loc = engine.get_current_location()
        player.current_location_id = corpse_loc.id

        # Create a test enemy with inventory
        enemy_item = Item(name="Dark Blade", description="A wicked sword.", item_type=ItemType.WEAPON, value=30)
        game_state.items[enemy_item.id] = enemy_item

        test_enemy = NPC(
            name="Test Bandit",
            description="A rough-looking thug.",
            role=NPCRole.COMMONER,
            current_location_id=corpse_loc.id,
            home_location_id=corpse_loc.id,
            mood=0.0,
            goal=NPCGoal.ATTACK_PLAYER,
            level=2,
        )
        test_enemy.stats.health = 50
        test_enemy.stats.max_health = 50
        test_enemy.inventory.append(enemy_item.id)
        test_enemy.currency["gold"] = 15
        test_enemy.base_stats = test_enemy.stats.model_copy()
        game_state.characters[test_enemy.id] = test_enemy

        old_xp = player.experience
        old_level = player.level
        old_gold = player.currency.get("gold", 0)
        features_before = len(corpse_loc.notable_features)

        # Simulate killing the enemy
        engine.handle_combat_reward(test_enemy)
        del game_state.characters[test_enemy.id]

        # Verify XP message
        xp_msgs = [m for m in engine.pending_messages if "Gained" in m and "XP" in m]
        if xp_msgs:
            report.observe(f"XP message: {xp_msgs[0]} — OK")
        else:
            report.bug("No XP message", f"Expected 'Gained X XP' in pending_messages")

        # Verify XP increased (may wrap around if level-up consumed XP)
        leveled_up = player.level > old_level
        if player.experience > old_xp or leveled_up:
            if leveled_up:
                report.observe(f"XP triggered level-up: Lvl {old_level}→{player.level}, XP now {player.experience} — OK")
            else:
                report.observe(f"XP increased: {old_xp} → {player.experience} — OK")
        else:
            report.bug("XP not increased", f"Still {player.experience}")

        # Verify gold looted
        new_gold = player.currency.get("gold", 0)
        if new_gold > old_gold:
            report.observe(f"Gold looted: {old_gold} → {new_gold} — OK")
        else:
            report.warn("No gold looted (enemy may have had 0)")

        # Verify corpse created
        corpse_features = [f for f in corpse_loc.notable_features if f.metadata.get("corpse")]
        if corpse_features:
            corpse = corpse_features[-1]
            report.observe(f"Corpse created: {corpse.name} — OK")
            if corpse.contained_items:
                report.observe(f"Corpse has {len(corpse.contained_items)} lootable item(s) — OK")
            else:
                report.bug("Empty corpse", "Corpse should contain enemy's inventory")
        else:
            report.bug("No corpse", "NotableFeature with corpse metadata not created")

        engine.pending_messages.clear()

    except Exception as e:
        report.crash("Corpse/XP system failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 17: DM Effects System
    # ====================================================================
    print("\n--- PHASE 17: DM Effects System ---")
    try:
        game_state = engine.game_state
        player = game_state.session.player_character
        dm_loc = engine.get_current_location()
        player.current_location_id = dm_loc.id
        engine.pending_messages.clear()

        # Damage the player so heal has room to work
        player.stats.health = max(1, player.stats.health - 50)
        old_health = player.stats.health
        old_gold = player.currency.get("gold", 0)
        old_hour = game_state.session.game_time.hour
        old_items_count = len(dm_loc.items)
        old_features_count = len(dm_loc.notable_features)

        # Run DM command
        context = engine.build_context_for_ai()
        narrative = await engine.process_ai_command("rest by the fire", context, model)
        report.action(f"DM command result: {narrative[:80]}")

        if narrative and len(narrative) > 5:
            report.observe("Narrative returned — OK")
        else:
            report.bug("No narrative", f"Got: '{narrative}'")

        # Verify heal
        if player.stats.health > old_health:
            report.observe(f"Heal applied: {old_health} → {player.stats.health} — OK")
        else:
            report.bug("Heal not applied", f"Health still {player.stats.health}")

        # Verify time advanced
        new_hour = game_state.session.game_time.hour
        if new_hour != old_hour:
            report.observe(f"Time advanced: hour {old_hour} → {new_hour} — OK")
        else:
            report.bug("Time not advanced", "Hour unchanged")

        # Verify gold given
        new_gold = player.currency.get("gold", 0)
        if new_gold > old_gold:
            report.observe(f"Gold given: {old_gold} → {new_gold} — OK")
        else:
            report.bug("Gold not given", f"Still {new_gold}")

        # Verify item spawned
        if len(dm_loc.items) > old_items_count:
            new_item_id = dm_loc.items[-1]
            new_item = game_state.items.get(new_item_id)
            if new_item:
                report.observe(f"Item spawned: {new_item.name} — OK")
            else:
                report.bug("Spawned item not in game_state", str(new_item_id))
        else:
            report.bug("No item spawned", "Location items unchanged")

        # Verify feature added
        if len(dm_loc.notable_features) > old_features_count:
            new_feat = dm_loc.notable_features[-1]
            report.observe(f"Feature added: {new_feat.name} — OK")
        else:
            report.bug("No feature added", "Notable features unchanged")

        # Verify DM messages in pending
        dm_msgs = [m for m in engine.pending_messages if m.startswith("DM:")]
        report.observe(f"DM effect messages: {len(dm_msgs)}")

        engine.pending_messages.clear()

        # --- Guardrail tests ---
        # Damage can't kill
        player.stats.health = 5
        engine._apply_ai_effect({"type": "damage", "amount": 9999})
        if player.stats.health >= 1:
            report.observe(f"Damage guardrail: HP={player.stats.health} (min 1) — OK")
        else:
            report.bug("Damage killed player", f"HP={player.stats.health}")

        # Gold cap
        old_gold = player.currency.get("gold", 0)
        engine._apply_ai_effect({"type": "give_gold", "amount": 9999})
        gained = player.currency.get("gold", 0) - old_gold
        if gained <= 50:
            report.observe(f"Gold cap: gained {gained} (max 50) — OK")
        else:
            report.bug("Gold cap exceeded", f"Gained {gained}")

        # Invalid effect type
        result = engine._apply_ai_effect({"type": "fly_to_moon"})
        if result is None:
            report.observe("Invalid effect ignored — OK")
        else:
            report.bug("Invalid effect not ignored", f"Got: {result}")

        # Restore health for subsequent phases
        player.stats.health = player.stats.max_health

    except Exception as e:
        report.crash("DM effects failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 19: Hunt & Forage
    # ====================================================================
    print("\n--- PHASE 19: Hunt & Forage ---")
    try:
        game_state = engine.game_state
        player = game_state.session.player_character

        # Find a GeneralLocation (not a building) from the locations dict
        hunt_loc = None
        for loc_obj in game_state.locations.values():
            if isinstance(loc_obj, GeneralLocation):
                hunt_loc = loc_obj
                break
        if not hunt_loc:
            report.bug("No GeneralLocation found for hunt test", "Need a GeneralLocation to set terrain")
            raise Exception("No GeneralLocation")
        hunt_loc.general_type = GeneralLocationType.FOREST
        player.current_location_id = hunt_loc.id
        report.action(f"Set location to forest type for hunting tests")

        # Test hunt — should cost stamina and advance time
        player.stats.stamina = player.stats.max_stamina
        old_stamina = player.stats.stamina
        old_hour = game_state.session.game_time.hour
        old_inv_size = len(player.inventory)

        # Run hunt multiple times to get at least one success
        hunt_success = False
        for _ in range(20):
            player.stats.stamina = player.stats.max_stamina
            success, msg = engine.hunt()
            if success and "obtained" in msg:
                hunt_success = True
                break

        if hunt_success:
            report.action(f"Hunt success: {msg}")
            if len(player.inventory) <= old_inv_size:
                report.bug("Hunt didn't add item to inventory", f"Inventory size unchanged after successful hunt")
        else:
            report.warning("Hunt never succeeded in 20 attempts (possible but unlikely at 70% chance)")

        # Verify stamina cost
        player.stats.stamina = player.stats.max_stamina
        engine.hunt()
        stamina_after = player.stats.stamina
        expected = player.stats.max_stamina - engine.HUNT_STAMINA_COST
        if stamina_after != expected:
            report.bug("Hunt stamina cost wrong", f"Expected {expected}, got {stamina_after}")
        else:
            report.action(f"Hunt stamina cost correct: {engine.HUNT_STAMINA_COST}")

        # Test forage
        player.stats.stamina = player.stats.max_stamina
        forage_success = False
        for _ in range(20):
            player.stats.stamina = player.stats.max_stamina
            success, msg = engine.forage()
            if success and "found" in msg:
                forage_success = True
                break

        if forage_success:
            report.action(f"Forage success: {msg}")
        else:
            report.warning("Forage never succeeded in 20 attempts (possible but unlikely at 85% chance)")

        # Verify forage stamina cost
        player.stats.stamina = player.stats.max_stamina
        engine.forage()
        stamina_after = player.stats.stamina
        expected = player.stats.max_stamina - engine.FORAGE_STAMINA_COST
        if stamina_after != expected:
            report.bug("Forage stamina cost wrong", f"Expected {expected}, got {stamina_after}")
        else:
            report.action(f"Forage stamina cost correct: {engine.FORAGE_STAMINA_COST}")

        # Test invalid terrain — city location should reject hunt/forage
        hunt_loc.general_type = GeneralLocationType.CITY_CENTER
        success, msg = engine.hunt()
        if success:
            report.bug("Hunt succeeded in city", "Should return False for city terrain")
        else:
            report.action(f"Hunt correctly rejected in city: {msg}")
        success, msg = engine.forage()
        if success:
            report.bug("Forage succeeded in city", "Should return False for city terrain")
        else:
            report.action(f"Forage correctly rejected in city: {msg}")

        # Test insufficient stamina
        hunt_loc.general_type = GeneralLocationType.FOREST
        player.stats.stamina = 0
        success, msg = engine.hunt()
        if success:
            report.bug("Hunt succeeded with 0 stamina", "Should reject when too tired")
        else:
            report.action(f"Hunt correctly rejected with no stamina: {msg}")
        success, msg = engine.forage()
        if success:
            report.bug("Forage succeeded with 0 stamina", "Should reject when too tired")
        else:
            report.action(f"Forage correctly rejected with no stamina: {msg}")

        # Restore
        player.stats.stamina = player.stats.max_stamina
        hunt_loc.general_type = GeneralLocationType.FOREST

    except Exception as e:
        report.crash("Hunt/forage failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 20: Surprise Examine Events
    # ====================================================================
    print("\n--- PHASE 20: Surprise Examine Events ---")
    try:
        game_state = engine.game_state
        player = game_state.session.player_character
        loc = engine.get_current_location()
        player.stats.health = player.stats.max_health

        # Create a plain feature (no special metadata)
        plain_feature = NotableFeature(
            name="Mysterious Stone Pillar",
            detailed_description="A weathered stone pillar covered in moss.",
            metadata={},
        )
        loc.notable_features.append(plain_feature)

        # Roll for surprise until we get one (force many attempts)
        got_surprise = False
        old_gold = player.currency.get("gold", 0)
        old_health = player.stats.health
        old_items = len(loc.items)
        old_chars = len(game_state.characters)

        for attempt in range(200):
            # Reset the feature for each attempt
            test_feature = NotableFeature(
                name=f"Test Pillar {attempt}",
                detailed_description="A weathered stone pillar.",
                metadata={},
            )
            result = engine.roll_examine_surprise(test_feature)
            if result:
                got_surprise = True
                report.action(f"Surprise triggered on attempt {attempt + 1}: {result[:80]}...")
                # Verify feature got marked
                if not test_feature.metadata.get("examined"):
                    report.bug("Feature not marked as examined", "metadata['examined'] should be True")
                if not test_feature.metadata.get("surprise_event"):
                    report.bug("Feature not marked with surprise_event", "metadata['surprise_event'] should be True")
                break
            # Even without surprise, examined should be set
            if not test_feature.metadata.get("examined"):
                report.bug("Feature not marked examined on no-surprise", "metadata['examined'] should be True even without surprise")
                break

        if not got_surprise:
            report.warning("No surprise event in 200 attempts (extremely unlikely at 20% chance)")

        # Verify special features are skipped
        puzzle_feature = NotableFeature(name="Puzzle", metadata={"puzzle": True})
        campfire_feature = NotableFeature(name="Campfire", metadata={"campfire": True})
        corpse_feature = NotableFeature(name="Corpse", metadata={"corpse": True})
        already_examined = NotableFeature(name="Old", metadata={"examined": True})

        for feat, label in [(puzzle_feature, "puzzle"), (campfire_feature, "campfire"),
                            (corpse_feature, "corpse"), (already_examined, "already examined")]:
            result = engine.roll_examine_surprise(feat)
            if result is not None:
                report.bug(f"Surprise triggered on {label} feature", "Special features should be skipped")
            else:
                report.action(f"Surprise correctly skipped {label} feature")

        # Test each event type by checking state changes across many rolls
        gold_event = False
        item_event = False
        trap_event = False
        ambush_event = False

        for i in range(500):
            f = NotableFeature(name=f"Barrel {i}", detailed_description="Old barrel.", metadata={})
            player.stats.health = player.stats.max_health
            pre_gold = player.currency.get("gold", 0)
            pre_items = len(loc.items)
            pre_chars = len(game_state.characters)
            pre_health = player.stats.health

            result = engine.roll_examine_surprise(f)
            if not result:
                continue

            if "gold" in result.lower():
                gold_event = True
                if player.currency.get("gold", 0) <= pre_gold:
                    report.bug("Gold event didn't add gold", result)
            elif "glints" in result.lower() or "it's a" in result.lower():
                item_event = True
                if len(loc.items) <= pre_items:
                    report.bug("Item event didn't add item", result)
            elif "trap" in result.lower():
                trap_event = True
                if player.stats.health >= pre_health:
                    report.bug("Trap didn't deal damage", result)
                if player.stats.health < 1:
                    report.bug("Trap killed the player", "Health should never go below 1")
            elif "leaps out" in result.lower() or "attack" in result.lower():
                ambush_event = True
                if len(game_state.characters) <= pre_chars:
                    report.bug("Ambush didn't spawn NPC", result)

            if gold_event and item_event and trap_event and ambush_event:
                break

        for event_name, triggered in [("gold", gold_event), ("item", item_event),
                                       ("trap", trap_event), ("ambush", ambush_event)]:
            if triggered:
                report.action(f"Surprise {event_name} event verified")
            else:
                report.warning(f"Surprise {event_name} event never triggered in 500 rolls")

        # Verify trap can't kill — set health to 1
        player.stats.health = 1
        for i in range(500):
            f = NotableFeature(name=f"Danger {i}", metadata={})
            engine.roll_examine_surprise(f)
            if player.stats.health < 1:
                report.bug("Trap killed player at 1 HP", f"Health dropped to {player.stats.health}")
                break
        if player.stats.health >= 1:
            report.action("Trap correctly cannot kill player (tested at 1 HP)")

        # Restore health
        player.stats.health = player.stats.max_health

    except Exception as e:
        report.crash("Surprise examine failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ====================================================================
    # PHASE 21: Code-level bug detection (static checks)
    # ====================================================================
    print("\n--- PHASE 21: Static Analysis ---")

    # Check for get_item_lore method (referenced in cli.py but might not exist)
    if not hasattr(engine, 'get_item_lore'):
        report.bug("Missing method: get_item_lore",
                   "cli.py:616 calls engine.get_item_lore() but this method does not exist in GameEngine")

    # Check load_game uses old schema
    import inspect
    load_source = inspect.getsource(engine.load_game)
    if "world_grid" in load_source and "region_grids" not in load_source:
        report.bug("load_game uses outdated schema",
                   "engine.py load_game() references 'world_grid' key but the model uses 'region_grids'. "
                   "Also constructs WorldGrid with positional args (same bug as the original report).")

    # Check if CharacterClass is imported in cli.py
    try:
        cli_path = Path("cli.py")
        if cli_path.exists():
            cli_source = cli_path.read_text()
            # Extract the import block from models (may span multiple lines)
            import re as _re
            import_block = _re.search(r'from models import \((.*?)\)', cli_source, _re.DOTALL)
            imported_names = import_block.group(1) if import_block else ""
            if "CharacterClass" not in imported_names:
                uses = [l.strip() for l in cli_source.split('\n') if 'CharacterClass' in l and 'import' not in l and l.strip()]
                if uses:
                    report.bug("Missing import: CharacterClass in cli.py",
                              f"cli.py uses CharacterClass but doesn't import it. Lines: {uses}")
    except Exception:
        pass

    # Check that ai_provider returns raw content (not double-parsed)
    ai_source = inspect.getsource(engine.ai.__class__.generate_response)
    if "json.dumps" in ai_source and "parse_llm_json" in ai_source:
        report.bug("Double JSON parsing in ai_provider",
                   "generate_response() parses JSON then re-serializes with json.dumps. "
                   "Engine will parse it again, causing double-parse.")

    # Check that apply_equipment_effects uses base_stats properly
    equip_source = inspect.getsource(engine.apply_equipment_effects)
    if "base_stats" not in equip_source:
        report.bug("Equipment aliasing bug",
                   "apply_equipment_effects does not reference base_stats. "
                   "Calling it multiple times will compound stat bonuses.")

    # Check complete_quest doesn't use phantom player.stats.experience
    quest_source = inspect.getsource(engine.complete_quest)
    if "player.stats.experience" in quest_source:
        report.bug("Phantom stats.experience field",
                   "complete_quest references player.stats.experience which doesn't exist on CharacterStats.")

    return report


async def main():
    print("=" * 70)
    print("  AIVENTURE AUTOMATED PLAYTEST")
    print("=" * 70)

    report = await run_playtest()
    report.print_report()


if __name__ == "__main__":
    asyncio.run(main())
