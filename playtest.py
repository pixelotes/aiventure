"""
Automated playtest script for AIVenture.
Uses a mock AI provider to run a full game session without requiring a real LLM server.
Reports bugs, crashes, and gameplay observations.
"""
import asyncio
import sys
import traceback
import random
from uuid import uuid4
from pathlib import Path
from typing import List, Optional, Any, Dict

from ai_provider import AIProvider
from engine import GameEngine
from models import (
    Direction, CharacterType, QuestStatus, ItemType, NPC,
    LocationType, GeneralLocation
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
                "goal_region_type": "Crystal Citadel"
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
    # PHASE 11: Code-level bug detection (static checks)
    # ====================================================================
    print("\n--- PHASE 11: Static Analysis ---")

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
