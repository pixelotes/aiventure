from __future__ import annotations
import asyncio
import logging
import sys
import random
import difflib
import inspect
import time
from uuid import UUID, uuid4
from pathlib import Path
from typing import Optional, List

from config import settings
from models import (
    Direction, CharacterType, CharacterClass, QuestStatus, Item, NPC,
    LocationType, ItemType, GeneralLocation, Coordinates, ServiceType
)
from ai_provider import LMStudioProvider
from utils import Colors, ThinkingSpinner
from engine import GameEngine, EngineResult

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

class CommandRegistry:
    """Registry for CLI commands to avoid large if-elif blocks"""
    def __init__(self):
        self.commands = {}
        self.aliases = {}

    def register(self, names, handler, help_text=""):
        if isinstance(names, str):
            names = [names]
        primary = names[0]
        self.commands[primary] = {"handler": handler, "help": help_text}
        for name in names:
            self.aliases[name] = primary

    def get_handler(self, name):
        primary = self.aliases.get(name)
        return self.commands.get(primary, {}).get("handler") if primary else None

class GameCLI:
    """Command line interface for the game"""
    def __init__(self):
        ai_provider = LMStudioProvider(settings.ollama_url, settings.ollama_model, settings.ollama_timeout)
        self.engine = GameEngine(ai_provider)
        self.running = False
        self.selected_model = ""
        self.ac_socket_port = 9999
        self._last_save_time = time.time()
        self.registry = CommandRegistry()
        self._register_commands()
        self._start_autocomplete_server()

    def _register_commands(self):
        """Register all game commands"""
        self.registry.register(["look", "l"], self.display_location, "See your current location")
        self.registry.register(["map"], self.display_map, "Show the regional map")
        self.registry.register(["time"], lambda args: self.display_time(args) or True, "Check the current time")
        self.registry.register(["status", "stat"], lambda args: self.display_character_status(args) or True, "Show character status")
        self.registry.register(["inventory", "inv", "i"], lambda args: self.show_inventory(args) or True, "Show inventory")
        self.registry.register(["go"], self.handle_movement, "Move in a direction")
        for d in Direction:
            self.registry.register([d.value], lambda args, d=d: self.handle_movement(d.value), f"Move {d.value}")
        self.registry.register(["talk"], self.handle_talk, "Talk to an NPC")
        self.registry.register(["ask"], self.handle_ask, "Ask an NPC about a topic")
        self.registry.register(["examine"], self.handle_examine, "Examine something closely")
        self.registry.register(["equip", "wield"], lambda args: print(self.engine.equip_item(args)) or True, "Equip an item")
        self.registry.register(["unequip"], lambda args: print(self.engine.unequip_item(args)) or True, "Unequip an item")
        self.registry.register(["pick", "take", "grab"], self.handle_pickup, "Pick up an item")
        self.registry.register(["drop"], self.handle_drop, "Drop an item")
        self.registry.register(["use"], self.handle_use, "Use an item")
        self.registry.register(["attack"], lambda args: self.handle_combat_action("attack", args), "Attack a target")
        self.registry.register(["parry"], lambda args: self.handle_combat_action("parry", args), "Parry an attack")
        self.registry.register(["flee"], lambda args: self.handle_combat_action("flee", args), "Attempt to flee")
        self.registry.register(["wait", "sleep"], self.handle_wait, "Wait for a duration")
        self.registry.register(["complete"], self.handle_complete_quest, "Complete/turn in a quest")
        self.registry.register(["read", "lore", "study"], self.handle_read, "Read or study lore")
        self.registry.register(["buy"], self.handle_buy, "Buy an item from a shopkeeper")
        self.registry.register(["sell"], self.handle_sell, "Sell an item to a shopkeeper")
        self.registry.register(["shop"], self.handle_shop, "Browse a shopkeeper's wares")
        self.registry.register(["enter"], self.handle_enter, "Enter a building")
        self.registry.register(["cook"], self.handle_cook, "Cook ingredients at a campfire")
        self.registry.register(["save"], self.save_game, "Save your progress")
        self.registry.register(["load"], self.load_game, "Load a saved game")
        self.registry.register(["ai"], self.handle_ai_command, "Execute a free-form AI action")
        self.registry.register(["help", "h"], lambda args: self.display_help(args) or True, "Show this help message")
    
    def _start_autocomplete_server(self):
        """Start a side-channel server for autocomplete suggestions"""
        import socket
        import threading
        
        def server_loop():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind(('127.0.0.1', self.ac_socket_port))
                server.listen(5)
                while True:
                    conn, _ = server.accept()
                    try:
                        data = conn.recv(1024).decode('utf-8').strip()
                        if data:
                            import json
                            suggestions = self.get_autocomplete_suggestions(data)
                            response = json.dumps({"suggestions": suggestions})
                            conn.sendall(response.encode('utf-8'))
                    finally:
                        conn.close()
            except Exception as e:
                # Silently fail if port is taken or other issues, 
                # though in produce we might want logging
                pass
            finally:
                server.close()

        threading.Thread(target=server_loop, daemon=True).start()
    
    def _drain_pending_messages(self):
        for msg in self.engine.pending_messages:
            if msg.startswith("DANGER:"):
                print(f"\n{Colors.RED}{Colors.BOLD}⚔️  {msg}{Colors.ENDC}")
            elif msg.startswith("PREMONITION:"):
                print(f"\n{Colors.HEADER}{Colors.BOLD}👁  {msg}{Colors.ENDC}")
            elif msg.startswith("OMINOUS:"):
                print(f"\n{Colors.RED}🌑 {msg}{Colors.ENDC}")
            elif msg.startswith("DISCOVERY:"):
                print(f"\n{Colors.CYAN}{Colors.BOLD}🔍 {msg}{Colors.ENDC}")
            elif msg.startswith("WORLD EVENT:"):
                print(f"\n{Colors.HEADER}{Colors.BOLD}🌍 {msg}{Colors.ENDC}")
            elif msg.startswith("QUEST ENCOUNTER:"):
                print(f"\n{Colors.GREEN}{Colors.BOLD}📜 {msg}{Colors.ENDC}")
            elif msg.startswith("Quest completed:") or msg.startswith("Quest ready"):
                print(f"\n{Colors.GREEN}✅ {msg}{Colors.ENDC}")
            elif msg.startswith("Gained") or msg.startswith("LEVEL UP"):
                print(f"\n{Colors.GREEN}{Colors.BOLD}⭐ {msg}{Colors.ENDC}")
            elif msg.startswith("DM:"):
                print(f"\n{Colors.CYAN}🎲 {msg[3:].strip()}{Colors.ENDC}")
            else:
                print(f"\n{Colors.YELLOW}📢 {msg}{Colors.ENDC}")
        self.engine.pending_messages.clear()

    def display_header(self):
        print("\n" + "="*60 + "\n    AI-POWERED TEXT ADVENTURE ENGINE\n" + "="*60)
    async def display_location(self, args: str = ""):
        if not self.engine.game_state: return
        location = self.engine.get_current_location()
        region = self.engine.game_state.session.world.regions.get(self.engine.game_state.session.current_region_id)
        
        icon = "📍"
        if "cave" in location.name.lower(): icon = "🕳️"
        elif "tower" in location.name.lower(): icon = "🏰"
        elif location.location_type == LocationType.REGION: icon = "🗺️"
        
        region_name_display = f" [{region.name}]" if region else ""
        print(f"\n{Colors.BLUE}{Colors.BOLD}{icon}  {location.name.upper()}{region_name_display} ({location.coordinates.x}, {location.coordinates.y}){Colors.ENDC}")
        print(f"{location.description}")
        
        env_details = []
        if getattr(location, 'atmosphere', None): env_details.append(f"The atmosphere is {location.atmosphere}")
        if getattr(location, 'ambient_sounds', None): env_details.append(f"You hear {', '.join(location.ambient_sounds)}")
        if getattr(location, 'ambient_smells', None): env_details.append(f"You smell {', '.join(location.ambient_smells)}")
        if getattr(location, 'weather', None): env_details.append(f"The weather is {location.weather}")
        if getattr(location, 'temperature', None) and location.temperature != "moderate": env_details.append(f"It feels {location.temperature}")
            
        if env_details: print(" ".join([f"{detail}." for detail in env_details]))
        
        if region and region.active_events:
            print(f"{Colors.BOLD}{Colors.RED}🌍 REGIONAL EVENT: {', '.join(region.active_events)}{Colors.ENDC}")
            
        if self.engine.game_state.session.active_global_events:
            for event in self.engine.game_state.session.active_global_events:
                if event.is_active:
                    print(f"{Colors.BOLD}{Colors.HEADER}🌌 WORLD EVENT: {event.name.upper()}{Colors.ENDC}")
                    print(f"   {event.description}")
            
        buildings = [f for f in location.notable_features if f.metadata.get("enterable")]
        other_features = [f for f in location.notable_features if not f.metadata.get("enterable")]
        if buildings:
            print(f"Buildings you can enter: {', '.join([f'{Colors.GREEN}{b.name}{Colors.ENDC}' for b in buildings])}.")
        if other_features:
            feature_strs = []
            for f in other_features:
                if f.metadata.get("campfire"):
                    feature_strs.append(f"{Colors.RED}[Campfire]{Colors.ENDC} {Colors.CYAN}{f.name}{Colors.ENDC}")
                elif f.metadata.get("puzzle") and not f.metadata.get("solved"):
                    feature_strs.append(f"{Colors.YELLOW}[?]{Colors.ENDC} {Colors.CYAN}{f.name}{Colors.ENDC}")
                elif f.metadata.get("puzzle") and f.metadata.get("solved"):
                    feature_strs.append(f"{Colors.GREEN}[Solved]{Colors.ENDC} {Colors.CYAN}{f.name}{Colors.ENDC}")
                elif f.metadata.get("corpse"):
                    feature_strs.append(f"{Colors.RED}[Corpse]{Colors.ENDC} {Colors.CYAN}{f.name}{Colors.ENDC}")
                else:
                    feature_strs.append(f"{Colors.CYAN}{f.name}{Colors.ENDC}")
            print(f"You notice: {', '.join(feature_strs)}.")
        if location.items:
            item_list = []
            for item_id in location.items:
                if item_id in self.engine.game_state.items:
                    item = self.engine.game_state.items[item_id]
                    color = Colors.CYAN
                    if item.rarity == "rare": color = Colors.BLUE
                    elif item.rarity in ["epic", "legendary"]: color = Colors.HEADER
                    item_list.append(f"{color}{item.name}{Colors.ENDC}")
            if item_list: print(f"On the ground you see: {', '.join(item_list)}.")
        
        npcs = [char for char in self.engine.game_state.characters.values() if char.current_location_id == location.id and getattr(char, 'character_type', None) != CharacterType.PLAYER]
        if npcs: print(f"People here: {', '.join([f'{Colors.YELLOW}{npc.name}{Colors.ENDC}' for npc in npcs])}.")
        
        visible_exits = sorted([conn.direction.value.title() for conn in location.connections if conn.is_visible])
        if visible_exits: print(f"Exits: {', '.join([f'{Colors.GREEN}{d}{Colors.ENDC}' for d in visible_exits])}")
        else: print("There are no obvious exits.")

    def display_intro_screen(self):
        if not self.engine.game_state: return
        world = self.engine.game_state.session.world
        player = self.engine.game_state.session.player_character
        quest_id = player.active_quests[0]
        quest = self.engine.game_state.quests[quest_id]

        print("\n" + "="*60)
        print(f"{Colors.HEADER}{Colors.BOLD}📖 AN ADVENTURE BEGINS...{Colors.ENDC}".center(60 + 10)) # Adjust for color chars
        print("="*60)
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}THE WORLD{Colors.ENDC}")
        print(f"The world of {Colors.BOLD}{world.name}{Colors.ENDC} is {world.description}")
        
        print(f"\n{Colors.YELLOW}{Colors.BOLD}THE HERO{Colors.ENDC}")
        print(f"{player.background_lore}")
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}THE QUEST{Colors.ENDC}")
        print(f"You are on a quest for {Colors.BOLD}{quest.name}{Colors.ENDC}:")
        print(f"{quest.description}")
        
        print("\n" + "="*60)
        input(f"\n{Colors.BOLD}Press Enter to start your journey...{Colors.ENDC}")
    
    def display_character_status(self, args: str = ""):
        if not self.engine.game_state: return
        player = self.engine.game_state.session.player_character
        stats = player.stats
        xp_threshold = player.level * 100
        xp_pct = min(100, int((player.experience / max(1, xp_threshold)) * 100))
        bar_len = 20
        filled = int(bar_len * xp_pct / 100)
        xp_bar = f"{'█' * filled}{'░' * (bar_len - filled)}"
        print(f"\n💤 {player.name} (Lvl {player.level} {player.character_class.value.title()})")
        print(f"   XP: {player.experience}/{xp_threshold} [{xp_bar}] {xp_pct}%")
        print(f"   Health: {stats.health}/{stats.max_health} | Stamina: {stats.stamina}/{stats.max_stamina} | Gold: {player.currency.get('gold', 0)}")
        
        active_quests = [self.engine.game_state.quests[qid] for qid in player.active_quests if qid in self.engine.game_state.quests]
        if active_quests:
            print(f"   Active Quests: {len(active_quests)}")
            for quest in active_quests:
                status_icons = {"active": "⏳", "completed": "✅"}
                icon = status_icons.get(quest.status.value, "❓")
                print(f"     {icon} {quest.name}: {quest.objectives[0].description}")
                if getattr(quest, 'location_hint', None) and quest.status == QuestStatus.ACTIVE:
                    print(f"        Hint: \"{quest.location_hint}\"")
        
        passives = []
        if player.character_class == CharacterClass.WARRIOR: passives.append("Sturdy (↑ Stamina Recovery)")
        elif player.character_class == CharacterClass.MAGE: passives.append("Arcane Flow (↑ Mana Recovery)")
        if passives:
            print(f"   {Colors.BLUE}Passives: {', '.join(passives)}{Colors.ENDC}")
        
        if player.conditions:
            print(f"   {Colors.RED}Conditions: {', '.join(player.conditions)}{Colors.ENDC}")

        if player.temporary_effects:
            buff_strs = [f"+{e.get('bonus',0)} {e.get('stat','?')} ({e.get('remaining_minutes',0)}min)" for e in player.temporary_effects]
            print(f"   {Colors.GREEN}Active Buffs: {', '.join(buff_strs)}{Colors.ENDC}")

        if player.quests_completed > 0: print(f"   Quests completed: {player.quests_completed}")

    def display_time(self, args: str = ""):
        if not self.engine.game_state: return
        gt = self.engine.game_state.session.game_time
        print(f"\n⏳ It is {gt.time_of_day.value} on day {gt.day} of the {gt.season.title()}. Time: {gt.hour:02d}:{gt.minute:02d}.")

    async def display_map(self, args: str = ""):
        if not self.engine.game_state: return
        player = self.engine.game_state.session.player_character
        curr_region_id = self.engine.game_state.session.current_region_id
        discovered_locs = [loc for loc in self.engine.game_state.locations.values() if loc.visit_count > 0 and loc.parent_id == curr_region_id]
        if not discovered_locs: print("No exploration yet in this region."); return

        min_x = min(loc.coordinates.x for loc in discovered_locs)
        max_x = max(loc.coordinates.x for loc in discovered_locs)
        min_y = min(loc.coordinates.y for loc in discovered_locs)
        max_y = max(loc.coordinates.y for loc in discovered_locs)

        BOX_W, BOX_H, H_SPAC = 12, 3, 3
        can_w = (max_x - min_x + 1) * (BOX_W + H_SPAC)
        can_h = (max_y - min_y + 1) * (BOX_H + 1)
        canvas = [[' ' for _ in range(can_w)] for _ in range(can_h)]

        for loc in discovered_locs:
            cx = (loc.coordinates.x - min_x) * (BOX_W + H_SPAC)
            cy = (loc.coordinates.y - min_y) * (BOX_H + 1)
            is_curr = (loc.id == player.current_location_id)
            d_name = loc.name.upper()[:BOX_W-2].center(BOX_W-2)
            line = f"|*{d_name[1:-1]}*|" if is_curr else f"|{d_name}|"
            
            row = cy + (BOX_H // 2)
            for j, char in enumerate(line): canvas[row][cx+j] = char
            
            sy, sx = row, cx + BOX_W // 2
            for conn in loc.connections:
                tloc = self.engine.game_state.locations.get(conn.target_location_id)
                # Fog of War: Only show connections to visited locations
                if tloc and tloc.visit_count > 0:
                    if conn.direction == Direction.EAST:
                        for i in range(H_SPAC): canvas[sy][sx + BOX_W//2 + i] = '-'
                    elif conn.direction == Direction.SOUTH:
                        for i in range(2): canvas[sy + 1 + i][sx] = '|'

        print("\n--- Map ---")
        for r in canvas: print("".join(r).rstrip())
        print("-----------")

    def display_help(self, args: str = ""):
        commands = {
            "go <dir>": "Move around", "look": "See location", "map": "Show map", 
            "time": "Check time", "inv": "Show items", "status": "Show hero info",
            "talk <npc>": "Chat", "ask <npc> about <topic>": "Inquire", 
            "examine <target>": "Look closely", "pick <item>": "Get item", 
            "drop <item>": "Lose item", "equip <item>": "Wield/Wear", 
            "use <item>": "Activate", "cook": "Cook at campfire",
            "enter <building>": "Enter a building",
            "shop": "Browse wares", "buy <item>": "Buy from merchant",
            "sell <item>": "Sell to merchant", "ai <text>": "Free-form action", "save/load": "Manage files", "quit": "Leave game"
        }
        print("\n📋 Available Commands:")
        for c, d in commands.items(): print(f"  {c:<25} - {d}")
    
    async def process_command(self, command: str) -> bool:
        command = command.strip()
        if not command: return True
        
        parts = command.split()
        cmd, args = parts[0].lower(), " ".join(parts[1:])
        
        if cmd in ["quit", "exit", "q"]: return False
        
        # Passive commands that don't count as "actions" for DM/quest triggers
        PASSIVE_CMDS = {"look", "l", "map", "time", "status", "stat", "inventory",
                        "inv", "i", "help", "h", "save", "load"}

        handler = self.registry.get_handler(cmd)
        if handler:
            result = handler(args)
            if inspect.isawaitable(result):
                await result

            # Increment action counters for non-passive commands
            if self.engine.game_state and cmd not in PASSIVE_CMDS:
                self.engine.game_state.session.dm_action_counter += 1
                self.engine.game_state.session.actions_since_last_quest += 1
                self.engine._tick_npcs()

            self._drain_pending_messages()
            # Auto-save
            if self.engine.game_state and (time.time() - self._last_save_time) >= settings.auto_save_interval:
                p = Path("saves") / f"{self.engine.game_state.session.session_name}.json"
                if await self.engine.save_game(p):
                    self._last_save_time = time.time()
            return True
        
        # Command not found, try fuzzy matching
        valid_cmds = list(self.registry.aliases.keys()) + ["quit", "exit", "q"]
        matches = difflib.get_close_matches(cmd, valid_cmds, n=1, cutoff=0.6)
        if matches:
            print(f"\n❌ Unknown command '{cmd}備. Did you mean '{matches[0]}'? (Type 'help' for commands)")
        else:
            print(f"\n❌ Unknown command '{cmd}'. (Type 'help' for a list of available commands)")
        return True

    def get_autocomplete_suggestions(self, partial_input: str) -> List[str]:
        """Get context-aware autocomplete suggestions"""
        parts = partial_input.lstrip().split()
        if not parts or partial_input.endswith(" "):
            # We are looking for an argument for a command, or it's empty
            if not parts:
                return [
                    "look", "map", "time", "status", "inventory", "go", "talk",
                    "ask", "examine", "equip", "unequip", "pick", "drop", "use",
                    "attack", "enter", "cook", "save", "load"
                ]
            
            cmd = parts[0].lower()
            remaining = " ".join(parts[1:]) if len(parts) > 1 else ""
            
            # Context-aware suggestions based on command
            if cmd in ["go", "north", "south", "east", "west"]:
                return [d.value for d in Direction]
            
            if cmd in ["pick", "take", "grab", "examine"]:
                loc = self.engine.get_current_location()
                items = [self.engine.game_state.items[iid].name for iid in loc.items if iid in self.engine.game_state.items]
                features = [f.name for f in loc.notable_features]
                return sorted(list(set(items + features)))

            if cmd in ["drop", "use", "equip", "unequip", "examine"]:
                player = self.engine.game_state.session.player_character
                items = [self.engine.game_state.items[iid].name for iid in player.inventory if iid in self.engine.game_state.items]
                return sorted(items)
                
            if cmd in ["talk", "ask", "attack"]:
                loc = self.engine.get_current_location()
                npcs = [char.name for char in self.engine.game_state.characters.values()
                        if char.current_location_id == loc.id and getattr(char, 'character_type', None) != CharacterType.PLAYER]
                return sorted(npcs)

            if cmd == "enter":
                loc = self.engine.get_current_location()
                return sorted([f.name for f in loc.notable_features if f.metadata.get("enterable")])

            return []
        
        # We are completing the last part of the input
        last_part = parts[-1].lower()
        if len(parts) == 1:
            # Completing a command
            valid_cmds = [
                "quit", "exit", "help", "look", "map", "time", "status", "inventory",
                "go", "talk", "ask", "examine", "equip", "unequip", "pick", "take",
                "grab", "drop", "use", "attack", "enter", "cook", "wait", "sleep", "complete", "save", "load",
                "north", "south", "east", "west", "ai"
            ]
            return [c for c in valid_cmds if c.startswith(last_part)]
        
        # Completing an argument
        cmd = parts[0].lower()
        arg_prefix = " ".join(parts[1:]).lower()
        
        options = []
        if cmd in ["go"]:
             options = [d.value for d in Direction]
        elif cmd in ["pick", "take", "grab", "examine"]:
            loc = self.engine.get_current_location()
            options += [self.engine.game_state.items[iid].name for iid in loc.items if iid in self.engine.game_state.items]
            options += [f.name for f in loc.notable_features]
        
        if cmd in ["drop", "use", "equip", "unequip", "examine"]:
            player = self.engine.game_state.session.player_character
            options += [self.engine.game_state.items[iid].name for iid in player.inventory if iid in self.engine.game_state.items]
            
        if cmd in ["talk", "ask", "attack"]:
            loc = self.engine.get_current_location()
            options += [char.name for char in self.engine.game_state.characters.values()
                        if char.current_location_id == loc.id and getattr(char, 'character_type', None) != CharacterType.PLAYER]

        if cmd == "enter":
            loc = self.engine.get_current_location()
            options += [f.name for f in loc.notable_features if f.metadata.get("enterable")]

        return [o for o in sorted(list(set(options))) if o.lower().startswith(arg_prefix)]

    async def handle_movement(self, direction_str: str):
        try:
            direction = Direction(direction_str.lower())
            success, message = await self.engine.move_player(direction, self.selected_model)
            
            if not success and message.startswith("TRAVEL_CONFIRM:"):
                region_name = message.split(":", 1)[1]
                choice = input(f"\n🌍 Do you want to travel to {region_name}? (y/n): ").strip().lower()
                if choice == 'y':
                    success, message = await self.engine.move_player(direction, self.selected_model, confirmed=True)
                else:
                    print("You decide to stay where you are.")
                    return

            print(f"\n{message}")
            if success: await self.display_location()
        except ValueError: print(f"Invalid direction: {direction_str}")

    async def handle_enter(self, building_name: str):
        if not building_name:
            loc = self.engine.get_current_location()
            enterable = [f.name for f in loc.notable_features if f.metadata.get("enterable")]
            if enterable:
                print(f"Enter what? Available: {', '.join(enterable)}")
            else:
                print("There is nothing to enter here.")
            return
        with ThinkingSpinner(ThinkingSpinner.ENTERING):
            success, message = await self.engine.enter_building(building_name, self.selected_model)
        print(f"\n{message}")
        if success:
            await self.display_location()

    async def handle_cook(self, args: str):
        loc = self.engine.get_current_location()
        campfire = next((f for f in loc.notable_features if f.metadata.get("campfire")), None)
        if not campfire:
            print("There is no campfire here to cook at.")
            return

        player = self.engine.game_state.session.player_character
        food_items = []
        seen = set()
        for item_id in player.inventory:
            item = self.engine.game_state.items.get(item_id)
            if item and item.name in self.engine.FOOD_INGREDIENT_NAMES and item.name not in seen:
                seen.add(item.name)
                food_items.append(item)

        if not food_items:
            print("You have no food ingredients to cook with.")
            return

        print(f"\n{Colors.CYAN}Campfire Cooking{Colors.ENDC}")
        print("Available ingredients:")
        for i, item in enumerate(food_items, 1):
            print(f"  {i}. {item.name}")

        print("\nChoose 1-3 ingredients (e.g. '1 2 3'). Type 'cancel' to stop.")
        choice = input("Ingredients: ").strip()
        if choice.lower() == "cancel":
            print("You step away from the campfire.")
            return

        try:
            indices = [int(c) - 1 for c in choice.split()]
            if not (1 <= len(indices) <= 3):
                print("Choose 1 to 3 ingredients.")
                return
            selected_ids = []
            for idx in indices:
                if 0 <= idx < len(food_items):
                    selected_ids.append(food_items[idx].id)
                else:
                    print(f"Invalid selection: {idx + 1}")
                    return
        except ValueError:
            print("Invalid input. Use numbers separated by spaces.")
            return

        with ThinkingSpinner(ThinkingSpinner.COOKING):
            success, msg, meal = await self.engine.cook_items(selected_ids, self.selected_model)
        if success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}Chef's Result:{Colors.ENDC} {msg}")
        else:
            print(f"\n{Colors.RED}{msg}{Colors.ENDC}")

    async def handle_wait(self, args: str):
        minutes = 30
        if args.isdigit(): minutes = int(args)
        await self.engine.advance_time(minutes, self.selected_model)
        print(f"\nYou wait for {minutes} minutes...")
        self.display_time()
        await self.display_location()

    def show_inventory(self, args: str = ""):
        player = self.engine.game_state.session.player_character
        print(f"\n🎒 {player.name}'s Inventory:")
        if not player.inventory: print("  (empty)"); return
        for item_id in player.inventory:
            item = self.engine.game_state.items.get(item_id)
            if item:
                eq = " (eq)" if item.id in player.equipped_items.values() else ""
                stack = f" x{item.current_stack_size}" if item.stackable and item.current_stack_size > 1 else ""
                print(f"  • {item.name}{stack}{eq} - {item.description}")

    async def handle_examine(self, target_name: str):
        if not target_name: return
        item = self.engine.find_item_in_inventory(target_name) or self.engine.find_item_in_location(target_name)
        if item:
            print(f"\nYou examine the {item.name}:\n{item.description}")
            if item.contained_items:
                revealed = []
                for iid in list(item.contained_items):
                    sub = self.engine.game_state.items.get(iid)
                    if sub: 
                        revealed.append(sub.name); self.engine.get_current_location().items.append(iid)
                if revealed: 
                    print(f"Inside you find: {', '.join(revealed)}.")
                    item.contained_items.clear()
            return
        loc = self.engine.get_current_location()
        feature = next((f for f in loc.notable_features if target_name.lower() in f.name.lower()), None)
        if feature:
            if not feature.detailed_description:
                feature.detailed_description = await self.handle_ai_command(f"Describe the {feature.name} in detail.", print_response=False)
            print(f"\n🎭 {feature.detailed_description}")
            if feature.metadata.get("puzzle"):
                if feature.metadata.get("solved"):
                    print(f"{Colors.GREEN}(Solved){Colors.ENDC}")
                else:
                    hint = feature.metadata.get("solution_hint", "")
                    if hint:
                        print(f"{Colors.YELLOW}Hint: {hint}{Colors.ENDC}")
            if feature.contained_items:
                revealed = []
                for iid in list(feature.contained_items):
                    sub = self.engine.game_state.items.get(iid)
                    if sub:
                        revealed.append(sub.name); loc.items.append(iid)
                if revealed:
                    print(f"Inside you find: {', '.join(revealed)}.")
                    feature.contained_items.clear()
            # Reveal hidden dungeon connections
            if feature.metadata.get("dungeon_entrance"):
                hidden_dir = feature.metadata.get("hidden_connection_direction", "down")
                try:
                    reveal_direction = Direction(hidden_dir)
                except ValueError:
                    reveal_direction = Direction.DOWN
                for conn in loc.connections:
                    if conn.direction == reveal_direction and not conn.is_visible:
                        conn.is_visible = True
                        conn.is_passable = True
                        print(f"\n{Colors.GREEN}{Colors.BOLD}You've discovered a hidden passage leading {reveal_direction.value}!{Colors.ENDC}")
                        feature.metadata.pop("dungeon_entrance", None)
                        break
            return
        print("Nothing special to see.")

    async def handle_talk(self, npc_name: str):
        loc = self.engine.get_current_location()
        npc = next((c for c in self.engine.game_state.characters.values() if c.current_location_id == loc.id and npc_name.lower() in c.name.lower() and isinstance(c, NPC)), None)
        if not npc:
            await self.handle_ai_command(f"chitchat with {npc_name}")
            return
        await self.handle_ai_command(f"chitchat with {npc.name}", target_npc=npc)

    async def handle_ask(self, args: str):
        if " about " not in args: print("Ask who about what? (e.g. ask guard about rumor)"); return
        npc_name, topic = [p.strip() for p in args.split(" about ", 1)]
        loc = self.engine.get_current_location()
        npc = next((c for c in self.engine.game_state.characters.values() if c.current_location_id == loc.id and npc_name.lower() in c.name.lower() and isinstance(c, NPC)), None)
        if not npc: print(f"No one named {npc_name} here."); return
        
        if topic == 'rumor':
            with ThinkingSpinner():
                rumor = await self.engine.generate_rumor(npc, self.selected_model)
            print(f"🎭 {npc.name}: \"{rumor}\"")
        elif topic == 'quest':
            with ThinkingSpinner():
                quest = await self.engine.generate_quest(npc, self.selected_model)
            if quest: print(f"🎭 {npc.name}: \"{quest.description}\"\nNew Quest: {quest.name}")
            else: print(f"🎭 {npc.name}: \"Nothing right now.\"")
        elif topic == 'services':
            print(f"🎭 {npc.name}: \"I can help with: {', '.join([s.name for s in npc.services_offered]) or 'Nothing'}\"")
        else:
            msg = await self.engine.handle_service_request(npc, topic, self.selected_model)
            if "doesn't offer" in msg: await self.handle_ai_command(f"ask {npc.name} about {topic}", target_npc=npc)
            else: print(msg)

    async def handle_pickup(self, item_name: str):
        msg = self.engine.pickup_item(item_name)
        print(f"\n{msg}")
        self.engine.check_quest_progress()
        self._drain_pending_messages()
        await self.display_location()

    async def handle_drop(self, item_name: str):
        player = self.engine.game_state.session.player_character
        item = self.engine.find_item_in_inventory(item_name)
        if item:
            success, msg = self.engine.remove_item_from_inventory(player, item_name, 1)
            if success:
                new_item = Item.model_validate(item.model_dump()); new_item.id = uuid4()
                self.engine.game_state.items[new_item.id] = new_item
                self.engine.get_current_location().items.append(new_item.id)
                print(f"Dropped {item.name}.")
            else: print(msg)

    async def handle_use(self, args: str):
        if " with " in args:
            t_n, tg_n = [p.strip() for p in args.split(" with ", 1)]
            tool = self.engine.find_item_in_inventory(t_n)
            if not tool: print(f"No {t_n} in your inventory."); return
            
            loc = self.engine.get_current_location()
            
            # 1. Try to find target item (for combinations)
            target_item = self.engine.find_item_in_inventory(tg_n)
            if target_item:
                success, msg, result = await self.engine.combine_items(tool.id, target_item.id, self.selected_model)
                print(f"\n{msg}")
                if success and result:
                    self.engine.game_state.items[result.id] = result
                    player = self.engine.game_state.session.player_character
                    player.inventory.append(result.id)
                    player.inventory.remove(tool.id)
                    player.inventory.remove(target_item.id)
                return

            # 2. Try to find target feature
            target_feature = next((f for f in loc.notable_features if tg_n.lower() in f.name.lower()), None)
            if target_feature:
                # Check if it's a puzzle
                if target_feature.metadata.get("puzzle"):
                    if target_feature.metadata.get("solved"):
                        print("\nThis puzzle has already been solved.")
                    else:
                        success, msg = self.engine.attempt_solve_puzzle(target_feature, tool)
                        if success:
                            print(f"\n{Colors.GREEN}{Colors.BOLD}PUZZLE SOLVED!{Colors.ENDC}")
                        print(f"\n{msg}")
                    return

                # Check predefined interactions
                if target_feature.id in tool.interactions:
                    print(f"\n{tool.interactions[target_feature.id]}")
                    if target_feature.contained_items:
                        rev = []
                        for iid in list(target_feature.contained_items):
                            sub = self.engine.game_state.items.get(iid)
                            if sub: rev.append(sub.name); loc.items.append(iid)
                        if rev: print(f"Found: {', '.join(rev)}."); target_feature.contained_items.clear()
                    return
                
                # Try AI-driven interaction
                success, msg = await self.engine.use_item_on_feature(tool.id, target_feature.id, self.selected_model)
                print(f"\n{msg}")
                return

            # 3. Try items in location (if not a combination or feature)
            target_obj = self.engine.find_item_in_location(tg_n)
            if target_obj:
                if target_obj.id in tool.interactions:
                    print(f"\n{tool.interactions[target_obj.id]}")
                else:
                    print("Nothing happens.")
                return
            
            print(f"No '{tg_n}' here to use with '{t_n}'.")
        else:
            item = self.engine.find_item_in_inventory(args)
            if item and item.self_use_effect_description:
                print(f"\n{item.self_use_effect_description}")
                if item.use_effects:
                    applied = self.engine.apply_item_effects(
                        self.engine.game_state.session.player_character, item
                    )
                    if applied:
                        print(f"  Effects: {applied}")
            else:
                print("\nCan't use that alone. Try 'use <item> with <target>'.")

    async def handle_combat_action(self, action: str, target_name: str):
        loc = self.engine.get_current_location()
        
        if not self.engine.in_combat:
            target = next((c for c in self.engine.game_state.characters.values() if c.current_location_id == loc.id and target_name.lower() in c.name.lower()), None)
            if not target: print(f"{Colors.RED}Target not found.{Colors.ENDC}"); return
            self.engine.in_combat = True
            self.engine.combat_opponents = [target]
        
        target = self.engine.combat_opponents[0]
        results = await self.engine.execute_combat_turn(target.id, action)
        
        print(f"\n⚔️  {Colors.BOLD}Combat Turn{Colors.ENDC}")
        print(f"   {results['player_msg']}")
        print(f"   {results['enemy_msg']}")

        # Check player death
        death_msg = self.engine.check_player_death()
        if death_msg:
            print(f"\n{Colors.RED}{Colors.BOLD}{death_msg}{Colors.ENDC}")
            await self.display_location()
            return

        if results["victory"] or results["fled"] or target.stats.health <= 0:
            self.engine.in_combat = False
            self.engine.combat_opponents = []

    async def handle_complete_quest(self, args: str):
        npc_name = args.replace("with", "").strip()
        loc = self.engine.get_current_location()
        npc = next((c for c in self.engine.game_state.characters.values() if c.current_location_id == loc.id and npc_name.lower() in c.name.lower() and isinstance(c, NPC)), None)
        if not npc: print("No one here matches that name."); return
        
        player = self.engine.game_state.session.player_character
        completable = [self.engine.game_state.quests[qid] for qid in player.active_quests if qid in self.engine.game_state.quests and self.engine.game_state.quests[qid].status == QuestStatus.COMPLETED and self.engine.game_state.quests[qid].giver_id == npc.id]
        
        if completable:
            q = completable[0]
            print(f"\n🎁 {Colors.BOLD}Quest Completion: {q.name}{Colors.ENDC}")
            print(f"1. Turn in to {npc.name} (Honorable)")
            print(f"2. Sell to a rival (Dishonorable - More gold, lose reputation)")
            
            choice = input("\nChoice (1/2): ").strip()
            if choice == "2":
                # Regional Rival logic (simplified)
                player.currency["gold"] = player.currency.get("gold", 0) + int(q.rewards.currency.get("gold", 0) * 1.5)
                if npc.faction:
                    player.faction_standings[npc.faction] = player.faction_standings.get(npc.faction, 0) - 10
                player.active_quests.remove(q.id)
                player.completed_quests.append(q.id)
                q.status = QuestStatus.TURNED_IN
                print(f"\n{Colors.RED}You sold the items to a rival. {npc.name} won't be happy.{Colors.ENDC}")
            else:
                print(await self.engine.complete_quest(q.id))
        else: print(f"{npc.name} has no quests for you to return.")

    def _find_merchant_at_location(self) -> Optional[NPC]:
        """Find an NPC with buy/sell services at the current location"""
        loc = self.engine.get_current_location()
        for char in self.engine.game_state.characters.values():
            if (isinstance(char, NPC) and char.current_location_id == loc.id
                    and any(s.service_type == ServiceType.BUY_SELL for s in char.services_offered)):
                return char
        return None

    async def handle_shop(self, npc_name: str = ""):
        if npc_name:
            loc = self.engine.get_current_location()
            npc = next((c for c in self.engine.game_state.characters.values() if c.current_location_id == loc.id and npc_name.lower() in c.name.lower() and isinstance(c, NPC)), None)
        else:
            npc = self._find_merchant_at_location()
        if not npc or not any(s.service_type == ServiceType.BUY_SELL for s in npc.services_offered):
            print("No merchant here."); return
        if not npc.shop_inventory:
            print(f"{npc.name}: \"I have nothing to sell right now.\""); return
        print(f"\n🛒 {npc.name}'s Wares:")
        for item_id in npc.shop_inventory:
            item = self.engine.game_state.items.get(item_id)
            if item:
                price = int(item.value * npc.prices_modifier)
                print(f"  • {item.name} - {price}g - {item.description}")

    async def handle_buy(self, item_name: str):
        npc = self._find_merchant_at_location()
        if not npc: print("No merchant here."); return
        result = self.engine.buy_item_from_npc(npc, item_name)
        print(f"\n{result}")

    async def handle_sell(self, item_name: str):
        npc = self._find_merchant_at_location()
        if not npc: print("No merchant here."); return
        result = self.engine.sell_item_to_npc(npc, item_name)
        print(f"\n{result}")

    async def handle_read(self, item_name: str):
        item = self.engine.find_item_in_inventory(item_name)
        if not item: print("You don't have that item."); return
        
        with ThinkingSpinner(ThinkingSpinner.STUDYING):
            lore = await self.engine.get_item_lore(item.id, self.selected_model)
        print(f"{Colors.HEADER}{Colors.BOLD}📖 {item.name} Lore:{Colors.ENDC}")
        print(f"{lore}")

    async def save_game(self, args: str = ""):
        p = Path("saves") / f"{self.engine.game_state.session.session_name}.json"
        if await self.engine.save_game(p): print(f"Saved to {p}")
        else: print("Save failed.")

    async def load_game(self, args: str = ""):
        d = Path("saves")
        if not d.exists(): print("No saves."); return
        files = list(d.glob("*.json"))
        for i, f in enumerate(files): print(f"{i+1}. {f.stem}")
        try:
            c = input("\nLoad #: ").strip()
            if c.isdigit() and int(c) <= len(files):
                if await self.engine.load_game(files[int(c)-1]): await self.display_location()
            else: print("Invalid selection.")
        except Exception as e: print(f"Load failed: {e}")

    async def handle_ai_command(self, command: str, target_npc: Optional[NPC] = None, print_response: bool = True) -> Optional[str]:
        if not self.engine.game_state: return None

        if target_npc:
            # NPC-targeted commands: narrative-only (existing behavior)
            player_char = self.engine.game_state.session.player_character
            full_command = f"{command} (Respond as the NPC in 1-2 sentences. Never speak for {player_char.name}.)"
            context = self.engine.build_context_for_ai(target_npc=target_npc)
            with ThinkingSpinner():
                response = await self.engine.ai.generate_response(full_command, context)
            if print_response: print(f"\n🎭 {response}")
            await self.engine.update_npc_memory(target_npc, command, response, self.selected_model)
            return response
        else:
            # World/action commands: DM system with game effects
            context = self.engine.build_context_for_ai()
            with ThinkingSpinner():
                narrative = await self.engine.process_ai_command(command, context, self.selected_model)
            if print_response and narrative: print(f"\n🎭 {narrative}")
            return narrative

    async def main_loop(self):
        self.display_header()
        if not await self.show_main_menu(): return
        self.running = True
        while self.running:
            try:
                cmd = input("\n> ").strip()
                if cmd and not await self.process_command(cmd): self.running = False
            except (KeyboardInterrupt, EOFError): break
        print("\nThanks for playing!")

    async def show_main_menu(self) -> bool:
        while True:
            print("\n--- Main Menu ---\n1. Start New Game\n2. Load Game\n3. Exit")
            choice = input("> ").strip()
            if choice == "1": return await self.start_new_game()
            elif choice == "2": await self.load_game(); return bool(self.engine.game_state)
            elif choice == "3": return False

    async def start_new_game(self) -> bool:
        print("\n🌟 Starting New Adventure")
        models = await self.engine.ai.get_available_models()
        if not models:
            print(f"\n{Colors.RED}❌ Connection error.{Colors.ENDC}")
            print(f"Failed to connect to AI provider at {settings.ollama_url}")
            print("Please ensure your AI server (LM Studio or Ollama) is running and reachable.")
            return False
        self.selected_model = settings.ollama_model if settings.ollama_model in models else models[0]
        p_name = input("Character name: ").strip() or "Adventurer"
        s_name = input("Adventure name: ").strip() or "Adventure"

        classes = [c for c in CharacterClass if c != CharacterClass.COMMONER]
        print("\nChoose your class:")
        for i, c in enumerate(classes, 1):
            print(f"  {i}. {c.value.title()}")
        choice = input("Class #: ").strip()
        try:
            selected_class = classes[int(choice) - 1]
        except (ValueError, IndexError):
            selected_class = CharacterClass.WARRIOR
            print(f"Defaulting to {selected_class.value.title()}.")

        await self.engine.create_new_game(p_name, uuid4(), s_name, self.selected_model, selected_class)
        self.display_intro_screen()
        await self.display_location()
        return True

if __name__ == "__main__":
    asyncio.run(GameCLI().main_loop())
