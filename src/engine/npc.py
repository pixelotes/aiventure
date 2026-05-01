from __future__ import annotations
from typing import Optional
import random

from models import NPC, NPCGoal, PlayerCharacter

llm_logger = __import__('logging').getLogger("llm_responses")


class NPCMixin:

    async def update_npc_memory(self, npc: NPC, player_msg: str, ai_msg: str, model_name: str) -> None:
        """Summarize the interaction and update NPC memory and mood"""
        current_memory = npc.interaction_summary or "No previous interactions."
        prompt = (f"Summarize the relationship and recent interaction between {npc.name} and the player. "
                  f"Previous Memory: {current_memory}\n"
                  f"New interaction: Player said '{player_msg}', {npc.name} replied '{ai_msg}'.\n"
                  "Respond with a JSON object containing: 'summary' (concise sentence) and 'sentiment' (float from -1.0 to 1.0 based on player's tone).")

        try:
            data = await self._generate_and_validate(prompt, model_name)
            npc.interaction_summary = data.get('summary', 'No notable interaction.')[:500]
            sentiment = data.get('sentiment', 0.0)
            # Update mood gradually
            npc.mood = max(0.0, min(1.0, npc.mood + (sentiment * 0.1)))
        except Exception:
            # Fallback
            new_summary = await self.ai.generate_response(f"Summarize: {player_msg} -> {ai_msg}", is_content_generation=False, model_name=model_name)
            npc.interaction_summary = new_summary[:500]

    async def generate_rumor(self, npc: NPC, model_name: str) -> str:
        context = self.build_context_for_ai()
        prompt = f"The player is asking {npc.name} ({npc.description}) for a rumor. Based on the context, generate a short, intriguing rumor this NPC might know. Respond with just the rumor itself, as a single sentence."
        rumor = await self.ai.generate_response(prompt, context, model_name=model_name)
        npc.rumors.append(rumor)
        return rumor

    async def get_item_lore(self, item_id, model_name: str) -> str:
        item = self.game_state.items.get(item_id)
        if not item:
            return "You can't learn anything about that."
        context = self.build_context_for_ai()
        prompt = f"The player is studying '{item.name}' ({item.description}). Generate a short lore passage (2-3 sentences) about this item's history and significance."
        return await self.ai.generate_response(prompt, context, model_name=model_name)

    async def handle_service_request(self, npc: NPC, service_name: str, model_name: str) -> str:
        if not npc.services_offered:
            return f"{npc.name} doesn't offer any services."
        service = next((s for s in npc.services_offered if service_name.lower() in s.name.lower()), None)
        if not service:
            return f"{npc.name} doesn't offer '{service_name}'."

        from models import ServiceType
        player = self.game_state.session.player_character
        if service.service_type == ServiceType.BUY_SELL:
            return f"{npc.name}: \"{service.description} Use 'buy <item>' or 'sell <item>' to trade.\""
        elif service.service_type == ServiceType.REST:
            cost = sum(service.cost.values())
            if player.currency.get("gold", 0) < cost:
                return f"{npc.name}: \"You need {cost} gold.\""
            player.currency["gold"] -= cost
            player.stats.health, player.stats.mana, player.stats.stamina = player.stats.max_health, player.stats.max_mana, player.stats.max_stamina
            await self.advance_time(480, model_name)
            return f"{npc.name}: \"Sleep well.\" (Restored and time advanced)"
        elif service.service_type == ServiceType.HEAL:
            cost = service.cost.get("gold", 1)
            if player.currency.get("gold", 0) < cost:
                return f"{npc.name}: \"You need {cost} gold.\""
            player.currency["gold"] -= cost
            player.stats.health = min(player.stats.max_health, player.stats.health + 30)
            return f"{npc.name}: \"Healed.\""
        return f"{npc.name}: \"{service.description}\""

    def _update_npc_positions(self) -> None:
        """Move NPCs based on the time of day"""
        gt = self.game_state.session.game_time
        # Daytime: 7:00 to 19:00
        is_work_time = 7 <= gt.hour < 19

        for char in self.game_state.characters.values():
            if isinstance(char, NPC) and char.goal == NPCGoal.NONE:
                target_loc = char.work_location_id if (is_work_time and char.work_location_id) else char.home_location_id
                if target_loc and char.current_location_id != target_loc:
                    char.previous_location_id = char.current_location_id
                    char.current_location_id = target_loc

    def _tick_npcs(self) -> None:
        """Tick all goal-driven NPCs. Called after each non-passive player action. No AI calls."""
        if not self.game_state:
            return
        player = self.game_state.session.player_character
        player_loc = player.current_location_id

        npcs = [c for c in list(self.game_state.characters.values())
                if isinstance(c, NPC) and c.goal != NPCGoal.NONE]

        for npc in npcs:
            if npc.id not in self.game_state.characters:
                continue
            npc.ticks_alive += 1
            same_loc = (npc.current_location_id == player_loc)

            if npc.max_ticks > 0 and npc.ticks_alive >= npc.max_ticks:
                self._despawn_npc(npc, "vanishes without a trace")
                continue

            if npc.goal == NPCGoal.ATTACK_PLAYER:
                self._tick_attack(npc, player, same_loc)
            elif npc.goal == NPCGoal.DELIVER_MESSAGE:
                self._tick_message(npc, player, same_loc)
            elif npc.goal == NPCGoal.FOLLOW_PLAYER:
                self._tick_follow(npc, player)
            elif npc.goal == NPCGoal.FLEE:
                self._tick_flee(npc, player, same_loc)

    def _tick_attack(self, npc: NPC, player: PlayerCharacter, same_loc: bool) -> None:
        if not same_loc:
            if npc.ticks_alive <= 5:
                npc.previous_location_id = npc.current_location_id
                npc.current_location_id = player.current_location_id
                self.pending_messages.append(f"DANGER: {npc.name} has tracked you down!")
            return
        if not self.in_combat:
            self.in_combat = True
            self.combat_opponents = [npc]
            self.pending_messages.append(f"DANGER: {npc.name} attacks you!")
        elif npc not in self.combat_opponents:
            self.pending_messages.append(f"DANGER: {npc.name} is circling, waiting to strike!")

    def _tick_message(self, npc: NPC, player: PlayerCharacter, same_loc: bool) -> None:
        if not same_loc:
            npc.previous_location_id = npc.current_location_id
            npc.current_location_id = player.current_location_id
            self.pending_messages.append(f"A figure approaches — {npc.name} has arrived.")
            return
        if not npc.goal_data.get("delivered"):
            msg = npc.goal_data.get("message", "...")
            self.pending_messages.append(f'{npc.name} speaks: "{msg}"')
            npc.goal_data["delivered"] = True
        else:
            if npc.is_transient:
                self._despawn_npc(npc, "slips away into the shadows")
            else:
                npc.goal = NPCGoal.NONE

    def _tick_follow(self, npc: NPC, player: PlayerCharacter) -> None:
        if npc.current_location_id != player.current_location_id:
            npc.previous_location_id = npc.current_location_id
            npc.current_location_id = player.current_location_id

    def _tick_flee(self, npc: NPC, player: PlayerCharacter, same_loc: bool) -> None:
        if same_loc:
            loc = self.game_state.locations.get(npc.current_location_id)
            if loc and loc.connections:
                valid = [c for c in loc.connections if c.is_passable]
                if valid:
                    target = random.choice(valid)
                    npc.previous_location_id = npc.current_location_id
                    npc.current_location_id = target.target_location_id
                    self.pending_messages.append(f"{npc.name} flees in panic!")
        if npc.is_transient and npc.ticks_alive >= 3:
            self._despawn_npc(npc, "has disappeared")

    def _despawn_npc(self, npc: NPC, flavor: str = "vanishes") -> None:
        self.pending_messages.append(f"{npc.name} {flavor}.")
        if npc.id in self.game_state.characters:
            del self.game_state.characters[npc.id]
