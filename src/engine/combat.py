from __future__ import annotations
from typing import Dict, Any, Optional
import random
from uuid import UUID

from models import BaseCharacter, NotableFeature

llm_logger = __import__('logging').getLogger("llm_responses")


class CombatMixin:

    async def execute_combat_turn(self, target_id: UUID, action: str = "attack") -> Dict[str, Any]:
        """Execute a combat turn with strategic elements"""
        if not self.game_state:
            return {"msg": "No session"}
        player = self.game_state.session.player_character
        target = self.game_state.characters.get(target_id)
        if not target:
            return {"msg": "Target not found"}

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
        else:  # Attack
            hit_chance = 0.7 + (player.stats.dexterity * 0.01) - (target.stats.dexterity * 0.005)
            if random.random() < hit_chance:
                dmg = random.randint(1, 8) + (player.stats.strength // 3) + player.stats.damage_bonus
                target.stats.health -= dmg
                results["player_msg"] = f"You hit {target.name} for {dmg} damage."
                if random.random() < 0.2:  # 20% bleed chance on hit
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
        if "parrying" in player.conditions:
            player.conditions.remove("parrying")

        for char in [player, target]:
            if "bleeding" in char.conditions:
                bleed_dmg = 2
                char.stats.health -= bleed_dmg
                msg = f" (Bleeding: -{bleed_dmg} HP)"
                if char.id == player.id:
                    results["player_msg"] += msg
                else:
                    results["enemy_msg"] += msg

        return results

    def handle_combat_reward(self, target: BaseCharacter):
        player = self.game_state.session.player_character
        exp = target.level * 20
        player.experience += exp
        self.pending_messages.append(f"Gained {exp} XP from defeating {target.name}.")

        # Loot NPC gold
        npc_gold = target.currency.get("gold", 0) if hasattr(target, 'currency') else 0
        if npc_gold > 0:
            player.currency["gold"] = player.currency.get("gold", 0) + npc_gold
            self.pending_messages.append(f"Found {npc_gold} gold on {target.name}.")

        # Create lootable corpse
        loot_items = [iid for iid in target.inventory if iid in self.game_state.items]
        loc = self.get_current_location()
        if loot_items:
            corpse = NotableFeature(
                name=f"Remains of {target.name}",
                detailed_description=f"The fallen body of {target.name}. Something might be worth searching.",
                contained_items=loot_items,
                metadata={"corpse": True, "original_npc_name": target.name},
            )
            loc.notable_features.append(corpse)
            self.pending_messages.append(f"You notice the remains of {target.name}. Try 'examine remains' to search.")

        lvl_msg = self.check_level_up()
        if lvl_msg:
            self.pending_messages.append(lvl_msg)

    def check_level_up(self) -> Optional[str]:
        """Check and apply level-up if enough XP accumulated"""
        player = self.game_state.session.player_character
        xp_threshold = player.level * 100
        if player.experience < xp_threshold:
            return None
        player.experience -= xp_threshold
        player.level += 1
        # Boost base stats
        player.base_stats.max_health += 10
        player.base_stats.max_stamina += 5
        player.base_stats.max_mana += 5
        # Class-specific level bonus (smaller than initial)
        for stat, mod in self.CLASS_STAT_MODIFIERS.get(player.character_class, {}).items():
            if stat.startswith("max_"):
                continue  # already handled above
            current = getattr(player.base_stats, stat, 0)
            setattr(player.base_stats, stat, current + max(1, mod // 2))
        # Heal to new max
        self.apply_equipment_effects(player)
        player.stats.health = player.stats.max_health
        player.stats.mana = player.stats.max_mana
        player.stats.stamina = player.stats.max_stamina
        return f"LEVEL UP! You are now level {player.level}!"

    def check_player_death(self) -> Optional[str]:
        """Check if player died and handle respawn"""
        player = self.game_state.session.player_character
        if player.stats.health > 0:
            return None
        player.deaths += 1
        # Respawn at starting region (0,0)
        start_region_id = self.game_state.session.world.starting_region_id
        start_grid = self.game_state.session.region_grids.get(start_region_id)
        if start_grid:
            respawn_loc = start_grid.get_location_id(0, 0)
            if respawn_loc:
                player.current_location_id = respawn_loc
                self.game_state.session.current_region_id = start_region_id
        # Penalties
        player.stats.health = player.stats.max_health // 2
        player.stats.mana = 0
        player.stats.stamina = player.stats.max_stamina // 2
        gold_loss = player.currency.get("gold", 0) // 10
        player.currency["gold"] = max(0, player.currency.get("gold", 0) - gold_loss)
        # Clear combat
        self.in_combat = False
        self.combat_opponents = []
        return f"YOU DIED! (Death #{player.deaths}) Lost {gold_loss} gold. You awaken weakened..."
