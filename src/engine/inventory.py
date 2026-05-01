from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4

from models import (
    CharacterClass, CharacterStats, BaseCharacter, PlayerCharacter, NPC, NPCRole,
    ItemType, Item, Service, ServiceType,
)

llm_logger = __import__('logging').getLogger("llm_responses")


class InventoryMixin:

    ITEM_CATALOG = [
        # Weapons
        {"name": "Iron Sword", "description": "A sturdy iron blade.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"attack_bonus": 2, "damage_bonus": 3}, "value": 25},
        {"name": "Oak Staff", "description": "A gnarled wooden staff crackling with faint energy.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"intelligence": 2, "attack_bonus": 1}, "value": 20},
        {"name": "Steel Dagger", "description": "A sharp, lightweight dagger.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"dexterity": 1, "damage_bonus": 2}, "value": 15},
        {"name": "War Hammer", "description": "A heavy hammer that packs a devastating blow.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"strength": 2, "damage_bonus": 4}, "value": 30},
        {"name": "Hunting Bow", "description": "A recurve bow suited for both hunting and combat.", "item_type": ItemType.WEAPON, "equipment_slot": "weapon", "stat_modifiers": {"dexterity": 2, "damage_bonus": 2}, "value": 22},
        # Armor
        {"name": "Leather Armor", "description": "Light but durable leather protection.", "item_type": ItemType.ARMOR, "equipment_slot": "armor", "stat_modifiers": {"armor_class": 2}, "value": 20},
        {"name": "Chainmail", "description": "Interlocking metal rings providing solid defense.", "item_type": ItemType.ARMOR, "equipment_slot": "armor", "stat_modifiers": {"armor_class": 4, "dexterity": -1}, "value": 40},
        {"name": "Mage Robes", "description": "Enchanted robes that enhance magical ability.", "item_type": ItemType.ARMOR, "equipment_slot": "armor", "stat_modifiers": {"armor_class": 1, "intelligence": 2, "max_mana": 10}, "value": 30},
        # Shields
        {"name": "Wooden Shield", "description": "A round wooden shield reinforced with iron.", "item_type": ItemType.ARMOR, "equipment_slot": "shield", "stat_modifiers": {"armor_class": 2}, "value": 12},
        # Consumables
        {"name": "Health Potion", "description": "A red potion that restores 30 health.", "item_type": ItemType.CONSUMABLE, "consumable": True, "self_use_effect_description": "You drink the potion and feel warmth spread through your body. (+30 HP)", "use_effects": ["heal:30"], "value": 10, "stackable": True, "stack_size": 5},
        {"name": "Stamina Tonic", "description": "A green tonic that restores 30 stamina.", "item_type": ItemType.CONSUMABLE, "consumable": True, "self_use_effect_description": "You drink the tonic and feel a surge of energy. (+30 Stamina)", "use_effects": ["stamina:30"], "value": 8, "stackable": True, "stack_size": 5},
        {"name": "Mana Elixir", "description": "A blue elixir that restores 25 mana.", "item_type": ItemType.CONSUMABLE, "consumable": True, "self_use_effect_description": "You drink the elixir and feel your magical reserves replenish. (+25 Mana)", "use_effects": ["mana:25"], "value": 12, "stackable": True, "stack_size": 5},
        # Food Ingredients (for cooking)
        {"name": "Wild Berries", "description": "A handful of sweet, plump berries.", "item_type": ItemType.MATERIAL, "value": 2, "stackable": True, "stack_size": 10, "weight": 0.2},
        {"name": "Forest Mushrooms", "description": "Earthy brown mushrooms with a rich aroma.", "item_type": ItemType.MATERIAL, "value": 3, "stackable": True, "stack_size": 10, "weight": 0.3},
        {"name": "Healing Herbs", "description": "Fragrant green herbs known for their restorative properties.", "item_type": ItemType.MATERIAL, "value": 4, "stackable": True, "stack_size": 10, "weight": 0.1},
        {"name": "Raw Meat", "description": "A cut of fresh game meat.", "item_type": ItemType.MATERIAL, "value": 5, "stackable": True, "stack_size": 5, "weight": 0.5},
        {"name": "Fresh Fish", "description": "A river fish, still glistening.", "item_type": ItemType.MATERIAL, "value": 4, "stackable": True, "stack_size": 5, "weight": 0.4},
        {"name": "Spicy Peppers", "description": "Small red peppers that tingle the tongue.", "item_type": ItemType.MATERIAL, "value": 3, "stackable": True, "stack_size": 10, "weight": 0.1},
        {"name": "Honeycomb", "description": "A golden chunk of wild honeycomb.", "item_type": ItemType.MATERIAL, "value": 5, "stackable": True, "stack_size": 5, "weight": 0.3},
        {"name": "Mountain Root", "description": "A tough, knotted root with warming properties.", "item_type": ItemType.MATERIAL, "value": 4, "stackable": True, "stack_size": 10, "weight": 0.2},
    ]

    FOOD_INGREDIENT_NAMES = {
        "Wild Berries", "Forest Mushrooms", "Healing Herbs", "Raw Meat",
        "Fresh Fish", "Spicy Peppers", "Honeycomb", "Mountain Root",
    }

    COOKING_BUFF_CAPS = {
        "strength": 5, "dexterity": 5, "constitution": 5,
        "intelligence": 5, "wisdom": 5, "charisma": 5,
        "armor_class": 3, "attack_bonus": 3, "damage_bonus": 3,
        "max_health": 30, "max_stamina": 20, "max_mana": 20,
    }

    CLASS_STARTER_GEAR = {
        CharacterClass.WARRIOR: ["Iron Sword", "Wooden Shield", "Health Potion"],
        CharacterClass.MAGE: ["Oak Staff", "Mage Robes", "Mana Elixir"],
        CharacterClass.ROGUE: ["Steel Dagger", "Leather Armor", "Health Potion"],
        CharacterClass.CLERIC: ["War Hammer", "Leather Armor", "Mana Elixir"],
        CharacterClass.RANGER: ["Hunting Bow", "Leather Armor", "Health Potion"],
        CharacterClass.BARD: ["Steel Dagger", "Mage Robes", "Stamina Tonic"],
        CharacterClass.COMMONER: ["Health Potion"],
    }

    CLASS_STAT_MODIFIERS = {
        CharacterClass.WARRIOR: {"strength": 4, "constitution": 3, "max_health": 30, "max_stamina": 20},
        CharacterClass.MAGE: {"intelligence": 5, "wisdom": 2, "max_mana": 40, "max_health": -10},
        CharacterClass.ROGUE: {"dexterity": 5, "charisma": 1, "damage_bonus": 2},
        CharacterClass.CLERIC: {"wisdom": 4, "constitution": 2, "max_mana": 20, "max_health": 10},
        CharacterClass.RANGER: {"dexterity": 3, "wisdom": 2, "constitution": 2, "max_stamina": 20},
        CharacterClass.BARD: {"charisma": 5, "dexterity": 2, "intelligence": 1, "max_mana": 15},
    }

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
        """Apply equipment bonuses to character stats (reads base_stats, writes stats)"""
        if not isinstance(character, (PlayerCharacter, NPC)):
            return

        base = character.base_stats
        bonuses = self.calculate_equipment_bonuses(character)

        character.stats.strength = max(1, base.strength + bonuses.strength)
        character.stats.dexterity = max(1, base.dexterity + bonuses.dexterity)
        character.stats.constitution = max(1, base.constitution + bonuses.constitution)
        character.stats.intelligence = max(1, base.intelligence + bonuses.intelligence)
        character.stats.wisdom = max(1, base.wisdom + bonuses.wisdom)
        character.stats.charisma = max(1, base.charisma + bonuses.charisma)
        character.stats.armor_class = max(1, base.armor_class + bonuses.armor_class)
        character.stats.attack_bonus = base.attack_bonus + bonuses.attack_bonus
        character.stats.damage_bonus = base.damage_bonus + bonuses.damage_bonus

        character.stats.max_health = max(1, base.max_health + (bonuses.constitution * 5))
        character.stats.max_mana = max(1, base.max_mana + (bonuses.intelligence * 3))

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
            if remaining_quantity <= 0:
                break
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

    def apply_item_effects(self, character: BaseCharacter, item: Item) -> str:
        """Apply use_effects from a consumable item. Returns description of effects."""
        if not item.use_effects:
            return ""
        effects_applied = []
        for effect_str in item.use_effects:
            parts = effect_str.split(":")
            if len(parts) < 2:
                continue
            if parts[0] == "heal":
                amount = int(parts[1])
                character.stats.health = min(character.stats.max_health, character.stats.health + amount)
                effects_applied.append(f"+{amount} HP")
            elif parts[0] == "stamina":
                amount = int(parts[1])
                character.stats.stamina = min(character.stats.max_stamina, character.stats.stamina + amount)
                effects_applied.append(f"+{amount} Stamina")
            elif parts[0] == "mana":
                amount = int(parts[1])
                character.stats.mana = min(character.stats.max_mana, character.stats.mana + amount)
                effects_applied.append(f"+{amount} Mana")
            elif parts[0] == "buff" and len(parts) == 4:
                self.apply_temporary_effect(character, {
                    "stat": parts[1], "bonus": int(parts[2]),
                    "remaining_minutes": int(parts[3]), "source": item.name,
                })
                effects_applied.append(f"+{parts[2]} {parts[1]} ({parts[3]}min)")
        if item.consumable:
            self.remove_item_from_inventory(character, item.name, 1)
        return ", ".join(effects_applied)

    def apply_temporary_effect(self, character: BaseCharacter, effect: Dict) -> None:
        """Add a temporary stat buff. Replaces existing effect from same source+stat."""
        character.temporary_effects = [
            e for e in character.temporary_effects
            if not (e.get("stat") == effect["stat"] and e.get("source") == effect.get("source"))
        ]
        character.temporary_effects.append(effect)
        stat = effect["stat"]
        bonus = effect["bonus"]
        if hasattr(character.stats, stat):
            setattr(character.stats, stat, getattr(character.stats, stat) + bonus)

    def expire_temporary_effects(self, character: BaseCharacter, minutes_elapsed: int) -> List[str]:
        """Tick down temporary effects and remove expired ones. Returns expiry messages."""
        expired_msgs = []
        still_active = []
        for effect in character.temporary_effects:
            effect["remaining_minutes"] = effect.get("remaining_minutes", 0) - minutes_elapsed
            if effect["remaining_minutes"] <= 0:
                stat = effect.get("stat", "")
                bonus = effect.get("bonus", 0)
                source = effect.get("source", "Unknown")
                if hasattr(character.stats, stat):
                    setattr(character.stats, stat, max(1, getattr(character.stats, stat) - bonus))
                expired_msgs.append(f"The effect of {source} (+{bonus} {stat}) has worn off.")
            else:
                still_active.append(effect)
        character.temporary_effects = still_active
        return expired_msgs

    def _create_catalog_item(self, name: str) -> Optional[Item]:
        """Create an item from the catalog by name"""
        template = next((t for t in self.ITEM_CATALOG if t["name"] == name), None)
        if not template:
            return None
        item = Item(**template)
        self.game_state.items[item.id] = item
        return item

    def _populate_shop_inventory(self, npc: NPC) -> None:
        """Give shopkeepers/merchants a starting inventory from the catalog"""
        if npc.role == NPCRole.SHOPKEEPER:
            stock = ["Iron Sword", "Leather Armor", "Wooden Shield", "Health Potion", "Stamina Tonic"]
        elif npc.role == NPCRole.MERCHANT:
            stock = ["Chainmail", "Mage Robes", "War Hammer", "Hunting Bow", "Health Potion", "Mana Elixir"]
        else:
            return
        for item_name in stock:
            item = self._create_catalog_item(item_name)
            if item:
                npc.shop_inventory.append(item.id)

    def _create_services_for_npc(self, npc: NPC) -> None:
        if npc.role == NPCRole.SHOPKEEPER:
            npc.services_offered.append(Service(service_type=ServiceType.BUY_SELL, name="General Goods", description="I buy and sell various items.", cost={"gold": 0}))
            self._populate_shop_inventory(npc)
        elif npc.role == NPCRole.MERCHANT:
            npc.services_offered.append(Service(service_type=ServiceType.BUY_SELL, name="Trade Goods", description="I deal in fine wares and exotic items.", cost={"gold": 0}))
            self._populate_shop_inventory(npc)
        elif npc.role == NPCRole.INNKEEPER:
            npc.services_offered.extend([
                Service(service_type=ServiceType.REST, name="Room for the Night", description="A warm bed and a hot meal.", cost={"gold": 2, "silver": 5}),
                Service(service_type=ServiceType.HEAL, name="Herbal Remedies", description="Basic healing herbs and tonics.", cost={"gold": 1})
            ])
        elif npc.role == NPCRole.CRAFTSMAN:
            npc.services_offered.append(Service(service_type=ServiceType.REPAIR, name="Item Repair", description="I can mend your broken equipment.", cost={"gold": 5}))

    def find_item_in_inventory(self, item_name: str) -> Optional[Item]:
        if not self.game_state:
            return None
        for item_id in self.game_state.session.player_character.inventory:
            item = self.game_state.items.get(item_id)
            if item and item_name.lower() in item.name.lower():
                return item
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
        if not self.game_state:
            return None
        for item_id in self.get_current_location().items:
            item = self.game_state.items.get(item_id)
            if item and item_name.lower() in item.name.lower():
                return item
        return None

    def equip_item(self, item_name: str) -> str:
        item = self.find_item_in_inventory(item_name)
        if not item or not item.equipment_slot:
            return "Can't equip."
        player = self.game_state.session.player_character
        if item.equipment_slot in player.equipped_items:
            old_id = player.equipped_items[item.equipment_slot]
            old = self.game_state.items.get(old_id)
            msg = f"Unequipped {old.name}. " if old else ""
        else:
            msg = ""
        player.equipped_items[item.equipment_slot] = item.id
        self.apply_equipment_effects(player)
        return msg + f"Equipped {item.name}."

    def unequip_item(self, item_name: str) -> str:
        player = self.game_state.session.player_character
        item = next((self.game_state.items.get(iid) for iid in player.equipped_items.values() if iid in self.game_state.items and item_name.lower() in self.game_state.items[iid].name.lower()), None)
        if not item:
            return "Not equipped."
        slot = next(s for s, iid in player.equipped_items.items() if iid == item.id)
        del player.equipped_items[slot]
        self.apply_equipment_effects(player)
        return f"Unequipped {item.name}."

    def _get_price_mod(self) -> float:
        """Return price_mod from current region's active events (default 1.0)."""
        session = self.game_state.session
        region = session.world.regions.get(session.current_region_id)
        return region.event_modifiers.get("price_mod", 1.0) if region else 1.0

    def buy_item_from_npc(self, npc: NPC, item_name: str) -> str:
        item = next((self.game_state.items.get(iid) for iid in npc.shop_inventory if iid in self.game_state.items and item_name.lower() in self.game_state.items[iid].name.lower()), None)
        if not item:
            return f"{npc.name} doesn't have that."
        player = self.game_state.session.player_character
        price = int(item.value * npc.prices_modifier * self._get_price_mod())
        if player.currency.get("gold", 0) < price:
            return f"Need {price} gold."
        player.currency["gold"] -= price
        player.inventory.append(item.id)
        npc.shop_inventory.remove(item.id)
        return f"Bought {item.name}."

    def sell_item_to_npc(self, npc: NPC, item_name: str) -> str:
        item = self.find_item_in_inventory(item_name)
        if not item:
            return "You don't have it."
        if item.item_type == ItemType.QUEST_ITEM:
            return "Can't sell quest items."

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

            total_mod = npc.prices_modifier * faction_mod * mood_mod * self._get_price_mod()
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

    async def combine_items(self, item1_id, item2_id, model_name: str):
        item1 = self.game_state.items.get(item1_id)
        item2 = self.game_state.items.get(item2_id)
        if not item1 or not item2:
            return False, "Item not found.", None

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
                from models import ItemRarity
                new_item = Item(
                    name=data.get('name', 'Crafted Item'),
                    description=data.get('description', 'Something you made.'),
                    item_type=ItemType(self._coerce_enum(data.get('item_type', 'material'), ItemType)),
                    value=data.get('value', 10),
                    rarity=ItemRarity(self._coerce_enum(data.get('rarity', 'common'), ItemRarity))
                )
                self.game_state.items[new_item.id] = new_item

                # Consume materials
                inventory = self.game_state.session.player_character.inventory
                if item1.id in inventory:
                    inventory.remove(item1.id)
                if item2.id in inventory:
                    inventory.remove(item2.id)

                msg = f"You successfully crafted {new_item.name} from {item1.name} and {item2.name}!"
                return True, msg, new_item
            except Exception:
                return False, f"You tried to combine {item1.name} and {item2.name}, but nothing useful came of it.", None

        return False, "These items don't seem like they would work together.", None
