from .base import BaseEngine, EngineResult
from .inventory import InventoryMixin
from .combat import CombatMixin
from .world_gen import WorldGenMixin
from .npc import NPCMixin
from .quests import QuestMixin
from .dm import VirtualDMMixin
from .activities import ActivitiesMixin
from .persistence import PersistenceMixin

class GameEngine(
    InventoryMixin,
    CombatMixin,
    WorldGenMixin,
    NPCMixin,
    QuestMixin,
    VirtualDMMixin,
    ActivitiesMixin,
    PersistenceMixin,
    BaseEngine,
):
    pass

__all__ = ["GameEngine", "EngineResult"]
