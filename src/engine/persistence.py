from __future__ import annotations
import json
from pathlib import Path
from uuid import UUID

from models import CompleteGameState, WorldGrid

llm_logger = __import__('logging').getLogger("llm_responses")


class PersistenceMixin:

    async def save_game(self, filepath: Path) -> bool:
        if not self.game_state:
            return False
        try:
            # GameState is a BaseModel, model_dump(mode='json') handles everything
            # as long as sub-models like WorldGrid are also BaseModels.
            game_dict = self.game_state.model_dump(mode='json')

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(game_dict, f, indent=2, default=str)
            return True
        except Exception as e:
            llm_logger.error(f"Save error: {e}")
            import traceback
            llm_logger.error(traceback.format_exc())
            return False

    async def load_game(self, filepath: Path) -> bool:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                game_dict = json.load(f)
            # Migration: old saves used 'world_grid', new uses 'region_grids'
            session_data = game_dict['session']
            if 'world_grid' in session_data and 'region_grids' not in session_data:
                starting_id = session_data.get('current_region_id') or next(iter(session_data.get('world', {}).get('regions', {})), None)
                if starting_id:
                    session_data['region_grids'] = {starting_id: session_data.pop('world_grid')}
                else:
                    session_data.pop('world_grid', None)
            grids_data = session_data.get('region_grids', {})
            for region_id_str, grid_data in grids_data.items():
                w, h = grid_data['width'], grid_data['height']
                raw_grid = [[UUID(cell) if cell else None for cell in row] for row in grid_data['grid']]
                grids_data[region_id_str] = WorldGrid(width=w, height=h, grid=raw_grid)
            game_dict['session']['region_grids'] = grids_data
            self.game_state = CompleteGameState.model_validate(game_dict)
            return True
        except Exception as e:
            llm_logger.error(f"Load error: {e}")
            return False
