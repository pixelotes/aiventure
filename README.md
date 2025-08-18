# AI-Powered Text Adventure Engine

A sophisticated, AI-powered text adventure game built in Python that creates immersive, procedurally-generated fantasy worlds. The game leverages local Large Language Models via Ollama to provide dynamic storytelling, intelligent NPCs, and contextual responses to player actions.

## ✨ Features

### 🎮 **Core Gameplay**
- **Dynamic World Generation**: Procedurally created fantasy worlds with interconnected locations
- **Intelligent AI Responses**: Context-aware storytelling powered by local LLMs via Ollama
- **Rich Environmental Details**: Atmospheric descriptions with ambient sounds, smells, and weather
- **Interactive Command System**: Natural language commands with comprehensive help system

### 🗺️ **World & Exploration**
- **Hierarchical World Structure**: Worlds → Regions → Cities → Districts → Buildings → Rooms
- **Grid-Based Navigation**: 3x3 starting region with compass-based movement
- **Dynamic Location Discovery**: Locations reveal details as you explore them
- **Visual ASCII Map**: Track your exploration with an in-game map system
- **Environmental Storytelling**: Each location has unique atmosphere, features, and secrets

### 👥 **Characters & NPCs**
- **Complex Character System**: Detailed stats, equipment, and progression
- **Intelligent NPCs**: AI-powered conversations with unique personalities and dialogue styles
- **NPC Roles & Services**: Shopkeepers, innkeepers, quest givers, and craftsmen
- **Dynamic Relationships**: Reputation and faction systems
- **Character Classes**: Warriors, mages, rogues, clerics, rangers, and bards

### 📦 **Items & Equipment**
- **Comprehensive Item System**: Weapons, armor, consumables, tools, and quest items
- **Equipment Bonuses**: Stat modifications from equipped gear
- **Item Stacking**: Intelligent inventory management for stackable items
- **Magical Properties**: Enchantments, special abilities, and cursed items
- **Interactive Objects**: Use items together to solve puzzles and unlock secrets

### 🎯 **Quests & Objectives**
- **Dynamic Quest Generation**: AI-created fetch quests with procedural placement
- **Quest Progression Tracking**: Active and completed quest monitoring
- **Multi-Objective Quests**: Complex objectives with branching paths
- **Contextual Hints**: Location clues to help find quest items
- **Meaningful Rewards**: Experience, currency, and unique items

### 🛍️ **Economy & Services**
- **NPC Services**: Buy/sell items, rest at inns, repair equipment, healing services
- **Dynamic Pricing**: Price modifiers based on NPC relationships
- **Currency System**: Gold, silver, and copper coins
- **Shop Inventories**: NPCs maintain their own stock of goods

### 💾 **Persistence & Management**
- **JSON Save System**: Complete game state preservation
- **Auto-save Feature**: Automatic game saving on exit
- **Session Management**: Named game sessions with timestamps
- **World State Tracking**: All changes persist between sessions

### 🎲 **Advanced Features**
- **Hidden Item Discovery**: Search notable features to find concealed items
- **Combat System**: Turn-based combat with stats and equipment bonuses
- **Time Progression**: Day/night cycle affecting gameplay
- **Puzzle Mechanics**: Use items together to solve environmental challenges
- **AI Rumor Generation**: Dynamic rumors and information from NPCs

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** - The game is built with modern Python
2. **Ollama** - Local LLM server for AI responses
   ```bash
   # Install Ollama (see https://ollama.ai)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   
   # Pull a recommended model
   ollama pull qwen2.5-coder:1.5b
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-text-adventure-engine
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the game** (optional):
   ```bash
   cp .env.example .env
   # Edit .env to customize Ollama URL, model, etc.
   ```

### Running the Game

```bash
python -m src.cli
```

### Docker Support

The game includes full Docker support with a web terminal interface:

```bash
docker-compose up -d
```

This provides a web-based terminal using xterm.js, allowing you to play the game through your browser.

## 🎮 How to Play

### Basic Commands

- **Movement**: `go north`, `south`, `east`, `west` or just `n`, `s`, `e`, `w`
- **Exploration**: `look`, `examine <object>`, `map`
- **Inventory**: `inventory`, `pick up <item>`, `drop <item>`
- **Equipment**: `equip <item>`, `unequip <item>`
- **Interaction**: `use <item>`, `use <item> with <target>`

### Advanced Commands

- **NPCs**: `talk <name>`, `ask <npc> about <topic>`
- **Services**: `ask <npc> about services`, `buy <item>`, `sell <item>`
- **Quests**: `ask <npc> about quest`, `complete quest <npc>`
- **Combat**: `attack <target>`
- **System**: `status`, `time`, `save`, `load`, `help`

### Tips for New Players

1. **Start by exploring** - Use `look` to examine your surroundings
2. **Talk to NPCs** - They provide quests, services, and valuable information
3. **Check notable features** - Use `examine` on interesting objects to find hidden items
4. **Manage your inventory** - Equip better gear as you find it
5. **Follow quest hints** - NPCs give location clues for fetch quests
6. **Use the map** - Track your exploration with the `map` command

## ⚙️ Configuration

### Environment Variables

Key settings in `.env`:

```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:1.5b
OLLAMA_TIMEOUT=600

# Game Settings
SAVES_DIRECTORY=saves
LOG_LEVEL=INFO
LOG_FILE=llm_debug.log
```

### Recommended Ollama Models

- **qwen2.5-coder:1.5b** - Fast, efficient, good for most gameplay
- **llama3.1:8b** - More detailed responses, slower
- **mistral:7b** - Balanced performance and quality

## 🏗️ Architecture

### Game Structure

```
src/
├── cli.py          # Main game engine and CLI interface
├── config.py       # Configuration management
└── ...

saves/              # JSON save files
stories/           # Pre-defined story content (future)
docker/            # Docker configuration with xterm.js
```

### Core Components

- **GameEngine**: Central game logic and state management
- **CompleteGameState**: Comprehensive world state model
- **Pydantic Models**: Type-safe data structures for all game entities
- **AI Integration**: Seamless Ollama LLM integration for dynamic content

## 🛠️ Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
```

### Adding New Features

The game is designed with extensibility in mind:

- **New Item Types**: Extend the `ItemType` enum and add handling logic
- **New NPC Roles**: Add roles to `NPCRole` and implement service logic
- **New Quest Types**: Extend quest generation and completion systems
- **New Locations**: Add location types and generation logic

## 🎯 Technical Highlights

- **Type Safety**: Comprehensive Pydantic models ensure data integrity
- **Async Design**: Non-blocking AI calls for responsive gameplay  
- **Flexible AI Integration**: Easy to swap LLM models and providers
- **Rich Game State**: Everything persists - locations, NPCs, quests, inventory
- **Modular Architecture**: Clean separation of concerns for easy extension

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama is running: `ollama serve`
   - Check the URL in configuration
   - Verify the model is downloaded: `ollama list`

2. **Slow Response Times**
   - Try a smaller model like `qwen2.5-coder:1.5b`
   - Increase `OLLAMA_TIMEOUT` in configuration
   - Check system resources

3. **Save File Issues**
   - Ensure `saves/` directory exists
   - Check file permissions
   - Verify JSON structure isn't corrupted

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🚗 Roadmap

### Planned Features

- [ ] **Web Frontend**: Browser-based interface with rich graphics
- [ ] **Database Backend**: PostgreSQL/SQLite support for larger worlds
- [ ] **Multi-language Support**: Localization for different languages
- [ ] **Advanced Combat**: Tactical combat with positioning and abilities
- [ ] **Crafting System**: Item creation and enhancement
- [ ] **Settlement Building**: Player-owned locations and NPCs
- [ ] **Multiple Worlds**: Portal system between different realms
- [ ] **Player Companions**: Recruitable NPCs with AI personalities

### Current Status

This is a fully-featured single-player text adventure engine with sophisticated AI integration. The game provides hours of emergent gameplay through procedural generation and intelligent NPC interactions.

---

**Ready to embark on your AI-powered adventure? Install Ollama, clone the repo, and let the magic begin!** ✨