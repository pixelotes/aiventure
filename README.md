"""
# AI-Powered Text Adventure Engine

A modern, AI-powered text adventure engine built with Python, FastAPI, and Ollama. This project provides both a command-line interface and a REST API for creating immersive text-based adventures.

## Features

- **AI-Powered Storytelling**: Uses local LLM via Ollama for dynamic content generation
- **Persistent Game Worlds**: JSON-based save system with complex world states
- **RESTful API**: Complete REST API for building web frontends
- **Command Line Interface**: Full-featured CLI for direct gameplay
- **Modular Architecture**: Extensible system for locations, characters, items, and quests
- **Random Content Generation**: Dynamic world, city, and character generation
- **Multi-session Support**: Handle multiple concurrent game sessions

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-text-adventure-engine
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment configuration:
```bash
cp .env.example .env
```

5. Start Ollama and pull the model:
```bash
ollama serve
ollama pull llama3.1:8b
```

### Running the Application

#### Command Line Interface
```bash
python -m src.cli
```

#### REST API Server
```bash
python -m src.server
# or
uvicorn src.server:app --reload
```

#### Using Docker
```bash
docker-compose up -d
```

## API Documentation

Once the server is running, visit:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Key Endpoints

- `POST /games` - Create new game
- `GET /games/{session_id}` - Get game state
- `POST /games/{session_id}/actions` - Perform player action
- `POST /games/{session_id}/move` - Move player
- `GET /games/{session_id}/inventory` - Get inventory
- `POST /games/{session_id}/save` - Save game

## Project Structure

```
ai-text-adventure-engine/
├── src/
│   ├── models/          # Data models and schemas
│   ├── core/           # Core game logic
│   ├── api/            # FastAPI routes
│   ├── services/       # Business logic services
│   ├── utils/          # Utility functions
│   ├── cli.py          # Command line interface
│   └── server.py       # FastAPI server
├── saves/              # Game save files
├── stories/           # Pre-defined story files
├── tests/             # Test suite
├── requirements.txt   # Python dependencies
├── docker-compose.yml # Docker configuration
└── README.md         # This file
```

## Game World Structure

The engine supports a hierarchical world structure:

- **World** → **Regions** → **Cities** → **Districts** → **Buildings** → **Rooms**
- **Characters**: Players, NPCs, monsters with full stats and AI
- **Items**: Equipment, consumables, quest items with detailed properties
- **Quests**: Complex quest system with objectives and branching

## Configuration

Key configuration options in `.env`:

- `OLLAMA_URL`: Ollama server URL
- `OLLAMA_MODEL`: LLM model to use
- `API_PORT`: Server port
- `SAVES_DIRECTORY`: Where to store game saves
- `MAX_SESSIONS`: Maximum concurrent sessions

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black src/
isort src/
```

### Type Checking
```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on GitHub or contact the development team.

## Roadmap

- [ ] Web frontend interface
- [ ] Database backend option
- [ ] Multi-language support
