import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Ollama Configuration
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:1.5b"
    ollama_timeout: int = 600 # Increased timeout for generation
    
    # Game Configuration
    saves_directory: Path = Path("saves")
    stories_directory: Path = Path("stories")
    max_sessions: int = 100
    auto_save_interval: int = 300  # seconds
    
    # Security
    secret_key: str = "your-secret-key-here-change-in-production"
    access_token_expire_minutes: int = 30
    
    # Database (for future expansion)
    database_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = "llm_debug.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Get settings instance
settings = Settings()
