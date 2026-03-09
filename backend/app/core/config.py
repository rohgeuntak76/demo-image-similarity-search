import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # For OpenAI API or compatible servers
    OPENAI_API_KEY: Optional[str] = "DUMMY_KEY"
    
    # For self-hosted VLM with vLLM
    VLM_BASE_URL: Optional[str] = "http://localhost:8000/v1"
    VLM_MODEL_NAME: Optional[str] = "Qwen/Qwen3-VL-8B-Instruct"

    # File paths and directories
    FONT_PATH: str = "fonts/NanumGothic-Regular.ttf"
    BASE_INDEX_DIR: str = "data/assets"
    INDEXED_IMAGE_DIR: str = "data/images/indexed"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
