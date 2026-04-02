from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Outfit Extractor"
    VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"
    PORT: int = 8000
    
    # Image Constraints
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_PIXELS: int = 1024
    
    # Hugging Face
    HF_TOKEN: Optional[str] = None
    
    # Models
    SEGFORMER_MODEL: str = "sayeed99/segformer_b3_clothes"
    BIREFNET_MODEL: str = "ZhengPeng7/BiRefNet_lite"
    
    # Device (Fixed to CPU as per request)
    DEVICE: str = "cpu"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
