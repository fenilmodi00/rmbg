from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

class ExtractionResponse(BaseModel):
    transparent_image_b64: str = Field(..., description="Base64 encoded transparent PNG")
    white_bg_image_b64: str = Field(..., description="Base64 encoded white bg PNG")
    labels_found: List[str] = Field(..., description="Clothing labels detected")
    inference_time_ms: int
    image_width: int
    image_height: int

class ExtractFromURLRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="Public URL of the image to extract from")
    output_format: Optional[str] = Field("transparent", description="Selected output variant (default=transparent)")

class HealthResponse(BaseModel):
    status: str
    segformer_loaded: bool
    birefnet_loaded: bool
    device: str
    version: str
