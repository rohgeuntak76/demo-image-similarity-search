from pydantic import BaseModel, Field
from typing import List

class SearchResult(BaseModel):
    filename: str
    similarity: float
    year: int
    image_base64: str

class SearchResponse(BaseModel):
    query_image: str
    results: List[SearchResult]
