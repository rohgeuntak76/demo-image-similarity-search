# Image Similarity Search API

This project provides a FastAPI-based API for finding similar images and generating analysis reports. It uses a pre-trained ResNet-50 model to extract feature embeddings, FAISS for efficient similarity searching, and a configurable Vision Language Model (VLM) for generating detailed analysis reports.

The application is configured to connect to a local [vLLM](https://github.com/vllm-project/vllm) server running an OpenAI-compatible API, but can also be pointed to the official OpenAI API.

## Project Structure

```
.
├── app
│   ├── core
│   │   ├── config.py
│   │   └── __init__.py
│   ├── models
│   │   ├── schemas.py
│   │   └── __init__.py
│   ├── services
│   │   ├── image_analysis.py
│   │   └── __init__.py
│   ├── main.py
│   └── __init__.py
├── data
│   ├── assets
│   │   └── NanumGothic.ttf
│   ├── images
│   │   ├── indexed
│   │   └── query
├── .env
├── requirements.txt
└── README.md
```

## How to Set Up and Run

### 1. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory. You can configure it to use either a local vLLM server or the OpenAI API.

**For a local vLLM server (Default):**

Update the `.env` file with the URL of your vLLM server and the model identifier.

```
# --- For Local vLLM Server ---
VLM_BASE_URL="http://localhost:8000/v1"
VLM_MODEL_NAME="Qwen/Qwen2-VL-8B-Instruct"

# --- Asset Paths ---
FONT_PATH="data/assets/NanumGothic.ttf"
```

**For the OpenAI API:**

Comment out the vLLM variables and add your OpenAI API key.

```
# --- For OpenAI API ---
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

# --- For Local vLLM Server ---
# VLM_BASE_URL="http://localhost:8000/v1"
# VLM_MODEL_NAME="Qwen/Qwen2-VL-8B-Instruct"

# --- Asset Paths ---
FONT_PATH="data/assets/NanumGothic.ttf"
```

### 3. Run the Application

Once the dependencies are installed and the configuration is set, run the application using `uvicorn`:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

Access the interactive API documentation at `http://127.0.0.1:8000/docs`.

-   **`POST /index`**: Upload multiple images to build the search index.
-   **`POST /search`**: Upload a query image to find the top 3 most similar images.
-   **`POST /generate-report`**: Upload a query image to generate a PDF analysis report using the configured VLM.

## Code Implementation

<details>
<summary><code>app/main.py</code></summary>

```python
import os
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from app.services.image_analysis import image_analysis_service
from app.models.schemas import SearchResponse, SearchResult
from app.core.config import settings

app = FastAPI(title="Image Similarity Search API")

@app.post("/index", status_code=201)
async def create_index(files: List[UploadFile] = File(...)):
    # ... (code unchanged)
    
@app.post("/search", response_model=SearchResponse)
async def search_similar_images(file: UploadFile = File(...)):
    # ... (code unchanged)

@app.post("/generate-report")
async def generate_analysis_report(file: UploadFile = File(...)):
    # ... (code unchanged)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Similarity Search API. Use /docs to see the API documentation."}
```
</details>

<details>
<summary><code>app/services/image_analysis.py</code></summary>

```python
import os
import numpy as np
from PIL import Image
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from fpdf import FPDF
import openai
import base64
from io import BytesIO
from typing import List, Tuple

from app.core.config import settings

class ImageAnalysisService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._get_transform()
        self.model = self._load_model()
        
        os.makedirs(os.path.dirname(settings.INDEX_PATH), exist_ok=True)
        os.makedirs(settings.INDEXED_IMAGE_DIR, exist_ok=True)

    # ... (helper methods unchanged)

    def generate_gpt_summary(self, query_path: str, similar_paths: List[str], similarities: List[float]) -> str:
        summary_prompt = f"""
        Analyze the following weather chart images to create a structured report.
        
        1.  Summarize the key weather features of the [Query Image (2025)].
        2.  For each of the [Top-3 Similar Weather Charts], describe its main features, its similarities and differences compared to the query image, and its similarity score (%).
            - Top 1 Similarity: {similarities[0]:.1f}%
            - Top 2 Similarity: {similarities[1]:.1f}%
            - Top 3 Similarity: {similarities[2]:.1f}%
        3.  Provide a comprehensive comparison of all four images, highlighting common patterns and notable differences.
        4.  Explain the meteorological basis for the high similarity.
        5.  Include any other relevant observations.

        Please write the report in Korean at an expert level, using clear sections or tables.
        """

        # ... (image_to_base64 unchanged)

        query_img_b64 = image_to_base64(query_path)
        similar_imgs_b64 = [image_to_base64(p) for p in similar_paths]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": summary_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{query_img_b64}"}},
                ] + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in similar_imgs_b64]
            }
        ]
        
        client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.VLM_BASE_URL,
        )
        try:
            response = client.chat.completions.create(
                model=settings.VLM_MODEL_NAME,
                messages=messages,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to generate VLM summary: {e}"

    # ... (create_pdf_report unchanged)

image_analysis_service = ImageAnalysisService()
```
</details>

<details>
<summary><code>app/core/config.py</code></summary>

```python
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # For OpenAI API or compatible servers
    OPENAI_API_KEY: Optional[str] = "DUMMY_KEY"
    
    # For self-hosted VLM with vLLM
    VLM_BASE_URL: Optional[str] = "http://localhost:8000/v1"
    VLM_MODEL_NAME: Optional[str] = "Qwen/Qwen2-VL-8B-Instruct"

    # File paths and directories
    FONT_PATH: str = "data/assets/NanumGothic.ttf"
    INDEX_PATH: str = "data/assets/weather_index.faiss"
    NAMES_PATH: str = "data/assets/weather_names.npy"
    EMBEDDINGS_PATH: str = "data/assets/weather_embeddings.npy"
    INDEXED_IMAGE_DIR: str = "data/images/indexed"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
```
</details>

<details>
<summary><code>app/models/schemas.py</code></summary>

```python
from pydantic import BaseModel, Field
from typing import List

class SearchResult(BaseModel):
    filename: str
    similarity: float

class SearchResponse(BaseModel):
    query_image: str
    results: List[SearchResult]
```
</details>
