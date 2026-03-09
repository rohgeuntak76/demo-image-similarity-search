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
│   ├── fonts
│   │   └── NanumGothic-Regular.ttf
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
VLM_MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

# --- Asset Paths ---
FONT_PATH="data/fonts/NanumGothic-Regular.ttf"
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
FONT_PATH="data/fonts/NanumGothic.ttf"
```

### 3. Run the Application

Once the dependencies are installed and the configuration is set, run the application using `uvicorn`:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

Access the interactive API documentation at `http://127.0.0.1:8000/docs`.

-   **`POST /index/create`**: Deletes pre-existing index and creates from scratch for a specific year.
    *   **Parameters**: `year` (query parameter, integer) - The year for which to create the index.
-   **`POST /index/update`**: Appends uploaded images into an existing FAISS index for a specific year.
    *   **Parameters**: `year` (query parameter, integer) - The year for which to update the index.
-   **`GET /index/available`**: Returns a list of years for which indices have been created.
-   **`POST /search`**: Upload a query image to find the top 3 most similar images across specified years.
    *   **Parameters**: `years` (query parameter, list of integers) - The years to search within.
-   **`POST /report/generate`**: Upload a query image to generate a PDF analysis report using the configured VLM, searching across specified years.
    *   **Parameters**: 
        *   `years` (query parameter, list of integers) - The years to include in the search.
        *   `file` (multipart form-data) - The query image file.
        *   `prompt` (multipart form-data, optional) - Custom prompt for the VLM analysis.

