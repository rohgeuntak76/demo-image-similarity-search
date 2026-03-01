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
    """
    Upload multiple images to build the search index.
    - Clears the existing indexed images and assets.
    - Saves the new images.
    - Creates and saves a new FAISS index.
    """
    # Clear previous data
    if os.path.exists(settings.INDEXED_IMAGE_DIR):
        shutil.rmtree(settings.INDEXED_IMAGE_DIR)
    os.makedirs(settings.INDEXED_IMAGE_DIR)
    
    if os.path.exists(os.path.dirname(settings.INDEX_PATH)):
        for item in os.listdir(os.path.dirname(settings.INDEX_PATH)):
            os.remove(os.path.join(os.path.dirname(settings.INDEX_PATH), item))

    image_paths = []
    # Read and save input images
    for file in files:
        file_path = os.path.join(settings.INDEXED_IMAGE_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(file_path)
    
    try:
        # create_index() will save FAISS index in settings.INDEX_PATH, embeddings in settings.EMBEDDINGS_PATH, file name in settings.NAMES_PATH
        image_analysis_service.create_index(image_paths)
        return {"message": f"{len(image_paths)} images indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create index: {e}")

@app.post("/search", response_model=SearchResponse)
async def search_similar_images(file: UploadFile = File(...)):
    """
    Upload a query image to find the top 3 most similar images.
    Returns a JSON response with similarity scores and filenames.
    """
    query_dir = os.path.join("data", "images", "query")
    os.makedirs(query_dir, exist_ok=True)
    # save query image 
    query_path = os.path.join(query_dir, file.filename)
    with open(query_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        similarities, similar_paths = image_analysis_service.search_similar(query_path)
        
        results = [
            SearchResult(filename=os.path.basename(path), similarity=sim)
            for path, sim in zip(similar_paths, similarities)
        ]
        
        return SearchResponse(query_image=file.filename, results=results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {e}")

@app.post("/generate-report")
async def generate_analysis_report(file: UploadFile = File(...)):
    """
    Upload a query image to generate a full PDF analysis report.
    - Finds similar images.
    - Generates a summary with OpenAI's GPT-4o.
    - Creates and returns a PDF file.
    """
    query_dir = os.path.join("data", "images", "query")
    os.makedirs(query_dir, exist_ok=True)
    
    query_path = os.path.join(query_dir, file.filename)
    with open(query_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        similarities, similar_paths = image_analysis_service.search_similar(query_path)
        gpt_summary = image_analysis_service.generate_gpt_summary(query_path, similar_paths, similarities)
        pdf_path = image_analysis_service.create_pdf_report(query_path, similar_paths, similarities, gpt_summary)
        
        return FileResponse(pdf_path, media_type='application/pdf', filename=os.path.basename(pdf_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Similarity Search API. Use /docs to see the API documentation."}

