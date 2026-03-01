import os
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse

from app.services.image_analysis import image_analysis_service
from app.models.schemas import SearchResponse, SearchResult
from app.core.config import settings

app = FastAPI(title="Image Similarity Search API")

@app.post("/index", status_code=201)
async def create_index(year: int, files: List[UploadFile] = File(...)):
    """
    Upload multiple images to build the search index for a specific year.
    - Clears the existing indexed images and assets for the given year.
    - Saves the new images.
    - Creates and saves a new FAISS index for the year.
    """
    # Clear previous data
    if os.path.exists(settings.INDEXED_IMAGE_DIR):
        shutil.rmtree(settings.INDEXED_IMAGE_DIR)
    os.makedirs(settings.INDEXED_IMAGE_DIR)
    
    year_index_dir = os.path.join(settings.BASE_INDEX_DIR, str(year))
    if os.path.exists(year_index_dir):
        shutil.rmtree(year_index_dir)
    os.makedirs(year_index_dir, exist_ok=True)

    image_paths = []
    # Read and save input images
    for file in files:
        file_path = os.path.join(settings.INDEXED_IMAGE_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(file_path)
    
    try:
        # create_index() will save FAISS index in settings.INDEX_PATH, embeddings in settings.EMBEDDINGS_PATH, file name in settings.NAMES_PATH
        image_analysis_service.create_index(image_paths, year)
        return {"message": f"{len(image_paths)} images indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create index: {e}")

@app.post("/search", response_model=SearchResponse)
async def search_similar_images(years: List[int] = Query(...), file: UploadFile = File(...)):
    """
    Upload a query image to find the top 3 most similar images.
    Returns a JSON response with similarity scores and filenames.
    """
    query_dir = os.path.join("backend", "data", "images", "query")
    os.makedirs(query_dir, exist_ok=True)
    # save query image 
    query_path = os.path.join(query_dir, file.filename)
    with open(query_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        all_results = []
        for year in years:
            try:
                similarities, similar_paths = image_analysis_service.search_similar(query_path, year)
                for path, sim in zip(similar_paths, similarities):
                    all_results.append(SearchResult(filename=os.path.basename(path), similarity=sim))
            except FileNotFoundError as e:
                # Log the error or handle it as appropriate, for now, we'll just skip this year
                print(f"Skipping year {year} due to: {e}")
            except Exception as e:
                print(f"Error searching year {year}: {e}")
        
        # Sort all collected results by similarity and get top 3
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        top_k_results = all_results[:3] # Assuming k=3 as per original endpoint description
        
        return SearchResponse(query_image=file.filename, results=top_k_results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform search: {e}")

@app.post("/generate-report")
async def generate_analysis_report(years: List[int] = Query(...), file: UploadFile = File(...)):
    """
    Upload a query image to generate a full PDF analysis report.
    - Finds similar images.
    - Generates a summary with OpenAI's GPT-4o.
    - Creates and returns a PDF file.
    """
    query_dir = os.path.join("backend", "data", "images", "query")
    os.makedirs(query_dir, exist_ok=True)
    
    query_path = os.path.join(query_dir, file.filename)
    with open(query_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        all_results_raw = [] # Store raw (similarity, path) tuples
        for year in years:
            try:
                similarities_year, similar_paths_year = image_analysis_service.search_similar(query_path, year)
                for i in range(len(similarities_year)):
                    all_results_raw.append((similarities_year[i], similar_paths_year[i]))
            except FileNotFoundError as e:
                print(f"Skipping year {year} for report generation due to: {e}")
            except Exception as e:
                print(f"Error searching year {year} for report generation: {e}")
        
        # Sort all collected results by similarity and get top 3
        all_results_raw.sort(key=lambda x: x[0], reverse=True)
        top_k_raw_results = all_results_raw[:3] # Assuming k=3
        
        similarities = [res[0] for res in top_k_raw_results]
        similar_paths = [res[1] for res in top_k_raw_results]

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

