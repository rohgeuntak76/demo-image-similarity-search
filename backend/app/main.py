import os
import shutil
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.services.image_analysis import image_analysis_service
from app.models.schemas import SearchResponse, SearchResult
from app.core.config import settings

app = FastAPI(title="Image Similarity Search API")

# Mount the data directory to serve indexed images
app.mount("/data", StaticFiles(directory="data"), name="data")

def remove_file(path: str):
    """Helper function to remove a file from the filesystem."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error removing file {path}: {e}")

def save_uploaded_images(files: List[UploadFile], year: int) -> List[str]:
    """Helper to save uploaded images to a year-specific directory."""
    year_img_dir = os.path.join(settings.INDEXED_IMAGE_DIR, str(year))
    os.makedirs(year_img_dir, exist_ok=True)
    image_paths = []
    for file in files:
        file_path = os.path.join(year_img_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(file_path)
    return image_paths

@app.post("/index/create", status_code=201)
async def create_index(year: int, files: List[UploadFile] = File(...)):
    """
    Delete pre-existing index and create from scratch for a specific year.
    - Clears the existing indexed images and assets for the given year.
    - Saves the new images.
    - Creates and saves a new FAISS index for the year.
    """
    # Clear previous assets (index, embeddings, names)
    year_index_dir = os.path.join(settings.BASE_INDEX_DIR, str(year))
    if os.path.exists(year_index_dir):
        shutil.rmtree(year_index_dir)
    
    # Clear previous images for this year
    year_img_dir = os.path.join(settings.INDEXED_IMAGE_DIR, str(year))
    if os.path.exists(year_img_dir):
        shutil.rmtree(year_img_dir)
    
    image_paths = save_uploaded_images(files, year)
    
    try:
        image_analysis_service.create_index(image_paths, year)
        return {"message": f"Index for year {year} created with {len(image_paths)} images."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create index: {e}")

@app.post("/index/update", status_code=200)
async def update_index(year: int, files: List[UploadFile] = File(...)):
    """
    Append uploaded images into existing FAISS index for a specific year.
    """
    image_paths = save_uploaded_images(files, year)
    
    try:
        image_analysis_service.update_index(image_paths, year)
        return {"message": f"Index for year {year} updated with {len(image_paths)} images."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update index: {e}")

@app.get("/index/available", response_model=List[int])
async def get_available_indices():
    """
    Return a list of years for which a FAISS index has been created.
    """
    if not os.path.exists(settings.BASE_INDEX_DIR):
        return []
    
    years = []
    for entry in os.listdir(settings.BASE_INDEX_DIR):
        if os.path.isdir(os.path.join(settings.BASE_INDEX_DIR, entry)) and entry.isdigit():
            # Check if index.faiss exists in the directory
            if os.path.exists(os.path.join(settings.BASE_INDEX_DIR, entry, "index.faiss")):
                years.append(int(entry))
    
    return sorted(years)

@app.post("/search", response_model=SearchResponse)
async def search_similar_images(years: List[int] = Query(...), file: UploadFile = File(...)):
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
        all_results = []
        for year in years:
            try:
                similarities, similar_paths = image_analysis_service.search_similar(query_path, year)
                for path, sim in zip(similar_paths, similarities):
                    all_results.append(SearchResult(filename=os.path.basename(path), similarity=sim, year=year))
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
    finally:
        if os.path.exists(query_path):
            os.remove(query_path)

@app.post("/report/generate")
async def generate_analysis_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    prompt: Optional[str] = Form(None),
    similar_images_json: str = Form(...)
):
    """
    Generate a full PDF analysis report using specific search results.
    - Uses images provided by the frontend.
    - Generates a summary with VLM.
    - Creates and returns a PDF file.
    - Deletes temporary PDF after sending.
    """
    query_dir = os.path.join("data", "images", "query")
    os.makedirs(query_dir, exist_ok=True)
    
    query_path = os.path.join(query_dir, file.filename)
    with open(query_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Parse the similar images provided by the frontend
        try:
            results_data = json.loads(similar_images_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid search results format: {e}")

        if not results_data:
             raise HTTPException(status_code=400, detail="No similar images provided.")

        # Reconstruct paths and collect similarities
        similarities = []
        similar_paths = []
        
        for res in results_data:
            filename = res.get("filename")
            year = res.get("year")
            similarity = res.get("similarity")
            
            if filename and year is not None:
                # Reconstruct path based on year-specific directory structure
                path = os.path.join(settings.INDEXED_IMAGE_DIR, str(year), filename)
                if os.path.exists(path):
                    similar_paths.append(path)
                    similarities.append(similarity if similarity is not None else 0.0)
                else:
                    print(f"Warning: Image path not found: {path}")

        if not similar_paths:
            raise HTTPException(status_code=404, detail="None of the specified similar images were found on the server.")

        vlm_summary = image_analysis_service.generate_vlm_summary(query_path, similar_paths, similarities, user_prompt=prompt)
        pdf_path = image_analysis_service.create_pdf_report(query_path, similar_paths, similarities, vlm_summary)
        
        # Schedule the PDF file for deletion after the response is sent
        background_tasks.add_task(remove_file, pdf_path)
        
        return FileResponse(pdf_path, media_type='application/pdf', filename=os.path.basename(pdf_path))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")
    finally:
        if os.path.exists(query_path):
            os.remove(query_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Similarity Search API. Use /docs to see the API documentation."}

