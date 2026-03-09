# Image Similarity Search Monorepo

This repository contains a full-stack application for searching similar weather charts and generating AI-powered analysis reports. It leverages ResNet-50 for feature extraction, FAISS for efficient similarity searching, and a Vision Language Model (VLM) for expert report generation.

## Project Structure

-   **`backend/`**: A FastAPI-based REST API that handles image embedding, FAISS indexing, similarity searching, and PDF report generation using a locally deployed VLM.
-   **`frontend/`**: A Streamlit-based web dashboard that provides an intuitive interface for managing indexes, previewing similar images, and interactive report generation.

## Getting Started

### 1. Backend Setup
Navigate to the `backend/` directory to set up the API and configure your VLM environment.
```bash
cd backend
# Refer to backend/README.md for detailed instructions
```

### 2. Frontend Setup
Navigate to the `frontend/` directory to set up the web dashboard.
```bash
cd frontend
# Refer to frontend/README.md for detailed instructions
```

## Docker Deployment

Both components are Dockerized for consistent deployment across environments. You can build and run them individually or use them as part of a larger containerized stack.

- **Backend**: `docker build -t image-search-backend backend/`
- **Frontend**: `docker build -t image-search-frontend frontend/`

Refer to the respective `README.md` files in each directory for specific Docker run commands and environment variable configurations.
