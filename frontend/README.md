# Image Similarity Search Frontend

This is a Streamlit-based web interface for the Image Similarity Search API. It allows users to manage FAISS indexes and generate detailed AI-powered weather chart analysis reports.

## Features

-   **FAISS Index Management**: Create or update image search indexes for specific years.
-   **2-Step Report Generation**:
    1.  **Search & Preview**: Upload a query image and preview the top 3 similar charts from the database.
    2.  **Interactive Analysis**: Review and edit the AI prompt, then generate a PDF report with VLM (Vision Language Model) analysis.
-   **PDF Preview**: View the generated report directly in the browser.

## Setup & Local Development

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```
    The UI will be available at `http://localhost:8501`.

## Docker Setup

To build and run the frontend using Docker:

1.  **Build the Image**:
    ```bash
    docker build -t image-search-frontend .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 8501:8501 -e BACKEND_URL="http://host.docker.internal:8000" image-search-frontend
    ```
    *Note: Use `http://host.docker.internal:8000` if your backend is running on the host machine (Mac/Windows). On Linux, use the host's IP address.*

## Configuration

The frontend connects to the backend API. You can configure the backend URL using the `BACKEND_URL` environment variable.

-   **Default**: `http://localhost:8000`
-   **Example**: `BACKEND_URL="http://api.example.com" streamlit run app.py`
