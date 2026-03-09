# Frontend Application

This directory contains the Streamlit-based frontend application for the Image Similarity Search project. It provides a user interface to interact with the backend API, allowing you to create FAISS indexes and generate analysis reports.

## Setup

1.  Navigate into the `frontend/` directory:
    ```bash
    cd frontend
    ```
2.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the Streamlit application, run the following command from within the `frontend/` directory:

```bash
streamlit run app.py
```

The application will typically open in your web browser at `http://localhost:8501`.

## Configuration

The frontend application communicates with the backend API. By default, it expects the backend to be running at `http://localhost:8000`. If your backend is running at a different address, you can configure it using the `BACKEND_URL` environment variable:

```bash
BACKEND_URL="http://your-backend-address:port" streamlit run app.py
```
