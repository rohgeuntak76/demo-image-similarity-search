import streamlit as st
import requests
import os
import json
import base64

# Configuration for the backend API
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
LOGO_PATH = os.getenv("LOGO_PATH", "./logo/logo.png")

st.set_page_config(page_title="Image Similarity Search UI", layout="wide")
st.logo(image=LOGO_PATH,size="large")
st.markdown(
    """
    <style>
        [alt=Logo] {
            height: 3.5rem; /* Adjust size here */
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Image Similarity Search Dashboard")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Create FAISS Index", "Generate Report"])

# --- Create/Update FAISS Index Tab ---
if page == "Create FAISS Index":
    st.header("Manage FAISS Index")
    st.write("Upload images to create a new FAISS index or update an existing one for a specific year.")

    year = st.number_input("Enter the year for the index:", min_value=2000, max_value=2100, value=2024, step=1)
    
    # Initialize a key for the file uploader to allow clearing it
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    uploaded_files = st.file_uploader(
        "Choose images...", 
        type=["png", "jpg", "jpeg", "gif"], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if st.button("Clear Uploaded Files"):
        st.session_state["uploader_key"] += 1
        st.rerun()

    col1, col2 = st.columns(2)
    
    CHUNK_SIZE = 1000

    def process_in_chunks(files, year, mode):
        total_files = len(files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, total_files, CHUNK_SIZE):
            chunk = files[i:i + CHUNK_SIZE]
            chunk_files = [("files", (file.name, file.getvalue(), file.type)) for file in chunk]
            
            # Determine endpoint: first chunk uses 'mode', subsequent chunks always use 'update'
            current_mode = mode if i == 0 else "update"
            endpoint = f"{BACKEND_URL}/index/{current_mode}?year={year}"
            
            status_text.text(f"Processing batch {i//CHUNK_SIZE + 1}... ({min(i + CHUNK_SIZE, total_files)}/{total_files})")
            
            try:
                response = requests.post(endpoint, files=chunk_files, timeout=600)
                if response.status_code not in [200, 201]:
                    st.error(f"Failed at batch starting with file {i}: {response.text}")
                    return False
            except Exception as e:
                st.error(f"Error during batch processing: {e}")
                return False
                
            progress_bar.progress(min((i + CHUNK_SIZE) / total_files, 1.0))
            
        status_text.text("Indexing complete!")
        return True

    with col1:
        if st.button("Create Index (Overwrite)"):
            if uploaded_files:
                if process_in_chunks(uploaded_files, year, "create"):
                    st.success(f"FAISS index for year {year} created successfully with {len(uploaded_files)} images!")
            else:
                st.warning("Please upload at least one image.")

    with col2:
        if st.button("Update Index (Append)"):
            if uploaded_files:
                if process_in_chunks(uploaded_files, year, "update"):
                    st.success(f"FAISS index for year {year} updated successfully with {len(uploaded_files)} images!")
            else:
                st.warning("Please upload at least one image.")

# --- Generate Report Tab ---
elif page == "Generate Report":
    st.header("Generate Analysis Report")
    st.write("Upload a query image and select years to generate a detailed analysis report.")

    # Fetch available years from backend
    try:
        available_years_response = requests.get(f"{BACKEND_URL}/index/available")
        if available_years_response.status_code == 200:
            available_years = available_years_response.json()
        else:
            available_years = []
            st.error("Failed to fetch available years from backend.")
    except Exception as e:
        available_years = []
        st.error(f"Error connecting to backend: {e}")

    if not available_years:
        st.warning("No indices available. Please create an index first.")
    else:
        selected_years = st.multiselect("Select years to search within:", options=available_years, default=available_years)
        query_image = st.file_uploader("Choose a query image...", type=["png", "jpg", "jpeg", "gif"], accept_multiple_files=False)

        if st.button("Search & Preview Similar Images"):
            if query_image and selected_years:
                files = {"file": (query_image.name, query_image.getvalue(), query_image.type)}
                years_params = "&".join([f"years={y}" for y in selected_years])
                try:
                    response = requests.post(f"{BACKEND_URL}/search?{years_params}", files=files)
                    if response.status_code == 200:
                        st.session_state["search_results"] = response.json()
                        st.session_state["query_image_name"] = query_image.name
                        st.session_state["query_image_data"] = query_image.getvalue()
                        st.session_state["query_image_type"] = query_image.type
                        st.success("Search successful!")
                    else:
                        st.error(f"Search failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please upload a query image and select years.")

        # Show preview and prompt step if search results exist
        if "search_results" in st.session_state:
            st.subheader("Step 1: Preview Similar Images")
            results = st.session_state["search_results"]["results"]
            
            # Display query image and similar images
            cols = st.columns(4)
            with cols[0]:
                st.image(st.session_state["query_image_data"], caption="Query Image", width='stretch')
            
            for i, res in enumerate(results):
                with cols[i+1]:
                    # Use base64 data from the response for the preview
                    st.image(
                        f"data:image/png;base64,{res['image_base64']}", 
                        caption=f"Top {i+1} (Year: {res['year']}, Sim: {res['similarity']:.1f}%)", 
                        width='content'
                    )

            st.subheader("Step 2: Generate PDF Report")
            
            # Construct default prompt
            query_name = st.session_state["query_image_name"]
            sims = [res["similarity"] for res in results]
            
            # Fill with 0 if fewer than 3 results for safe formatting
            sim_scores = sims + [0.0] * (3 - len(sims))
            
            default_prompt = f"""
Analyze the following weather chart images and create a structured meteorological report.

### 1. Query Image Summary
Summarize the key weather features of the query image ({query_name}).

### 2. Top-3 Similar Weather Charts Analysis
For each retrieved chart, describe its main features, similarities, and differences compared to the query image. Reference these similarity scores:
- Top 1 Similarity: {sim_scores[0]:.1f}%
- Top 2 Similarity: {sim_scores[1]:.1f}%
- Top 3 Similarity: {sim_scores[2]:.1f}%

### 3. Comprehensive Analysis
Comprehensively compare the four images and explain the overall  similarities and differences. Describe overall similarity and overall differences only

### 4. Meteorological Interpretation
Explain the meteorological Interpretation for the high similarity among these charts.

### 5. Other Notes
Provide any other Relevant observations to complement the report

Please write the report entirely in Korean at an expert level.
            """

            user_prompt = st.text_area("VLM Analysis Prompt:", value=default_prompt, height=400)

            if st.button("Generate Final Report"):
                files = {"file": (st.session_state["query_image_name"], st.session_state["query_image_data"], st.session_state["query_image_type"])}
                
                # Send the exact search results shown in the preview to the backend
                # Strip image_base64 to reduce request size since backend only needs filenames
                stripped_results = [
                    {k: v for k, v in res.items() if k != "image_base64"} 
                    for res in st.session_state["search_results"]["results"]
                ]
                search_results_json = json.dumps(stripped_results)
                data = {
                    "prompt": user_prompt,
                    "similar_images_json": search_results_json
                }
                
                with st.spinner("Generating PDF report... This may take a moment."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/report/generate", 
                            files=files, 
                            data=data,
                            timeout=300
                        )
                        if response.status_code == 200:
                            st.success("Report generated successfully!")
                            st.session_state["generated_pdf"] = response.content
                        else:
                            st.error(f"Failed to generate report: {response.text}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")

        # Step 3: Preview & Download (Persistent)
        if "generated_pdf" in st.session_state:
            st.divider()
            st.subheader("Step 3: Preview & Download Report")
            
            pdf_bytes = st.session_state["generated_pdf"]
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # PDF Preview using iframe
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf" style="border: none;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            st.download_button(
                label="Download Report PDF",
                data=pdf_bytes,
                file_name=f"report_{st.session_state['query_image_name'].split('.')[0]}.pdf",
                mime="application/pdf"
            )
