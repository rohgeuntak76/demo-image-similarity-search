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
    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg", "gif"], accept_multiple_files=True)

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create Index (Overwrite)"):
            if uploaded_files:
                files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                
                try:
                    response = requests.post(f"{BACKEND_URL}/index/create?year={year}", files=files, timeout=300)
                    if response.status_code == 201:
                        st.success(f"FAISS index for year {year} created successfully!")
                    else:
                        st.error(f"Failed to create index: {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload at least one image.")

    with col2:
        if st.button("Update Index (Append)"):
            if uploaded_files:
                files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
                
                try:
                    response = requests.post(f"{BACKEND_URL}/index/update?year={year}", files=files, timeout=300)
                    if response.status_code == 200:
                        st.success(f"FAISS index for year {year} updated successfully!")
                    else:
                        st.error(f"Failed to update index: {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
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
                    img_url = f"{BACKEND_URL}/data/images/indexed/{res['year']}/{res['filename']}"
                    st.image(img_url, caption=f"Top {i+1} (Year: {res['year']}, Sim: {res['similarity']:.1f}%)", width='stretch')

            st.subheader("Step 2: Generate PDF Report")
            
            # Construct default prompt
            query_name = st.session_state["query_image_name"]
            sims = [res["similarity"] for res in results]
            
            # Fill with 0 if fewer than 3 results for safe formatting
            sim_scores = sims + [0.0] * (3 - len(sims))
            
            default_prompt = f"""Analyze the following weather chart images to create a structured report.

1.  Summarize the key weather features of the [Query Image ({query_name})].
2.  For each of the [Top-3 Similar Weather Charts], describe its main features, its similarities and differences compared to the query image, and its similarity score (%).
    - Top 1 Similarity: {sim_scores[0]:.1f}%
    - Top 2 Similarity: {sim_scores[1]:.1f}%
    - Top 3 Similarity: {sim_scores[2]:.1f}%
3.  Provide a comprehensive comparison of all four images, highlighting common patterns and notable differences.
4.  Explain the meteorological basis for the high similarity.
5.  Include any other relevant observations.

Please write the report in Korean at an expert level, using clear sections or tables."""

            user_prompt = st.text_area("VLM Analysis Prompt:", value=default_prompt, height=400)

            if st.button("Generate Final Report"):
                files = {"file": (st.session_state["query_image_name"], st.session_state["query_image_data"], st.session_state["query_image_type"])}
                
                # Send the exact search results shown in the preview to the backend
                search_results_json = json.dumps(st.session_state["search_results"]["results"])
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
