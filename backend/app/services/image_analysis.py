import os
import numpy as np
from PIL import Image
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import markdown
from weasyprint import HTML
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
        
        # Ensure data directories exist
        os.makedirs(settings.INDEXED_IMAGE_DIR, exist_ok=True)

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        resnet50 = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        model = torch.nn.Sequential(*list(resnet50.children())[:-1])
        model.to(self.device).eval()
        return model

    def extract_embedding(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(input_tensor).squeeze().cpu().numpy()
        return feat

    def create_index(self, image_paths: List[str], year: int):
        embeddings = []
        for path in image_paths:
            emb = self.extract_embedding(path)
            embeddings.append(emb)
        
        embeddings = np.vstack(embeddings).astype("float32")
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        # Construct paths with year-specific directory
        year_index_dir = os.path.join(settings.BASE_INDEX_DIR, str(year))
        os.makedirs(year_index_dir, exist_ok=True) 

        index_file_path = os.path.join(year_index_dir, f"index.faiss")
        embeddings_file_path = os.path.join(year_index_dir, f"embeddings.npy")
        names_file_path = os.path.join(year_index_dir, f"names.npy")

        faiss.write_index(index, index_file_path)
        np.save(embeddings_file_path, embeddings)
        np.save(names_file_path, np.array(image_paths))

    def update_index(self, image_paths: List[str], year: int):
        year_index_dir = os.path.join(settings.BASE_INDEX_DIR, str(year))
        index_file_path = os.path.join(year_index_dir, "index.faiss")
        embeddings_file_path = os.path.join(year_index_dir, "embeddings.npy")
        names_file_path = os.path.join(year_index_dir, "names.npy")

        if not os.path.exists(index_file_path):
            return self.create_index(image_paths, year)

        # Load existing index and metadata
        index = faiss.read_index(index_file_path)
        existing_embeddings = np.load(embeddings_file_path)
        existing_names = np.load(names_file_path).tolist()

        new_embeddings = []
        for path in image_paths:
            emb = self.extract_embedding(path)
            new_embeddings.append(emb)
        
        new_embeddings = np.vstack(new_embeddings).astype("float32")
        
        # Update FAISS index
        index.add(new_embeddings)
        
        # Combine with existing metadata
        combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
        existing_names.extend(image_paths)
        
        # Save updated data
        faiss.write_index(index, index_file_path)
        np.save(embeddings_file_path, combined_embeddings)
        np.save(names_file_path, np.array(existing_names))

    def search_similar(self, query_image_path: str, year: int, k: int = 3) -> Tuple[List[float], List[str]]:
        # Construct paths with year-specific directory
        year_index_dir = os.path.join(settings.BASE_INDEX_DIR, str(year))
        index_file_path = os.path.join(year_index_dir, f"index.faiss")
        names_file_path = os.path.join(year_index_dir, f"names.npy")

        if not os.path.exists(index_file_path):
            raise FileNotFoundError(f"FAISS index for year '{year}' not found. Please create it first.")

        index = faiss.read_index(index_file_path)
        names = np.load(names_file_path)
        
        query_vec = self.extract_embedding(query_image_path).reshape(1, -1).astype("float32")
        distances, indices = index.search(query_vec, k)
        
        max_dist = np.max(distances)
        similarity_percent = [100 * (1 - (d / max_dist)) for d in distances[0]]
        similar_paths = [names[idx] for idx in indices[0]]
        
        return similarity_percent, similar_paths

    def image_to_base64(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_vlm_summary(self, query_path: str, similar_paths: List[str], similarities: List[float], user_prompt: str = None) -> str:
        query_filename = os.path.basename(query_path)
        
        system_prompt = """
        You are a senior meteorologist and an expert in synoptic weather chart analysis. Your primary role is to analyze, compare, and interpret complex meteorological data from weather charts (such as surface pressure maps, isobaric charts, and frontal analyses).
        When analyzing these images, you must adhere strictly to the following scientific and formatting guidelines:
        1. **Analytical Depth:**
        - Explicitly identify key synoptic features: High (H) and Low (L) pressure centers, pressure gradients (isobar spacing), and prevailing wind directions.
        - Locate and classify frontal boundaries (cold, warm, stationary, or occluded fronts).
        - Recognize macro-weather patterns, especially those relevant to the East Asian region (e.g., Siberian High expansion, North Pacific High, Changma/monsoon fronts, or typhoons).
        2. **Comparative Logic:**
        - When comparing a query image to historical/similar images, focus on the structural alignment of pressure systems and fronts.
        - Explain *why* the retrieval algorithm flagged them as similar based on actual atmospheric physics, not just visual pixel overlap.
        3. **Tone and Language:**
        - Your tone must be highly objective, academic, and professional.
        - **Crucial:** You must generate the entire final response in expert-level Korean (기상청 및 대기과학 전공자 수준의 전문 용어 사용).
        4. **Formatting Strictness:**
        - You are strictly bound to output in pure Markdown. 
        - Use Markdown headers (###), bullet points (-), bold text (**text**), and Markdown tables (| Column | Column |).
        - CRITICAL: Output the raw Markdown directly. Do NOT wrap your response in ```markdown ... ``` code blocks.
        """

        if user_prompt:
            summary_prompt = user_prompt
        else:
            summary_prompt = f"""
            Analyze the following weather chart images and create a structured meteorological report.

            ### 1. Query Image Summary
            Summarize the key weather features of the query image ({query_filename}).

            ### 2. Top-3 Similar Weather Charts Analysis
            For each retrieved chart, describe its main features, similarities, and differences compared to the query image. Reference these similarity scores:
            - Top 1 Similarity: {similarities[0]:.1f}%
            - Top 2 Similarity: {similarities[1]:.1f}%
            - Top 3 Similarity: {similarities[2]:.1f}%

            ### 3. Comprehensive Comparison
            Provide a comparison of all four images, highlighting common patterns and notable differences. **Please use a Markdown table** to summarize the key comparative metrics.

            ### 4. Meteorological Basis
            Explain the meteorological basis for the high similarity among these charts.

            ### 5. Additional Relevant Observations
            Provide any other Relevant observations to complement the report

            Please write the report entirely in Korean at an expert level.
            """

        query_img_b64 = self.image_to_base64(query_path)
        similar_imgs_b64 = [self.image_to_base64(p) for p in similar_paths]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": summary_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{query_img_b64}"}},
                ] + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in similar_imgs_b64]
            }
        ]
        
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.VLM_BASE_URL)
        try:
            response = client.chat.completions.create(
                model=settings.VLM_MODEL_NAME,
                messages=messages,
                max_tokens=16384
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to generate VLM summary: {e}"

    def create_pdf_report(self, query_path: str, similar_paths: List[str], similarities: List[float], vlm_summary: str) -> str:
        html_content = markdown.markdown(vlm_summary, extensions=['tables'])
        query_uri = f"file://{os.path.abspath(query_path)}"
        
        similar_images_html = ""
        for i, (path, sim) in enumerate(zip(similar_paths, similarities)):
            img_uri = f"file://{os.path.abspath(path)}"
            similar_images_html += f"""
            <div class="image-box">
                <h4>Top {i+1} (Similarity: {sim:.1f}%)</h4>
                <img src="{img_uri}" alt="Similar Image {i+1}">
            </div>"""

        full_html = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
                body {{ font-family: 'Nanum Gothic', sans-serif; line-height: 1.6; color: #333; margin: 30px; }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                .header-img {{ width: 60%; max-width: 600px; margin-bottom: 10px; display: block; }}
                .image-container {{ display: flex; flex-wrap: wrap; gap: 2%; margin-bottom: 30px; }}
                .image-box {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; text-align: center; width: 48%; margin-bottom: 15px; page-break-inside: avoid; box-sizing: border-box; }}
                .image-box img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
                th {{ background-color: #ecf0f1; font-weight: bold; }}
                /* --- PDF Pagination Rules --- */
                .keep-together {{ page-break-inside: avoid; break-inside: avoid; }}
                .vlm-section {{ page-break-before: always; break-before: page; }} /* Pushes VLM to the next page */
            </style>
        </head>
        <body>
            <div class="keep-together">
                <h1 style="text-align: center; border-bottom: none;">유사 일기도 이미지 구조화 자동 보고서</h1>
                
                <h3>[쿼리 이미지] {os.path.basename(query_path)}</h3>
                <img class="header-img" src="{query_uri}">
                
                <h3>Top 3 Similar Charts</h3>
                <div class="image-container">{similar_images_html}</div>
            </div>
            <div class="vlm-section">
                    <h3>VLM Analysis - {settings.VLM_MODEL_NAME}</h3>
                    <div class="vlm-content">{html_content}</div>
            </div>
        </body>
        </html>"""

        pdf_filename = f"report_{os.path.basename(query_path).split('.')[0]}.pdf"
        os.makedirs("data", exist_ok=True)
        pdf_path = os.path.join("data", pdf_filename)
        HTML(string=full_html).write_pdf(pdf_path)
        return pdf_path

image_analysis_service = ImageAnalysisService()
