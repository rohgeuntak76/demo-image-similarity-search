import os
import numpy as np
from PIL import Image
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from fpdf import FPDF
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
        # os.makedirs(os.path.dirname(settings.INDEX_PATH), exist_ok=True) # No longer needed, year-specific directories are created in create_index
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
        os.makedirs(year_index_dir, exist_ok=True) # Ensure year directory exists

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

    def generate_vlm_summary(self, query_path: str, similar_paths: List[str], similarities: List[float], user_prompt: str = None) -> str:
        if user_prompt:
            summary_prompt = user_prompt
        else:
            query_filename = os.path.basename(query_path)
            summary_prompt = f"""
            Analyze the following weather chart images to create a structured report.

            1.  Summarize the key weather features of the [Query Image ({query_filename})].
            2.  For each of the [Top-3 Similar Weather Charts], describe its main features, its similarities and differences compared to the query image, and its similarity score (%).
                - Top 1 Similarity: {similarities[0]:.1f}%
                - Top 2 Similarity: {similarities[1]:.1f}%
                - Top 3 Similarity: {similarities[2]:.1f}%
            3.  Provide a comprehensive comparison of all four images, highlighting common patterns and notable differences.
            4.  Explain the meteorological basis for the high similarity.
            5.  Include any other relevant observations.

            Please write the report in Korean at an expert level, using clear sections or tables.
            """

        def image_to_base64(image_path):
            img = Image.open(image_path).convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        query_img_b64 = image_to_base64(query_path)
        similar_imgs_b64 = [image_to_base64(p) for p in similar_paths]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": summary_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{query_img_b64}"}},
                ] + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in similar_imgs_b64]
            }
        ]
        
        client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.VLM_BASE_URL,
        )
        try:
            response = client.chat.completions.create(
                model=settings.VLM_MODEL_NAME,
                messages=messages,
                # max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to generate VLM summary: {e}"

    def create_pdf_report(self, query_path: str, similar_paths: List[str], similarities: List[float], vlm_summary: str) -> str:
        
        class PDF_Korean(FPDF):
            def __init__(self):
                super().__init__()
                self.add_font("Nanum", "", settings.FONT_PATH, uni=True)
                self.set_font("Nanum", size=14)
            def header(self):
                self.set_font("Nanum", size=18)
                self.cell(0, 15, "Similar Weather Chart Analysis Report", ln=True, align='C')
                self.ln(5)

        pdf = PDF_Korean()
        pdf.add_page()
        
        pdf.set_font("Nanum", size=13)
        pdf.cell(0, 10, f"[Query Image] {os.path.basename(query_path)}", ln=True)
        pdf.image(query_path, w=90)
        pdf.ln(6)

        for i, path in enumerate(similar_paths):
            pdf.cell(0, 10, f"[Top {i+1}] {os.path.basename(path)} (Similarity: {similarities[i]:.1f}%)", ln=True)
            pdf.image(path, w=90)
            pdf.ln(4)

        pdf.set_font("Nanum", size=12)
        pdf.multi_cell(0, 10, "VLM Analysis:\n" + vlm_summary)

        pdf_filename = f"report_{os.path.basename(query_path).split('.')[0]}.pdf"
        pdf_path = os.path.join("data", pdf_filename)
        pdf.output(pdf_path)
        return pdf_path

image_analysis_service = ImageAnalysisService()
