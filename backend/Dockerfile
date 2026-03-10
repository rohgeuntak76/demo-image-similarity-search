# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for WeasyPrint and FAISS
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b \
    libpango1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ResNet-50 model weights
RUN python3 -c 'import torchvision.models as models; models.resnet50(weights="ResNet50_Weights.DEFAULT")'

# Copy the current directory contents into the container at /app
COPY app /app/app
COPY data /app/data
COPY fonts /app/fonts
COPY .env /app/.env

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the uvicorn server when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
