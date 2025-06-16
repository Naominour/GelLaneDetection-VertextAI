FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/models/ ./src/models/
COPY model.pth ./
COPY inference_preprocessing.py ./
COPY predictor.py ./

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8080"]
