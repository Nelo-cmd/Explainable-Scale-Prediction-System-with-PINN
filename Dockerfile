FROM python:3.10-slim

WORKDIR /app

# Install CPU-only PyTorch first (smaller image)
RUN pip install --no-cache-dir torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY static/ static/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
