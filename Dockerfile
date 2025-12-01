FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch first to avoid pulling CUDA in Docker
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py", "--demo", "--mock", "--artifacts-dir", "/app/artifacts"]