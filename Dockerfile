# ---- Base Image ----
FROM python:3.10-slim

# ---- Prevent Python from buffering + create app dir ----
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# ---- System Dependencies (for PyPDF, Torch etc) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy requirements ----
COPY requirements.txt .

# ---- Install Python dependencies ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy your whole project ----
COPY . .

# ---- Expose port ----
EXPOSE 8000

# ---- Start Gunicorn ----
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]

