FROM python:3.10-slim AS builder

WORKDIR /app

# Installer les dépendances système de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et les installer dans un virtualenv temporaire
COPY ./requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

# Copier uniquement les dépendances installées depuis le builder
COPY --from=builder /install /usr/local

# Copier le code de l'application
COPY ./app ./app

# Expose le port FastAPI
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
