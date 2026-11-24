
# ===== 1. Image de base =====
FROM python:3.11-slim

# ===== 2. Définir le dossier de travail =====
WORKDIR /app

# ===== 3. Copier et installer les dépendances =====
COPY model/requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt
# ===== 4. Copier le code et le modèle =====
COPY app.py .
COPY model/ ./model/

# ===== 5. Définir le chemin du modèle =====
ENV MODEL_URI=model

# ===== 6. Documenter le port de l'API =====
EXPOSE 8000

# ===== 7. Lancer l'application FastAPI =====
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
