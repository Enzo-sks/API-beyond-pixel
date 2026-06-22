FROM python:3.11-slim

WORKDIR /app

# Installer la bonne version de NumPy AVANT PyTorch
RUN pip install numpy==1.26.4

# Installer PyTorch CPU (qui respectera la version de NumPy déjà installée)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier le code
COPY . .

# Lancer FastAPI avec le port Railway
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
