# 🤖 Plateforme NLP avec Mistral et TorchServe

Plateforme complète d'orchestration NLP utilisant Mistral-7B-Instruct via TorchServe avec une API REST FastAPI.

## 📋 Table des matières

- [🚀 Démarrage rapide](#-démarrage-rapide)
- [🏗️ Architecture](#️-architecture)
- [📦 Installation](#-installation)
- [🔧 Configuration](#-configuration)
- [🚀 Utilisation](#-utilisation)
- [📡 API Endpoints](#-api-endpoints)
- [📊 Monitoring](#-monitoring)
- [🐳 Docker](#-docker)
- [🛠️ Développement](#️-développement)
- [📚 Documentation](#-documentation)

## 🚀 Démarrage rapide

### Prérequis

- Docker & Docker Compose
- Python 3.10+
- CUDA-compatible GPU (optionnel mais recommandé)
- 16GB+ RAM
- 50GB+ espace disque

### Installation en 1 commande

```bash
# Clone et déployement complet
git clone <your-repo>
cd nlp-platform-mistral
chmod +x deploy.sh
./deploy.sh dev setup
```

### Test rapide

```bash
# Test de génération de texte
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Explique-moi l'\''IA en 3 phrases", "max_tokens": 200}'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│                 │    │                  │    │                 │
│   Client Web    │◄──►│   FastAPI        │◄──►│   TorchServe    │
│   /Mobile/API   │    │   Orchestrator   │    │   + Mistral     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │                  │
                       │   Redis Cache    │
                       │   + Monitoring   │
                       │                  │
                       └──────────────────┘
```

### Composants principaux

1. **TorchServe** : Serveur de modèles ML avec Mistral-7B-Instruct
2. **FastAPI Orchestrator** : API REST avec gestion des requêtes, cache, métriques
3. **Redis** : Cache distribué pour les réponses
4. **Prometheus + Grafana** : Monitoring et métriques
5. **Custom Handler** : Gestionnaire optimisé pour Mistral

## 📦 Installation

### 1. Installation manuelle

```bash
# Clone du repository
git clone <your-repo>
cd nlp-platform-mistral

# Installation des dépendances
pip install -r requirements.txt

# Configuration des modèles
cd mistral
chmod +x convert_model.sh
./convert_model.sh

# Démarrage de TorchServe
cd ../torchserve
torchserve --start --model-store model-store --ts-config config.properties

# Démarrage de l'orchestrateur
python api_server.py
```

### 2. Installation avec Docker

```bash
# Démarrage avec Docker Compose
docker-compose up -d

# Vérification
docker-compose ps
curl http://localhost:8000/health
```

## 🔧 Configuration

### Fichiers de configuration principaux

#### `mistral/config/config.json`
Configuration du modèle Mistral avec paramètres de génération.

#### `torchserve/config.properties`
Configuration TorchServe (workers, mémoire, GPU).

#### `torchserve/custom_handler.py`
Handler personnalisé avec optimisations pour Mistral.

#### `docker-compose.yml`
Orchestration des services Docker.

### Variables d'environnement

```bash
# TorchServe
TORCHSERVE_URL=http://localhost:8080

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MAX_WORKERS=10

# GPU
CUDA_VISIBLE_DEVICES=0
```

## 🚀 Utilisation

### Démarrage des services

```bash
# Environnement de développement
./deploy.sh dev start

# Environnement de production
./deploy.sh prod start

# Vérification du status
./deploy.sh dev status
```

### Commandes utiles

```bash
# Logs en temps réel
./deploy.sh dev logs

# Test de fonctionnalité
./deploy.sh dev test

# Restart des services
./deploy.sh dev restart

# Sauvegarde
./deploy.sh dev backup

# Nettoyage
./deploy.sh dev cleanup
```

## 📡 API Endpoints

### Base URL: `http://localhost:8000`

### Santé et status

```http
GET /health          # Vérification de santé
GET /status          # Status complet du système
GET /metrics         # Métriques de performance
GET /models          # Modèles disponibles
```

### Génération de texte

```http
POST /generate
Content-Type: application/json

{
  "text": "Votre prompt ici",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

### Question-Réponse

```http
POST /question-answering
Content-Type: application/json

{
  "question": "Qu'est-ce que l'intelligence artificielle?",
  "context": "L'IA est une technologie...",
  "max_tokens": 300
}
```

### Résumé de texte

```http
POST /summarize
Content-Type: application/json

{
  "text": "Texte long à résumer...",
  "max_length": 150,
  "temperature": 0.3
}
```

### Traduction

```http
POST /translate
Content-Type: application/json

{
  "text": "Hello, how are you?",
  "target_language": "français",
  "max_tokens": 100
}
```

### Classification

```http
POST /classify
Content-Type: application/json

{
  "text": "Ce produit est fantastique!",
  "categories": ["positif", "négatif", "neutre"]
}
```

### Traitement par batch

```http
POST /batch
Content-Type: application/json

{
  "requests": [
    {
      "text": "Première requête",
      "task_type": "text_generation",
      "parameters": {"max_tokens": 100}
    },
    {
      "text": "Deuxième requête", 
      "task_type": "question_answering",
      "parameters": {"max_tokens": 150}
    }
  ],
  "priority": 2
}
```

## 📊 Monitoring

### Métriques disponibles

- **Performance** : Temps de réponse, throughput, latence
- **Utilisation** : CPU, GPU, mémoire
- **Cache** : Taux de hit, taille du cache
- **Erreurs** : Taux d'erreur, types d'erreurs
- **Modèles** : Utilisation des modèles, versions

### Dashboards

- **Grafana** : http://localhost:3000 (admin/admin123)
- **Prometheus** : http://localhost:9090
- **TorchServe Metrics** : http://localhost:8082

### Logs

```bash
# Logs de l'API
docker-compose logs nlp-orchestrator

# Logs de TorchServe
docker-compose logs torchserve

# Logs temps réel
docker-compose logs -f --tail=100
```

## 🐳 Docker

### Images utilisées

- `pytorch/torchserve:latest-gpu` : TorchServe avec support GPU
- `python:3.10-slim` : Base pour l'orchestrateur
- `redis:7-alpine` : Cache Redis
- `prom/prometheus:latest` : Métriques
- `grafana/grafana:latest` : Dashboards

### Volumes

```yaml
volumes:
  - ./torchserve/model-store:/home/model-server/model-store
  - ./logs:/app/logs
  - prometheus_data:/prometheus
  - grafana_data:/var/lib/grafana
  - redis_data:/data
```

### Network

Réseau Docker `172.20.0.0/16` pour communication inter-services.

## 🛠️ Développement

### Structure du projet

```
nlp-platform-mistral/
├── mistral/
│   ├── config/config.json
│   └── convert_model.sh
├── torchserve/
│   ├── config.properties
│   ├── custom_handler.py
│   └── model-store/
├── orchestration_client.py
├── api_server.py
├── docker-compose.yml
├── Dockerfile.orchestrator
├── requirements.txt
├── deploy.sh
└── README.md
```

### Développement local

```bash
# Installation en mode développement
pip install -r requirements.txt
pip install -e .

# Variables d'environnement
export TORCHSERVE_URL=http://localhost:8080
export LOG_LEVEL=DEBUG

# Démarrage en mode debug
python api_server.py --reload
```

### Tests

```bash
# Tests unitaires
pytest tests/

# Tests d'intégration
pytest tests/integration/

# Tests de charge
pytest tests/load/

# Coverage
pytest --cov=src tests/
```

### Contribution

1. Fork du repository
2. Création d'une branche feature
3. Tests et documentation
4. Pull request avec description détaillée

## 📚 Documentation

### API Documentation

- **OpenAPI/Swagger** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

### Exemples de code

#### Python

```python
import asyncio
from orchestration_client import NLPPlatformAPI, NLPRequest, TaskType

async def main():
    api = NLPPlatformAPI()
    
    # Génération simple
    response = await api.generate_text(
        "Explique-moi l'IA",
        max_tokens=200
    )
    print(response.generated_text)

asyncio.run(main())
```

#### cURL

```bash
# Génération de texte
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Bonjour", "max_tokens": 100}'

# Avec authentification (si activée)
curl -X POST "http://localhost:8000/generate" \
     -H "Authorization: Bearer your-token" \
     -H "Content-Type: application/json" \
     -d '{"text": "Bonjour"}'
```

#### JavaScript/Node.js

```javascript
const axios = require('axios');

async function generateText(prompt) {
  try {
    const response = await axios.post('http://localhost:8000/generate', {
      text: prompt,
      max_tokens: 200,
      temperature: 0.7
    });
    
    return response.data.data.generated_text;
  } catch (error) {
    console.error('Erreur:', error.response.data);
  }
}
```

## 🔧 Troubleshooting

### Problèmes courants

#### TorchServe ne démarre pas

```bash
# Vérifier les logs
docker-compose logs torchserve

# Vérifier l'espace disque
df -h

# Vérifier la mémoire
free -h

# Restart du service
docker-compose restart torchserve
```

#### Erreur GPU

```bash
# Vérifier NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Vérifier les drivers
nvidia-smi

# Mode CPU forcé
export CUDA_VISIBLE_DEVICES=""
```

#### Modèle non trouvé

```bash
# Reconvertir le modèle
cd mistral
./convert_model.sh

# Vérifier le model store
ls -la torchserve/model-store/

# Réenregistrer le modèle
curl -X POST "http://localhost:8081/models?url=mistral-7b-instruct.mar"
```

#### Problèmes de performance

```bash
# Métriques système
docker stats

# Métriques applicatives
curl http://localhost:8000/metrics

# Optimisation GPU
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
```

### Support

- **Issues GitHub** : Rapporter les bugs et demandes de fonctionnalités
- **Discussions** : Questions générales et aide
- **Wiki** : Documentation technique approfondie

## 📄 Licence

MIT License - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🚀 Roadmap

- [ ] Support multi-modèles (Llama2, Falcon, etc.)
- [ ] Interface web React
- [ ] API Gateway avec rate limiting
- [ ] Déploiement Kubernetes
- [ ] Support de fine-tuning
- [ ] Intégration avec vector databases
- [ ] Plugin système pour extensibilité

---

**Version** : 1.0.0  
**Dernière mise à jour** : Septembre 2025