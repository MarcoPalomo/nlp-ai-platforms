#!/bin/bash

# Script de déploiement complet pour la plateforme NLP Mistral/TorchServe
# Usage: ./deploy.sh [environment] [action]
# Environments: dev, staging, prod
# Actions: setup, start, stop, restart, logs, status

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT=${1:-dev}
ACTION=${2:-setup}
PROJECT_NAME="nlp-platform-mistral"

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Vérification des prérequis
check_prerequisites() {
    log "🔍 Vérification des prérequis..."
    
    # Docker
    if ! command -v docker &> /dev/null; then
        error "Docker n'est pas installé"
        exit 1
    fi
    
    # Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        if ! docker compose version &> /dev/null; then
            error "Docker Compose n'est pas installé"
            exit 1
        fi
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    
    # Python
    if ! command -v python3 &> /dev/null; then
        warn "Python3 non trouvé, certaines fonctionnalités peuvent être limitées"
    fi
    
    # NVIDIA Docker (pour GPU)
    if command -v nvidia-smi &> /dev/null; then
        if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
            warn "NVIDIA Docker runtime non configuré correctement"
        else
            info "✅ Support GPU détecté"
        fi
    else
        warn "Pas de GPU NVIDIA détecté, fonctionnement en mode CPU"
    fi
    
    log "✅ Prérequis vérifiés"
}

# Configuration de l'environnement
setup_environment() {
    log "🛠️ Configuration de l'environnement $ENVIRONMENT..."
    
    # Création des répertoires
    mkdir -p logs model-store workflow-store cache monitoring/{prometheus,grafana/{dashboards,datasources}}
    
    # Création du fichier .env
    cat > .env << EOF
# Configuration pour environnement: $ENVIRONMENT
ENVIRONMENT=$ENVIRONMENT
PROJECT_NAME=$PROJECT_NAME

# TorchServe
TORCHSERVE_URL=http://localhost:8080
TORCHSERVE_MANAGEMENT_URL=http://localhost:8081
TORCHSERVE_METRICS_URL=http://localhost:8082

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MAX_WORKERS=10
CACHE_SIZE=1000

# Base de données (optionnel)
REDIS_URL=redis://localhost:6379

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=admin123

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8

# Sécurité
API_KEY_ENABLED=false
JWT_SECRET_KEY=your-secret-key-here
EOF

    # Configuration Prometheus
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'torchserve'
    static_configs:
      - targets: ['torchserve:8082']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'nlp-api'
    static_configs:
      - targets: ['nlp-orchestrator:8001']
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

    # Dashboard Grafana de base
    cat > monitoring/grafana/dashboards/dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "NLP Platform Dashboard",
    "panels": [
      {
        "title": "Requests per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(requests_total[5m])",
            "legendFormat": "RPS"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "avg(response_time_seconds)",
            "legendFormat": "Avg Response Time"
          }
        ]
      }
    ]
  }
}
EOF

    log "✅ Environnement $ENVIRONMENT configuré"
}

# Construction des images Docker
build_images() {
    log "🏗️ Construction des images Docker..."
    
    # Construction de l'orchestrateur
    $DOCKER_COMPOSE_CMD build nlp-orchestrator
    
    log "✅ Images construites"
}

# Conversion et déploiement du modèle
deploy_model() {
    log "📦 Déploiement du modèle Mistral..."
    
    cd mistral
    
    # Rendre le script exécutable
    chmod +x convert_model.sh
    
    # Conversion du modèle
    ./convert_model.sh
    
    cd ..
    
    log "✅ Modèle déployé"
}

# Démarrage des services
start_services() {
    log "🚀 Démarrage des services..."
    
    # Démarrage avec Docker Compose
    $DOCKER_COMPOSE_CMD up -d
    
    # Attente du démarrage de TorchServe
    log "⏳ Attente du démarrage de TorchServe..."
    timeout=300
    elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if curl -f http://localhost:8081/ping >/dev/null 2>&1; then
            log "✅ TorchServe démarré"
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    
    if [ $elapsed -ge $timeout ]; then
        error "Timeout: TorchServe n'a pas démarré"
        exit 1
    fi
    
    # Enregistrement du modèle
    log "📝 Enregistrement du modèle..."
    curl -X POST "http://localhost:8081/models?url=mistral-7b-instruct.mar" || {
        warn "Erreur lors de l'enregistrement du modèle"
    }
    
    # Vérification de l'API
    log "🔍 Vérification de l'API..."
    timeout=60
    elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            log "✅ API NLP démarrée"
            break
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    
    log "🎉 Services démarrés avec succès!"
    echo ""
    echo "📡 Endpoints disponibles:"
    echo "  - API NLP: http://localhost:8000"
    echo "  - Documentation: http://localhost:8000/docs"
    echo "  - TorchServe: http://localhost:8080"
    echo "  - Gestion TorchServe: http://localhost:8081"
    echo "  - Métriques: http://localhost:8082"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin123)"
}

# Arrêt des services
stop_services() {
    log "🛑 Arrêt des services..."
    $DOCKER_COMPOSE_CMD down
    log "✅ Services arrêtés"
}

# Redémarrage des services
restart_services() {
    log "🔄 Redémarrage des services..."
    stop_services
    start_services
}

# Affichage des logs
show_logs() {
    log "📋 Logs des services..."
    $DOCKER_COMPOSE_CMD logs -f --tail=100
}

# Status des services
show_status() {
    log "📊 Status des services..."
    
    echo ""
    echo "🐳 Docker Compose:"
    $DOCKER_COMPOSE_CMD ps
    
    echo ""
    echo "🏥 Health Checks:"
    
    # TorchServe
    if curl -f http://localhost:8081/ping >/dev/null 2>&1; then
        echo "✅ TorchServe: Healthy"
    else
        echo "❌ TorchServe: Unhealthy"
    fi
    
    # API NLP
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ NLP API: Healthy"
        
        # Métriques
        echo ""
        echo "📈 Métriques:"
        curl -s http://localhost:8000/metrics | python3 -m json.tool 2>/dev/null || echo "Erreur récupération métriques"
    else
        echo "❌ NLP API: Unhealthy"
    fi
    
    # Modèles TorchServe
    echo ""
    echo "🤖 Modèles disponibles:"
    curl -s http://localhost:8081/models 2>/dev/null | python3 -m json.tool || echo "Aucun modèle ou TorchServe indisponible"
}

# Test de fonctionnalité
test_platform() {
    log "🧪 Test de la plateforme..."
    
    # Test simple de génération
    echo ""
    echo "📝 Test de génération de texte:"
    curl -X POST "http://localhost:8000/generate" \
         -H "Content-Type: application/json" \
         -d '{"text": "Bonjour, comment ça va?", "max_tokens": 50}' \
         2>/dev/null | python3 -m json.tool || echo "Erreur de test"
    
    echo ""
    echo "❓ Test de question-réponse:"
    curl -X POST "http://localhost:8000/question-answering" \
         -H "Content-Type: application/json" \
         -d '{"question": "Qu'\''est-ce que l'\''IA?", "max_tokens": 100}' \
         2>/dev/null | python3 -m json.tool || echo "Erreur de test"
}

# Nettoyage
cleanup() {
    log "🧹 Nettoyage..."
    
    # Arrêt des services
    $DOCKER_COMPOSE_CMD down -v --remove-orphans
    
    # Suppression des images (optionnel)
    read -p "Supprimer les images Docker? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi $(docker images "$PROJECT_NAME*" -q) 2>/dev/null || true
    fi
    
    # Nettoyage des volumes
    docker volume prune -f
    
    log "✅ Nettoyage terminé"
}

# Backup de la configuration
backup_config() {
    log "💾 Sauvegarde de la configuration..."
    
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Sauvegarde des fichiers de config
    cp -r mistral/config "$backup_dir/"
    cp torchserve/config.properties "$backup_dir/"
    cp docker-compose.yml "$backup_dir/"
    cp .env "$backup_dir/" 2>/dev/null || true
    
    # Archive
    tar -czf "${backup_dir}.tar.gz" "$backup_dir"
    rm -rf "$backup_dir"
    
    log "✅ Sauvegarde créée: ${backup_dir}.tar.gz"
}

# Menu principal
show_usage() {
    echo "Usage: $0 [environment] [action]"
    echo ""
    echo "Environments:"
    echo "  dev      - Environnement de développement"
    echo "  staging  - Environnement de test"
    echo "  prod     - Environnement de production"
    echo ""
    echo "Actions:"
    echo "  setup    - Configuration initiale complète"
    echo "  start    - Démarrage des services"
    echo "  stop     - Arrêt des services"
    echo "  restart  - Redémarrage des services"
    echo "  logs     - Affichage des logs"
    echo "  status   - Status des services"
    echo "  test     - Test de fonctionnalité"
    echo "  backup   - Sauvegarde de la configuration"
    echo "  cleanup  - Nettoyage complet"
    echo ""
    echo "Exemples:"
    echo "  $0 dev setup     # Configuration développement"
    echo "  $0 prod start    # Démarrage production"
    echo "  $0 staging test  # Test staging"
}

# Fonction principale
main() {
    echo "=========================================="
    echo "  🤖 NLP Platform Mistral/TorchServe"
    echo "  Environment: $ENVIRONMENT"
    echo "  Action: $ACTION"
    echo "=========================================="
    echo ""
    
    cd "$SCRIPT_DIR"
    
    case $ACTION in
        setup)
            check_prerequisites
            setup_environment
            build_images
            deploy_model
            start_services
            test_platform
            echo ""
            log "🎉 Setup complet terminé!"
            echo ""
            echo "🚀 Plateforme prête à l'utilisation:"
            echo "  - API: http://localhost:8000/docs"
            echo "  - Monitoring: http://localhost:3000"
            echo ""
            ;;
        start)
            check_prerequisites
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        test)
            test_platform
            ;;
        backup)
            backup_config
            ;;
        cleanup)
            cleanup
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Gestion des signaux
trap 'echo ""; log "🛑 Interruption détectée, nettoyage..."; cleanup; exit 1' INT TERM

# Validation des paramètres
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    error "Environnement invalide: $ENVIRONMENT"
    show_usage
    exit 1
fi

# Exécution
main "$@"