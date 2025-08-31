#!/bin/bash

# Script de conversion et packaging du modèle Mistral pour TorchServe
# Usage: ./convert_model.sh [model_path] [output_dir]

set -e

# Configuration par défaut
MODEL_NAME="mistral-7b-instruct"
MODEL_PATH=${1:-"mistralai/Mistral-7B-Instruct-v0.1"}
OUTPUT_DIR=${2:-"../torchserve/model-store"}
HANDLER_PATH="../torchserve/custom_handler.py"
CONFIG_PATH="config/config.json"

echo "🚀 Conversion du modèle Mistral pour TorchServe"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

# Vérification des dépendances
check_dependencies() {
    echo "📦 Vérification des dépendances..."
    
    if ! command -v torch-model-archiver &> /dev/null; then
        echo "❌ torch-model-archiver non trouvé. Installation..."
        pip install torchserve torch-model-archiver torch-workflow-archiver
    fi
    
    python -c "import transformers" 2>/dev/null || {
        echo "❌ transformers non trouvé. Installation..."
        pip install transformers accelerate bitsandbytes
    }
    
    echo "✅ Dépendances vérifiées"
}

# Téléchargement et préparation du modèle
prepare_model() {
    echo "📥 Préparation du modèle..."
    
    # Création du répertoire temporaire
    TEMP_DIR="/tmp/mistral_conversion"
    rm -rf $TEMP_DIR
    mkdir -p $TEMP_DIR
    
    # Script Python pour télécharger et sauvegarder le modèle
    cat > $TEMP_DIR/download_model.py << 'EOF'
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_path = sys.argv[1]
output_path = sys.argv[2]

print(f"Téléchargement du modèle: {model_path}")

try:
    # Chargement du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Chargement du modèle
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Sauvegarde
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Création du fichier de métadonnées
    metadata = {
        "model_name": "mistral-7b-instruct",
        "model_path": model_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "torch_dtype": "float16",
        "vocab_size": tokenizer.vocab_size,
        "max_position_embeddings": getattr(model.config, 'max_position_embeddings', 4096)
    }
    
    with open(os.path.join(output_path, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Modèle sauvegardé dans: {output_path}")
    
except Exception as e:
    print(f"❌ Erreur lors du téléchargement: {e}")
    sys.exit(1)
EOF

    python $TEMP_DIR/download_model.py "$MODEL_PATH" "$TEMP_DIR/model"
    echo "✅ Modèle préparé"
}

# Création de l'archive MAR
create_mar_archive() {
    echo "📦 Création de l'archive MAR..."
    
    # Création du répertoire de sortie
    mkdir -p $OUTPUT_DIR
    
    # Suppression de l'ancienne archive si elle existe
    rm -f $OUTPUT_DIR/${MODEL_NAME}.mar
    
    # Création de l'archive avec torch-model-archiver
    torch-model-archiver \
        --model-name $MODEL_NAME \
        --version 1.0 \
        --serialized-file $TEMP_DIR/model/pytorch_model.bin \
        --handler $HANDLER_PATH \
        --extra-files "$TEMP_DIR/model/config.json,$TEMP_DIR/model/tokenizer.json,$TEMP_DIR/model/tokenizer_config.json,$TEMP_DIR/model/special_tokens_map.json,$CONFIG_PATH,$TEMP_DIR/model/model_metadata.json" \
        --export-path $OUTPUT_DIR \
        --force
    
    echo "✅ Archive MAR créée: $OUTPUT_DIR/${MODEL_NAME}.mar"
}

# Validation de l'archive
validate_archive() {
    echo "🔍 Validation de l'archive..."
    
    if [ -f "$OUTPUT_DIR/${MODEL_NAME}.mar" ]; then
        file_size=$(du -h "$OUTPUT_DIR/${MODEL_NAME}.mar" | cut -f1)
        echo "✅ Archive créée avec succès ($file_size)"
        
        # Test de l'archive avec torch-model-archiver
        torch-model-archiver --export-path /tmp/test_extract --extract $OUTPUT_DIR/${MODEL_NAME}.mar
        echo "✅ Archive validée"
        rm -rf /tmp/test_extract
    else
        echo "❌ Erreur: Archive non créée"
        exit 1
    fi
}

# Nettoyage
cleanup() {
    echo "🧹 Nettoyage..."
    rm -rf $TEMP_DIR
    echo "✅ Nettoyage terminé"
}

# Fonction principale
main() {
    echo "========================================="
    echo "  Conversion Mistral pour TorchServe"
    echo "========================================="
    
    check_dependencies
    prepare_model
    create_mar_archive
    validate_archive
    cleanup
    
    echo ""
    echo "🎉 Conversion terminée avec succès!"
    echo "Archive disponible: $OUTPUT_DIR/${MODEL_NAME}.mar"
    echo ""
    echo "Prochaines étapes:"
    echo "1. Démarrer TorchServe: torchserve --start --model-store $OUTPUT_DIR"
    echo "2. Enregistrer le modèle: curl -X POST \"localhost:8081/models?url=${MODEL_NAME}.mar\""
    echo "3. Tester: curl -X POST \"localhost:8080/predictions/${MODEL_NAME}\" -H \"Content-Type: application/json\" -d '{\"text\": \"Hello, how are you?\"}'"
}

# Gestion des erreurs
trap cleanup ERR

# Exécution
main "$@"