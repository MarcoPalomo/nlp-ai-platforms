"""
Custom Handler pour Mistral avec TorchServe pour les tâches NLP avec Mistral-7B-Instruct
"""

import json
import logging
import os
import time
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import map_class_to_label, PredictionException

logger = logging.getLogger(__name__)


class MistralHandler(BaseHandler):
    """
    Handler personnalisé pour Mistral optimisé pour la production
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.config = None
        self.generation_config = None
        self.initialized = False
        
    def initialize(self, context):
        """
        Initialisation du modèle et du tokenizer
        """
        try:
            # Configuration du contexte
            self.context = context
            self.manifest = context.manifest
            properties = context.system_properties
            
            # Détermination du device
            if torch.cuda.is_available() and properties.get("gpu_id") is not None:
                self.device = torch.device(f"cuda:{properties.get('gpu_id')}")
                logger.info(f"Utilisation du GPU: {self.device}")
            else:
                self.device = torch.device("cpu")
                logger.info("Utilisation du CPU")
            
            # Chargement de la configuration
            model_dir = properties.get("model_dir")
            config_path = os.path.join(model_dir, "config.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("Configuration chargée depuis config.json")
            else:
                # Configuration par défaut
                self.config = {
                    "max_length": 4096,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "max_new_tokens": 512
                }
                logger.info("Configuration par défaut utilisée")
            
            # Chargement du tokenizer
            logger.info("Chargement du tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Configuration du pad_token si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Chargement du modèle
            logger.info("Chargement du modèle Mistral...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Configuration de génération
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.get("max_new_tokens", 512),
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                top_k=self.config.get("top_k", 50),
                repetition_penalty=self.config.get("repetition_penalty", 1.1),
                do_sample=self.config.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                use_cache=True
            )
            
            self.model.eval()
            self.initialized = True
            
            logger.info("✅ Modèle Mistral initialisé avec succès")
            logger.info(f"Device: {self.device}")
            logger.info(f"Model dtype: {self.model.dtype}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
            raise e
    
    def preprocess(self, requests: List[Dict]) -> List[Dict]:
        """
        Préprocessing des requêtes
        """
        try:
            processed_requests = []
            
            for request in requests:
                # Extraction des données
                if isinstance(request, dict):
                    if "body" in request:
                        data = request["body"]
                    else:
                        data = request
                else:
                    data = {"text": str(request)}
                
                # Extraction du texte
                if isinstance(data, (bytes, bytearray)):
                    data = json.loads(data.decode("utf-8"))
                elif isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        data = {"text": data}
                
                # Validation et formatage
                text = data.get("text", "")
                if not text:
                    raise PredictionException("Champ 'text' requis dans la requête")
                
                # Application du template de conversation Mistral
                if not text.startswith("<s>[INST]"):
                    formatted_text = f"<s>[INST] {text} [/INST]"
                else:
                    formatted_text = text
                
                # Paramètres de génération personnalisés
                generation_params = {
                    "max_new_tokens": data.get("max_new_tokens", self.config.get("max_new_tokens", 512)),
                    "temperature": data.get("temperature", self.config.get("temperature", 0.7)),
                    "top_p": data.get("top_p", self.config.get("top_p", 0.9)),
                    "top_k": data.get("top_k", self.config.get("top_k", 50)),
                    "repetition_penalty": data.get("repetition_penalty", self.config.get("repetition_penalty", 1.1)),
                    "do_sample": data.get("do_sample", self.config.get("do_sample", True))
                }
                
                processed_requests.append({
                    "text": formatted_text,
                    "generation_params": generation_params,
                    "original_data": data
                })
            
            logger.info(f"Préprocessing terminé pour {len(processed_requests)} requêtes")
            return processed_requests
            
        except Exception as e:
            logger.error(f"Erreur lors du préprocessing: {str(e)}")
            raise PredictionException(f"Erreur de préprocessing: {str(e)}")
    
    def inference(self, processed_requests: List[Dict]) -> List[Dict]:
        """
        Inférence avec le modèle Mistral
        """
        try:
            results = []
            
            for request in processed_requests:
                start_time = time.time()
                
                # Tokenisation
                inputs = self.tokenizer(
                    request["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.get("max_length", 4096)
                ).to(self.device)
                
                # Génération avec paramètres personnalisés
                generation_config = GenerationConfig(
                    **request["generation_params"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    use_cache=True
                )
                
                with torch.no_grad():
                    # Génération
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                    
                    # Décodage en excluant les tokens d'entrée
                    input_length = inputs.input_ids.shape[1]
                    generated_tokens = outputs[0][input_length:]
                    
                    response_text = self.tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                
                inference_time = time.time() - start_time
                
                # Métriques
                metrics = {
                    "inference_time": round(inference_time, 3),
                    "input_tokens": input_length,
                    "output_tokens": len(generated_tokens),
                    "total_tokens": len(outputs[0]),
                    "tokens_per_second": round(len(generated_tokens) / inference_time, 2)
                }
                
                results.append({
                    "generated_text": response_text.strip(),
                    "metrics": metrics,
                    "parameters_used": request["generation_params"]
                })
                
                logger.info(f"Inférence réussie en {inference_time:.3f}s - {metrics['tokens_per_second']} tokens/s")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'inférence: {str(e)}")
            raise PredictionException(f"Erreur d'inférence: {str(e)}")
    
    def postprocess(self, inference_results: List[Dict]) -> List[Dict]:
        """
        Post-processing des résultats
        """
        try:
            processed_results = []
            
            for result in inference_results:
                # Nettoyage du texte généré
                generated_text = result["generated_text"]
                
                # Suppression des artefacts de tokenisation
                generated_text = generated_text.replace("</s>", "").strip()
                
                # Format de réponse standardisé
                response = {
                    "status": "success",
                    "generated_text": generated_text,
                    "metadata": {
                        "model": "mistral-7b-instruct",
                        "version": "1.0",
                        "timestamp": time.time(),
                        "metrics": result["metrics"],
                        "parameters": result["parameters_used"]
                    }
                }
                
                processed_results.append(response)
            
            logger.info(f"Post-processing terminé pour {len(processed_results)} résultats")
            return processed_results
            
        except Exception as e:
            logger.error(f"Erreur lors du post-processing: {str(e)}")
            return [{
                "status": "error",
                "error": f"Erreur de post-processing: {str(e)}",
                "generated_text": "",
                "metadata": {
                    "model": "mistral-7b-instruct",
                    "timestamp": time.time()
                }
            }]
    
    def handle(self, data, context):
        """
        Point d'entrée principal pour les requêtes
        """
        try:
            if not self.initialized:
                raise PredictionException("Modèle non initialisé")
            
            # Pipeline complet
            preprocessed = self.preprocess(data)
            inference_results = self.inference(preprocessed)
            postprocessed = self.postprocess(inference_results)
            
            return postprocessed
            
        except Exception as e:
            logger.error(f"Erreur dans handle: {str(e)}")
            return [{
                "status": "error",
                "error": str(e),
                "generated_text": "",
                "metadata": {
                    "model": "mistral-7b-instruct",
                    "timestamp": time.time()
                }
            }]


# Point d'entrée pour TorchServe
_service = MistralHandler()


def handle(data, context):
    """
    Fonction handle pour TorchServe
    """
    return _service.handle(data, context)