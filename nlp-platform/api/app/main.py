"""
Serveur FastAPI - Exposition des endpoints REST pour interagir avec Mistral via TorchServe
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime

# Import du client d'orchestration
from orchestration_client import NLPPlatformAPI, NLPRequest, TaskType, NLPResponse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialisation FastAPI
app = FastAPI(
    title="NLP Platform API",
    description="API d'orchestration pour Mistral avec TorchServe",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance globale de l'API NLP
nlp_api = None

# Modèles Pydantic pour l'API
class TextGenerationRequest(BaseModel):
    text: str = Field(..., description="Texte d'entrée")
    max_tokens: int = Field(512, ge=1, le=2048, description="Nombre max de tokens")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Température de génération")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p pour nucleus sampling")
    top_k: int = Field(50, ge=1, le=100, description="Top-k pour sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Pénalité de répétition")
    priority: int = Field(1, ge=1, le=3, description="Priorité (1=normal, 2=high, 3=urgent)")

class QuestionAnsweringRequest(BaseModel):
    question: str = Field(..., description="Question à poser")
    context: str = Field("", description="Contexte optionnel")
    max_tokens: int = Field(300, ge=1, le=1024)
    temperature: float = Field(0.5, ge=0.1, le=1.5)

class SummarizationRequest(BaseModel):
    text: str = Field(..., description="Texte à résumer")
    max_length: int = Field(150, ge=50, le=500, description="Longueur max du résumé")
    temperature: float = Field(0.3, ge=0.1, le=1.0)

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Texte à traduire")
    target_language: str = Field("français", description="Langue cible")
    max_tokens: int = Field(512, ge=1, le=1024)

class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Texte à classifier")
    categories: Optional[List[str]] = Field(None, description="Catégories possibles")
    max_tokens: int = Field(100, ge=1, le=300)

class BatchRequest(BaseModel):
    requests: List[Dict[str, Any]] = Field(..., description="Liste de requêtes")
    priority: int = Field(1, ge=1, le=3)

class APIResponse(BaseModel):
    status: str
    data: Any
    request_id: str
    processing_time: float
    timestamp: str

# Dépendance pour l'API NLP
async def get_nlp_api():
    global nlp_api
    if nlp_api is None:
        torchserve_url = os.getenv("TORCHSERVE_URL", "http://localhost:8080")
        nlp_api = NLPPlatformAPI(torchserve_url)
    return nlp_api

# Routes de l'API
@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    logger.info("🚀 Démarrage de l'API NLP Platform")
    
    # Création des répertoires de logs
    os.makedirs("/app/logs", exist_ok=True)
    
    # Test de connexion à TorchServe
    try:
        api = await get_nlp_api()
        health = await api.health_check()
        if health["status"] == "healthy":
            logger.info("✅ Connexion à TorchServe établie")
        else:
            logger.warning("⚠️ TorchServe non disponible au démarrage")
    except Exception as e:
        logger.error(f"❌ Erreur de connexion TorchServe: {e}")

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "NLP Platform API avec Mistral",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check(api: NLPPlatformAPI = Depends(get_nlp_api)):
    """Vérification de santé du système"""
    try:
        health_status = await api.health_check()
        return JSONResponse(
            status_code=200 if health_status["status"] == "healthy" else 503,
            content=health_status
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/generate", response_model=APIResponse)
async def generate_text(
    request: TextGenerationRequest,
    api: NLPPlatformAPI = Depends(get_nlp_api)
):
    """Génération de texte avec Mistral"""
    try:
        start_time = datetime.now()
        
        response = await api.generate_text(
            text=request.text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        return APIResponse(
            status=response.status,
            data={
                "generated_text": response.generated_text,
                "metadata": response.metadata
            },
            request_id=response.request_id,
            processing_time=response.processing_time,
            timestamp=start_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur génération: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question-answering", response_model=APIResponse)
async def answer_question(
    request: QuestionAnsweringRequest,
    api: NLPPlatformAPI = Depends(get_nlp_api)
):
    """Réponse à une question"""
    try:
        start_time = datetime.now()
        
        response = await api.answer_question(
            question=request.question,
            context=request.context,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return APIResponse(
            status=response.status,
            data={
                "answer": response.generated_text,
                "metadata": response.metadata
            },
            request_id=response.request_id,
            processing_time=response.processing_time,
            timestamp=start_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur Q&A: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=APIResponse)
async def summarize_text(
    request: SummarizationRequest,
    api: NLPPlatformAPI = Depends(get_nlp_api)
):
    """Résumé de texte"""
    try:
        start_time = datetime.now()
        
        response = await api.summarize_text(
            text=request.text,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return APIResponse(
            status=response.status,
            data={
                "summary": response.generated_text,
                "metadata": response.metadata
            },
            request_id=response.request_id,
            processing_time=response.processing_time,
            timestamp=start_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur résumé: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate", response_model=APIResponse)
async def translate_text(
    request: TranslationRequest,
    api: NLPPlatformAPI = Depends(get_nlp_api)
):
    """Traduction de texte"""
    try:
        start_time = datetime.now()
        
        response = await api.translate_text(
            text=request.text,
            target_language=request.target_language,
            max_new_tokens=request.max_tokens
        )
        
        return APIResponse(
            status=response.status,
            data={
                "translation": response.generated_text,
                "target_language": request.target_language,
                "metadata": response.metadata
            },
            request_id=response.request_id,
            processing_time=response.processing_time,
            timestamp=start_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur traduction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=APIResponse)
async def classify_text(
    request: ClassificationRequest,
    api: NLPPlatformAPI = Depends(get_nlp_api)
):
    """Classification de texte"""
    try:
        start_time = datetime.now()
        
        response = await api.classify_text(
            text=request.text,
            categories=request.categories,
            max_new_tokens=request.max_tokens
        )
        
        return APIResponse(
            status=response.status,
            data={
                "classification": response.generated_text,
                "categories": request.categories,
                "metadata": response.metadata
            },
            request_id=response.request_id,
            processing_time=response.processing_time,
            timestamp=start_time.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=List[APIResponse])
async def batch_process(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    api: NLPPlatformAPI = Depends(get_nlp_api)
):
    """Traitement par batch"""
    try:
        start_time = datetime.now()
        
        # Conversion des requêtes
        nlp_requests = []
        for req_data in request.requests:
            task_type = TaskType(req_data.get("task_type", "text_generation"))
            nlp_req = NLPRequest(
                text=req_data["text"],
                task_type=task_type,
                parameters=req_data.get("parameters", {}),
                metadata=req_data.get("metadata", {}),
                priority=request.priority
            )
            nlp_requests.append(nlp_req)
        
        # Traitement par batch
        responses = await api.batch_process(nlp_requests)
        
        # Conversion en format API
        api_responses = []
        for response in responses:
            api_resp = APIResponse(
                status=response.status,
                data={
                    "generated_text": response.generated_text,
                    "task_type": response.task_type.value,
                    "metadata": response.metadata
                },
                request_id=response.request_id,
                processing_time=response.processing_time,
                timestamp=start_time.isoformat()
            )
            api_responses.append(api_resp)
        
        return api_responses
        
    except Exception as e:
        logger.error(f"Erreur batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics(api: NLPPlatformAPI = Depends(get_nlp_api)):
    """Métriques de la plateforme"""
    try:
        metrics = api.get_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Erreur métriques: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache(api: NLPPlatformAPI = Depends(get_nlp_api)):
    """Nettoyage du cache"""
    try:
        api.orchestrator.clear_cache()
        return {"message": "Cache nettoyé avec succès"}
    except Exception as e:
        logger.error(f"Erreur nettoyage cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """Liste des modèles disponibles"""
    try:
        # Appel à l'API de gestion TorchServe
        import aiohttp
        management_url = os.getenv("TORCHSERVE_URL", "http://localhost:8080").replace("8080", "8081")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{management_url}/models") as response:
                if response.status == 200:
                    models = await response.json()
                    return {"models": models}
                else:
                    return {"models": [], "error": "TorchServe non disponible"}
                    
    except Exception as e:
        logger.error(f"Erreur liste modèles: {e}")
        return {"models": [], "error": str(e)}

# Routes de monitoring avancé
@app.get("/status")
async def system_status(api: NLPPlatformAPI = Depends(get_nlp_api)):
    """Status complet du système"""
    try:
        health = await api.health_check()
        metrics = api.get_metrics()
        
        return {
            "system_health": health,
            "performance_metrics": metrics,
            "api_version": "1.0.0",
            "uptime": datetime.now().isoformat(),
            "environment": {
                "torchserve_url": os.getenv("TORCHSERVE_URL", "http://localhost:8080"),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "max_workers": os.getenv("MAX_WORKERS", "10")
            }
        }
    except Exception as e:
        logger.error(f"Erreur status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gestion des erreurs globales
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire d'erreurs global"""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Erreur interne du serveur",
            "timestamp": datetime.now().isoformat()
        }
    )

# Middleware de logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware de logging des requêtes"""
    start_time = time.time()
    
    # Log de la requête entrante
    logger.info(f"📥 {request.method} {request.url.path}")
    
    # Traitement de la requête
    response = await call_next(request)
    
    # Log de la réponse
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# Fonctions utilitaires
def format_error_response(error: str, request_id: str = None) -> Dict:
    """Formatage des réponses d'erreur"""
    return {
        "status": "error",
        "error": error,
        "request_id": request_id,
        "timestamp": datetime.now().isoformat()
    }

# Point d'entrée principal
if __name__ == "__main__":
    # Configuration du serveur
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"🌐 Démarrage du serveur sur {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
        log_level="info"
    )
# Inclusion du routeur NLP (chat + ner)
app.include_router(nlp.router, prefix="/nlp", tags=["NLP"])

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

"""
POST    /ner    Appel à TorchServe NER
POST    /chat   Appel à Mistral/vLLM
GET     /health Vérification de l’API
"""