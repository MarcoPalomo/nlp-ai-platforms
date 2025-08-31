# ================================
# 💡 NLP Platform - Makefile
# ================================

.PHONY: all minikube helm api clean stop status logs build-model

# Necessary to run minikube with the right parameters
minikube:
	@echo "🚀 [minikube] Starting Minikube..."
	minikube start --cpus=6 --memory=12g --driver=docker --addons=ingress

helm:
	@echo "📦 [helm] Deploying Helm charts..."
	helm upgrade --install torchserve ./helm-charts/torchserve --values ./helm-charts/torchserve/values.yaml
	helm upgrade --install mistral ./helm-charts/mistral --values ./helm-charts/mistral/values.yaml
	helm upgrade --install nlp-api ./helm-charts/nlp-api --values ./helm-charts/nlp-api/values.yaml

api:
	@echo "🔌 [api] Forwarding port 8000 for nlp-api service..."
	kubectl port-forward svc/nlp-api 8000:8000

stop:
	@echo "🛑 [minikube] Stopping Minikube..."
	minikube stop

# delete all parameters
clean:
	@echo "🧹 [clean] Removing all helm releases..."
	-helm uninstall torchserve || true
	-helm uninstall mistral || true
	-helm uninstall nlp-api || true

status:
	@echo "📊 [status] Kubernetes Resources:"
	kubectl get pods -A
	kubectl get svc -A

logs:
	kubectl logs deployment/nlp-api -c nlp-api --tail=100 -f

build-model:
	@echo "🧠 Building TorchServe model archive..."
	./scripts/build_model.sh
