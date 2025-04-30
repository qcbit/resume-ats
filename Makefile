install-brew:
	/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

install-tools:
	brew install kind
	brew install kubectl
	brew install kubectx
	brew install k9s
	brew install helm
	brew install uv
	brew install python@3.11.4

# Variables
KIND_CLUSTER_NAME := resume-ats-dev-cluster
KIND_CONFIG_FILE := kind-config.yaml

# Create a Kind cluster
.PHONY: kind-create
kind-create:
	kind create cluster --name $(KIND_CLUSTER_NAME) --config deployment/$(KIND_CONFIG_FILE)

# Delete the Kind cluster
.PHONY: kind-delete
kind-delete:
	kind delete cluster --name $(KIND_CLUSTER_NAME)

# Get cluster info
.PHONY: kind-info
kind-info:
	kubectl cluster-info --context kind-$(KIND_CLUSTER_NAME)

# Load a Docker image into the Kind cluster
.PHONY: kind-load-image
kind-load-image:
	@echo "Usage: make kind-load-image IMAGE=<image-name>"
	kind load docker-image $(IMAGE) --name $(KIND_CLUSTER_NAME)

.PHONY: kind-ingress-install
kind-ingress-install:
	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.1/deploy/static/provider/kind/deploy.yaml

.PHONY: kind-ingress-uninstall
kind-ingress-uninstall:
	kubectl delete -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.1/deploy/static/provider/kind/deploy.yaml

.PHONY: kind-ingress-patch
kind-ingress-patch:
	kubectl patch svc ingress-nginx-controller -n ingress-nginx -p '{"spec": {"type": "NodePort"}}'

.PHONY: portmap-ingress
portmap-ingress:
	@echo "Port mapping ingress-nginx-controller to localhost:30211"
	kubectl apply -f deployment/ingress-service.yaml

deploy-ingress: kind-ingress-install kind-ingress-patch portmap-ingress
	@echo "Ingress installed and patched successfully."

# This is a workaround for the issue with the ingress-nginx-controller service not being reachable
port-forward-ingress:
	kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8080:80

deploy-backend-services:
	@echo "Deploying backend services..."
	$(MAKE) -C backend deploy

deploy-frontend-services:
	@echo "Deploying frontend services..."
	$(MAKE) -C frontend deploy