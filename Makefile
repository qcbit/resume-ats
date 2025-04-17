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
KIND_CLUSTER_NAME := resume-dev-cluster
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