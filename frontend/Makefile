# Variables
FRONTEND_IMAGE_NAME := resume-frontend
IMAGE_TAG := latest
KIND_CLUSTER_NAME ?= resume-ats-dev-cluster
FRONTEND_YAML := deployment/dev/frontend.yaml

.PHONY: build-frontend
build-frontend:
	@echo "Building frontend Docker image $(FRONTEND_IMAGE_NAME):$(IMAGE_TAG)..."
	docker build -t $(FRONTEND_IMAGE_NAME):$(IMAGE_TAG) -f docker/frontend-ui.Dockerfile ./frontend-ui

.PHONY: load-frontend
load-frontend:
	@echo "Loading frontend image $(FRONTEND_IMAGE_NAME):$(IMAGE_TAG) into Kind cluster $(KIND_CLUSTER_NAME)..."
	kind load docker-image $(FRONTEND_IMAGE_NAME):$(IMAGE_TAG) --name $(KIND_CLUSTER_NAME)

.PHONY: delete-frontend
delete-frontend:
	@echo "Deleting existing frontend..."
	kubectl delete -f $(FRONTEND_YAML) --ignore-not-found=true

.PHONY: deploy-frontend
deploy-frontend:
	@echo "Deploying frontend..."
	kubectl apply -f $(FRONTEND_YAML)
	kubectl apply -f deployment/dev/frontend-ingress.yaml

# Variables
BACKEND_IMAGE_NAME := resume-backend
KIND_CLUSTER_NAME ?= resume-ats-dev-cluster
BACKEND_YAML := deployment/dev/backend.yaml # Path relative to this Makefile
DOCKERFILE_PATH := docker/backend.Dockerfile # Path relative to this Makefile

.PHONY: build-backend
build-backend:
	@echo "Building backend Docker image $(BACKEND_IMAGE_NAME):$(IMAGE_TAG)..."
	# Run docker build from the frontend/backend directory
	docker build -t $(BACKEND_IMAGE_NAME):$(IMAGE_TAG) -f $(DOCKERFILE_PATH) ./backend

.PHONY: load-backend
load-backend:
	@echo "Loading backend image $(BACKEND_IMAGE_NAME):$(IMAGE_TAG) into Kind cluster $(KIND_CLUSTER_NAME)..."
	kind load docker-image $(BACKEND_IMAGE_NAME):$(IMAGE_TAG) --name $(KIND_CLUSTER_NAME)

.PHONY: delete-backend
delete-backend:
	@echo "Deleting existing backend deployment and service..."
	kubectl delete -f $(BACKEND_YAML) --ignore-not-found=true

.PHONY: deploy-backend
deploy-backend:
	@echo "Deploying backend server..."
	kubectl apply -f $(BACKEND_YAML)

.PHONY: delete
delete: delete-backend delete-frontend
	@echo "Frontend deployment deleted successfully."

# Full deploy sequence: delete, build, load, deploy
.PHONY: deploy
deploy: delete-backend build-backend load-backend deploy-backend delete-frontend build-frontend load-frontend deploy-frontend
	@echo "Frontend deployed successfully."