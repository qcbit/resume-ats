export KIND_CLUSTER_NAME ?= resume-ats-dev-cluster

build-jtd:
	@echo "Building the docker image for the job title detector"
	docker build -t job-title-detector:latest -f docker/JTDetector.Dockerfile .

load-jtd:
	@echo "Loading the docker image for the job title detector into kind"
	kind load docker-image job-title-detector:latest --name $(KIND_CLUSTER_NAME)

delete-jtd:
	@echo "Deleting the job title detector service from ${KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/jtd.yaml || true

deploy-jtd: delete-jtd build-jtd load-jtd
	@echo "Deploying the job title detector service to ${KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/jtd.yaml

PORT ?= 5000

build-kwe-openai:
	@echo "Building the docker image for the keywords extractor"
	docker build --build-arg PORT=$(PORT) -t keywords-extractor-openai:latest -f docker/KWExtractorOpenAI.Dockerfile .

load-kwe-openai:
	@echo "Loading the docker image for the keywords extractor into kind"
	kind load docker-image keywords-extractor-openai:latest --name $(KIND_CLUSTER_NAME)

delete-kwe-openai:
	@echo "Deleting the keywords extractor service from ${KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/kwe-openai.yaml || true

deploy-kwe-openai: delete-kwe-openai build-kwe-openai load-kwe-openai
	@echo "Deploying the keywords extractor to ${KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/kwe-openai.yaml

build-kwe-llm-q:
	@echo "Building the docker image for the keywords extractor"
	docker buildx build --platform linux/amd64 -t keywords-extractor-llm-q:latest -f docker/KWExtractorLLMQ.Dockerfile . --load

load-kwe-llm-q:
	@echo "Loading the docker image for the keywords extractor into kind"
	kind load docker-image keywords-extractor-llm-q:latest --name $(KIND_CLUSTER_NAME)

delete-kwe-llm-q:
	@echo "Deleting the keywords extractor service from ${KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/kwe.yaml || true

deploy-kwe-llm-q: build-kwe-llm-q load-kwe-llm-q
	@echo "Deploying the keywords extractor to ${KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/kwe.yaml

build-kwe-ollama:
	@echo "Building the docker image for the keywords extractor"
	docker build -t keywords-extractor-ollama:latest -f docker/KWExtractorOllama.Dockerfile .
#	docker buildx build -t keywords-extractor-ollama:latest -f docker/KWExtractorOllama.Dockerfile . --load

load-kwe-ollama:
	@echo "Loading the docker image for the keywords extractor into kind"
	kind load docker-image keywords-extractor-ollama:latest --name $(KIND_CLUSTER_NAME)

delete-kwe-ollama:
	@echo "Deleting the keywords extractor service from ${KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/kwe.yaml || true

deploy-kwe-ollama: build-kwe-ollama load-kwe-ollama
	@echo "Deploying the keywords extractor to ${KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/kwe.yaml

delete-ollama-service:
	@echo "Deleting the ollama service from ${KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/ollama.yaml || true

build-ollama-service:
	@echo "Building the docker image for the ollama service"
	docker build -t ollama-service:latest -f docker/OllamaService.Dockerfile .

load-ollama-service:
	@echo "Loading the docker image for the ollama service into kind"
	kind load docker-image ollama-service:latest --name $(KIND_CLUSTER_NAME)

deploy-ollama-service: delete-ollama-service build-ollama-service load-ollama-service
	@echo "Deploying the ollama service to ${KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/ollama.yaml

build-match-scorer:
	@echo "Building the docker image for the match scorer service"
	docker build -t match-scorer:latest -f docker/MatchScorer.Dockerfile .

load-match-scorer:
	@echo "Loading the docker image for the match scorer service into kind"
	kind load docker-image match-scorer:latest --name $(KIND_CLUSTER_NAME)

delete-match-scorer:
	@echo "Deleting the match scorer service from ${KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/match-scorer.yaml || true

deploy-match-scorer: delete-match-scorer build-match-scorer load-match-scorer
	@echo "Deploying the match-scorer service to ${KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/match-scorer.yaml

delete-backend:
	@echo "Deleting the backend services from ${KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/jtd.yaml || true
	kubectl delete -f deployment/dev/kwe-openai.yaml || true
	kubectl delete -f deployment/dev/match-scorer.yaml || true

deploy: delete-backend deploy-jtd deploy-kwe-openai deploy-match-scorer
