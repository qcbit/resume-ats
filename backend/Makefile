export KIND_CLUSTER_NAME ?= resume-ats-dev-cluster

build-jtd:
	@echo "Building the docker image for the job title detector"
	docker build -t job-title-detector:latest -f docker/JTDetector.Dockerfile .

load-jtd:
	@echo "Loading the docker image for the job title detector into kind"
	kind load docker-image job-title-detector:latest --name $(KIND_CLUSTER_NAME)

delete-jtd:
	@echo "Deleting the job title detector service from {KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/jtd.yaml || true

deploy-jtd: delete-jtd build-jtd load-jtd
	@echo "Deploying the job title detector service to {KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/jtd.yaml

build-kwe:
	@echo "Building the docker image for the keywords extractor"
	docker build -t keywords-extractor:latest -f docker/KWExtractor.Dockerfile .

load-kwe:
	@echo "Loading the docker image for the keywords extractor into kind"
	kind load docker-image keywords-extractor:latest --name $(KIND_CLUSTER_NAME)

delete-kwe:
	@echo "Deleting the keywords extractor service from {KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/kwe.yaml || true

deploy-kwe: build-kwe load-kwe
	@echo "Deploying the keywords extractor to {KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/kwe.yaml

build-kwe-llm-q:
	@echo "Building the docker image for the keywords extractor"
	docker build -t keywords-extractor-llm-q:latest -f docker/KWExtractorLLMQ.Dockerfile .

load-kwe-llm-q:
	@echo "Loading the docker image for the keywords extractor into kind"
	kind load docker-image keywords-extractor-llm-q:latest --name $(KIND_CLUSTER_NAME)

delete-kwe-llm-q:
	@echo "Deleting the keywords extractor service from {KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/kwe.yaml || true

deploy-kwe-llm-q: build-kwe-llm-q load-kwe-llm-q
	@echo "Deploying the keywords extractor to {KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/kwe.yaml