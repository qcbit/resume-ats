export KIND_CLUSTER_NAME ?= resume-ats-dev-cluster

build-jtd:
	@echo "Building the docker image for the job title detector"
	docker build -t job-title-detector:latest -f docker/JTDetector.Dockerfile .

load-jtd:
	@echo "Loading the docker image for the job title detector into kind"
	kind load docker-image job-title-detector:latest --name $(KIND_CLUSTER_NAME)

deploy-jtd: build-jtd load-jtd
	@echo "Deleting the job title detector from {KIND_CLUSTER_NAME}"
	kubectl delete -f deployment/dev/jtd.yaml || true
	@echo "Deploying the job title detector to {KIND_CLUSTER_NAME}"
	kubectl apply -f deployment/dev/jtd.yaml