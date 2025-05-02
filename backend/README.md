# Backend Services for Resume-ATS

## Overview

This backend system provides a microservices architecture for resume analysis and job matching. It's designed to extract meaningful information from resumes and job descriptions, then analyze their compatibility using advanced AI techniques.

## Services

The backend contains the following services:

- **Job Title Detector**: Extracts job titles from job descriptions using embedding similarity and fuzzy matching
- **Keywords Extractor (OpenAI)**: Extracts relevant skills and keywords from job descriptions using OpenAI models
- **Match Scorer (OpenAI)**: Evaluates the match between resumes and job descriptions using Llama 3 LLM

## Getting Started

### Prerequisites

- Docker installed and running
- Kind (Kubernetes in Docker) installed
- kubectl command-line tool

## Setup

```sh
make deploy-backend
```

## Teardown

```sh
make delete-backend
```

See Makefile for other commands.

## Service Architecture

```sh
Frontend UI → Backend Server → [Microservices]
                                ├── Job Title Detector (5001)
                                ├── Keywords Extractor (5002)
                                └── Match Scorer (5003)
```

## Testing

Accessing Services

Services are accessible within the cluster via their service names:

- Job Title Detector: http://job-title-detector:5001
- Keywords Extractor: http://keywords-extractor:5002
- Match Scorer: http://match-scorer:5003

External Access

To test services from your local machine:

```sh
# Port-forward a service
kubectl port-forward service/job-title-detector 5001:5000
kubectl port-forward service/keywords-extractor 5002:5000
kubectl port-forward service/match-scorer 5003:5000
```

## Configuration

Each service has its own configuration options. See the individual service READMEs for detailed configuration instructions:

- Job Title Detector
- Keywords Extractor (OpenAI)
- Match Scorer (OpenAI)

## Troubleshooting

- Missing service connection: Check if all services are running with ```kubectl get pods```
- API errors: Check logs with ```kubectl logs <pod-name>```

## Links

[Frontend Documentation](../frontend/README.md)

[Back to Main README](../README.md)