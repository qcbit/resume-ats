# Resume-ATS: AI-Powered Resume Scoring Assistant

The AI-powered Resume/ATS Scoring Assistant automates screening and scoring resumes using large language models (LLMs). It evaluates resumes against job descriptions, offering feedback and recommendations to improve efficiency and accuracy in applicant tracking systems (ATS).

## Architecture

This project uses a microservices architecture with:

- **Frontend UI** - React-based interface for resume uploads and results
- **Backend Server** - Node.js/Express for file handling and service orchestration
- **AI Services**:
  - Job Title Detector - Extracts job titles from the job descriptions
  - Keywords Extractor - Uses LLM to identify relevant skills and keywords
  - Match Scorer - Uses LLM to evaluate resume-job compatibility

## Getting Started

Prerequisites

- macOS (for brew installation)
- Docker Desktop
- Internet connection
- OpenAI API key (for keyword extraction)

1. Install Homebrew

```make install-brew```

2. Install required tools

```make install-tools```

3. Create a KinD Kubernetes cluster.

```make kind-create```

4. Deploy the ingress.

```make deploy-ingress```

Note: The ingress status will initially be stuck in Pending.

5. Label the node for ingress

```sh
# Get the node name
kubectl get nodes

# Add the ingress-ready label (replace with your node name)
kubectl label node resume-ats-dev-cluster-control-plane ingress-ready=true
```

6. Create the OpenAI API key secret

```sh
kubectl create secret generic openai-api-key \
  --from-literal=OPENAI_API_KEY=your-actual-openai-api-key
```

7. Deploy backend microservices

```make deploy-backend-services```

8.  Deploy frontend services

```make deploy-frontend-services```

9.  Access the page <http://localhost:30211>.

If 404, the reapply the ingress manifest

```sh
kubectl apply -f deployment/ingress.yaml
kubectl get ingress -n default
```

You should see:

```sh
NAME             CLASS    HOSTS       ADDRESS     PORTS   AGE
resume-ingress   <none>   localhost   localhost   80      5s
```
