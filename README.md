# resume-ats
The AI-powered Resume/ATS Scoring Assistant automates screening and scoring resumes using large language models (LLMs). It evaluates resumes against job descriptions, offering feedback and recommendations to improve efficiency and accuracy in applicant tracking systems (ATS).

# Getting Started

1. Run **make install-brew**.
2. Run **make install-tools**.
3. To bring up the cluster, run **make kind-create**.
4. Run **make deploy-ingress**. The ingress status will be stuck in Pending.
5. Run **kubectl get nodes**. Copy the control plane name, e.g. resume-ats-dev-cluster-control-plane.
6. Add a label: **kubectl label node resume-ats-dev-cluster-control-plane ingress-ready=true**. After the label, the status should be Running.
7. You need an OpenAI API Token <https://platform.openai.com>. Once obtained, run **kubectl create secret generic openai-api-key \
  --from-literal=OPENAI_API_KEY=your-actual-openai-api-key**.
8. Run **make deploy-backend-services**.
9. Run **make deploy-frontend-services**.
10. Access the page <http://localhost:30211>.
Note: If 404, the reapply the ingress manifest: **kubectl apply -f deployment/ingress.yaml**
then verify **kubectl get ingress -n default**

You should see something like

```sh
NAME             CLASS    HOSTS       ADDRESS     PORTS   AGE
resume-ingress   <none>   localhost   localhost   80      5s
```

Now try to access the page again.

## Training the models

1. cd backend
2. source .venv/bin/activate
3. cd ..
4. python3 scripts/keyword_extraction_model.py
5. python3 scripts/fine_tune_roberta.py
