# 🚀 Production Deployment Guide (100% Free Stack)

This guide walks you through deploying your arXiv Hybrid RAG backend and connecting it to your live Vercel frontend for **$0 / Free Forever**.

---

## 🌐 Current Status
- **Frontend**: ✅ Already live on Vercel at [https://arxiz-assistant.vercel.app/](https://arxiz-assistant.vercel.app/)
- **Repository**: [https://github.com/ashuydv-05/HYBRID-RAG-EVAL](https://github.com/ashuydv-05/HYBRID-RAG-EVAL)

---

## 📋 3 Quick Steps to Complete Production Deployment

```
┌─────────────────────────────────┐
│ Step 1: Qdrant Cloud (Free 1GB) │ ──► Create free cluster & run python script/seed_cloud.py
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│ Step 2: Deploy Backend          │ ──► Deploy to Hugging Face Spaces (16GB RAM) or Render
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│ Step 3: Link Vercel Frontend    │ ──► Set NEXT_PUBLIC_API_URL on Vercel Settings & Redeploy
└─────────────────────────────────┘
```

---

## Step 1: Setup Free Qdrant Cloud (Vector Database)

1. Go to [https://cloud.qdrant.io/](https://cloud.qdrant.io/) and create a **Free Tier** cluster (1GB RAM free forever, no credit card required).
2. Once the cluster is created, copy:
   - **Cluster URL**: e.g., `https://xxxxxx-xxxx.us-east4-0.gcp.cloud.qdrant.io:6333`
   - **API Key**: Click *Data Access Control* / *API Keys* -> generate a key.
3. Seed your paper embeddings from your local terminal to Qdrant Cloud in 1 command:
   ```bash
   python script/seed_cloud.py --url "https://your-cluster-id.cloud.qdrant.io:6333" --api-key "your_qdrant_api_key" --recreate
   ```
   *(This will create the `arxiv_papers` collection and upload all vectors with progress bars).*

---

## Step 2: Deploy FastAPI Backend (Free 16GB RAM on Hugging Face Spaces)

We recommend **Hugging Face Spaces** because it gives **16 GB RAM + 2 vCPUs for FREE** (which runs PyTorch & Cross-Encoders with zero memory constraints).

### Option A: Hugging Face Spaces (Recommended - 16 GB RAM)
1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Configure:
   - **Space Name**: `arxiv-rag-backend`
   - **Space SDK**: Choose **Docker** -> **Blank**
   - **Space Hardware**: **Free (CPU basic · 2 vCPU · 16 GB RAM)**
   - **Visibility**: **Public**
3. Go to **Settings** -> **Variables and secrets** -> **New secret**, and add:
   - `GROQ_API_KEY` = `your_groq_api_key`
   - `QDRANT_URL` = `https://your-cluster-id.cloud.qdrant.io:6333`
   - `QDRANT_API_KEY` = `your_qdrant_api_key`
   - `ALLOWED_ORIGINS` = `https://arxiz-assistant.vercel.app,http://localhost:3000`
4. Deploy the code to Hugging Face Spaces:
   - In your local terminal, add Hugging Face remote and push:
     ```bash
     # Replace <YOUR_HF_USERNAME> with your Hugging Face username
     git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/arxiv-rag-backend
     git push -u hf main
     ```
5. Your Backend URL will be:
   ```
   https://<YOUR_HF_USERNAME>-arxiv-rag-backend.hf.space
   ```
   *(API endpoints live at `https://<YOUR_HF_USERNAME>-arxiv-rag-backend.hf.space/api`)*

---

### Option B: Render Web Service (Alternative)
1. Go to [https://dashboard.render.com/](https://dashboard.render.com/) -> **New +** -> **Web Service**.
2. Connect your GitHub repository `ashuydv-05/HYBRID-RAG-EVAL`.
3. Configure:
   - **Name**: `arxiv-backend`
   - **Environment**: **Docker**
   - **Instance Type**: **Free**
   - **Health Check Path**: `/api/health`
4. Add Environment Variables:
   - `GROQ_API_KEY` = `your_groq_api_key`
   - `QDRANT_URL` = `https://your-cluster-id.cloud.qdrant.io:6333`
   - `QDRANT_API_KEY` = `your_qdrant_api_key`
   - `ALLOWED_ORIGINS` = `https://arxiz-assistant.vercel.app,http://localhost:3000`
5. Click **Create Web Service**. Your API URL will be:
   ```
   https://arxiv-backend.onrender.com
   ```

---

## Step 3: Link Vercel Frontend to Backend

Now connect your live Vercel frontend to the backend API:

1. Open your [Vercel Dashboard](https://vercel.com/dashboard) and click on your project **`arxiz-assistant`**.
2. Go to **Settings** -> **Environment Variables**.
3. Add or update the following variable:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**:
     - If using Hugging Face: `https://<YOUR_HF_USERNAME>-arxiv-rag-backend.hf.space/api`
     - If using Render: `https://arxiv-backend.onrender.com/api`
   - **Target**: Check *Production*, *Preview*, and *Development*.
4. Go to **Deployments** tab on Vercel -> Click on the `...` menu on the latest deployment -> Click **Redeploy** (with unchecked "Use existing Build Cache" to bake in the new URL).

---

## Step 4: Automated CI/CD (GitHub Actions)

Your repository now has automated CI/CD configured in [`.github/workflows/ci.yml`](file:///Users/ashuyadav/Desktop/HYBRID%20RAG/.github/workflows/ci.yml).

### Enable Automated Hugging Face Sync:
To automatically update your Hugging Face Space whenever you `git push` to `main`:
1. Go to your GitHub repository: `https://github.com/ashuydv-05/HYBRID-RAG-EVAL` -> **Settings** -> **Secrets and variables** -> **Actions**.
2. Add the following secrets:
   - `HF_TOKEN`: Your Hugging Face User Access Token (from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with `Write` access).
   - `HF_SPACE`: `<your_hf_username>/arxiv-rag-backend` (e.g. `ashuydv/arxiv-rag-backend`).
   - `GROQ_API_KEY`: Your Groq API key (for automated backend test suites).

---

## 🔍 Verification & Health Check

Test that all components are online:

1. **Backend Health Check**:
   ```bash
   curl -s https://<YOUR_BACKEND_URL>/api/health | jq .
   ```
   Expected response:
   ```json
   {
     "status": "healthy",
     "version": "2.0.0",
     "components": {
       "workflow": "healthy",
       "workflow_graph": "healthy"
     }
   }
   ```

2. **Frontend Test**:
   Visit [https://arxiz-assistant.vercel.app/](https://arxiz-assistant.vercel.app/) and type:
   > *"What are the key findings of attention mechanism in transformer models?"*
   
   The assistant will retrieve relevant arXiv context from Qdrant Cloud, run reasoning via Groq LLM, and stream the response to the user interface!
