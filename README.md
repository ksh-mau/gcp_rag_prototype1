GCP RAG System Prototype

1. Architecture Overview

Key Python Scripts:

scripts/ingest_docs.py:
- Reads TXT documents from a Google Cloud Storage (GCS) bucket.
- Chunks text into passages.
- Generates embeddings via a Vertex AI Embedding Model.
- Upserts embeddings and metadata to Vertex AI Vector Search.

scripts/query_rag.py:
- Accepts a CLI query.
- Generates query embedding.
- Searches Vector Search for relevant chunks.
- Constructs a prompt with context + query.
- Gets an answer from a Vertex AI LLM.
- Displays the answer and cited sources.

2. Core Google Cloud Services Used
- Google Cloud Storage (GCS)
- Vertex AI Embedding Model (e.g., textembedding-gecko@003)
- Vertex AI Vector Search (Vector Store)
- Vertex AI LLM (e.g., text-bison@002)

3. System Prerequisites
- Google Cloud Platform (GCP) Account with billing enabled.
- Google Cloud SDK (gcloud CLI) installed and configured (recommended).
- Python 3.10+ (Python 3.12.x stable from python.org recommended).
- pip (Python package installer).
- Git (optional, for cloning).

4. GCP Environment Setup

4.1. Project Setup
- Ensure you have a GCP Project. Note the Project ID.
- Enable Billing for the project.

4.2. Enable APIs
Enable the following APIs for your project:
- Vertex AI API (aiplatform.googleapis.com)
- Cloud Storage API (storage.googleapis.com)
- Service Usage API (serviceusage.googleapis.com)

Command using gcloud:
```bash
gcloud services enable aiplatform.googleapis.com storage.googleapis.com serviceusage.googleapis.com --project=YOUR_PROJECT_ID
```

4.3. Service Account and Key
- Navigate to IAM & Admin > Service Accounts in the GCP Console.
- Click "+ CREATE SERVICE ACCOUNT".
- Name: e.g., rag-system-sa
- Click "CREATE AND CONTINUE".
- Grant Roles:
  - Vertex AI User
  - Storage Admin
  - Service Usage Consumer
- Click "CONTINUE", then "DONE".
- Select the created service account, go to the "KEYS" tab.
- Click "ADD KEY" > "Create new key", choose JSON, click "CREATE".
- Rename the downloaded JSON key file to service-account-key.json and place it in the project root directory (gcp_rag_prototype/).

4.4. Google Cloud Storage (GCS) Bucket
- Navigate to Cloud Storage > Buckets.
- Click "+ CREATE".
- Name: Globally unique (e.g., your-unique-rag-bucket). Record this for config.py.
- Location type: Region.
- Location: Select your desired GCP region (e.g., us-central1, us-east1). This region will be used for other services too. Record for config.py.
- Defaults for Storage class (Standard) and Access control (Uniform) are acceptable.
- Ensure "Enforce public access prevention" is checked.
- Click "CREATE".

4.5. Vertex AI Vector Search Setup
- Navigate to Vertex AI > Vector Search.

Create Index:
- Click "+ CREATE INDEX".
- Display name: e.g., rag-prototype-index.
- Region: Same as your GCS bucket.
- GCS folder URI: e.g., gs://YOUR_GCS_BUCKET_NAME/vector_index_data/.
- Algorithm type: Brute Force (for prototype) or Tree-AH algorithm.
- Dimensions: 768.
- (If Tree-AH) Approximate neighbors count: e.g., 10.
- Update method: Batch.
- (Advanced Options) Distance measure type: COSINE_DISTANCE.
- Click "CREATE". Wait for completion.
- Note the Index ID (Resource ID) for config.py (VECTOR_STORE_INDEX_ID).

Create Index Endpoint:
- Go to the "Index Endpoints" tab in Vector Search.
- Click "+ CREATE INDEX ENDPOINT".
- Index endpoint name: e.g., rag-prototype-endpoint.
- Region: Same as your Index.
- Access: Public endpoint.
- Click "CREATE". Wait for completion (5-15 min).
- Note its numeric ID for config.py (VECTOR_STORE_INDEX_ENDPOINT_ID).

Deploy Index to Endpoint:
- Select your created Index Endpoint.
- Click "DEPLOY INDEX".
- Deployed index display name: e.g., deployed-rag-index.
- Index: Select your created rag-prototype-index.
- Machine type: Choose a small type (e.g., n1-standard-2).
- Minimum/Maximum replica count: Set both to 1.
- Click "DEPLOY". Wait for completion (15-45+ min).
- Note the specific ID of this deployed index for config.py (VECTOR_STORE_DEPLOYED_INDEX_ID).

5. Local Environment Setup

5.1. Project Files
- Ensure all project files are in a root directory (e.g., gcp_rag_prototype/).

5.2. Python Virtual Environment
From the project root directory:
- Create: python -m venv venv
- Activate:
  - Windows (PowerShell): Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .
env\Scripts ctivate
  - Windows (CMD): venv\Scripts ctivate
  - macOS/Linux: source venv/bin/activate

Your terminal prompt should now show (venv).

5.3. Install Dependencies
With the venv active:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
If numpy or shapely fail to build on Windows, ensure "Microsoft C++ Build Tools" are installed.

5.4. Application Configuration
- In the config/ directory, copy config_template.py to config.py.
- Edit config/config.py and fill in all YOUR_..._HERE placeholders with your actual GCP resource names, IDs, and desired settings, especially:
  - PROJECT_ID
  - REGION
  - GCS_BUCKET_NAME
  - VECTOR_STORE_INDEX_ID
  - VECTOR_STORE_INDEX_ENDPOINT_ID
  - VECTOR_STORE_DEPLOYED_INDEX_ID
- Ensure SERVICE_ACCOUNT_KEY_PATH points to service-account-key.json in the project root.

6. Application Execution

6.1. Prepare and Upload Documents
- Place .txt files (e.g., example.txt from data/sample_documents/) into the root of the GCS bucket specified in config.py.
(Currently, ingest_docs.py is hardcoded to process a file named "example.txt" from the bucket root).

6.2. Ingest Documents
From the project root, with venv active:
```bash
python scripts/ingest_docs.py
```

6.3. Query Documents
After successful ingestion (allow a few minutes for Vector Search to fully index):
From the project root, with venv active:
```bash
python scripts/query_rag.py "Your question here"
```

Example:
```bash
python scripts/query_rag.py "What are primary colors?"
```

7. Cost Management
Key services incurring costs:
- Vertex AI Embedding Model (per 1k characters/tokens).
- Vertex AI LLM (per 1k input/output characters/tokens).
- Vertex AI Vector Search (indexing, endpoint hosting (hourly), querying).
- Google Cloud Storage (minimal for small files).

To manage costs, UNDEPLOY the index from the Vector Search Index Endpoint or DELETE the Index Endpoint when not actively testing.

8. Troubleshooting
- Module/Import Errors: Ensure venv is active, requirements.txt installed correctly, and using a stable Python version from python.org.
- GCP Auth Errors (401/403): Check service-account-key.json path, IAM roles, and API enablement. Allow time for IAM propagation.
- GCP API/Model Not Found (404) / Quota Errors (429): Verify APIs are enabled, region in config.py is correct and supports the specified models, and check project quotas.
- Vector Store Errors: Confirm all Vector Store IDs in config.py are correct and match deployed resources. Ensure index deployment to endpoint is complete and active.
