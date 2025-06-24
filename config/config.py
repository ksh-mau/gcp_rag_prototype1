# config.py 

# --- GCP Project Settings ---
PROJECT_ID = "ragapplication-460303"  # e.g., "my-rag-project-123"
REGION = "us-east1"                   # e.g., "us-central1", "us-east1", "europe-west1" 
                                         # Choose a region where Vertex AI models and Vector Search are available

# --- Service Account Key ---
# Path relative to the project root (gcp_rag_prototype/)
# The JSON key file for your service account.
SERVICE_ACCOUNT_KEY_PATH = "service-account-key.json" 

# --- Google Cloud Storage (GCS) ---
GCS_BUCKET_NAME = "gcs-rag-bucket" # Must be globally unique

# --- Vertex AI Model Settings ---
# See: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
EMBEDDING_MODEL_NAME = "text-embedding-005" # Or other embedding models like "text-multilingual-embedding-002", "text-embedding-004"
LLM_MODEL_NAME = "text-bison@002"               # Or other text generation models like "text-bison", "gemini-1.0-pro"

# --- Vertex AI Vector Search (Vector Store) IDs ---
# These will be populated after you create the Index and Index Endpoint in GCP.
# The ID of the Index *resource* itself (e.g., a long number or the name you gave it)
VECTOR_STORE_INDEX_ID = "6420681713281662976"       
VECTOR_STORE_INDEX_ENDPOINT_ID = "6806478353235705856" 
VECTOR_STORE_DEPLOYED_INDEX_ID = "deployed_rag_prototype_ind_1748947406404" 

# --- RAG Pipeline Settings ---
# For basic_word_chunker, these are approximate word counts.
# For more advanced token-based chunkers, this would be token counts.
CHUNK_SIZE = 250        # Approximate words per chunk
CHUNK_OVERLAP = 30      # Approximate word overlap between chunks
TOP_K_RESULTS = 3       # Number of relevant chunks to retrieve for context