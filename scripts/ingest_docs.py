# scripts/ingest_docs.py
import os
import sys
import uuid

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
config_dir = os.path.join(project_root, 'config')
if config_dir not in sys.path: sys.path.insert(0, config_dir)

try:
    from config import GCS_BUCKET_NAME, CHUNK_SIZE, CHUNK_OVERLAP, \
                       PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH # Ensure these are imported
except ImportError:
    print("ERROR (ingest_docs.py): Could not import configurations from config.py. Make sure config_template.py was renamed and filled.")
    exit(1)

from gcp_clients.storage_client import GCSClient
from document_processor.chunking import basic_word_chunker
from gcp_clients.vertex_ai_client import VertexAIClient # Uses 'vertexai' import
from gcp_clients.vector_store_client import VectorStoreClient # Uses real SDK

def process_and_ingest_documents():
    print("--- Starting Document Ingestion Process (Using Real SDKs) ---")

    try:
        print("Initializing GCS Client...")
        gcs_client = GCSClient(project_id=PROJECT_ID, service_account_key_path=SERVICE_ACCOUNT_KEY_PATH)
        # Optional: Check or create bucket if you want the script to do this.
        # gcs_client.check_or_create_bucket(GCS_BUCKET_NAME, location=REGION) 
        
        print("Initializing Vertex AI Client (for embeddings)...")
        vertex_ai_client = VertexAIClient() 
        
        print("Initializing Vector Store Client...")
        vector_store_client = VectorStoreClient() 
    except Exception as e:
        print(f"FATAL: Failed to initialize clients: {e}")
        import traceback
        traceback.print_exc()
        return

    # For Phase 1, focusing on TXT files.
    # This example processes a predefined list. A real version might list .txt files from a GCS prefix.
    # Ensure these files exist in the root of your GCS_BUCKET_NAME
    # For simplicity, we process only one known file first.
    # Upload "example.txt" to the root of your GCS_BUCKET_NAME for this to work.
    files_to_process = ["example.txt"] 
    
    print(f"Target files for processing: {files_to_process}")
    
    all_embeddings_for_vector_store = []

    for doc_name in files_to_process:
        print(f"\nProcessing document: '{doc_name}' from GCS bucket: '{GCS_BUCKET_NAME}'")
        
        text_content = gcs_client.download_text_file(GCS_BUCKET_NAME, doc_name)
        
        if text_content is None or not text_content.strip():
            print(f"Could not download or content is empty for '{doc_name}'. Skipping.")
            continue
        
        print(f"Successfully downloaded '{doc_name}'. Content length: {len(text_content)} chars.")
        
        print(f"Chunking '{doc_name}' (size: {CHUNK_SIZE} words, overlap: {CHUNK_OVERLAP} words)...")
        chunks = basic_word_chunker(text_content, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"Document '{doc_name}' split into {len(chunks)} chunks.")

        if not chunks:
            print(f"No chunks generated for '{doc_name}'. Skipping embedding generation.")
            continue

        print(f"Generating embeddings for {len(chunks)} chunks from '{doc_name}'...")
        # Using RETRIEVAL_DOCUMENT as the task type for document chunks.
        chunk_embeddings = vertex_ai_client.get_text_embeddings(chunks, task_type="RETRIEVAL_DOCUMENT")

        if chunk_embeddings is None or len(chunk_embeddings) != len(chunks):
            print(f"Error generating embeddings for '{doc_name}' or mismatch in count. Skipping.")
            continue
        
        successful_embeddings_count = 0
        for i, chunk_text in enumerate(chunks):
            if chunk_embeddings[i] is not None:
                # Create a unique ID for each chunk
                chunk_id = f"{doc_name}_chunk_{uuid.uuid4()}" 
                
                embedding_data = {
                    "id": chunk_id,
                    "embedding": chunk_embeddings[i],
                    "metadata": { # Store useful metadata for filtering or display
                        "source_document_name": doc_name,
                        "chunk_index": str(i), # Vector store often expects string values for metadata
                        "text_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text # Preview, not full text for metadata
                    }
                }
                all_embeddings_for_vector_store.append(embedding_data)
                successful_embeddings_count +=1
            else:
                print(f"  Skipping chunk {i} of '{doc_name}' due to missing embedding for that specific chunk.")
        
        print(f"Successfully generated {successful_embeddings_count} embeddings for chunks of '{doc_name}'.")

    if all_embeddings_for_vector_store:
        print(f"\nUpserting {len(all_embeddings_for_vector_store)} total embeddings to Vector Store...")
        # Note: Vertex AI Vector Search upsert can take a few minutes to reflect.
        success = vector_store_client.upsert_embeddings(all_embeddings_for_vector_store)
        if success:
            print("All embeddings successfully upserted to Vector Store.")
        else:
            print("Failed to upsert embeddings to Vector Store.")
    else:
        print("\nNo valid embeddings were generated to upsert.")

    print("\n--- Document Ingestion Process Finished ---")

if __name__ == '__main__':
    # Ensure all necessary configurations are set
    required_configs = [
        PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH, GCS_BUCKET_NAME,
        CHUNK_SIZE, CHUNK_OVERLAP
    ]
    if not all(required_configs) or any("YOUR_" in str(val) for val in required_configs if isinstance(val, str)):
        print("ERROR: One or more critical configurations (PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH, GCS_BUCKET_NAME, CHUNK_SIZE, CHUNK_OVERLAP) are missing or still have placeholder values in config.py.")
    else:
        process_and_ingest_documents()