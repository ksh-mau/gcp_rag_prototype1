# scripts/query_rag.py
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path: sys.path.insert(0, src_path)
config_dir = os.path.join(project_root, 'config')
if config_dir not in sys.path: sys.path.insert(0, config_dir)

try:
    from config import TOP_K_RESULTS, PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH # Ensure these are imported
except ImportError:
    print("ERROR (query_rag.py): Could not import configurations from config.py. Make sure config_template.py was renamed and filled.")
    exit(1)

from gcp_clients.vertex_ai_client import VertexAIClient # Uses 'vertexai' import
from gcp_clients.vector_store_client import VectorStoreClient # Uses real SDK

def search_and_answer(user_query: str):
    print("--- Starting RAG Query Process (Using Real SDKs) ---")

    try:
        print("Initializing Vertex AI Client (for embeddings & LLM)...")
        vertex_ai_client = VertexAIClient()
        
        print("Initializing Vector Store Client...")
        vector_store_client = VectorStoreClient()
    except Exception as e:
        print(f"FATAL: Failed to initialize clients: {e}")
        import traceback
        traceback.print_exc()
        return

    # 1. Generate embedding for the user query
    print(f"\nGenerating embedding for query: '{user_query}'")
    query_embedding_list = vertex_ai_client.get_text_embeddings([user_query], task_type="RETRIEVAL_QUERY")

    if not query_embedding_list or not query_embedding_list[0]:
        print("Failed to generate embedding for the query. Cannot proceed.")
        return
    query_embedding = query_embedding_list[0]
    print("Query embedding generated successfully.")

    # 2. Search Vector Store for relevant document chunks
    print(f"\nSearching Vector Store for top {TOP_K_RESULTS} relevant chunks...")
    # find_neighbors returns a list of MatchNeighbor objects
    neighbors = vector_store_client.find_neighbors(query_embedding=query_embedding, num_neighbors=TOP_K_RESULTS)

    if not neighbors:
        print("No relevant document chunks found in the Vector Store for your query.")
        # Optional: Call LLM with just the query for a general answer, or state no context found.
        print("\nAttempting to answer with LLM based on general knowledge (no specific context found)...")
        llm_response = vertex_ai_client.get_llm_completion(prompt=user_query)
        if llm_response:
            print("\n--- LLM Answer (General Knowledge) ---")
            print(llm_response)
        else:
            print("Sorry, I could not generate an answer for your query.")
        return

    print(f"Found {len(neighbors)} relevant neighbor(s).")

    # 3. Construct context from retrieved chunks
    # The MatchNeighbor object itself doesn't directly contain the original text.
    # The 'id' of the MatchNeighbor is the datapoint_id we used during upsert.
    # Our 'id' was like "doc_name_chunk_uuid".
    # The metadata like 'text_preview' or 'source_document_name' was stored with the datapoint in the Vector Store's own storage,
    # but not directly returned by the `match` method unless the index is configured for it and the client supports it easily.
    # For a robust RAG, you'd typically:
    #   a) Store full chunk text in a separate metadata store (like Firestore, GCS JSONs) keyed by chunk_id.
    #   b) Or, if Vector Store allows storing sufficient metadata and returning it with results, use that.
    # For this simplified version, we'll assume the 'text_preview' in our Vector Store datapoint's
    # metadata restriction could be used or we fetch it based on ID if we had stored it elsewhere.
    # The current `VectorStoreClient.find_neighbors` does NOT return metadata directly with MatchNeighbor.
    # We will need to enhance VectorStoreClient or our data strategy for full context.
    #
    # SIMPLIFICATION FOR NOW: We don't have the text from the neighbors directly.
    # A real RAG would fetch this text based on neighbor.id.
    # For this illustrative script, we'll just show the IDs.
    # To make it *runnable* for a demo, we need to get the text.
    # Let's assume for now that we modify VectorStoreClient.find_neighbors
    # or the ingestion to store text in a way that the mock/simplified query can get it.
    # For now, the `VectorStoreClient.find_neighbors` has been updated to simulate returning metadata.
    # And the `RealVectorStoreClient.upsert_embeddings` was updated to store some metadata as restrictions.
    # However, the `match` API does not return `restricts` by default with neighbors.
    # This part will need significant refinement for a production RAG.
    
    context_parts = []
    source_documents_cited = set()

    print("\nRetrieved Chunks (IDs and Distances):")
    for neighbor in neighbors:
        print(f"  ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
        # To get the actual text for the context, you would typically:
        # 1. Have stored the text along with the embedding OR
        # 2. Store a reference (like GCS path and byte offset) and retrieve it now.
        # 3. Some vector databases allow returning metadata fields. Vertex AI's `match` primarily returns ID and distance.
        # For this simplified example, we can't directly reconstruct full context from just the ID without an external lookup.
        # Let's assume for the sake of continuing the flow, we have a way to get text from ID.
        # This would be a TODO: Implement fetching chunk text by ID.
        # For demo, we'll make up context based on the ID.
        context_parts.append(f"Retrieved information related to ID {neighbor.id}.") # Placeholder context
        
        # Attempt to parse source document from ID if it follows "filename_chunk_..."
        if "_chunk_" in neighbor.id:
            source_documents_cited.add(neighbor.id.split("_chunk_")[0])

    if not context_parts:
        # This case should ideally be handled by the "No relevant document chunks found" above.
        print("Could not form context from retrieved neighbors.")
        return

    context_string = "\n".join(context_parts)
    print(f"\nConstructed Context (Placeholder): {context_string}") # This will be placeholder context

    # 4. Construct prompt for LLM
    prompt_for_llm = (
        "You are a helpful AI assistant. Please answer the user's question based ONLY on the following "
        "provided context. If the context does not contain the information to answer the question, "
        "please state that you don't have enough information from the provided documents.\n\n"
        f"CONTEXT:\n{context_string}\n\n"
        f"QUESTION:\n{user_query}\n\n"
        "ANSWER:"
    )
    print("\nSending prompt to LLM...")

    # 5. Get completion from LLM
    llm_response = vertex_ai_client.get_llm_completion(prompt=prompt_for_llm)

    # 6. Display result
    print("\n--- RAG System Answer ---")
    if llm_response:
        print(llm_response)
        if source_documents_cited:
            print("\nSources:")
            for src_doc in sorted(list(source_documents_cited)):
                print(f"- {src_doc}")
    else:
        print("Sorry, I could not generate an answer for your query based on the retrieved context.")

    print("\n--- RAG Query Process Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("user_query", type=str, help="The question you want to ask.")
    args = parser.parse_args()

    if not all([PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH, TOP_K_RESULTS]):
         print("ERROR: One or more critical configurations are missing or still have placeholder values in config.py.")
    else:
        search_and_answer(args.user_query)