# src/gcp_clients/vertex_ai_client.py
import os
import sys
from typing import List, Optional, Any
import time # For potential retry logic if needed

# Use the direct 'vertexai' import
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput, TextGenerationModel
    SDK_AVAILABLE = True
    print("VertexAIClient: Successfully imported 'vertexai' and language model classes.")
except ImportError as e:
    SDK_AVAILABLE = False
    print(f"CRITICAL ERROR (VertexAIClient): Could not import 'vertexai' or its language model classes: {e}")
    print("Ensure 'google-cloud-aiplatform' is correctly installed and provides 'vertexai' namespace.")
    # Define dummy classes if real SDK not available, to allow script parsing
    class TextEmbeddingModel: pass # type: ignore
    class TextEmbeddingInput: pass # type: ignore
    class TextGenerationModel: pass # type: ignore

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_dir = os.path.join(project_root, 'config')
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

try:
    from config import PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
except ImportError:
    print("ERROR (VertexAIClient): Could not import configurations from config.py.")
    # Provide dummy values or raise error if config is essential at module load time
    PROJECT_ID = REGION = SERVICE_ACCOUNT_KEY_PATH = EMBEDDING_MODEL_NAME = LLM_MODEL_NAME = None


class VertexAIClient:
    _instance = None
    _vertex_ai_initialized = False

    def __new__(cls):
        if not SDK_AVAILABLE:
            raise ImportError("Vertex AI SDK components not available. Cannot create VertexAIClient.")
        if cls._instance is None:
            cls._instance = super(VertexAIClient, cls).__new__(cls)
            if not all([PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME]):
                raise ValueError("VertexAIClient: Missing one or more required configurations (PROJECT_ID, REGION, etc.).")

            cls._instance.project_id = PROJECT_ID
            cls._instance.region = REGION
            cls._instance.embedding_model_name = EMBEDDING_MODEL_NAME
            cls._instance.llm_model_name = LLM_MODEL_NAME
            
            credentials_full_path = os.path.join(project_root, SERVICE_ACCOUNT_KEY_PATH)
            if not os.path.exists(credentials_full_path):
                raise FileNotFoundError(f"VertexAIClient: Service account key not found at {credentials_full_path}")
            
            # Set environment variable for ADC, also pass explicitly if init supports it
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_full_path
            
            if not VertexAIClient._vertex_ai_initialized:
                try:
                    print(f"VertexAIClient: Initializing vertexai for project '{cls._instance.project_id}', location '{cls._instance.region}'...")
                    vertexai.init(project=cls._instance.project_id, location=cls._instance.region) # credentials picked up from env
                    VertexAIClient._vertex_ai_initialized = True
                    print("VertexAIClient: vertexai.init() successful.")
                except Exception as e_init:
                    print(f"VertexAIClient: Error during vertexai.init(): {e_init}")
                    raise

            try:
                print(f"VertexAIClient: Loading embedding model '{cls._instance.embedding_model_name}'...")
                cls._instance.embedding_model = TextEmbeddingModel.from_pretrained(cls._instance.embedding_model_name)
                print("VertexAIClient: Embedding model loaded.")
            except Exception as e_load_embed:
                print(f"VertexAIClient: Error loading embedding model: {e_load_embed}")
                cls._instance.embedding_model = None # Ensure it's None if loading failed
            
            cls._instance.llm_model = None # Load on first use

        return cls._instance

    def get_text_embeddings(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT", output_dimensionality: Optional[int] = None) -> List[Optional[List[float]]]:
        if self.embedding_model is None:
            print("VertexAIClient: Embedding model not available.")
            return [None] * len(texts)
        if not texts: return []

        print(f"VertexAIClient: Generating embeddings for {len(texts)} texts (task: {task_type})...")
        try:
            inputs = [TextEmbeddingInput(text, task_type) for text in texts]
            kwargs = {}
            if output_dimensionality: # Check model compatibility for this param
                kwargs["output_dimensionality"] = output_dimensionality
            
            embeddings_response = self.embedding_model.get_embeddings(inputs, **kwargs)
            print("VertexAIClient: Embeddings generated successfully.")
            return [emb.values for emb in embeddings_response]
        except Exception as e:
            print(f"VertexAIClient: Error generating embeddings: {e}")
            import traceback
            traceback.print_exc()
            return [None] * len(texts)

    def get_llm_completion(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 1024, top_p: float = 0.8, top_k: int = 40) -> Optional[str]:
        if self.llm_model is None:
            try:
                print(f"VertexAIClient: Loading LLM model '{self.llm_model_name}'...")
                self.llm_model = TextGenerationModel.from_pretrained(self.llm_model_name)
                print("VertexAIClient: LLM model loaded.")
            except Exception as e_load_llm:
                print(f"VertexAIClient: Error loading LLM model: {e_load_llm}")
                return None
        
        print(f"VertexAIClient: Getting LLM completion for prompt (first 80 chars): '{prompt[:80]}...'")
        try:
            response = self.llm_model.predict(
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            print("VertexAIClient: LLM completion received.")
            return response.text
        except Exception as e:
            print(f"VertexAIClient: Error during LLM prediction: {e}")
            import traceback
            traceback.print_exc()
            return None