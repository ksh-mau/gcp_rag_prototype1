# src/gcp_clients/vector_store_client.py
import os
import sys
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import IndexDatapoint # For constructing datapoints

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_dir = os.path.join(project_root, 'config')
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

try:
    from config import PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH, \
                       VECTOR_STORE_INDEX_ENDPOINT_ID, VECTOR_STORE_DEPLOYED_INDEX_ID
except ImportError:
    print("ERROR (VectorStoreClient): Could not import configurations from config.py.")
    PROJECT_ID = REGION = SERVICE_ACCOUNT_KEY_PATH = VECTOR_STORE_INDEX_ENDPOINT_ID = VECTOR_STORE_DEPLOYED_INDEX_ID = None

class VectorStoreClient:
    _instance = None
    _vertex_ai_initialized_by_vector_client = False 

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreClient, cls).__new__(cls)
            if not all([PROJECT_ID, REGION, SERVICE_ACCOUNT_KEY_PATH, VECTOR_STORE_INDEX_ENDPOINT_ID, VECTOR_STORE_DEPLOYED_INDEX_ID]):
                raise ValueError("VectorStoreClient: Missing one or more required configurations.")
            if "YOUR_ACTUAL" in str(VECTOR_STORE_INDEX_ENDPOINT_ID) or "YOUR_ACTUAL" in str(VECTOR_STORE_DEPLOYED_INDEX_ID):
                raise ValueError("VectorStoreClient: Placeholder values found for Vector Store IDs in config.py.")


            cls._instance.project_id = PROJECT_ID
            cls._instance.region = REGION
            cls._instance.index_endpoint_id = VECTOR_STORE_INDEX_ENDPOINT_ID
            cls._instance.deployed_index_id = VECTOR_STORE_DEPLOYED_INDEX_ID

            credentials_full_path = os.path.join(project_root, SERVICE_ACCOUNT_KEY_PATH)
            if not os.path.exists(credentials_full_path):
                 raise FileNotFoundError(f"VectorStoreClient: Service account key not found at {credentials_full_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_full_path # Ensure ADC is set

            # Initialize aiplatform if VertexAIClient hasn't already done it
            # This check is a bit simplistic as it relies on a global state of aiplatform SDK.
            # A more robust approach might involve a shared initialization flag or module.
            try:
                if not aiplatform.constants.PROJECT_NAME: # A way to check if aiplatform.init was called
                    print(f"VectorStoreClient: Initializing vertexai for project '{cls._instance.project_id}', location '{cls._instance.region}'...")
                    aiplatform.init(project=cls._instance.project_id, location=cls._instance.region)
                    VectorStoreClient._vertex_ai_initialized_by_vector_client = True
                    print("VectorStoreClient: vertexai.init() successful.")
            except Exception as e_init:
                print(f"VectorStoreClient: Warning during aiplatform.init() (may have been initialized elsewhere): {e_init}")

            try:
                print(f"VectorStoreClient: Connecting to Index Endpoint ID '{cls._instance.index_endpoint_id}'...")
                cls._instance.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=cls._instance.index_endpoint_id
                )
                print("VectorStoreClient: Connected to Index Endpoint.")
            except Exception as e_connect:
                print(f"VectorStoreClient: Error connecting to Index Endpoint: {e_connect}")
                raise
        return cls._instance

    def upsert_embeddings(self, embeddings_with_metadata: List[Dict[str, Any]]) -> bool:
        if not embeddings_with_metadata:
            print("VectorStoreClient: No embeddings provided to upsert.")
            return True
        
        datapoints = []
        for item in embeddings_with_metadata:
            if not ("id" in item and "embedding" in item):
                print(f"VectorStoreClient: Skipping invalid item for upsert (missing id or embedding): {item.get('id', 'N/A')}")
                continue
            
            # Construct restrictions from metadata
            restricts = []
            if "metadata" in item and isinstance(item["metadata"], dict):
                for key, value in item["metadata"].items():
                    # Vector store restricts to string values for allow_list
                    restricts.append(IndexDatapoint.Restriction(namespace=str(key), allow_list=[str(value)]))

            datapoints.append(
                IndexDatapoint(
                    datapoint_id=item["id"],
                    feature_vector=item["embedding"],
                    restricts=restricts if restricts else None,
                )
            )
        
        if not datapoints:
            print("VectorStoreClient: No valid datapoints to upsert after processing.")
            return False

        print(f"VectorStoreClient: Upserting {len(datapoints)} datapoints to deployed index '{self.deployed_index_id}'...")
        try:
            self.index_endpoint.upsert_datapoints(
                datapoints=datapoints,
                deployed_index_id=self.deployed_index_id
            )
            print("VectorStoreClient: Datapoints upserted successfully.")
            return True
        except Exception as e:
            print(f"VectorStoreClient: Error upserting datapoints: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_neighbors(self, query_embedding: List[float], num_neighbors: int) -> List[aiplatform.MatchingEngineIndexEndpoint.MatchNeighbor]:
        print(f"VectorStoreClient: Finding {num_neighbors} neighbors...")
        try:
            response = self.index_endpoint.match(
                deployed_index_id=self.deployed_index_id,
                queries=[query_embedding],
                num_neighbors=num_neighbors,
            )
            # response is a list of lists of MatchNeighbor objects.
            # Since we send one query, we take the first list.
            neighbors = response[0] if response else []
            print(f"VectorStoreClient: Found {len(neighbors)} neighbors.")
            return neighbors
        except Exception as e:
            print(f"VectorStoreClient: Error finding neighbors: {e}")
            import traceback
            traceback.print_exc()
            return []