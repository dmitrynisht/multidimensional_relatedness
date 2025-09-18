import chromadb
import torch
import numpy as np


class ChromaEmbeddingStore:
    def __init__(self, db_name="sage_dynamic", db_path: str = None):
        if db_path is None:
            raise ValueError("Path to ChromaDB embedding storage must be provided")

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=db_name)
        self.db_name = db_name

    def store(self, embeddings, prefix="node"):
        vectors = embeddings.cpu().numpy()
        ids = [f"{prefix}_{i}" for i in range(vectors.shape[0])]
        self.collection.add(
            ids=ids,
            embeddings=vectors.tolist(),
            metadatas=[{"node_id": i} for i in range(vectors.shape[0])],
        )

        # Safe check for empty or missing data 
        results = self.collection.get(include=["embeddings", "metadatas"])
        if results.get("embeddings") is None or len(results["embeddings"]) == 0:
            print("No embeddings stored in ChromaDB collection.")
        else:
            print(f"In ChromaDB collection Stored {len(ids)} embeddings.")

    def load(self):
        results = self.collection.get(include=["embeddings", "metadatas"])

        # Safe check for empty or missing data
        if results.get("embeddings") is None or len(results["embeddings"]) == 0:
            raise ValueError("No embeddings found in ChromaDB collection.")

        if results.get("metadatas") is None or len(results["metadatas"]) == 0:
            raise ValueError("No metadata found in ChromaDB collection.")

        node_ids = [meta["node_id"] for meta in results["metadatas"]]
        embeddings = results["embeddings"]
        sorted_items = sorted(zip(node_ids, embeddings), key=lambda x: x[0])
        array = np.array([vec for _, vec in sorted_items])
        
        return torch.tensor(array)

    def clear(self):
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name
        )
        print(f"[Chroma] Collection '{self.collection.name}' cleared.")

    def delete_named_collection(self, collection_name: str):
        """
        Deletes a specific collection by name (without affecting the current one).
        """
        if collection_name == self.collection.name:
            raise ValueError("Use .clear() to delete and reset the active collection.")
        
        self.client.delete_collection(name=collection_name)
        print(f"[Chroma] Deleted collection: '{collection_name}'")
        
    def list_collections(self):
        """
        Returns a list of collection names available under the current client path.
        """

        return [coll for coll in self.client.list_collections()]


def get_chroma_client(db_name: str = "sage_dynamic", db_path: str = None):
    """
    Initializes and returns a ChromaDB client and collection.

    Args:
        db_name: Name of the vector DB (used as collection name).
        base_path: Local filesystem path for ChromaDB persistent storage.

    Returns:
        (client, collection)
    """
    if db_path is None:
        raise ValueError("Path to ChromaDB embedding storage must be provided")

    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(name=db_name)

    return chroma_client, chroma_collection


def store_final_embeddings(embeddings, collection, prefix="node"):
    """
    Stores final embeddings (after GRU) into a ChromaDB collection.

    Args:
        embeddings: A torch.Tensor of shape [num_nodes, embedding_dim].
        collection: A ChromaDB collection object.
        prefix: Optional prefix to use for ID naming (e.g., 'node_0', 'node_1', ...).
    """
    vectors = embeddings.cpu().numpy()
    ids = [f"{prefix}_{i}" for i in range(vectors.shape[0])]
    collection.add(
        ids=ids,
        embeddings=vectors.tolist(),
        metadatas=[{"node_id": i} for i in range(vectors.shape[0])],
    )
    print(f"Stored {len(ids)} embeddings into ChromaDB.")


def load_all_embeddings(collection, prefix="node"):
    """
    Loads all embeddings from ChromaDB and returns them as a tensor.

    Args:
        collection: A ChromaDB collection object.
        prefix: Optional prefix used for ID naming.

    Returns:
        torch.Tensor of shape [num_nodes, embedding_dim]
    """
    results = collection.get()
    node_ids = [item["node_id"] for item in results["metadatas"]]
    embeddings = results["embeddings"]

    # Ensure consistent ordering by sorting node IDs
    sorted_items = sorted(zip(node_ids, embeddings), key=lambda x: x[0])
    sorted_embeddings = np.array([vec for _, vec in sorted_items])

    return torch.tensor(sorted_embeddings)
