from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

def setup_qdrant():
    client = QdrantClient(host="localhost", port=6333)

    client.recreate_collection(
        collection_name="knowledge_bank",
        vectors_config=VectorParams(
            size=384,              
            distance=Distance.COSINE
        )
    )
    print("✅ Collection 'knowledge_bank' berhasil dibuat / diperbarui!")
    collections = client.get_collections()
    print("📌 Daftar collection sekarang:", collections)

if __name__ == "__main__":
    setup_qdrant()
