from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import CONFIG
import logging

app = Flask(__name__)


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API WA Bot ---
@app.route("/api/search", methods=["POST"])
def search():
    try:
        data = request.json
        question = data["question"]
        wa_number = data.get("wa_number", "unknown")

        # Cari jawaban top 3
        hits = qdrant.search(
            collection_name="knowledge_bank",
            query_vector=model.encode(question).tolist(),
            limit=3
        )

        if not hits:
            return jsonify({
                "status": "not_found",
                "message": "Tidak ditemukan pertanyaan yang relevan"
            })

        similar_questions = []
        for hit in hits:
            similar_questions.append({
                "id": hit.id,
                "question": hit.payload["question"],
                "answer": hit.payload["answer"],
                "category": hit.payload.get("category"),
                "similarity_score": hit.score
            })

        return jsonify({
            "status": "success",
            "data": {
                "similar_questions": similar_questions,
                "metadata": {
                    "total_found": len(hits),
                    "wa_number": wa_number,
                    "original_question": question
                }
            }
        })

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# --- API WA Management ---
@app.route("/api/sync", methods=["POST"])
def sync_data():
    try:
        data = request.json
        action = data["action"]
        content = data.get("content")

        # Bulk sync semua data
        if action == "bulk_sync":
            data_list = content
            points = []
            for item in data_list:
                vector = model.encode(item["question"]).tolist()
                points.append({
                    "id": item["id"],
                    "vector": vector,
                    "payload": {
                        "mysql_id": item["id"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "category": item.get("category")
                    }
                })
            qdrant.upsert(collection_name="knowledge_bank", points=points)
            logger.info(f"Bulk sync completed: {len(points)} records synchronized")
            return jsonify({
                "status": "success",
                "message": f"Berhasil sinkronisasi {len(points)} data Knowledge Bank",
                "total_synced": len(points)
            })

        # Tambah data baru
        elif action == "add":
            vector = model.encode(content["question"]).tolist()
            mysql_id = content["id"]
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": mysql_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": mysql_id,
                        "question": content["question"],
                        "answer": content["answer"],
                        "category": content.get("category")
                    }
                }]
            )
            logger.info(f"Data added to Knowledge Bank with ID: {mysql_id}")
            return jsonify({"status": "success", "message": "Data berhasil ditambahkan ke Knowledge Bank", "id": mysql_id})

        # Update data
        elif action == "update":
            mysql_id = content["id"]
            vector = model.encode(content["question"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": mysql_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": mysql_id,
                        "question": content["question"],
                        "answer": content["answer"],
                        "category": content.get("category")
                    }
                }]
            )
            logger.info(f"Data updated in Knowledge Bank with ID: {mysql_id}")
            return jsonify({"status": "success", "message": "Data berhasil diupdate di Knowledge Bank"})

        # Hapus data
        elif action == "delete":
            mysql_id = int(content["id"])
            try:
                points_selector = models.PointIdsList(points=[mysql_id])
                qdrant.delete(
                    collection_name="knowledge_bank",
                    points_selector=points_selector,
                    wait=True
                )
                logger.info(f"Data deleted from Knowledge Bank with ID: {mysql_id}")
                return jsonify({"status": "success", "message": "Data berhasil dihapus dari Knowledge Bank"})
            except Exception as delete_error:
                logger.error(f"Failed to delete data with ID {mysql_id}: {str(delete_error)}")
                return jsonify({"status": "error", "message": f"Gagal menghapus data: {str(delete_error)}"}), 500

        else:
            return jsonify({"status": "error", "message": f"Action '{action}' tidak dikenali"}), 400

    except Exception as e:
        logger.error(f"Error in sync_data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host=CONFIG["api"]["host"], port=CONFIG["api"]["port"], debug=True)
