from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import CONFIG
import logging

app = Flask(__name__)

# Load model E5
model = SentenceTransformer("intfloat/multilingual-e5-small")

# Setup Qdrant client
qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper untuk error response ---
def error_response(error_type, message, detail=None, code=500):
    payload = {
        "status": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code


# --- API WA Bot ---
@app.route("/api/search", methods=["POST"])
def search():
    try:
        data = request.json
        if not data or "question" not in data:
            return error_response("ValidationError", "Field 'question' wajib diisi", code=400)

        question = data["question"]
        wa_number = data.get("wa_number", "unknown")
        category_id = data.get("category_id")  # optional
        min_similarity = 0.65

        logger.info(f"[SEARCH] User={wa_number}, Question='{question}', CategoryID={category_id}")

        # Tambahkan filter kategori jika ada
        query_filter = None
        if category_id:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="category_id",
                    match=models.MatchValue(value=category_id)
                )]
            )
            logger.info(f"[SEARCH] Filter kategori diterapkan: {category_id}")

        # Encode query (E5 format)
        query_vector = model.encode("query: " + question).tolist()

        # HYBRID SEARCH: gabungkan semantic + keyword
        hits = qdrant.query_points(
            collection_name="knowledge_bank",
            prefetch=[
                # Semantic vector search
                models.Prefetch(
                    query=query_vector,
                    limit=50
                ),
                # Keyword lexical search
                models.Prefetch(
                    query=models.MatchText(
                        key="question",
                        text=question
                    ),
                    limit=50
                )
            ],
            fusion=models.Fusion.RRF,
            limit=3,
            query_filter=query_filter
        )

        if not hits or not hits.points:
            msg = "Tidak ada data ditemukan"
            logger.info(f"[SEARCH] {msg}")
            return jsonify({"status": "not_found", "message": msg}), 404

        # Filter berdasarkan similarity threshold
        filtered_hits = [h for h in hits.points if h.score >= min_similarity]

        logger.info(f"[SEARCH] Hasil setelah filter similarity >= {min_similarity}: {len(filtered_hits)}")
        for h in filtered_hits:
            logger.info(f"   - ID={h.id}, Score={h.score:.4f}")

        if not filtered_hits:
            msg = "Tidak ada hasil cukup relevan untuk pertanyaan Anda."
            return jsonify({"status": "low_confidence", "message": msg}), 200

        # Format hasil
        similar_questions = [
            {
                "question": hit.payload["question"],
                "answer_id": hit.payload["answer_id"],
                "category_id": hit.payload.get("category_id"),
                "similarity_score": hit.score
            }
            for hit in filtered_hits
        ]

        return jsonify({
            "status": "success",
            "data": {
                "similar_questions": similar_questions,
                "metadata": {
                    "total_found": len(similar_questions),
                    "wa_number": wa_number,
                    "original_question": question,
                    "category_used": category_id
                }
            }
        })

    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return error_response("ServerError", "Terjadi kesalahan internal pada server", detail=str(e))


# --- API WA Management ---
@app.route("/api/sync", methods=["POST"])
def sync_data():
    try:
        data = request.json
        if not data or "action" not in data:
            return error_response("ValidationError", "Field 'action' wajib diisi", code=400)

        action = data["action"]
        content = data.get("content")

        # Bulk sync semua data
        if action == "bulk_sync":
            if not isinstance(content, list):
                return error_response("ValidationError", "Content harus berupa list untuk bulk_sync", code=400)

            points = []
            for item in content:
                vector = model.encode("passage: " + item["question"]).tolist()
                points.append({
                    "id": item["id"],
                    "vector": vector,
                    "payload": {
                        "mysql_id": item["id"],
                        "question": item["question"],
                        "answer_id": item["answer_id"],
                        "category_id": item.get("category_id")
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
            vector = model.encode("passage: " + content["question"]).tolist()
            mysql_id = content["id"]
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": mysql_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": mysql_id,
                        "question": content["question"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id")
                    }
                }]
            )
            logger.info(f"Data added to Knowledge Bank with ID: {mysql_id}")
            return jsonify({"status": "success", "message": "Data berhasil ditambahkan ke Knowledge Bank", "id": mysql_id})

        # Update data
        elif action == "update":
            mysql_id = content["id"]
            vector = model.encode("passage: " + content["question"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": mysql_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": mysql_id,
                        "question": content["question"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id")
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
                return error_response("DeleteError", f"Gagal menghapus data dengan ID {mysql_id}", detail=str(delete_error))

        else:
            return error_response("ValidationError", f"Action '{action}' tidak dikenali", code=400)

    except Exception as e:
        logger.error(f"Error in sync_data: {str(e)}", exc_info=True)
        return error_response("ServerError", "Terjadi kesalahan internal saat sinkronisasi", detail=str(e))


if __name__ == "__main__":
    app.run(host=CONFIG["api"]["host"], port=CONFIG["api"]["port"], debug=True)
