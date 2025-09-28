from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import CONFIG
import logging

app = Flask(__name__)

# Load model E5
model = SentenceTransformer("/home/kominfo/models/intfloat-multilingual-e5-small")


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

        # Filter kategori (opsional)
        query_filter = None
        if category_id:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="category_id",
                    match=models.MatchValue(value=category_id)
                )]
            )

        # --- Dense semantic search ---
        query_vector = model.encode("query: " + question).tolist()
        dense_hits = qdrant.search(
            collection_name="knowledge_bank",
            query_vector=query_vector,
            limit=5,
            query_filter=query_filter
        )

        # --- Keyword search (fallback text search) ---
        keyword_hits, _ = qdrant.scroll(
            collection_name="knowledge_bank",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="question",
                        match=models.MatchText(text=question)
                    )
                ]
            ),
            limit=5
        )

        # --- Gabungkan hasil (RRF sederhana) ---
        combined = {}
        for i, h in enumerate(dense_hits):
            score = 1 / (i + 1)  # Reciprocal Rank Fusion
            combined[h.id] = {"hit": h, "score": combined.get(h.id, {}).get("score", 0) + score}

        for i, h in enumerate(keyword_hits):
            score = 1 / (i + 1)
            if h.id in combined:
                combined[h.id]["score"] += score
            else:
                combined[h.id] = {"hit": h, "score": score}

        # Urutkan hasil
        sorted_hits = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:3]

        if not sorted_hits:
            return jsonify({"status": "not_found", "message": "Tidak ada data ditemukan"}), 404

        # Ambil hasil dengan threshold
        results = []
        for item in sorted_hits:
            h = item["hit"]
            score = item["score"]
            # pakai .score kalau ada (dense), kalau tidak pakai skor gabungan
            raw_score = getattr(h, "score", score)
            if raw_score >= min_similarity:
                results.append({
                    "question": h.payload["question"],
                    "answer_id": h.payload["answer_id"],
                    "category_id": h.payload.get("category_id"),
                    "similarity_score": float(raw_score)
                })

        if not results:
            return jsonify({"status": "low_confidence", "message": "Tidak ada hasil cukup relevan"}), 200

        return jsonify({
            "status": "success",
            "data": {
                "similar_questions": results,
                "metadata": {
                    "total_found": len(results),
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

        # Bulk sync
        if action == "bulk_sync":
            if not isinstance(content, list):
                return error_response("ValidationError", "Content harus berupa list", code=400)

            points = []
            for item in content:
                vector = model.encode("passage: " + item["question"]).tolist()
                points.append({
                    "id": item["id"],
                    "vector": vector,
                    "payload": {
                        "mysql_id": item["id"],
                        "question": item["question"],  # penting buat keyword search
                        "answer_id": item["answer_id"],
                        "category_id": item.get("category_id")
                    }
                })

            qdrant.upsert(collection_name="knowledge_bank", points=points)

            # index text untuk hybrid
            qdrant.create_payload_index(
                collection_name="knowledge_bank",
                field_name="question",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                )
            )

            return jsonify({
                "status": "success",
                "message": f"Sinkronisasi {len(points)} data (hybrid enabled)",
                "total_synced": len(points)
            })

        # Add
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
            return jsonify({"status": "success", "message": "Data berhasil ditambahkan", "id": mysql_id})

        # Update
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
            return jsonify({"status": "success", "message": "Data berhasil diupdate"})

        # Delete
        elif action == "delete":
            mysql_id = int(content["id"])
            qdrant.delete(
                collection_name="knowledge_bank",
                points_selector=models.PointIdsList(points=[mysql_id]),
                wait=True
            )
            return jsonify({"status": "success", "message": "Data berhasil dihapus"})

        else:
            return error_response("ValidationError", f"Action '{action}' tidak dikenali", code=400)

    except Exception as e:
        logger.error(f"Error in sync_data: {str(e)}", exc_info=True)
        return error_response("ServerError", "Kesalahan internal saat sinkronisasi", detail=str(e))


if __name__ == "__main__":
    app.run(host=CONFIG["api"]["host"], port=CONFIG["api"]["port"], debug=True)
