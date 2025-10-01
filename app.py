from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import CONFIG
import logging

app = Flask(__name__)

# Load model E5 (pakai path lokal)
model = SentenceTransformer("/home/kominfo/models/e5-small-local")

# Setup Qdrant client
qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stopwords sederhana (bisa diperluas sesuai kebutuhan)
STOPWORDS = {
    "apa", "bagaimana", "cara", "untuk", "dan", "atau", "yang", "dengan",
    "di", "ke", "dari", "buat", "membuat", "mendaftar", "dimana", "kapan",
    "berapa", "siapa", "adalah", "itu", "ini", "saya", "kamu"
}

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

# --- Helper overlap dengan stopword removal ---
def tokenize_and_filter(text: str):
    return [w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) > 2]

def keyword_overlap(query: str, candidate: str) -> float:
    q_tokens = set(tokenize_and_filter(query))
    c_tokens = set(tokenize_and_filter(candidate))
    if not q_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)


# --- API WA Bot ---
@app.route("/api/search", methods=["POST"])
def search():
    try:
        data = request.json
        if not data or "question" not in data:
            return error_response("ValidationError", "Field 'question' wajib diisi", code=400)

        question = data["question"].strip()
        wa_number = data.get("wa_number", "unknown")
        category_id = data.get("category_id")  # optional

        # Threshold minimal
        min_similarity = 0.85
        min_overlap = 0.3
        min_final_score = 0.80

        # Validasi panjang pertanyaan
        if len(question.split()) < 2:
            return jsonify({
                "status": "low_confidence",
                "message": "Pertanyaan terlalu singkat, mohon lebih spesifik"
            }), 200

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

        # --- Keyword search (BM25 sederhana via scroll + text index) ---
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

        # --- Gabungkan hasil (Reciprocal Rank Fusion) ---
        combined = {}
        for i, h in enumerate(dense_hits):
            score = 1 / (i + 1)
            combined[str(h.id)] = {"hit": h, "score": combined.get(str(h.id), {}).get("score", 0) + score}

        for i, h in enumerate(keyword_hits):
            score = 1 / (i + 1)
            if str(h.id) in combined:
                combined[str(h.id)]["score"] += score
            else:
                combined[str(h.id)] = {"hit": h, "score": score}

        sorted_hits = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:3]

        if not sorted_hits:
            return jsonify({"status": "not_found", "message": "Tidak ada data ditemukan"}), 404

        # --- Seleksi hasil dengan threshold + overlap ---
        results = []
        for item in sorted_hits:
            h = item["hit"]
            dense_score = getattr(h, "score", 0.0)
            overlap = keyword_overlap(question, h.payload["question"])
            final_score = 0.7 * dense_score + 0.3 * overlap

            if dense_score >= min_similarity and overlap >= min_overlap and final_score >= min_final_score:
                results.append({
                    "question": h.payload["question"],
                    "answer_id": h.payload["answer_id"],
                    "category_id": h.payload.get("category_id"),
                    "dense_score": float(dense_score),
                    "overlap_score": float(overlap),
                    "final_score": float(final_score)
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
                point_id = str(item["id"])
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": point_id,
                        "question": item["question"],
                        "answer_id": item["answer_id"],
                        "category_id": item.get("category_id")
                    }
                })

            qdrant.upsert(collection_name="knowledge_bank", points=points)

            # Index text
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
            point_id = str(content["id"])
            vector = model.encode("passage: " + content["question"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": point_id,
                        "question": content["question"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id")
                    }
                }]
            )
            return jsonify({"status": "success", "message": "Data berhasil ditambahkan", "id": point_id})

        # Update
        elif action == "update":
            point_id = str(content["id"])
            vector = model.encode("passage: " + content["question"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": point_id,
                        "question": content["question"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id")
                    }
                }]
            )
            return jsonify({"status": "success", "message": "Data berhasil diupdate"})

        # Delete
        elif action == "delete":
            point_id = str(content["id"])
            qdrant.delete(
                collection_name="knowledge_bank",
                points_selector=models.PointIdsList(points=[point_id]),
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
