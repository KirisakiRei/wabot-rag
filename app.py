from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import CONFIG
import logging

app = Flask(__name__)

# ===== Model & Client =====
# Pakai model lokal agar tidak download ulang
model = SentenceTransformer("/home/kominfo/models/e5-small-local")

qdrant = QdrantClient(
    host=CONFIG["qdrant"]["host"],
    port=CONFIG["qdrant"]["port"]
)

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Stopwords (untuk overlap booster ringan) =====
STOPWORDS = {
    "apa", "bagaimana", "cara", "untuk", "dan", "atau", "yang", "dengan",
    "di", "ke", "dari", "buat", "mengurus", "membuat", "mendaftar", "dimana", "kapan",
    "berapa", "adalah", "itu", "ini", "saya", "kamu"
}

# ===== Helpers =====
def error_response(error_type, message, detail=None, code=500):
    payload = {"status": "error", "error": {"type": error_type, "message": message}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code

def tokenize_and_filter(text: str):
    # lowercase, split sederhana; buang stopwords & token sangat pendek
    return [w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) > 2]

def keyword_overlap(query: str, candidate: str) -> float:
    """
    Overlap Jaccard (0..1) setelah stopword removal.
    Dipakai hanya sebagai booster kecil, bukan filter keras.
    """
    q_tokens = set(tokenize_and_filter(query))
    c_tokens = set(tokenize_and_filter(candidate))
    if not q_tokens or not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)

# ===== API: WA Bot =====
@app.route("/api/search", methods=["POST"])
def search():
    try:
        data = request.json
        if not data or "question" not in data:
            return error_response("ValidationError", "Field 'question' wajib diisi", code=400)

        question = data["question"].strip()
        wa_number = data.get("wa_number", "unknown")
        category_id = data.get("category_id")  # optional

        # --- Threshold & bobot (disetel supaya dense jadi faktor utama) ---
        DENSE_ACCEPT = 0.88          # jika dense >= ini → langsung lolos meski overlap rendah
        FINAL_ACCEPT = 0.85          # kalau final >= ini → lolos
        OVERLAP_ALPHA = 0.15         # booster overlap kecil (dense + alpha*overlap)

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

        # --- Dense semantic search (E5) ---
        query_vector = model.encode("query: " + question).tolist()
        dense_hits = qdrant.search(
            collection_name="knowledge_bank",
            query_vector=query_vector,
            limit=5,
            query_filter=query_filter
        )

        # --- Keyword search (text index via MatchText) ---
        # Catatan: ini bukan BM25 murni, tapi text index Qdrant cukup membantu kata kunci literal
        keyword_hits, _ = qdrant.scroll(
            collection_name="knowledge_bank",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="question",
                    match=models.MatchText(text=question)
                )]
            ),
            limit=5
        )

        # --- Gabungkan kandidat via RRF sederhana ---
        combined = {}
        for i, h in enumerate(dense_hits):
            rrf_score = 1 / (i + 1)
            combined[str(h.id)] = {"hit": h, "rrf": combined.get(str(h.id), {}).get("rrf", 0) + rrf_score}

        for i, h in enumerate(keyword_hits):
            rrf_score = 1 / (i + 1)
            if str(h.id) in combined:
                combined[str(h.id)]["rrf"] += rrf_score
            else:
                combined[str(h.id)] = {"hit": h, "rrf": rrf_score}

        # Ambil top-N gabungan untuk dinilai dengan skor akhir
        top_items = sorted(combined.values(), key=lambda x: x["rrf"], reverse=True)[:3]

        if not top_items:
            return jsonify({"status": "not_found", "message": "Tidak ada data ditemukan"}), 404

        results = []
        debug_candidates = []

        for item in top_items:
            h = item["hit"]
            dense_score = float(getattr(h, "score", 0.0))  # pada hasil keyword, .score mungkin tidak ada
            overlap = float(keyword_overlap(question, h.payload["question"]))
            final_score = dense_score + OVERLAP_ALPHA * overlap

            debug_candidates.append({
                "id": str(h.id),
                "question": h.payload.get("question"),
                "dense_score": round(dense_score, 4),
                "overlap_score": round(overlap, 4),
                "final_score": round(final_score, 4)
            })

            # Aturan lolos:
            # 1) Dense sangat tinggi → lolos langsung (robust ke sinonim)
            # 2) Jika tidak, pakai final_score (dense + booster overlap)
            if dense_score >= DENSE_ACCEPT or final_score >= FINAL_ACCEPT:
                results.append({
                    "question": h.payload["question"],
                    "answer_id": h.payload["answer_id"],
                    "category_id": h.payload.get("category_id"),
                    "dense_score": dense_score,
                    "overlap_score": overlap,
                    "final_score": final_score
                })

        if not results:
            # Kirim debug score supaya mudah tuning di sisi klien/log
            logger.info(f"[LOW_CONF] DEBUG CANDIDATES: {debug_candidates}")
            return jsonify({
                "status": "low_confidence",
                "message": "Tidak ada hasil cukup relevan",
                "debug": {
                    "thresholds": {
                        "dense_accept": DENSE_ACCEPT,
                        "final_accept": FINAL_ACCEPT,
                        "overlap_alpha": OVERLAP_ALPHA
                    },
                    "candidates": debug_candidates
                }
            }), 200

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

# ===== API: WA Management (sync) =====
@app.route("/api/sync", methods=["POST"])
def sync_data():
    try:
        data = request.json
        if not data or "action" not in data:
            return error_response("ValidationError", "Field 'action' wajib diisi", code=400)

        action = data["action"]
        content = data.get("content")

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
                        "question": item["question"],   # penting untuk text index
                        "answer_id": item["answer_id"],
                        "category_id": item.get("category_id")
                    }
                })

            qdrant.upsert(collection_name="knowledge_bank", points=points)

            # Pastikan ada text index utk field 'question'
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

            logger.info(f"[SYNC] {len(points)} data disinkronisasi + text index OK")
            return jsonify({
                "status": "success",
                "message": f"Sinkronisasi {len(points)} data berhasil",
                "total_synced": len(points)
            })

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
