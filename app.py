from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import CONFIG
import logging
import requests
import json
import re
import time

app = Flask(__name__)

# ==========================================================
# 🔹 LOAD MODEL DAN CLIENT
# ==========================================================
model = SentenceTransformer("/home/kominfo/models/e5-small-local")
qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# 🔹 UTILITY DAN SETUP
# ==========================================================
STOPWORDS = {
    "apa", "bagaimana", "cara", "untuk", "dan", "atau", "yang", "dengan",
    "di", "ke", "dari", "buat", "mengurus", "membuat", "mendaftar",
    "dimana", "kapan", "berapa", "adalah", "itu", "ini", "saya", "kamu",
    "siapa", "kepala", "kota", "medan"
}

CATEGORY_KEYWORDS = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": ["ktp","kependudukan","kk","akta","kelahiran","kematian","domisili"],
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": ["bpjs","kesehatan","rsud","puskesmas","klinik","vaksin","obat"],
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": ["sekolah","guru","siswa","ppdb","beasiswa","pendidikan"],
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": ["pengaduan","izin","pelayanan","bantuan","masyarakat","usaha"],
    "0196f6b9-ba96-70f1-a930-3b89e763170f": ["kepala dinas","kadis","sekretaris","jabatan","struktur"],
    "01970829-1054-72b2-bb31-16a34edd84fc": ["aturan","peraturan","perwali","perda","perpres","hukum"],
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": ["lokasi","alamat","kantor"],
    "001970853-dd2e-716e-b90c-c4f79270f700": ["tugas","fungsi","profil","visi","misi"]
}

CATEGORY_NAMES = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": "Kependudukan",
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": "Kesehatan",
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": "Pendidikan",
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": "Layanan Masyarakat",
    "0196f6b9-ba96-70f1-a930-3b89e763170f": "Struktur Organisasi",
    "01970829-1054-72b2-bb31-16a34edd84fc": "Peraturan",
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": "Lokasi Fasilitas Pemerintahan Kota Medan",
    "001970853-dd2e-716e-b90c-c4f79270f700": "Profil"
}

def detect_category(q):
    ql = q.lower()
    for cid, kws in CATEGORY_KEYWORDS.items():
        if any(kw in ql for kw in kws):
            return {"id": cid, "name": CATEGORY_NAMES[cid]}
    return None

def normalize_text(t):
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def clean_location_terms(t):
    t = re.sub(r"\bdi\s+kota\s+medan\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdi\s+medan\b", "", t, flags=re.IGNORECASE)
    return t.strip()

def tokenize_and_filter(t):
    return [w.lower() for w in t.split() if w.lower() not in STOPWORDS and len(w) > 2]

def keyword_overlap(a, b):
    A, B = set(tokenize_and_filter(a)), set(tokenize_and_filter(b))
    return len(A & B) / len(A | B) if A and B else 0.0

# ==========================================================
# 🔹 AI RELEVANCE CHECK (Post Validation)
# ==========================================================
def check_relevance_ai(user_q, rag_q):
    try:
        url = "https://dekallm.cloudeka.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6FaPtqd1W5aj0z_-AbsKBA",
            "Content-Type": "application/json"
        }
        system_prompt = """
Anda bertugas memeriksa apakah hasil pencarian RAG sesuai dengan maksud pertanyaan pengguna.
Balas hanya dalam JSON berikut:
{
 "relevant": true/false,
 "reason": "...",
 "reformulated_question": "..."
}
Kriteria:
- relevan jika hasil menjawab inti pertanyaan.
- jika berbeda konteks, terlalu umum, atau salah fokus, set relevant=false dan bantu ubah pertanyaan dengan versi lebih jelas.
"""
        user_prompt = f"User Question: {user_q}\nRAG Result: {rag_q}"
        payload = {
            "model": "meta/llama-4-maverick-instruct",
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            "temperature": 0.1,
            "top_p": 0.5
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"relevant": True, "reason": "-", "reformulated_question": ""}
    except Exception as e:
        logger.error(f"[AI-Relevance] Error: {e}", exc_info=True)
        return {"relevant": True, "reason": "AI check failed", "reformulated_question": ""}

# ==========================================================
# 🔹 SEARCH ENDPOINT
# ==========================================================
@app.route("/api/search", methods=["POST"])
def search():
    try:
        t0 = time.time()
        data = request.json
        if not data or "question" not in data:
            return error_response("ValidationError", "Field 'question' wajib diisi", code=400)
        user_q = data["question"].strip()
        wa = data.get("wa_number", "unknown")

        # --- normalize
        question = normalize_text(clean_location_terms(user_q))
        category = detect_category(question)
        cat_id = category["id"] if category else None

        # --- embedding
        t_emb = time.time()
        qvec = model.encode("query: " + question).tolist()
        emb_time = time.time() - t_emb

        # --- qdrant
        t_qd = time.time()
        filt = None
        if cat_id:
            filt = models.Filter(must=[models.FieldCondition(key="category_id", match=models.MatchValue(value=cat_id))])
        dense_hits = qdrant.search("knowledge_bank", query_vector=qvec, limit=5, query_filter=filt)
        qd_time = time.time() - t_qd

        if not dense_hits:
            return jsonify({"status":"low_confidence","message":"Tidak ada hasil ditemukan"}),200

        # --- ambil kandidat teratas
        top_hit = dense_hits[0]
        top_q = top_hit.payload["question"]
        dense_score = float(top_hit.score)

        # --- relevance check pakai AI
        t_ai = time.time()
        relevance = check_relevance_ai(user_q, top_q)
        ai_time = time.time() - t_ai

        # --- jika tidak relevan, reformulasi dan retry sekali
        if not relevance.get("relevant", True):
            new_q = relevance.get("reformulated_question", "")
            if new_q:
                logger.info(f"[RETRY] Reformulating: {new_q}")
                new_vec = model.encode("query: " + new_q).tolist()
                dense_hits = qdrant.search("knowledge_bank", query_vector=new_vec, limit=5, query_filter=filt)
                if dense_hits:
                    top_hit = dense_hits[0]
                    top_q = top_hit.payload["question"]
                    dense_score = float(top_hit.score)
                    question = new_q

        # --- build result
        result = []
        for h in dense_hits[:3]:
            overlap = keyword_overlap(question, h.payload["question"])
            result.append({
                "question": h.payload["question"],
                "answer_id": h.payload["answer_id"],
                "category_id": h.payload.get("category_id"),
                "dense_score": float(h.score),
                "overlap_score": float(overlap)
            })

        total_time = round(time.time()-t0,3)

        return jsonify({
            "status": "success",
            "data": {
                "similar_questions": result,
                "metadata": {
                    "wa_number": wa,
                    "original_question": user_q,
                    "final_question": question,
                    "dense_score_top": dense_score,
                    "ai_reason": relevance.get("reason", "-"),
                    "ai_reformulated": relevance.get("reformulated_question", "-"),
                    "category": category["name"] if category else "Global"
                }
            },
            "timing": {
                "ai_sec": round(ai_time,3),
                "embedding_sec": round(emb_time,3),
                "qdrant_sec": round(qd_time,3),
                "total_sec": total_time
            }
        })

    except Exception as e:
        logger.error(f"Error in search: {e}", exc_info=True)
        return error_response("ServerError","Kesalahan internal",detail=str(e))

# ==========================================================
# 🔹 ERROR RESPONSE
# ==========================================================
def error_response(t, msg, detail=None, code=500):
    payload = {"status":"error","error":{"type":t,"message":msg}}
    if detail: payload["error"]["detail"]=detail
    return jsonify(payload),code

# API SYNC
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
                        "question": item["question"],
                        "answer_id": item["answer_id"],
                        "category_id": item.get("category_id")
                    }
                })
           
            qdrant.upsert(collection_name="knowledge_bank", points=points)

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
            return jsonify({"status": "success", "message": "Data berhasil diperbarui"})

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
