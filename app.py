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
# 🔹 STOPWORDS DAN KATEGORI
# ==========================================================
STOPWORDS = {
    "apa","bagaimana","cara","untuk","dan","atau","yang","dengan","di","ke","dari",
    "buat","mengurus","membuat","mendaftar","dimana","kapan","berapa","adalah","itu",
    "ini","saya","kamu","siapa","kepala","kota","medan"
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

# ==========================================================
# 🔹 UTILITIES
# ==========================================================
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
# 🔹 AI FILTER (Pre-Processing)
# ==========================================================
def ai_filter_pre(question):
    """AI menilai apakah pertanyaan relevan dengan konteks Pemko Medan"""
    try:
        url = "https://dekallm.cloudeka.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6FaPtqd1W5aj0z_-AbsKBA",
            "Content-Type": "application/json"
        }
        system_prompt = """
Anda adalah filter AI untuk pertanyaan seputar Pemerintah Kota Medan.
Putuskan apakah pertanyaan pengguna relevan dengan konteks pemerintahan, dinas, layanan publik, atau fasilitas di Kota Medan.

Kriteria valid (true):
- Menyebut dinas, kepala dinas, pejabat, struktur, visi, misi, tugas, fungsi, layanan publik.
- Berkaitan dengan administrasi, perizinan, pendidikan, kesehatan, bantuan sosial, pajak, fasilitas umum di Kota Medan.

Kriteria tidak valid (false):
- Menyebut wilayah di luar Kota Medan.
- Tidak ada hubungannya dengan pemerintahan, layanan publik, atau terlalu pendek.

Balas hanya JSON:
{
 "valid": true/false,
 "reason": "...",
 "suggestion": "...",
 "clean_question": "..."
}
"""
        payload = {
            "model": "meta/llama-4-maverick-instruct",
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": question.strip()}
            ],
            "temperature": 0.0,
            "top_p": 0.6
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        data = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        if not match:
            return {"valid": True, "clean_question": question}
        return json.loads(match.group(0))
    except Exception as e:
        logger.error(f"[AI-Filter] error: {e}")
        return {"valid": True, "clean_question": question}

# ==========================================================
# 🔹 AI POST CHECK (Relevance)
# ==========================================================
def ensure_interrogative_form(text):
    """Pastikan hasil reformulasi berbentuk pertanyaan"""
    if not text:
        return text
    lower = text.lower().strip()
    if lower.startswith(("bagaimana","apa","dimana","siapa","kapan")):
        return text
    if any(w in lower for w in ["prosedur","langkah","cara"]):
        return "Bagaimana " + text.strip().rstrip("?") + "?"
    return "Bagaimana cara " + text.strip().rstrip("?") + "?"

def ai_check_relevance(user_q, rag_q):
    try:
        url = "https://dekallm.cloudeka.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6FaPtqd1W5aj0z_-AbsKBA",
            "Content-Type": "application/json"
        }
        system_prompt = """
Anda bertugas memeriksa apakah hasil pencarian RAG sesuai dengan maksud pertanyaan pengguna.
Balas hanya JSON:
{
 "relevant": true/false,
 "reason": "...",
 "reformulated_question": "..."
}
- Relevan jika hasil menjawab inti pertanyaan user.
- Jika tidak sesuai konteks, bantu ubah pertanyaan agar lebih natural.
- Reformulasi harus berbentuk pertanyaan tanya yang wajar.
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
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            parsed["reformulated_question"] = ensure_interrogative_form(parsed.get("reformulated_question",""))
            return parsed
        return {"relevant": True, "reason": "-", "reformulated_question": ""}
    except Exception as e:
        logger.error(f"[AI-Post] error: {e}")
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
            return jsonify({"error":"Field 'question' wajib diisi"}),400

        user_q = data["question"].strip()
        wa = data.get("wa_number","unknown")

        # ========== PRE FILTER AI ==========
        t_pre = time.time()
        pre = ai_filter_pre(user_q)
        t_pre_time = time.time()-t_pre

        if not pre.get("valid",True):
            return jsonify({
                "status":"low_confidence",
                "message": pre.get("reason","Pertanyaan tidak relevan"),
                "ai_debug": pre,
                "timing": {"ai_domain_sec":round(t_pre_time,3)}
            }),200

        question = normalize_text(clean_location_terms(pre.get("clean_question",user_q)))
        category = detect_category(question)
        cat_id = category["id"] if category else None

        # ========== EMBEDDING ==========
        t_emb = time.time()
        qvec = model.encode("query: "+question).tolist()
        emb_time = time.time()-t_emb

        # ========== QDRANT ==========
        t_qd = time.time()
        filt = None
        if cat_id:
            filt = models.Filter(must=[models.FieldCondition(key="category_id",match=models.MatchValue(value=cat_id))])
        dense_hits = qdrant.search("knowledge_bank", query_vector=qvec, limit=5, query_filter=filt)
        qd_time = time.time()-t_qd

        if not dense_hits:
            return jsonify({"status":"low_confidence","message":"Tidak ada hasil ditemukan"}),200

        # ========== RELEVANCE AI POST CHECK ==========
        t_post = time.time()
        relevance = ai_check_relevance(user_q, dense_hits[0].payload["question"])
        t_post_time = time.time()-t_post

        # Jika tidak relevan → reformulasi dan retry
        if not relevance.get("relevant",True):
            reformulated = relevance.get("reformulated_question","")
            if reformulated:
                logger.info(f"[RETRY] Reformulating: {reformulated}")
                qvec = model.encode("query: "+reformulated).tolist()
                dense_hits = qdrant.search("knowledge_bank", query_vector=qvec, limit=5, query_filter=filt)
                question = reformulated

        # ========== DENSE + OVERLAP LOGIC ==========
        results, rejected = [], []
        for h in dense_hits[:5]:
            dense = float(h.score)
            overlap = keyword_overlap(question, h.payload["question"])
            note, accepted = None, False
            if dense >= 0.90:
                accepted, note = True, "auto_accepted_by_dense"
            elif 0.80 <= dense < 0.90 and overlap > 0.2:
                accepted, note = True, "accepted_by_overlap"

            item = {
                "question": h.payload["question"],
                "answer_id": h.payload["answer_id"],
                "category_id": h.payload.get("category_id"),
                "dense_score": dense,
                "overlap_score": overlap,
                "note": note or "-"
            }
            (results if accepted else rejected).append(item)

        total_time = time.time()-t0

        # ========== RESPONSE ==========
        if not results:
            return jsonify({
                "status":"low_confidence",
                "message":"Tidak ada hasil cukup relevan",
                "ai_debug": relevance,
                "debug_rejected": rejected,
                "timing":{
                    "ai_domain_sec":round(t_pre_time,3),
                    "ai_relevance_sec":round(t_post_time,3),
                    "embedding_sec":round(emb_time,3),
                    "qdrant_sec":round(qd_time,3),
                    "total_sec":round(total_time,3)
                }
            }),200

        return jsonify({
            "status":"success",
            "data":{
                "similar_questions": results,
                "metadata":{
                    "wa_number": wa,
                    "original_question": user_q,
                    "final_question": question,
                    "dense_score_top": results[0]["dense_score"],
                    "category": category["name"] if category else "Global",
                    "ai_reason": relevance.get("reason","-"),
                    "ai_reformulated": relevance.get("reformulated_question","-")
                }
            },
            "timing":{
                "ai_domain_sec":round(t_pre_time,3),
                "ai_relevance_sec":round(t_post_time,3),
                "embedding_sec":round(emb_time,3),
                "qdrant_sec":round(qd_time,3),
                "total_sec":round(total_time,3)
            }
        })

    except Exception as e:
        logger.error(f"Error in search: {e}", exc_info=True)
        return jsonify({"error":"Kesalahan internal","detail":str(e)}),500


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
