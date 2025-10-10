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


model = SentenceTransformer("/home/kominfo/models/e5-small-local")
qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STOPWORDS = {
    "apa", "bagaimana", "cara", "untuk", "dan", "atau", "yang", "dengan",
    "di", "ke", "dari", "buat", "mengurus", "membuat", "mendaftar",
    "dimana", "kapan", "berapa", "adalah", "itu", "ini", "saya", "kamu",
    "siapa", "kepala", "kota", "medan"
}


CATEGORY_KEYWORDS = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": [  # Kependudukan
        "ktp", "kartu tanda penduduk", "kependudukan", "kk", "kartu keluarga", "akta", "kelahiran", "kematian", "domisili"
    ],
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": [  # Kesehatan
        "bpjs", "kesehatan", "rsud", "puskesmas", "klinik", "imunisasi", "vaksin", "stunting", "obat", "berobat"
    ],
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": [  # Pendidikan
        "sekolah", "guru", "siswa", "ppdb", "beasiswa", "pendidikan", "kuliah", "universitas", "murid", "kip"
    ],
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": [  # Layanan Masyarakat
        "pengaduan", "izin", "permohonan", "pelayanan", "bantuan", "masyarakat", "surat",
    ],
    "0196f6b9-ba96-70f1-a930-3b89e763170f": [  # Struktur Organisasi
        "kepala dinas", "kadis", "sekretaris", "jabatan", "struktur", "pimpinan"
    ],
    "01970829-1054-72b2-bb31-16a34edd84fc": [  # Peraturan
        "aturan", "peraturan", "perwali", "perda", "pergub", "perpres", "permen", "hukum"
    ],
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": [  # Lokasi Fasilitas Pemerintahan
        "lokasi", "alamat", "kantor", "posisi"
    ],
     "001970853-dd2e-716e-b90c-c4f79270f700": [  # Profil
        "tugas", "tupoksi", "peran", "fungsi", "profil", "visi", "misi"
    ]
    
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

def detect_category(question: str):
    q_lower = question.lower()
    for cat_id, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return {"id": cat_id, "name": CATEGORY_NAMES.get(cat_id, "Unknown")}
    return None

def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_location_terms(text: str) -> str:
    """Hilangkan frasa seperti 'di medan' atau 'di kota medan'"""
    text = re.sub(r"\bdi\s+kota\s+medan\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdi\s+medan\b", "", text, flags=re.IGNORECASE)
    return text.strip()

def tokenize_and_filter(text: str):
    return [w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) > 2]

def keyword_overlap(query: str, candidate: str) -> float:
    q_tokens = set(tokenize_and_filter(query))
    c_tokens = set(tokenize_and_filter(candidate))
    if not q_tokens or not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)

def preprocess_question_with_ai(question: str):
    try:
        url = "https://dekallm.cloudeka.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6FaPtqd1W5aj0z_-AbsKBA",
            "Content-Type": "application/json"
        }

        system_prompt = """
        Anda adalah filter AI untuk pertanyaan seputar layanan publik dan fasilitas Pemerintah Kota Medan.
        Tugas Anda:
        1. Nilai apakah pertanyaan relevan dengan topik pemerintahan, pelayanan publik, atau fasilitas di Kota Medan.
        2. Terima pertanyaan yang berkaitan dengan:
        - pengurusan dokumen (KTP, KK, akta, izin, NIB, UMKM, beasiswa, BPJS, dll.)
        - pertanyaan tentang dinas, kepala dinas, atau struktur organisasi Pemko Medan.
        3. Jika pertanyaan menyebut singkatan dinas (contoh: Disnaker, Dinkes, Dishub, Disdik, Disdukcapil, Kominfo, DLH, Satpol PP, BPBD, Bappeda), ubah ke bentuk lengkap HANYA bila konteksnya jelas terkait "dinas" atau "kepala dinas".
        Jangan ubah atau menebak jika konteksnya tidak berkaitan dengan pemerintahan.
        4. Tolak jika pertanyaan terlalu pendek (<3 kata) atau menyebut daerah di luar Kota Medan.
        5. Jawab HANYA dalam format JSON valid:
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
            "temperature":  0.0,
            "top_p": 0.6
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        ai_reply = data["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", ai_reply, re.DOTALL)

        if not match:
            return {"valid": True, "clean_question": question}

        parsed = json.loads(match.group(0))
        if parsed.get("valid") is False:
            logger.info(f"[AI-FILTER] ❌ Ditolak: {parsed.get('reason')}")
            return {"valid": False, "reason": parsed.get("reason"), "suggestion": parsed.get("suggestion")}
        return {"valid": True, "clean_question": normalize_text(parsed.get("clean_question", question))}

    except Exception as e:
        logger.error(f"[AI-FILTER] Gagal koneksi API: {str(e)}", exc_info=True)
        return {"valid": True, "clean_question": normalize_text(question)}


@app.route("/api/search", methods=["POST"])
def search():
    try:
        t0 = time.time()  # ⏱️ total start
        
        data = request.json
        if not data or "question" not in data:
            return error_response("ValidationError", "Field 'question' wajib diisi", code=400)

        raw_question = data["question"].strip()
        wa_number = data.get("wa_number", "unknown")

        # ========== ⏱️ AI Filter timing ==========
        t_ai_start = time.time()
        ai_filter = preprocess_question_with_ai(raw_question)
        ai_time = time.time() - t_ai_start

        if not ai_filter.get("valid", True):
            return jsonify({
                "status": "low_confidence",
                "message": ai_filter.get("reason", "Pertanyaan tidak valid"),
                "suggestion": ai_filter.get("suggestion", "Silakan ajukan pertanyaan yang lebih spesifik."),
                "timing": {
                    "ai_filter_sec": round(ai_time, 3),
                    "embedding_sec": 0.0,
                    "qdrant_sec": 0.0,
                    "total_sec": round(time.time() - t0, 3)
                }
            }), 200

        question = normalize_text(ai_filter.get("clean_question", raw_question))
        question = clean_location_terms(question)

        # ========== ⏱️ Deteksi kategori ==========
        category_data = detect_category(question)
        category_id = category_data["id"] if category_data else None

        # ========== ⏱️ Embedding timing ==========
        t_embed = time.time()
        query_vector = model.encode("query: " + question).tolist()
        embedding_time = time.time() - t_embed

        # ========== ⏱️ Qdrant search timing ==========
        t_qdrant = time.time()
        query_filter = None
        if category_id:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="category_id",
                    match=models.MatchValue(value=category_id)
                )]
            )

        dense_hits = qdrant.search(
            collection_name="knowledge_bank",
            query_vector=query_vector,
            limit=5,
            query_filter=query_filter
        )

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
        qdrant_time = time.time() - t_qdrant

        # === combine results (tanpa ubah logika lama) ===
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

        results, rejected = [], []
        for item in sorted_hits:
            h = item["hit"]
            dense_score = getattr(h, "score", 0.0)
            overlap = keyword_overlap(question, h.payload["question"])

            accepted = False
            note = None

            if dense_score >= 0.90:
                accepted = True
                note = "auto_accepted_by_dense"
            elif 0.80 <= dense_score < 0.90 and overlap > 0.2:
                accepted = True
                note = "accepted_by_overlap"

            if accepted:
                results.append({
                    "question": h.payload["question"],
                    "answer_id": h.payload["answer_id"],
                    "category_id": h.payload.get("category_id"),
                    "dense_score": float(dense_score),
                    "overlap_score": float(overlap),
                    "note": note
                })
            else:
                rejected.append({
                    "question": h.payload["question"],
                    "dense_score": float(dense_score),
                    "overlap_score": float(overlap)
                })

        total_time = time.time() - t0  # total waktu request

        # ========== RESPONSE ==========
        if not results:
            return jsonify({
                "status": "low_confidence",
                "message": "Tidak ada hasil cukup relevan",
                "debug_rejected": rejected,
                "timing": {
                    "ai_filter_sec": round(ai_time, 3),
                    "embedding_sec": round(embedding_time, 3),
                    "qdrant_sec": round(qdrant_time, 3),
                    "total_sec": round(total_time, 3)
                }
            }), 200

        return jsonify({
            "status": "success",
            "data": {
                "similar_questions": results,
                "metadata": {
                    "total_found": len(results),
                    "wa_number": wa_number,
                    "original_question": raw_question,
                    "normalized_question": question,
                    "ai_clean_question": ai_filter.get("clean_question", "(tidak ada perubahan)"),
                    "detected_category": category_data["name"] if category_data else "Global"
                }
            },
            "timing": {
                "ai_filter_sec": round(ai_time, 3),
                "embedding_sec": round(embedding_time, 3),
                "qdrant_sec": round(qdrant_time, 3),
                "total_sec": round(total_time, 3)
            }
        })

    except Exception as e:
        logger.error(f"Error in search: {str(e)}", exc_info=True)
        return error_response("ServerError", "Kesalahan internal", detail=str(e))

def error_response(error_type, message, detail=None, code=500):
    payload = {"status": "error", "error": {"type": error_type, "message": message}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code


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
