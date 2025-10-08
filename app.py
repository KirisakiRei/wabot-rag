from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import CONFIG
import logging
import requests
import json
import re

app = Flask(__name__)

# ==========================================================
# 🔹 LOAD MODEL DAN CLIENT
# ==========================================================
model = SentenceTransformer("/home/kominfo/models/e5-small-local")
qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])

# ==========================================================
# 🔹 LOGGING
# ==========================================================
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================================
# 🔹 STOPWORDS & NORMALIZER
# ==========================================================
STOPWORDS = {
    "apa", "bagaimana", "cara", "untuk", "dan", "atau", "yang", "dengan",
    "di", "ke", "dari", "buat", "mengurus", "membuat", "mendaftar",
    "dimana", "kapan", "berapa", "adalah", "itu", "ini", "saya", "kamu",
    "siapa", "kepala"
}

def normalize_text(text: str):
    """Hilangkan tanda baca & lowercase"""
    return re.sub(r"[^\w\s]", "", text).lower().strip()

def tokenize_and_filter(text: str):
    """Tokenisasi + hapus stopword"""
    return [w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) > 2]

def keyword_overlap(query: str, candidate: str) -> float:
    q_tokens = set(tokenize_and_filter(query))
    c_tokens = set(tokenize_and_filter(candidate))
    if not q_tokens or not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)


# ==========================================================
# 🔹 ERROR HANDLER
# ==========================================================
def error_response(error_type="ServerError", message="Terjadi kesalahan", code=500, detail=None):
    payload = {
        "status": "error",
        "error_type": error_type,
        "message": message
    }
    if detail:
        payload["detail"] = str(detail)
    return jsonify(payload), code


# ==========================================================
# 🔹 CATEGORY AUTO-DETECTION
# ==========================================================
CATEGORY_MAP = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": {  # Kependudukan
        "keywords": [
                     "ktp",
                     "nik",
                     "kk",
                     "kartu keluarga",
                     "e-ktp",
                     "domisili",
                     "kelahiran",
                     "kematian"
                    ],
        "name": "Kependudukan"
    },
    "0196f6b9-ba96-70f1-a930-3b89e763170f": {  # Struktur Organisasi
        "keywords": [
                     "kepala dinas",
                     "kadis",
                     "sekretaris",
                     "jabatan",
                     "struktur",
                     "pimpinan",
                     "pegawai",
                     "organisasi",
                     "staff",
                     "karyawan"
                    ],
        "name": "Struktur Organisasi"
    },
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": {  # Kesehatan
        "keywords": [
                     "puskesmas",
                     "rsud",
                     "klinik",
                     "vaksin",
                     "bpjs",
                     "obat",
                     "berobat",
                     "posyandu",
                     "stunting",
                     "imunisasi",
                     "hamil"
                    ],
        "name": "Kesehatan"
    },
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": {  # Pendidikan
        "keywords": [
                     "sekolah",
                     "guru",
                     "siswa",
                     "universitas",
                     "kuliah",
                     "beasiswa",
                     "mahasiswa",
                     "akademik",
                     "kurikulum",
                     "ppdb",
                     "sd",
                     "smp",
                     "smbp",
                     "sekolah negeri",
                     "prestasi",
                     "zonasi",
                     "afirmasi",
                     "kip"
                    ],
        "name": "Pendidikan"
    },
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": {  # Layanan Masyarakat
        "keywords": [
                     "pengaduan",
                     "izin",
                     "pelayanan",
                     "surat",
                     "permohonan",
                     "bantuan",
                     "masyarakat"
                    ],
        "name": "Layanan Masyarakat"
    },
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": {  # Lokasi Fasilitas 
        "keywords": [
                     "lokasi",
                     "alamat",
                     "posisi",
                     "kantor"
                    ],
        "name": "Lokasi Fasilitas Pemerintahan Kota Medan"
    },
    "01970829-1054-72b2-bb31-16a34edd84fc": {  # Peraturan
        "keywords": [
                     "aturan",
                     "perda",
                     "pergub",
                     "perwali",
                     "permen",
                     "perpres",
                     "peraturan",
                     "hukum"
                    ],
        "name": "Peraturan"
    },
    "001970853-dd2e-716e-b90c-c4f79270f700": {  # Profil
        "keywords": [
                     "visi",
                     "misi", 
                     "tupoksi", 
                     "profil", 
                     "tugas", 
                     "fungsi"
                    ],
        "name": "Profil"
    }
    
}

def detect_category(question: str):
    text = normalize_text(question)
    for cat_id, cat_data in CATEGORY_MAP.items():
        for kw in cat_data["keywords"]:
            if kw in text:
                return {"id": cat_id, "name": cat_data["name"]}
    return None


# ==========================================================
# 🔹 AI PREPROCESSING FILTER
# ==========================================================
def preprocess_question_with_ai(question: str):
    try:
        url = "https://dekallm.cloudeka.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6FaPtqd1W5aj0z_-AbsKBA",
            "Content-Type": "application/json"
        }

        system_prompt = """
Anda adalah model AI yang memfilter pertanyaan sebelum diteruskan ke sistem RAG FAQ Pemerintah Kota Medan.
Tugas Anda adalah menilai apakah pertanyaan layak diproses oleh sistem FAQ Pemko Medan.
Ikuti aturan berikut secara ketat:
1. Pertanyaan harus relevan dengan layanan publik, fasilitas pemerintahan, atau informasi seputar Kota Medan.
2. Tidak boleh terlalu pendek (contoh: 'KTP', 'izin', 'Medan').
3. Tidak boleh menyebut daerah di luar Kota Medan (Jakarta, Bandung, dll).
4. Jawaban harus berupa JSON:
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
            "temperature": 0.3,
            "top_p": 0.7
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


# ==========================================================
# 🔹 API SEARCH
# ==========================================================
@app.route("/api/search", methods=["POST"])
def search():
    try:
        data = request.json
        if not data or "question" not in data:
            return error_response("ValidationError", "Field 'question' wajib diisi", code=400)

        original_question = data["question"].strip()
        wa_number = data.get("wa_number", "unknown")

        # AI filter
        ai_filter = preprocess_question_with_ai(original_question)
        if not ai_filter.get("valid", True):
            return jsonify({
                "status": "low_confidence",
                "message": ai_filter.get("reason", "Pertanyaan tidak valid"),
                "suggestion": ai_filter.get("suggestion", "Silakan ajukan pertanyaan yang lebih spesifik.")
            }), 200

        question = ai_filter.get("clean_question", original_question)

        # Otomatis deteksi kategori
        category_data = detect_category(question)
        if category_data:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="category_id",
                    match=models.MatchValue(value=category_data["id"])
                )]
            )
            logger.info(f"[CATEGORY AUTO] 🏷 {category_data['name']} → dari pertanyaan '{original_question}'")
        else:
            query_filter = None
            logger.info(f"[CATEGORY AUTO] Tidak ditemukan kategori → pencarian global")

        # Dense Search
        query_vector = model.encode("query: " + question).tolist()
        dense_hits = qdrant.search(
            collection_name="knowledge_bank",
            query_vector=query_vector,
            limit=5,
            query_filter=query_filter
        )

        # Keyword fallback
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

        # Gabung hasil
        combined = {}
        for i, h in enumerate(dense_hits):
            combined[str(h.id)] = {"hit": h, "score": 1 / (i + 1)}
        for i, h in enumerate(keyword_hits):
            combined.setdefault(str(h.id), {"hit": h, "score": 0})
            combined[str(h.id)]["score"] += 1 / (i + 1)

        sorted_hits = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:3]
        if not sorted_hits:
            return jsonify({"status": "not_found", "message": "Tidak ada hasil ditemukan"}), 404

        results = []
        for item in sorted_hits:
            h = item["hit"]
            dense_score = getattr(h, "score", 0.0)
            overlap = keyword_overlap(question, h.payload["question"])
            if dense_score >= 0.80 or overlap > 0.2:
                results.append({
                    "question": h.payload["question"],
                    "answer_id": h.payload["answer_id"],
                    "category_id": h.payload.get("category_id"),
                    "dense_score": float(dense_score),
                    "overlap_score": float(overlap)
                })

        return jsonify({
            "status": "success",
            "data": {
                "similar_questions": results,
                "metadata": {
                    "total_found": len(results),
                    "wa_number": wa_number,
                    "original_question": original_question,
                    "normalized_question": question,
                    "detected_category": category_data["name"] if category_data else "Global"
                }
            }
        })

    except Exception as e:
        logger.error(f"Error in search API: {str(e)}", exc_info=True)
        return error_response("ServerError", "Kesalahan internal pada Server", detail=str(e))


# ==========================================================
# 🔹 API SYNC
# ==========================================================
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
                points.append({
                    "id": str(item["id"]),
                    "vector": vector,
                    "payload": {
                        "mysql_id": str(item["id"]),
                        "question": item["question"],
                        "answer_id": item["answer_id"],
                        "category_id": item.get("category_id")
                    }
                })
            qdrant.upsert(collection_name="knowledge_bank", points=points)
            return jsonify({
                "status": "success",
                "message": f"Sinkronisasi {len(points)} data berhasil"
            })

        elif action == "add":
            point_id = str(content["id"])
            vector = model.encode("passage: " + content["question"]).tolist()
            qdrant.upsert(collection_name="knowledge_bank", points=[{
                "id": point_id,
                "vector": vector,
                "payload": {
                    "mysql_id": point_id,
                    "question": content["question"],
                    "answer_id": content["answer_id"],
                    "category_id": content.get("category_id")
                }
            }])
            return jsonify({"status": "success", "message": "Data berhasil ditambahkan"})

        elif action == "update":
            point_id = str(content["id"])
            vector = model.encode("passage: " + content["question"]).tolist()
            qdrant.upsert(collection_name="knowledge_bank", points=[{
                "id": point_id,
                "vector": vector,
                "payload": {
                    "mysql_id": point_id,
                    "question": content["question"],
                    "answer_id": content["answer_id"],
                    "category_id": content.get("category_id")
                }
            }])
            return jsonify({"status": "success", "message": "Data berhasil diperbarui"})

        elif action == "delete":
            point_id = str(content["id"])
            qdrant.delete(collection_name="knowledge_bank",
                points_selector=models.PointIdsList(points=[point_id]), wait=True)
            return jsonify({"status": "success", "message": "Data berhasil dihapus"})

        else:
            return error_response("ValidationError", f"Action '{action}' tidak dikenali", code=400)

    except Exception as e:
        logger.error(f"Error in sync API: {str(e)}", exc_info=True)
        return error_response("ServerError", "Kesalahan internal saat sinkronasi", detail=str(e))


# ==========================================================
# 🔹 RUN
# ==========================================================
if __name__ == "__main__":
    app.run(host=CONFIG["api"]["host"], port=CONFIG["api"]["port"], debug=True)
