from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import CONFIG
import logging, requests, json, re, time

app = Flask(__name__)

# ==========================================================
# 🔹 LOAD MODEL DAN CLIENT
# ==========================================================
model = SentenceTransformer("/home/kominfo/models/e5-small-local")
qdrant = QdrantClient(
    host=CONFIG["qdrant"]["host"],
    port=CONFIG["qdrant"]["port"]
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# 🔹 STOPWORDS, SYNONYMS, DAN KATEGORI
# ==========================================================
STOPWORDS = {
    "apa", "bagaimana", "cara", "untuk", "dan", "atau", "yang",
    "dengan", "ke", "dari", "buat", "membuat", "mengurus",
    "mendaftar", "mencetak", "dimana", "kapan", "berapa",
    "adalah", "itu", "ini", "saya", "kamu"
}

SYNONYMS = {
    "ktp": ["kartu tanda penduduk"],
    "kk": ["kartu keluarga"],
    "kadis": ["kepala dinas"],
    "kominfo": ["dinas komunikasi dan informatika", "diskominfo"],
    "dukcapil": ["dinas kependudukan dan catatan sipil", "disdukcapil"],
    "dishub": ["dinas perhubungan"],
    "dinkes": ["dinas kesehatan"],
    "disnaker": ["dinas ketenagakerjaan"],
    "sktm": ["surat keterangan tidak mampu"],
    "siup": ["surat izin usaha perdagangan"],
    "umkm": ["usaha mikro kecil menengah"]
}

CATEGORY_KEYWORDS = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": [
        "ktp", "kk", "kartu keluarga", "kartu tanda penduduk",
        "akta", "kelahiran", "kematian", "domisili", "SKTM", "NIK"
    ],
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": [
        "bpjs", "rsud", "puskesmas", "klinik", "vaksin",
        "pengobatan", "berobat", "posyandu", "stunting", "imunisasi"
    ],
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": [
        "sekolah", "PPDB", "SPMB", "guru", "siswa", "beasiswa",
        "pendidikan", "prestasi", "zonasi", "afirmasi"
    ],
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": [
        "pengaduan", "izin", "siup", "bantuan", "masyarakat", "usaha"
    ],
    "0196f6b9-ba96-70f1-a930-3b89e763170f": [
        "kepala dinas", "kadis", "sekretaris", "jabatan", "struktur organisasi"
    ],
    "01970829-1054-72b2-bb31-16a34edd84fc": [
        "aturan", "peraturan", "perwali", "perda", "perpres", "hukum"
    ],
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": [
        "lokasi", "alamat", "kantor", "posisi"
    ],
    "001970853-dd2e-716e-b90c-c4f79270f700": [
        "tugas", "fungsi", "tupoksi", "profil", "visi", "misi"
    ]
}

CATEGORY_NAMES = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": "Kependudukan",
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": "Kesehatan",
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": "Pendidikan",
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": "Layanan Masyarakat",
    "0196f6b9-ba96-70f1-a930-3b89e763170f": "Struktur Organisasi",
    "01970829-1054-72b2-bb31-16a34edd84fc": "Peraturan",
    "0196f6c0-1178-733a-acd8-b8cb62eefe98":
        "Lokasi Fasilitas Pemerintahan Kota Medan",
    "001970853-dd2e-716e-b90c-c4f79270f700": "Profil"
}

ALL_KEYWORDS = set(sum(CATEGORY_KEYWORDS.values(), []))

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


def expand_terms(text):
    words = text.lower().split()
    expanded = []
    for w in words:
        expanded.append(w)
        if w in SYNONYMS:
            expanded.extend(SYNONYMS[w])
    return " ".join(expanded)


def tokenize_and_filter(t):
    return [
        w.lower() for w in t.split()
        if w.lower() not in STOPWORDS and len(w) > 2
    ]


def keyword_overlap(a, b):
    a_exp = expand_terms(a)
    b_exp = expand_terms(b)
    A, B = set(tokenize_and_filter(a_exp)), set(tokenize_and_filter(b_exp))
    return len(A & B) / len(A | B) if A and B else 0.0


def shorten_question(text):
    if not text:
        return text
    text = re.sub(r"yang diperlukan untuk\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"yang dibutuhkan untuk\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"yang harus dilakukan\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"untuk\s+mengurus\s+", "mengurus ", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 15:
        text = " ".join(words[:15]) + "..."
    return text.strip()


# ==========================================================
# 🔹 HARD FILTER LOKAL
# ==========================================================
def hard_filter_local(question: str):
    q = question.lower()
    q_norm = re.sub(r"[^\w\s]", " ", q)
    q_norm = re.sub(r"\s+", " ", q_norm)

    NON_MEDAN = [
        "jakarta", "bandung", "surabaya", "yogyakarta", "semarang",
        "siantar", "pematangsiantar", "pematang siantar",
        "binjai", "tebing", "tebing tinggi", "aceh", "padang",
        "pekanbaru", "riau", "deliserdang", "deli serdang",
        "langkat", "tanjung morawa", "belawan", "labuhanbatu"
    ]

    OPINI_WORDS = [
        "rajin", "malas", "ganteng", "cantik", "baik", "buruk",
        "terkenal", "paling", "ter", "terbaik", "terburuk",
        "terjelek", "terbodoh", "terrajin"
    ]

    for city in NON_MEDAN:
        if re.search(rf"\b{re.escape(city)}\b", q_norm):
            return {
                "valid": False,
                "reason": f"Pertanyaan menyebut daerah di luar Medan ({city.title()})",
                "clean_question": question
            }

    if any(re.search(rf"\b{re.escape(w)}\b", q_norm) for w in OPINI_WORDS):
        return {
            "valid": False,
            "reason": "Pertanyaan bersifat opini/personal, bukan layanan publik",
            "clean_question": question
        }

    if len(q_norm.split()) <= 2:
        return {
            "valid": False,
            "reason": "Pertanyaan terlalu pendek atau tidak jelas",
            "clean_question": question
        }

    return {
        "valid": True,
        "reason": "Lolos hard filter",
        "clean_question": question
    }


# ==========================================================
# 🔹 AI FILTER (PRE)
# ==========================================================
def ai_filter_pre(question: str):
    hard_check = hard_filter_local(question)
    if not hard_check["valid"]:
        logger.info(f"[HARD FILTER] ❌ {hard_check['reason']}")
        return hard_check

    try:
        url = "https://dekallm.cloudeka.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6FaPtqd1W5aj0z_-AbsKBA",
            "Content-Type": "application/json"
        }

        system_prompt = """
Anda adalah AI filter untuk pertanyaan terkait Pemerintah Kota Medan.

Petunjuk:
1. Balas HANYA dalam format JSON berikut:
   {"valid": true/false, "reason": "<penjelasan>", "clean_question": "<pertanyaan yang sudah dibersihkan>"}

2. Mark valid jika dan hanya jika pertanyaan membahas:
   - Dinas/instansi di bawah Pemko Medan
   - Layanan publik di Medan (KTP, SIM, pajak daerah, fasilitas kesehatan, pendidikan, dll)
   - Izin usaha/lingkungan/keramaian yang dikeluarkan Pemko Medan
   - Fasilitas umum milik Pemko Medan (taman, jalan, RSUD, dll)
   - Kebijakan atau program Pemerintah Kota Medan

3. Mark tidak valid jika:
   - Membahas daerah di luar Kota Medan (Jakarta, Bandung, Surabaya, Siantar, Tebing Tinggi, Kisaran, Deli Serdang, dll)
   - Membahas figur publik non-pemerintah (selebriti, influencer, dll)
   - Membahas topik pribadi, gosip, atau hal yang tidak berkaitan pemerintahan
   - Pertanyaan tidak jelas, ambigu, atau tidak relevan dengan Pemko Medan

4. Bersihkan pertanyaan di clean_question:
   - Hilangkan emoji, tanda baca berlebihan, kata tidak relevan, atau typo
   - Pastikan tetap dalam Bahasa Indonesia

5. Jika valid, isi reason dengan "Pertanyaan relevan dengan Pemko Medan".
   Jika tidak valid, isi reason dengan penjelasan singkat alasan penolakan.

CONTOH OUTPUT:
{"valid": true, "reason": "Pertanyaan relevan dengan Pemko Medan", "clean_question": "Bagaimana cara mengurus KTP di Medan?"}
{"valid": false, "reason": "Topik membahas daerah lain (Jakarta)", "clean_question": "Bagaimana cara mengurus KTP di Jakarta?"}

JANGAN BERIKAN PENJELASAN DI LUAR JSON.
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

        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        content = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {
            "valid": True,
            "reason": "AI tidak mengembalikan JSON",
            "clean_question": question
        }

        logger.info(f"[AI FILTER] ✅ Valid: {parsed['valid']} | Reason: {parsed['reason']}")
        return parsed

    except Exception as e:
        logger.error(f"[AI-Filter] {e}")
        return {"valid": True, "reason": "Fallback error AI Filter", "clean_question": question}


# ==========================================================
# 🔹 AI RELEVANCE CHECK (POST)
# ==========================================================
def ai_check_relevance(user_q: str, rag_q: str):
    try:
        url = "https://dekallm.cloudeka.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6FaPtqd1W5aj0z_-AbsKBA",
            "Content-Type": "application/json"
        }

        system_prompt = """
Tugas Anda mengevaluasi apakah hasil pencarian RAG sesuai dengan maksud
pertanyaan pengguna.
Balas hanya JSON:
{"relevant": true/false, "reason": "...", "reformulated_question": "..."}

Kriteria:
✅ Relevan jika topik sama (layanan publik, fasilitas, dokumen, kebijakan).
❌ Tidak relevan jika membahas jabatan/instansi berbeda,
   kota lain, atau konteks umum vs spesifik.
Jika tidak relevan, ubah pertanyaan jadi versi singkat berbentuk tanya
maks. 12 kata.
"""

        user_prompt = f"User: {user_q}\nRAG Result: {rag_q}"

        payload = {
            "model": "meta/llama-4-maverick-instruct",
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            "temperature": 0.1,
            "top_p": 0.5
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        content = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)

        parsed = json.loads(match.group(0)) if match else {
            "relevant": True, "reason": "-", "reformulated_question": ""
        }

        reform = parsed.get("reformulated_question", "").strip()
        if len(reform.split()) > 12:
            parsed["reformulated_question"] = " ".join(reform.split()[:12]) + "..."

        logger.info(f"[AI RELEVANCE] ✅ Relevant: {parsed['relevant']} | Reason: {parsed['reason']}")
        return parsed

    except Exception as e:
        logger.error(f"[AI-Post] {e}")
        return {"relevant": True, "reason": "AI relevance check failed", "reformulated_question": ""}


# ==========================================================
# 🔹 MAIN SEARCH PIPELINE
# ==========================================================
@app.route("/api/search", methods=["POST"])
def search():
    try:
        t0 = time.time()
        data = request.json or {}
        user_q = data.get("question", "").strip()
        wa = data.get("wa_number", "unknown")

        if not user_q:
            return jsonify({
                "status": "error",
                "message": "Field 'question' wajib diisi"
            }), 400

        # ---------- PRE FILTER ----------
        t_pre = time.time()
        pre = ai_filter_pre(user_q)
        t_pre_time = time.time() - t_pre

        if not pre.get("valid", True):
            return jsonify({
                "status": "low_confidence",
                "message": pre.get("reason", "Pertanyaan tidak relevan"),
                "data": {"similar_questions": []},
                "timing": {"ai_domain_sec": round(t_pre_time, 3)}
            }), 200

        # ---------- EMBEDDING ----------
        question = normalize_text(clean_location_terms(pre.get("clean_question", user_q)))
        category = detect_category(question)
        cat_id = category["id"] if category else None

        t_emb = time.time()
        qvec = model.encode("query: " + question).tolist()
        emb_time = time.time() - t_emb

        # ---------- QDRANT ----------
        t_qd = time.time()
        filt = models.Filter(must=[
            models.FieldCondition(key="category_id", match=models.MatchValue(value=cat_id))
        ]) if cat_id else None

        dense_hits = qdrant.search("knowledge_bank", query_vector=qvec, limit=5, query_filter=filt)
        if len(dense_hits) < 3:
            dense_hits = qdrant.search("knowledge_bank", query_vector=qvec, limit=5)
        qd_time = time.time() - t_qd

        # ---------- AI RELEVANCE ----------
        t_post = time.time()
        relevance = {}
        if dense_hits:
            relevance = ai_check_relevance(user_q, dense_hits[0].payload["question"])
        t_post_time = time.time() - t_post

        # ---------- SCORING ----------
        results, rejected = [], []
        for h in dense_hits:
            dense = float(h.score)
            overlap = keyword_overlap(question, h.payload["question"])
            note, accepted = "-", False
            if dense >= 0.88:
                accepted, note = True, "auto_accepted_by_dense"
            elif 0.82 <= dense < 0.88 and overlap >= 0.25:
                accepted, note = True, "accepted_by_overlap"
            item = {
                "question": h.payload["question"],
                "dense_score": dense,
                "overlap_score": overlap,
                "note": note
            }
            (results if accepted else rejected).append(item)

        total_time = time.time() - t0

        # ---------- RESPONSE ----------
        return jsonify({
            "status": "success" if results else "low_confidence",
            "message": "Hasil ditemukan" if results else "Tidak ada hasil cukup relevan",
            "data": {
                "similar_questions": results if results else rejected,
                "metadata": {
                    "wa_number": wa,
                    "original_question": user_q,
                    "final_question": question,
                    "category": category["name"] if category else "Global",
                    "ai_reason": relevance.get("reason", "-"),
                    "ai_reformulated": relevance.get("reformulated_question", "-")
                }
            },
            "timing": {
                "ai_domain_sec": round(t_pre_time, 3),
                "ai_relevance_sec": round(t_post_time, 3),
                "embedding_sec": round(emb_time, 3),
                "qdrant_sec": round(qd_time, 3),
                "total_sec": round(total_time, 3)
            }
        }), 200

    except Exception as e:
        logger.error(f"Error in search: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Kesalahan internal",
            "detail": str(e)
        }), 500


# ==========================================================
# 🔹 RUN SERVER
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


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
