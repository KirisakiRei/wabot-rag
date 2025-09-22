# 📖 Knowledge Bank API (Flask + Qdrant)

Sistem ini menyediakan API untuk **chatbot layanan publik** menggunakan pendekatan **RAG (Retrieval-Augmented Generation)**.  
Data pertanyaan dan jawaban disimpan di **Qdrant Vector Database**, lalu dapat dicari dengan embedding model (`MiniLM-L6-v2`).

---

## 🚀 Cara Akses API

- Base URL (contoh lokal): `http://localhost:5000`
- Base URL (contoh server): `http://<IP_SERVER>:5000`

## 📡 API Endpoints
1. **/api/search** → untuk WA Bot (mencari jawaban).
2. **/api/sync** → untuk WA Manajemen (sinkronisasi CRUD data).


### 1. 🔍 Cari Jawaban - Untuk WA BOT

**Endpoint**
```
POST /api/search
```

**Request Body JSON**
```json
{
  "question": "cara daftar KTP",
  "wa_number": "628xxxxxxxxxx",
  "category": "Kependudukan" //→ filter kategori
}
```

**Response (contoh berhasil)**
```json
{
  "status": "success",
  "data": {
    "similar_questions": [
      {
        "id": 15,
        "question": "Bagaimana cara membuat KTP baru?",
        "answer": "Datang ke Dukcapil dengan membawa KK dan dokumen pendukung.",
        "category": "Kependudukan",
        "similarity_score": 0.8123
      }
    ],
    "metadata": {
      "total_found": 1,
      "wa_number": "628xxxxxxxxxx",
      "original_question": "cara daftar KTP",
      "category_used": "Kependudukan"
    }
  }
}
```

**Kemungkinan Status**
- `success` → jawaban ditemukan
- `not_found` → tidak ada data di kategori
- `low_confidence` → ada data, tapi similarity < 0.65

---

### 2. 🔄 Sinkronisasi Data - Untuk WA MANAJEMEN

**Endpoint**
```
POST /api/sync
```

**Mapping Field**
Untuk memastikan konsistensi data:
- **Dari DB / Manajemen → API**  
  - `pertanyaan` → `question`  
  - `jawaban` → `answer`  
  - `kategori` → `category`  

- **Dari API → Qdrant** (payload):  
  ```json
  {
    "mysql_id": 1,
    "question": "Apa itu KIS?",
    "answer": "Kartu identitas peserta JKN-KIS.",
    "category": "Kesehatan"
  }
  ```

**Action yang tersedia:**
- `bulk_sync` → sinkronisasi semua data sekaligus
- `add` → tambah data baru
- `update` → update data lama
- `delete` → hapus data

#### 📌 Contoh `bulk_sync`
```json
{
  "action": "bulk_sync",
  "content": [
    {
      "id": 1,
      "question": "Apa itu Kartu Indonesia Sehat?",
      "answer": "KIS adalah kartu identitas peserta JKN.",
      "category": "Kesehatan"
    },
    {
      "id": 2,
      "question": "Bagaimana cara membuat KTP baru?",
      "answer": "Datang ke Dukcapil dengan membawa KK dan dokumen pendukung.",
      "category": "Kependudukan"
    }
  ]
}
```

#### 📌 Contoh `add`
```json
{
  "action": "add",
  "content": {
    "id": 3,
    "question": "Apa itu PPDB?",
    "answer": "PPDB adalah Pendaftaran Peserta Didik Baru.",
    "category": "Pendidikan"
  }
}
```

#### 📌 Contoh `update`
```json
{
  "action": "update",
  "content": {
    "id": 2,
    "question": "Bagaimana cara membuat KTP?",
    "answer": "Datang ke Dukcapil membawa KK dan akta lahir.",
    "category": "Kependudukan"
  }
}
```

#### 📌 Contoh `delete`
```json
{
  "action": "delete",
  "content": {
    "id": 2
  }
}
```

---

## 📊 Threshold Similarity

- Default: `0.65`  
- Jika hasil pencarian memiliki **similarity < 0.65**, maka API akan merespon:  
  ```json
  {
    "status": "low_confidence",
    "message": "Tidak ada jawaban dengan similarity ≥ 0.65"
  }
  ```


1. **WA Manajemen**  
   - Bertugas menambah, mengupdate, dan sinkronisasi data ke API.  
   - Pastikan field dikirim dengan nama **`question`, `answer`, `category`** meskipun di DB lokal namanya berbeda.

2. **WA Bot**  
   - Hanya perlu memanggil `/api/search` dengan pertanyaan user.  
   - Jika similarity < 0.65, sistem akan memberi respon *“Tidak ada jawaban cukup relevan”*.

3. **Kategori**  
   - Berguna untuk filter, pencarian akan difokuskan hanya pada kategori tersebut.  


## 📌 Contoh Tes via curl

### Search
```bash
curl -X POST http://localhost:5000/api/search -H "Content-Type: application/json" -d '{"question": "apa itu KIS?", "category": "Kesehatan"}'
```

### Bulk Sync
```bash
curl -X POST http://localhost:5000/api/sync -H "Content-Type: application/json" -d '{"action": "bulk_sync", "content": [{"id":1,"question":"Apa itu KIS?","answer":"Kartu Indonesia Sehat","category":"Kesehatan"}]}'
```

## ⚙️ Diagram Alur

```
User WA → WA Bot → /api/search → Flask API → Qdrant
WA Manajemen → /api/sync → Flask API → Qdrant
```
