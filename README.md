# 📖 Knowledge Bank API (Flask + Qdrant)

Sistem ini menyediakan API untuk **chatbot layanan publik** menggunakan pendekatan **RAG (Retrieval-Augmented Generation)**.  
Data pertanyaan disimpan di **Qdrant Vector Database**, sedangkan jawaban diambil menggunakan **`answer_id`** dari database WA manajemen.  

---

## 🚀 Cara Akses API

- Base URL (lokal): `http://localhost:5000`  
- Base URL (server): `http://<IP_SERVER>:5000`  

---

## 📡 API Endpoints
1. **`/api/search`** → untuk WA Bot (mencari pertanyaan mirip).  
2. **`/api/sync`** → untuk WA Manajemen (sinkronisasi CRUD data).  

---

## 1. 🔍 Cari Pertanyaan – Untuk WA BOT

**Endpoint**
```
POST /api/search
```

**Request Body JSON**
```json
{
  "question": "cara daftar KTP",
  "wa_number": "628xxxxxxxxxx"
}
```

### ✅ Response Success
```json
{
  "status": "success",
  "data": {
    "similar_questions": [
      {
        "id": 15,
        "question": "Bagaimana cara membuat KTP baru?",
        "answer_id": 101,
        "similarity_score": 0.8123
      }
    ],
    "metadata": {
      "total_found": 1,
      "wa_number": "628123456789",
      "original_question": "cara daftar KTP"
    }
  }
}
```

### ❌ Response Error

**Not Found**
```json
{
  "status": "not_found",
  "message": "Tidak ada data ditemukan"
}
```

**Low Confidence**
```json
{
  "status": "low_confidence",
  "message": "Tidak ada hasil cukup relevan untuk pertanyaan Anda."
}
```

**Validation Error**
```json
{
  "status": "error",
  "error": {
    "type": "ValidationError",
    "message": "Field 'question' wajib diisi"
  }
}
```

**Server Error**
```json
{
  "status": "error",
  "error": {
    "type": "ServerError",
    "message": "Terjadi kesalahan internal pada server",
    "detail": "Connection refused"
  }
}
```

---

## 2. 🔄 Sinkronisasi Data – Untuk WA MANAJEMEN

**Endpoint**
```
POST /api/sync
```

**Mapping Field**
Untuk memastikan konsistensi data:
- **Dari DB/Manajemen → API**  
  - `pertanyaan` → `question`  
  - `jawaban_id` → `answer_id`  

- **Dari API → Qdrant (payload)**  
  ```json
  {
    "mysql_id": 1,
    "question": "Apa itu KIS?",
    "answer_id": 200
  }
  ```

---

### a) 📌 Bulk Sync

**Request**
```json
{
  "action": "bulk_sync",
  "content": [
    {
      "id": 1,
      "question": "Apa itu Kartu Indonesia Sehat?",
      "answer_id": 200
    },
    {
      "id": 2,
      "question": "Bagaimana cara membuat KTP baru?",
      "answer_id": 201
    }
  ]
}
```

**Response Success**
```json
{
  "status": "success",
  "message": "Berhasil sinkronisasi 2 data Knowledge Bank",
  "total_synced": 2
}
```

---

### b) 📌 Add

**Request**
```json
{
  "action": "add",
  "content": {
    "id": 3,
    "question": "Apa itu PPDB?",
    "answer_id": 202
  }
}
```

**Response Success**
```json
{
  "status": "success",
  "message": "Data berhasil ditambahkan ke Knowledge Bank",
  "id": 3
}
```

---

### c) 📌 Update

**Request**
```json
{
  "action": "update",
  "content": {
    "id": 2,
    "question": "Bagaimana cara memperbarui KTP?",
    "answer_id": 201
  }
}
```

**Response Success**
```json
{
  "status": "success",
  "message": "Data berhasil diupdate di Knowledge Bank"
}
```

---

### d) 📌 Delete

**Request**
```json
{
  "action": "delete",
  "content": {
    "id": 2
  }
}
```

**Response Success**
```json
{
  "status": "success",
  "message": "Data berhasil dihapus dari Knowledge Bank"
}
```

---

### ❌ Response Error

**Action tidak valid**
```json
{
  "status": "error",
  "error": {
    "type": "ValidationError",
    "message": "Action 'remove_all' tidak dikenali"
  }
}
```

**Bulk Sync – Content salah format**
```json
{
  "status": "error",
  "error": {
    "type": "ValidationError",
    "message": "Content harus berupa list untuk bulk_sync"
  }
}
```

**Delete – ID tidak ditemukan**
```json
{
  "status": "error",
  "error": {
    "type": "DeleteError",
    "message": "Gagal menghapus data dengan ID 999",
    "detail": "Point not found"
  }
}
```

**Server Error**
```json
{
  "status": "error",
  "error": {
    "type": "ServerError",
    "message": "Terjadi kesalahan internal saat sinkronisasi",
    "detail": "Qdrant connection refused"
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
    "message": "Tidak ada hasil cukup relevan untuk pertanyaan Anda."
  }
  ```

---

## 📌 Contoh Tes via curl

### Search
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"question": "apa itu KIS?", "wa_number": "628123456789"}'
```

### Bulk Sync
```bash
curl -X POST http://localhost:5000/api/sync \
  -H "Content-Type: application/json" \
  -d '{"action": "bulk_sync", "content": [{"id":1,"question":"Apa itu KIS?","answer_id":200}]}'
```

---

## ⚙️ Diagram Alur

```
User WA → WA Bot → /api/search → Flask API → Qdrant
WA Manajemen → /api/sync → Flask API → Qdrant
```

1. **WA Manajemen**  
   - Bertugas menambah, mengupdate, dan sinkronisasi data ke API.  
   - Field yang dikirim: **`id`, `question`, `answer_id`**.  

2. **WA Bot**  
   - Memanggil `/api/search` dengan pertanyaan user.  
   - Menggunakan `answer_id` untuk mengambil jawaban final dari MySQL.  
