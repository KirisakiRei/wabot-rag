import requests
import json
import csv
import os
from datetime import datetime

# ===============================
# 🔹 Konfigurasi
# ===============================
API_URL = "http://localhost:5000/api/search"
LOG_DIR = "logs"

# Pastikan folder logs tersedia
os.makedirs(LOG_DIR, exist_ok=True)

# ===============================
# 🔹 Fungsi API Request
# ===============================
def ask_question(question, wa_number="628123456789"):
    payload = {
        "question": question,
        "wa_number": wa_number
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ===============================
# 🔹 Format angka skor
# ===============================
def fmt_score(val):
    if val is None:
        return "(N/A)"
    try:
        return f"{float(val):.3f}"
    except:
        return str(val)

# ===============================
# 🔹 Simpan ke file log harian
# ===============================
def save_log(wa_number, question, resp):
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOG_DIR, f"chatbot_log_{today}.csv")

    # Ambil data utama
    status = resp.get("status", "error")
    timing = resp.get("timing", {})
    meta = resp.get("data", {}).get("metadata", {})
    ai_debug = resp.get("ai_debug", {})

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wa_number": wa_number,
        "original_question": meta.get("original_question", question),
        "ai_clean_question": meta.get("ai_clean_question", ai_debug.get("ai_clean_question", "-")),
        "status": status,
        "category": meta.get("detected_category", "-"),
        "total_found": meta.get("total_found", 0),
        "response_time": timing.get("total_sec", 0),
        "ai_time": timing.get("ai_filter_sec", 0),
        "embedding_time": timing.get("embedding_sec", 0),
        "qdrant_time": timing.get("qdrant_sec", 0),
        "message": resp.get("message", "-"),
        "top_result": "-"
    }

    # Jika ada hasil relevan, ambil pertanyaan teratas
    if status == "success":
        results = resp["data"].get("similar_questions", [])
        if results:
            row["top_result"] = results[0]["question"]

    # Tulis ke file CSV (Excel-friendly)
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys(), delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ===============================
# 🔹 Tampilkan hasil di terminal
# ===============================
def pretty_print_response(resp, question, wa_number="628123456789"):
    print("=" * 60)
    status = resp.get("status")
    timing = resp.get("timing", {})

    if status == "success":
        meta = resp["data"]["metadata"]
        print(f"✅ STATUS: SUCCESS")
        print(f"⏱️ AI Filter    : {timing.get('ai_filter_sec', 0):.3f}s")
        print(f"🔢 Embedding    : {timing.get('embedding_sec', 0):.3f}s")
        print(f"📚 Qdrant Search: {timing.get('qdrant_sec', 0):.3f}s")
        print(f"⚡ Total         : {timing.get('total_sec', 0):.3f}s")
        print("-" * 60)
        print(f"📌 Original     : {meta.get('original_question', '-')}")
        print(f"🤖 AI Cleaned   : {meta.get('ai_clean_question', '-')}")
        print(f"📂 Category     : {meta.get('detected_category', '-')}")
        print(f"🔍 Total Found  : {meta.get('total_found', '-')}")
        print("-" * 60)
        for i, item in enumerate(resp["data"]["similar_questions"], 1):
            print(f"[{i}] Q: {item['question']}")
            print(f"    Dense     : {fmt_score(item.get('dense_score'))}")
            print(f"    Overlap   : {fmt_score(item.get('overlap_score'))}")
            print(f"    Note      : {item.get('note', '-')}")
            print()

    elif status == "low_confidence":
        print("⚠️ STATUS: LOW CONFIDENCE")
        print(f"⏱️ AI Filter    : {timing.get('ai_filter_sec', 0):.3f}s")
        print(f"⚡ Total         : {timing.get('total_sec', 0):.3f}s")
        print("-" * 60)
        print(f"Message   : {resp.get('message')}")
        print(f"Suggestion: {resp.get('suggestion', '-')}")
        print("-" * 60)

        # Debug AI info
        ai_debug = resp.get("ai_debug", {})
        if ai_debug:
            print("🤖 AI DEBUG:")
            print(f"   Cleaned    : {ai_debug.get('ai_clean_question', '-')}")
            print(f"   Reason     : {ai_debug.get('ai_reason', '-')}")
            print(f"   Suggestion : {ai_debug.get('ai_suggestion', '-')}")
            print("-" * 60)

        debug = resp.get("debug_rejected") or []
        if debug:
            print("🔎 Kandidat Terdekat:")
            for cand in debug:
                print(f"- Q: {cand.get('question', '-')}")
                print(f"    Dense   : {fmt_score(cand.get('dense_score'))}")
                print(f"    Overlap : {fmt_score(cand.get('overlap_score'))}")
                print()

    else:
        print("❌ STATUS: ERROR")
        print(json.dumps(resp, indent=2, ensure_ascii=False))
    print("=" * 60)

    # Simpan log
    save_log(wa_number, question, resp)

# ===============================
# 🔹 Mode interaktif terminal
# ===============================
if __name__ == "__main__":
    print("🤖 Chatbot Tester (Ketik 'exit' untuk keluar)")
    while True:
        q = input("\nAnda: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("👋 Keluar...")
            break
        resp = ask_question(q)
        pretty_print_response(resp, q)
