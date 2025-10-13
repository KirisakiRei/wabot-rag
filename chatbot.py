import requests
import json
import time
import csv
import os
from datetime import datetime

API_URL = "http://localhost:5000/api/search"
LOG_FILE = "chat_log.csv"

# ==========================================================
# 🔹 Utility: setup log file jika belum ada
# ==========================================================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "user_question", "status", "ai_reason", "ai_reformulated",
            "category", "dense_top", "overlap_top", "total_found",
            "ai_domain_sec", "ai_relevance_sec", "embedding_sec", "qdrant_sec", "total_sec"
        ])

# ==========================================================
# 🔹 Fungsi Helper
# ==========================================================
def fmt_time(sec):
    return f"{sec:.3f}s" if sec else "-"

def log_to_csv(row):
    """Simpan hasil ke file log untuk laporan performa"""
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ==========================================================
# 🔹 Chatbot CLI
# ==========================================================
def run_chatbot():
    print("\n🤖 Chatbot Tester (Ketik 'exit' untuk keluar)\n")

    while True:
        user_input = input("Anda: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("👋 Keluar...")
            break

        payload = {"question": user_input}

        try:
            t0 = time.time()
            resp = requests.post(API_URL, json=payload, timeout=60)
            data = resp.json()
            total_time = round(time.time() - t0, 3)
        except Exception as e:
            print(f"❌ Gagal koneksi ke API: {e}")
            continue

        print("=" * 60)

        status = data.get("status", "unknown").upper()
        timing = data.get("timing", {})

        print(f"⚙️ STATUS: {status}")
        print(f"⏱️ AI Filter: {fmt_time(timing.get('ai_domain_sec', 0))} | Relevance: {fmt_time(timing.get('ai_relevance_sec', 0))}")
        print(f"🔢 Embedding: {fmt_time(timing.get('embedding_sec', 0))} | Qdrant: {fmt_time(timing.get('qdrant_sec', 0))}")
        print(f"⚡ Total: {fmt_time(timing.get('total_sec', total_time))}")
        print("-" * 60)

        meta = data.get("data", {}).get("metadata", {})
        ai_debug = data.get("ai_debug", {})

        # ======================================================
        # ⚠️ LOW CONFIDENCE CASE
        # ======================================================
        if status.lower() == "low_confidence":
            print(f"⚠️ LOW CONFIDENCE")
            print(f"Message     : {data.get('message', '-')}")
            print(f"AI Reason   : {ai_debug.get('reason', '-')}")
            print(f"Suggestion  : {ai_debug.get('suggestion', '-')}")
            print(f"Reformulate : {ai_debug.get('reformulated_question', '-')}")
            print("-" * 60)

            rejected = data.get("debug_rejected", [])
            if rejected:
                print("🔎 Kandidat terdekat:")
                for r in rejected[:3]:
                    print(f"- Q: {r['question']}")
                    print(f"  Dense: {r['dense_score']:.3f} | Overlap: {r['overlap_score']:.3f}")
            print("=" * 60)

            # Simpan ke log
            log_to_csv([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_input,
                "LOW_CONFIDENCE",
                ai_debug.get("reason", "-"),
                ai_debug.get("reformulated_question", "-"),
                meta.get("category", "-"),
                "-", "-", len(rejected),
                timing.get("ai_domain_sec", 0),
                timing.get("ai_relevance_sec", 0),
                timing.get("embedding_sec", 0),
                timing.get("qdrant_sec", 0),
                timing.get("total_sec", total_time)
            ])
            continue

        # ======================================================
        # ✅ SUCCESS CASE
        # ======================================================
        print(f"📌 Original     : {meta.get('original_question', '-')}")
        print(f"🤖 Final        : {meta.get('final_question', '-')}")
        print(f"📂 Category     : {meta.get('category', '-')}")
        print(f"AI Reason       : {meta.get('ai_reason', '-')}")
        print(f"AI Reformulated : {meta.get('ai_reformulated', '-')}")
        print("-" * 60)

        sim = data.get("data", {}).get("similar_questions", [])
        if not sim:
            print("Tidak ada hasil relevan.")
        else:
            print(f"🔍 Total Found  : {len(sim)}")
            print()
            for i, s in enumerate(sim, 1):
                print(f"[{i}] Q: {s['question']}")
                print(f"    Dense: {s['dense_score']:.3f} | Overlap: {s['overlap_score']:.3f} | Note: {s.get('note','-')}")
                print()

        # Simpan ke log
        top_dense = sim[0]['dense_score'] if sim else "-"
        top_overlap = sim[0]['overlap_score'] if sim else "-"
        log_to_csv([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_input,
            "SUCCESS",
            meta.get("ai_reason", "-"),
            meta.get("ai_reformulated", "-"),
            meta.get("category", "-"),
            top_dense,
            top_overlap,
            len(sim),
            timing.get("ai_domain_sec", 0),
            timing.get("ai_relevance_sec", 0),
            timing.get("embedding_sec", 0),
            timing.get("qdrant_sec", 0),
            timing.get("total_sec", total_time)
        ])

        print("=" * 60)

# ==========================================================
# 🔹 Jalankan chatbot
# ==========================================================
if __name__ == "__main__":
    run_chatbot()
