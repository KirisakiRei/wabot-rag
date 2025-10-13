import requests
import csv
import datetime
import os
import time

API_URL = "http://127.0.0.1:5000/api/search"
LOG_FILE = "chat_log.csv"

# ===============================
# 🔹 Setup Logging ke CSV
# ===============================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "question", "status", "ai_reason", "ai_reformulated",
            "dense_top", "category", "ai_domain_sec", "ai_relevance_sec",
            "embedding_sec", "qdrant_sec", "total_sec"
        ])

def log_to_csv(data):
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.get("question", "-"),
            data.get("status", "-"),
            data.get("ai_reason", "-"),
            data.get("ai_reformulated", "-"),
            data.get("dense_top", "-"),
            data.get("category", "-"),
            data.get("ai_domain_sec", "-"),
            data.get("ai_relevance_sec", "-"),
            data.get("embedding_sec", "-"),
            data.get("qdrant_sec", "-"),
            data.get("total_sec", "-")
        ])

# ===============================
# 🔹 Fungsi Cetak Debug
# ===============================
def print_timing(t):
    print(f"⏱️ AI Filter: {t.get('ai_domain_sec',0):.3f}s | Relevance: {t.get('ai_relevance_sec',0):.3f}s")
    print(f"🔢 Embedding: {t.get('embedding_sec',0):.3f}s | Qdrant: {t.get('qdrant_sec',0):.3f}s")
    print(f"⚡ Total: {t.get('total_sec',0):.3f}s")

def fmt_score(s):
    return f"{float(s):.3f}" if isinstance(s, (float, int)) else "-"

# ===============================
# 🔹 Chatbot Loop
# ===============================
print("🤖 Chatbot Tester (Ketik 'exit' untuk keluar)\n")

while True:
    user_input = input("Anda: ").strip()
    if user_input.lower() in ["exit", "keluar", "quit"]:
        print("👋 Keluar...")
        break
    if not user_input:
        continue

    payload = {"question": user_input}
    start = time.time()
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        data = resp.json()
    except Exception as e:
        print(f"❌ Gagal konek ke API: {e}")
        continue

    end = time.time()
    status = data.get("status", "-")
    timing = data.get("timing", {})

    print("=" * 60)
    print(f"⚙️ STATUS: {status.upper()}")
    print_timing(timing)
    print("-" * 60)

    # LOW CONFIDENCE
    if status == "low_confidence":
        print("⚠️ LOW CONFIDENCE")
        print(f"Message     : {data.get('message', '-')}")
        ai_debug = data.get("ai_debug", {})
        print(f"AI Reason   : {ai_debug.get('reason', ai_debug.get('ai_reason','-'))}")
        print(f"Suggestion  : {ai_debug.get('suggestion', '-')}")
        print(f"Reformulate : {ai_debug.get('reformulated_question', ai_debug.get('clean_question','-'))}")
        print("-" * 60)

        rejected = data.get("debug_rejected", [])
        if rejected:
            print("🔎 Kandidat terdekat:")
            for cand in rejected:
                print(f"- Q: {cand.get('question','-')}")
                print(f"  Dense: {fmt_score(cand.get('dense_score'))} | Overlap: {fmt_score(cand.get('overlap_score'))}")
            print()
        print("=" * 60)

        log_to_csv({
            "question": user_input,
            "status": "low_confidence",
            "ai_reason": ai_debug.get("reason", "-"),
            "ai_reformulated": ai_debug.get("reformulated_question", "-"),
            "dense_top": "-",
            "category": "-",
            **timing
        })
        continue

    # SUCCESS
    if status == "success":
        meta = data.get("data", {}).get("metadata", {})
        results = data.get("data", {}).get("similar_questions", [])
        ai_reason = meta.get("ai_reason", "-")
        ai_reformulated = meta.get("ai_reformulated", "-")

        print(f"📌 Original     : {meta.get('original_question', '-')}")
        print(f"🤖 Final        : {meta.get('final_question', '-')}")
        print(f"📂 Category     : {meta.get('category', '-')}")
        print(f"AI Reason       : {ai_reason}")
        print(f"AI Reformulated : {ai_reformulated}")
        print("-" * 60)
        print(f"🔍 Total Found  : {meta.get('total_found', len(results))}\n")

        for i, r in enumerate(results, 1):
            print(f"[{i}] Q: {r['question']}")
            print(f"    Dense: {fmt_score(r['dense_score'])} | Overlap: {fmt_score(r['overlap_score'])} | Note: {r.get('note','-')}")
            print()
        print("=" * 60)

        log_to_csv({
            "question": user_input,
            "status": "success",
            "ai_reason": ai_reason,
            "ai_reformulated": ai_reformulated,
            "dense_top": results[0]["dense_score"] if results else "-",
            "category": meta.get("category", "-"),
            **timing
        })
    else:
        print(f"❌ Tidak diketahui: {data}")
