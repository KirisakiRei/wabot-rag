import requests
import json
import time

API_URL = "http://127.0.0.1:5000/api/search"

def fmt_score(s):
    return f"{s:.3f}" if isinstance(s, (int, float)) else "-"

print("🤖 Chatbot Tester (Ketik 'exit' untuk keluar)\n")

while True:
    user_q = input("Anda: ").strip()
    if user_q.lower() == "exit":
        print("👋 Keluar...")
        break

    payload = {"question": user_q}
    t0 = time.time()
    resp = requests.post(API_URL, json=payload)
    elapsed = time.time() - t0

    try:
        data = resp.json()
    except Exception as e:
        print(f"⚠️ Response error: {e}")
        print(resp.text)
        continue

    print("=" * 60)
    print(f"✅ STATUS: {data.get('status', 'unknown').upper()}")

    timing = data.get("timing", {})
    print(f"⏱️ AI Filter    : {timing.get('ai_sec', timing.get('ai_filter_sec', 0.0)):.3f}s")
    print(f"🔢 Embedding    : {timing.get('embedding_sec', 0.0):.3f}s")
    print(f"📚 Qdrant Search: {timing.get('qdrant_sec', 0.0):.3f}s")
    print(f"⚡ Total         : {timing.get('total_sec', elapsed):.3f}s")
    print("-" * 60)

    # 🔹 tampilkan metadata debug AI
    ai_debug = data.get("ai_debug") or data.get("data", {}).get("metadata", {})
    if ai_debug:
        print("🤖 AI DEBUG:")
        print(f"   Reason        : {ai_debug.get('ai_reason', '-')}")
        print(f"   Reformulated  : {ai_debug.get('ai_reformulated', '-')}")
        print(f"   Cleaned       : {ai_debug.get('ai_clean_question', '-')}")
        print()

    # 🔹 tampilkan metadata umum
    meta = data.get("data", {}).get("metadata", {})
    if meta:
        print(f"📌 Original     : {meta.get('original_question', '-')}")
        print(f"🤖 Final / Clean: {meta.get('final_question', meta.get('ai_clean_question', '-'))}")
        print(f"📂 Category     : {meta.get('category', meta.get('detected_category', '-'))}")
        print(f"🔍 Total Found  : {meta.get('total_found', '-')}")
        print("-" * 60)

    # 🔹 tampilkan hasil similarity
    results = data.get("data", {}).get("similar_questions", [])
    if results:
        for i, r in enumerate(results, 1):
            print(f"[{i}] Q: {r.get('question')}")
            print(f"    Dense     : {fmt_score(r.get('dense_score'))}")
            print(f"    Overlap   : {fmt_score(r.get('overlap_score'))}")
            print(f"    Note      : {r.get('note', '-')}")
            print()
    else:
        if data.get("status") == "low_confidence":
            print("⚠️ Tidak ada hasil relevan.")
            debug_rej = data.get("debug_rejected", [])
            if debug_rej:
                print("🔎 Kandidat Terdekat:")
                for r in debug_rej:
                    print(f"- Q: {r.get('question', '-')}")
                    print(f"    Dense   : {fmt_score(r.get('dense_score'))}")
                    print(f"    Overlap : {fmt_score(r.get('overlap_score'))}")
                    print()
    print("=" * 60)
    print(f"🕒 Total waktu (client): {elapsed:.3f}s\n")
