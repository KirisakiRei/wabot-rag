import requests
import json
import time

API_URL = "http://localhost:5000/api/search"


def ask_question(question, wa_number="628123456789"):
    """Kirim pertanyaan ke API dan kembalikan hasil JSON."""
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


def fmt_score(val):
    if val is None:
        return "(N/A)"
    try:
        return f"{float(val):.3f}"
    except:
        return str(val)


def pretty_print_response(resp):
    """Cetak hasil response dengan format yang mudah dibaca."""
    print("=" * 60)
    status = resp.get("status", "-").lower()

    timing = resp.get("timing", {})
    ai_debug = resp.get("ai_debug", {})

    if status == "success":
        print("✅ STATUS: SUCCESS")
        print(f"⏱️ AI Filter    : {timing.get('ai_filter_sec', 0):.3f}s")
        print(f"🔢 Embedding    : {timing.get('embedding_sec', 0):.3f}s")
        print(f"📚 Qdrant Search: {timing.get('qdrant_sec', 0):.3f}s")
        print(f"⚡ Total         : {timing.get('total_sec', 0):.3f}s")
        print("-" * 60)

        # tampilkan debug AI jika tersedia
        if ai_debug:
            print("🤖 AI DEBUG:")
            print(f"   Cleaned    : {ai_debug.get('ai_clean_question', '-')}")
            print(f"   Reason     : {ai_debug.get('ai_reason', '-')}")
            print(f"   Suggestion : {ai_debug.get('ai_suggestion', '-')}")
            print("-" * 60)

        meta = resp.get("data", {}).get("metadata", {})
        print(f"📌 Original     : {meta.get('original_question', '-')}")
        print(f"🤖 AI Cleaned   : {meta.get('ai_clean_question', '-')}")
        print(f"📂 Category     : {meta.get('detected_category', '-')}")
        print(f"🔍 Total Found  : {meta.get('total_found', 0)}")
        print("-" * 60)

        for i, item in enumerate(resp["data"]["similar_questions"], 1):
            print(f"[{i}] Q: {item.get('question', '-')}")
            print(f"    Dense     : {fmt_score(item.get('dense_score'))}")
            print(f"    Overlap   : {fmt_score(item.get('overlap_score'))}")
            print(f"    Note      : {item.get('note', '-')}")
            print()

    elif status == "low_confidence":
        print("⚠️ STATUS: LOW CONFIDENCE")
        print(f"⏱️ AI Filter    : {timing.get('ai_filter_sec', 0):.3f}s")
        print(f"🔢 Embedding    : {timing.get('embedding_sec', 0):.3f}s")
        print(f"📚 Qdrant Search: {timing.get('qdrant_sec', 0):.3f}s")
        print(f"⚡ Total         : {timing.get('total_sec', 0):.3f}s")
        print("-" * 60)
        print(f"Message   : {resp.get('message', '-')}")
        print(f"Suggestion: {resp.get('suggestion', '-')}")

        # tampilkan debug AI juga untuk analisa
        if ai_debug:
            print("-" * 60)
            print("🤖 AI DEBUG:")
            print(f"   Cleaned    : {ai_debug.get('ai_clean_question', '-')}")
            print(f"   Reason     : {ai_debug.get('ai_reason', '-')}")
            print(f"   Suggestion : {ai_debug.get('ai_suggestion', '-')}")
        
        debug = resp.get("debug_rejected", [])
        if debug:
            print("-" * 60)
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


if __name__ == "__main__":
    print("🤖 Chatbot Tester (ketik 'exit' untuk keluar)")
    while True:
        q = input("\nAnda: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("👋 Keluar...")
            break
        t_start = time.time()
        resp = ask_question(q)
        elapsed = time.time() - t_start
        pretty_print_response(resp)
        print(f"🕒 Total Waktu Request (client): {elapsed:.3f}s")
