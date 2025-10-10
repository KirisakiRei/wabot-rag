import requests
import json

API_URL = "http://localhost:5000/api/search"

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

def fmt_score(val):
    if val is None:
        return "(N/A)"
    try:
        return f"{float(val):.3f}"
    except:
        return str(val)

def pretty_print_response(resp):
    print("=" * 60)
    status = resp.get("status")

    if status == "success":
        print(f"✅ Status: {status}")
        meta = resp["data"]["metadata"]
        print(f"📌 Pertanyaan Asli    : {meta.get('original_question', '-')}")
        print(f"🧹 Normalized         : {meta.get('normalized_question', '-')}")
        print(f"🤖 AI Cleaned         : {meta.get('ai_clean_question', '(tidak ada perubahan)')}")
        print(f"📱 WA Number          : {meta.get('wa_number', '-')}")
        print(f"📂 Category Detected  : {meta.get('detected_category', '-')}")
        print(f"🔍 Total Found        : {meta.get('total_found', '-')}")
        print("-" * 60)

        for i, item in enumerate(resp["data"]["similar_questions"], 1):
            print(f"[{i}] Q: {item.get('question', '-')}")
            print(f"    Answer ID : {item.get('answer_id', '-')}")
            print(f"    Category  : {item.get('category_id', '-')}")
            print(f"    Dense     : {fmt_score(item.get('dense_score'))}")
            print(f"    Overlap   : {fmt_score(item.get('overlap_score'))}")
            print(f"    Note      : {item.get('note', '-')}")
            print()

    elif status == "low_confidence":
        print("⚠️ Low confidence")
        print(f"Message: {resp.get('message')}")
        if "suggestion" in resp:
            print(f"Suggestion: {resp['suggestion']}")
        print("-" * 60)
        debug_candidates = resp.get("debug_rejected") or resp.get("debug_top_candidates")
        if debug_candidates:
            print("🔎 Kandidat terdekat:")
            for cand in debug_candidates:
                print(f"- Q: {cand.get('question', '-')}")
                print(f"    Dense   : {fmt_score(cand.get('dense_score'))}")
                print(f"    Overlap : {fmt_score(cand.get('overlap_score'))}")
                print()

    else:
        print("❌ Error:")
        print(json.dumps(resp, indent=2, ensure_ascii=False))

    print("=" * 60)


if __name__ == "__main__":
    print("🤖 Chatbot Tester (ketik 'exit' untuk keluar)")
    while True:
        q = input("\nAnda: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("👋 Keluar...")
            break
        resp = ask_question(q)
        pretty_print_response(resp)
