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

def pretty_print_response(resp):
    print("=" * 60)
    status = resp.get("status")

    if status == "success":
        print(f"✅ Status: {status}")
        meta = resp["data"]["metadata"]
        print(f"📌 Pertanyaan Asli: {meta['original_question']}")
        print(f"📱 WA Number: {meta['wa_number']}")
        print(f"📂 Category: {meta['category_used']}")
        print(f"🔍 Total Found: {meta['total_found']}")
        print("-" * 60)
        for i, item in enumerate(resp["data"]["similar_questions"], 1):
            print(f"[{i}] Q: {item['question']}")
            print(f"    Answer ID : {item['answer_id']}")
            print(f"    Category  : {item.get('category_id')}")
            print(f"    Dense     : {item.get('dense_score')}")
            print(f"    Overlap   : {item.get('overlap_score')}")
            print(f"    Final     : {item.get('final_score')}")
            print()

    elif status == "low_confidence":
        print("⚠️ Low confidence")
        print(f"Message: {resp.get('message')}")
        print("-" * 60)
        # tampilkan kandidat debug (dense, overlap, final)
        debug_candidates = resp.get("debug_rejected") or resp.get("debug_top_candidates")
        if debug_candidates:
            print("🔎 Kandidat terdekat:")
            for cand in debug_candidates:
                print(f"- Q: {cand['question']}")
                print(f"    Dense   : {cand['dense_score']:.3f}")
                print(f"    Overlap : {cand['overlap_score']:.3f}")
                print(f"    Final   : {cand['final_score']:.3f}")
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
