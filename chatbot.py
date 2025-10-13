import requests
import json
import time

API_URL = "http://localhost:5000/api/search"

def print_header(title):
    print("=" * 70)
    print(title)
    print("=" * 70)

def print_step(title, content=None):
    print(f"\n🧩 {title}")
    print("-" * 70)
    if content:
        if isinstance(content, dict):
            for k, v in content.items():
                print(f"{k:20}: {v}")
        else:
            print(content)
    print("-" * 70)

def print_timing(t):
    print(f"⏱️  AI Filter     : {t.get('ai_domain_sec', '-')}s")
    print(f"🤖 AI Relevance  : {t.get('ai_relevance_sec', '-')}s")
    print(f"🔢 Embedding     : {t.get('embedding_sec', '-')}s")
    print(f"🗃️  Qdrant        : {t.get('qdrant_sec', '-')}s")
    print(f"⚡ Total Time    : {t.get('total_sec', '-')}s")

def print_candidates(results):
    print(f"\n🔍 Candidate Results ({len(results)} ditemukan)")
    print("-" * 70)
    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['question']}")
        print(f"    Dense: {r['dense_score']:.3f} | Overlap: {r['overlap_score']:.3f} | Note: {r['note']}")
    print("-" * 70)

def main():
    print("🤖 Chatbot Debugger (ketik 'exit' untuk keluar)\n")

    while True:
        question = input("Anda: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Keluar...")
            break

        payload = {"question": question}
        try:
            t0 = time.time()
            response = requests.post(API_URL, json=payload, timeout=120)
            elapsed = round(time.time() - t0, 2)

            print_header(f"FULL PIPELINE TRACE ({elapsed}s)")
            print_step("📥 Input Pertanyaan", question)

            if response.status_code != 200:
                print(f"❌ Server error: {response.status_code}")
                print(response.text)
                continue

            data = response.json()
            status = data.get("status", "").upper()

            # 1️⃣ PRE-PROCESSING
            print_step("🔍 PRE-PROCESSING (AI Filter / Specificity Check)")
            if status == "LOW_CONFIDENCE" and "ai_debug" in data:
                pre = data["ai_debug"]
                print(f"Valid        : {pre.get('valid', '-')}")
                print(f"Reason       : {pre.get('reason', '-')}")
                print(f"Clean Q      : {pre.get('clean_question', '-')}")
            else:
                timing = data.get("timing", {})
                print(f"AI Filter Time : {timing.get('ai_domain_sec', '-')}s (otomatis)")

            # 2️⃣ EMBEDDING
            print_step("🧠 EMBEDDING VECTOR")
            print("Embedding dilakukan dengan model e5-small-local (dalam API Flask)\n-> hasil vektor tidak ditampilkan untuk efisiensi.")

            # 3️⃣ QDRANT RETRIEVAL
            print_step("🗃️  QDRANT SEARCH")
            timing = data.get("timing", {})
            print(f"Qdrant Search Time : {timing.get('qdrant_sec', '-')}")
            if status == "SUCCESS":
                meta = data["data"]["metadata"]
                print(f"Category Detected  : {meta.get('category', '-')}")
                print(f"Top Dense Score    : {meta.get('dense_score_top', '-')}")
            elif "debug_rejected" in data:
                print("Qdrant hasil ditemukan namun tidak cukup relevan.")

            # 4️⃣ POST-PROCESSING (AI RELEVANCE)
            print_step("🤖 POST-PROCESSING (AI Relevance Check)")
            if status == "SUCCESS":
                meta = data["data"]["metadata"]
                print(f"AI Reason       : {meta.get('ai_reason', '-')}")
                print(f"Reformulated Q  : {meta.get('ai_reformulated', '-')}")
            elif status == "LOW_CONFIDENCE":
                if "ai_debug" in data:
                    dbg = data["ai_debug"]
                    print(f"AI Reason       : {dbg.get('reason', '-')}")
                    print(f"Reformulated Q  : {dbg.get('reformulated_question', '-')}")

            # 5️⃣ HASIL AKHIR
            print_step("📤 OUTPUT AKHIR")

            if status == "SUCCESS":
                meta = data["data"]["metadata"]
                print(f"STATUS        : SUCCESS")
                print(f"Original Q    : {meta.get('original_question', '-')}")
                print(f"Final Q       : {meta.get('final_question', '-')}")
                print(f"Category      : {meta.get('category', '-')}")
                print(f"AI Reform     : {meta.get('ai_reformulated', '-')}")
                print_candidates(data["data"]["similar_questions"])

            elif status == "LOW_CONFIDENCE":
                print("STATUS        : LOW CONFIDENCE")
                print(f"Message       : {data.get('message', '-')}")
                if "debug_rejected" in data:
                    print_candidates(data["debug_rejected"])
            else:
                print(f"STATUS        : {status or 'UNKNOWN'}")
                print(json.dumps(data, indent=2))

            # 6️⃣ TIMING
            print_step("⏱️  WAKTU EKSEKUSI")
            print_timing(data.get("timing", {}))

            print("\n" + "=" * 70 + "\n")

        except Exception as e:
            print(f"⚠️  ERROR: {str(e)}")
            print("=" * 60)
            print()

if __name__ == "__main__":
    main()
