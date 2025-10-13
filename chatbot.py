import requests, json, time

def run_chatbot():
    print("\n🤖 Chatbot Tester (Ketik 'exit' untuk keluar)\n")

    while True:
        user_input = input("Anda: ").strip()
        if user_input.lower() == "exit":
            print("👋 Keluar...")
            break

        payload = {"question": user_input}
        try:
            t0 = time.time()
            resp = requests.post("http://localhost:5000/api/search", json=payload, timeout=60)
            data = resp.json()
            total = round(time.time() - t0, 3)
        except Exception as e:
            print(f"❌ Error koneksi ke server: {e}")
            continue

        print("=" * 60)

        status = data.get("status", "unknown").upper()
        print(f"⚙️ STATUS: {status}")
        timing = data.get("timing", {})
        print(f"⏱️ AI Filter: {timing.get('ai_domain_sec',0)}s | Relevance: {timing.get('ai_relevance_sec',0)}s")
        print(f"🔢 Embedding: {timing.get('embedding_sec',0)}s | Qdrant: {timing.get('qdrant_sec',0)}s")
        print(f"⚡ Total: {timing.get('total_sec',total)}s")

        print("-" * 60)
        meta = data.get("data", {}).get("metadata", {})
        if "original_question" in meta:
            print(f"📌 Original: {meta['original_question']}")
        if "final_question" in meta:
            print(f"🤖 Final: {meta['final_question']}")
        if "category" in meta:
            print(f"📂 Category: {meta['category']}")

        print("-" * 60)

        # === HANDLE LOW CONFIDENCE ===
        if status.lower() == "low_confidence":
            reason = data.get("message", "-")
            ai_debug = data.get("ai_debug", {})
            print(f"⚠️ LOW CONFIDENCE")
            print(f"Reason     : {reason}")
            if "reason" in ai_debug:
                print(f"AI Reason  : {ai_debug.get('reason','-')}")
            if "reformulated_question" in ai_debug:
                print(f"AI Reform  : {ai_debug.get('reformulated_question','-')}")
            if "suggestion" in ai_debug:
                print(f"Suggestion : {ai_debug.get('suggestion','-')}")

            # tampilkan rejected list juga
            rej = data.get("debug_rejected", [])
            if rej:
                print("🔎 Kandidat terdekat:")
                for r in rej[:3]:
                    print(f"- Q: {r['question']}")
                    print(f"  Dense: {r['dense_score']:.3f} | Overlap: {r['overlap_score']:.3f}")
            print("=" * 60)
            continue

        # === HANDLE SUCCESS ===
        sim = data.get("data", {}).get("similar_questions", [])
        if not sim:
            print("Tidak ada hasil ditemukan.")
        else:
            print(f"🔍 Total Found: {len(sim)}\n")
            for i, s in enumerate(sim, 1):
                print(f"[{i}] Q: {s['question']}")
                print(f"    Dense: {s['dense_score']:.3f} | Overlap: {s['overlap_score']:.3f} | Note: {s.get('note','-')}")

        print("=" * 60)
