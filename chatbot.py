import requests
import json
import time
import os
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:5000/api/search"
LOG_FILE = "chatbot_log.xlsx"

# ==========================================================
# 🔹 Fungsi Cetak & Format
# ==========================================================
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

# ==========================================================
# 🔹 Fungsi Logging ke Excel
# ==========================================================
def log_to_excel(entry):
    """
    Menyimpan setiap hasil pipeline ke file Excel chatbot_log.xlsx
    """
    columns = [
        "Timestamp", "Status", "Original Question", "Final Question", "Category",
        "Dense Top", "AI Reason", "AI Reformulated",
        "Total Candidates", "Accepted", "Rejected",
        "AI Filter (s)", "AI Relevance (s)", "Embedding (s)", "Qdrant (s)", "Total Time (s)"
    ]

    # Convert single log jadi DataFrame
    df_new = pd.DataFrame([entry], columns=columns)

    # Jika file belum ada → buat baru
    if not os.path.exists(LOG_FILE):
        df_new.to_excel(LOG_FILE, index=False)
    else:
        df_existing = pd.read_excel(LOG_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(LOG_FILE, index=False)

# ==========================================================
# 🔹 Fungsi Utama
# ==========================================================
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
            timing = data.get("timing", {})

            # 1️⃣ PRE-PROCESSING
            print_step("🔍 PRE-PROCESSING (AI Filter / Specificity Check)")
            if status == "LOW_CONFIDENCE" and "ai_debug" in data:
                pre = data["ai_debug"]
                print(f"Valid        : {pre.get('valid', '-')}")
                print(f"Reason       : {pre.get('reason', '-')}")
                print(f"Clean Q      : {pre.get('clean_question', '-')}")
            else:
                print(f"AI Filter Time : {timing.get('ai_domain_sec', '-')}s (otomatis)")

            # 2️⃣ EMBEDDING
            print_step("🧠 EMBEDDING VECTOR")
            print("Embedding dilakukan dengan model e5-small-local (dalam API Flask)\n-> hasil vektor tidak ditampilkan untuk efisiensi.")

            # 3️⃣ QDRANT RETRIEVAL
            print_step("🗃️  QDRANT SEARCH")
            print(f"Qdrant Search Time : {timing.get('qdrant_sec', '-')}")
            if status == "SUCCESS":
                meta = data["data"]["metadata"]
                print(f"Category Detected  : {meta.get('category', '-')}")
                print(f"Top Dense Score    : {meta.get('dense_score_top', '-')}")
            elif "debug_rejected" in data:
                print("Qdrant hasil ditemukan namun tidak cukup relevan.")

            # 4️⃣ POST-PROCESSING
            print_step("🤖 POST-PROCESSING (AI Relevance Check)")
            ai_reason = "-"
            ai_reform = "-"
            if status == "SUCCESS":
                meta = data["data"]["metadata"]
                ai_reason = meta.get("ai_reason", "-")
                ai_reform = meta.get("ai_reformulated", "-")
                print(f"AI Reason       : {ai_reason}")
                print(f"Reformulated Q  : {ai_reform}")
            elif status == "LOW_CONFIDENCE":
                if "ai_debug" in data:
                    dbg = data["ai_debug"]
                    ai_reason = dbg.get("reason", "-")
                    ai_reform = dbg.get("reformulated_question", "-")
                    print(f"AI Reason       : {ai_reason}")
                    print(f"Reformulated Q  : {ai_reform}")

            # 5️⃣ HASIL AKHIR
            print_step("📤 OUTPUT AKHIR")
            accepted = 0
            rejected = 0
            dense_top = 0.0
            category = "-"
            final_q = question

            if status == "SUCCESS":
                meta = data["data"]["metadata"]
                category = meta.get("category", "-")
                final_q = meta.get("final_question", question)
                dense_top = meta.get("dense_score_top", 0)
                print(f"STATUS        : SUCCESS")
                print(f"Original Q    : {meta.get('original_question', '-')}")
                print(f"Final Q       : {final_q}")
                print(f"Category      : {category}")
                print(f"AI Reform     : {meta.get('ai_reformulated', '-')}")
                results = data["data"]["similar_questions"]
                print_candidates(results)
                accepted = len(results)
            elif status == "LOW_CONFIDENCE":
                print("STATUS        : LOW CONFIDENCE")
                print(f"Message       : {data.get('message', '-')}")
                rejected = len(data.get("debug_rejected", []))
                if "debug_rejected" in data:
                    print_candidates(data["debug_rejected"])

            # 6️⃣ TIMING
            print_step("⏱️  WAKTU EKSEKUSI")
            print_timing(timing)

            print("\n" + "=" * 70 + "\n")

            # 7️⃣ LOGGING KE EXCEL
            entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Status": status,
                "Original Question": question,
                "Final Question": final_q,
                "Category": category,
                "Dense Top": dense_top,
                "AI Reason": ai_reason,
                "AI Reformulated": ai_reform,
                "Total Candidates": accepted + rejected,
                "Accepted": accepted,
                "Rejected": rejected,
                "AI Filter (s)": timing.get("ai_domain_sec", 0),
                "AI Relevance (s)": timing.get("ai_relevance_sec", 0),
                "Embedding (s)": timing.get("embedding_sec", 0),
                "Qdrant (s)": timing.get("qdrant_sec", 0),
                "Total Time (s)": timing.get("total_sec", 0)
            }

            log_to_excel(entry)
            print(f"📝 Log tersimpan ke {LOG_FILE}\n")

        except Exception as e:
            print(f"⚠️  ERROR: {str(e)}")
            print("=" * 60)
            print()

# ==========================================================
# 🔹 Jalankan Program
# ==========================================================
if __name__ == "__main__":
    main()
