# 🚗 ClaimIQ — Multi-Agent AI Insurance Claims Processor

ClaimIQ is a multi-agent AI system that processes insurance claims the way a real claims adjuster would — checking policy coverage, assessing damage from photos, and flagging fraud risk — instead of just answering policy questions like most AI insurance projects do.

Submit a claim with a photo, click one button, and get back a real, explainable decision: **Approve**, **Deny**, or **Escalate for human review** — with the full reasoning shown, never a black box.

---

## 🎯 The Problem

Insurance claim processing today is slow and inconsistent. A human adjuster has to manually check policy coverage, review damage photos, and watch for fraud — a process that typically takes days and depends heavily on individual judgment. ClaimIQ demonstrates how independently trained, properly evaluated AI components can be combined into a single fast, transparent decision pipeline.

## 🧠 Why Multiple Agents Instead of One LLM Call

Most AI insurance projects are just a RAG chatbot answering policy questions. ClaimIQ goes further by treating each part of the real adjuster workflow as its own properly evaluated component:

- **Fraud is a numeric/behavioral pattern problem** — best solved with a classical ML model trained on historical claims, not an LLM guessing.
- **Damage severity needs a real trained visual classifier** — not an LLM describing a photo with no calibrated accuracy.
- **Policy coverage needs grounded, citable answers** — not an LLM answering from memory.
- **The final decision needs to be explainable** — a rule-based orchestrator combining all three signals, not a single opaque model, matters in a regulated industry like insurance.

---

## 🏗️ Architecture

```
User submits claim (description + photo) via Streamlit
        │
        ▼
FastAPI backend → stores claim in SQLite
        │
        ▼
User clicks "Process Claim with AI"
        │
        ├──► Policy RAG Agent (LangChain + FAISS + Gemini)
        │     → grounded coverage answer + citations
        │
        ├──► Fraud Risk Agent (Scikit-learn Random Forest)
        │     → fraud probability + risk label
        │
        ├──► Damage Assessment Agent (PyTorch, ResNet18 transfer learning)
        │     → severity classification + confidence
        │
        ▼
Orchestrator (rule-based decision engine)
        │
        ▼
Final decision + full reasoning, saved back to the claim record
```

---

## 🤖 The Four Agents

### 1. Policy RAG Agent
Answers coverage questions grounded in real policy documents (3 real, publicly sourced auto insurance policies from two different states plus a consumer guide), using chunking, embeddings, FAISS retrieval, and a strict citation-enforcing prompt to prevent hallucination.

### 2. Fraud Risk Agent
A Random Forest classifier trained on 15,420 real historical insurance claims, using `class_weight="balanced"` to address class imbalance (fraud is ~6% of claims). Prioritizes recall over precision — in fraud detection, missing real fraud is costlier than a false alarm a human can dismiss.

### 3. Damage Assessment Agent
A pretrained ResNet18 (ImageNet) with a custom trained classification head (transfer learning), classifying uploaded photos into minor / moderate / severe damage categories.

### 4. Orchestrator
A deliberately rule-based (not LLM-driven) decision engine that combines all three agents' outputs into one final, fully explainable decision. Rule-based orchestration was a deliberate choice: fully auditable, fast, and free of extra API calls — a legitimate production pattern used in real regulated decision systems.

**Decision rules, in priority order:**
1. Fraud probability ≥ 70% → escalate for manual fraud review
2. Coverage answer indicates a genuine exclusion → deny
3. Coverage answer is inconclusive (retrieval gap) → escalate for manual coverage review
4. Damage severity is severe → escalate for adjuster review, regardless of other signals
5. Moderate damage + fraud probability ≥ 40% → escalate for review
6. Otherwise → auto-approve

---

## 📊 Evaluation Results (real numbers, not estimates)

| Agent | Metric | Result |
|---|---|---|
| **Fraud Risk Agent** | Dataset | 15,420 claims, 5.99% fraud rate |
| | Recall (fraud class) | 72% |
| | Precision (fraud class) | 17% |
| | ROC-AUC | 0.824 |
| | Design choice | Tuned for recall over precision — missing fraud costs more than a false alarm a human reviews |
| **Damage Assessment Agent** | Dataset | 1,383 training / 248 validation images, balanced across 3 classes |
| | Overall accuracy | 67% (vs. 33% random baseline) |
| | Minor damage | Precision 0.70, Recall 0.79 |
| | Moderate damage | Precision 0.49, Recall 0.51 |
| | Severe damage | Precision 0.81, Recall 0.69 |
| | Key finding | "Moderate" is hardest to classify — it sits between two more visually distinct extremes, matching real-world ambiguity |
| **Policy RAG Agent** | Test approach | Manually verified against cross-document comparison questions (e.g., differing state-specific hit-and-run reporting windows) |
| | Result | Correctly retrieved and cited the right policy section in all tested cases |

---

## ⚠️ Known Limitations (honest, by design)

- **Damage model uses a frozen pretrained backbone**, not a fully fine-tuned CNN — a deliberate scope choice for a fast build; full fine-tuning is a natural next step for higher accuracy.
- **Coverage-decision logic uses keyword matching on free-text LLM output**, not structured output. This is fragile by nature — during testing, this surfaced two real edge cases (a hedged-but-covered answer being misread as a denial, and an "I don't know" answer being misread as coverage confirmation) that were found and fixed, but keyword matching on natural language will always have edge cases a structured output schema would avoid.
- **Fraud model precision is low (17%)** at its current threshold — a deliberate trade-off for higher recall, but means many false alarms would need human review in a real deployment.
- **The simplified 8-field fraud input** (used in the UI) fills the remaining ~22 model features with fixed defaults rather than collecting them all from the user — a usability trade-off for the demo.

---

## 🛠️ Tech Stack

Python, FastAPI, SQLite, Streamlit, LangChain, Google Gemini API, FAISS, Scikit-learn, PyTorch, torchvision

---

## 🚀 Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/SimpiSneha/ClaimIQ.git
cd ClaimIQ

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
cp .env.example .env
# edit .env and add your Gemini API key (get one free at aistudio.google.com)

# 4. Build the policy document index (one-time)
cd back_end
python build_index.py

# 5. Train the fraud model (one-time - requires data/fraud_data/fraud_oracle.csv)
python train_fraud_model.py

# 6. Train the damage model (one-time - requires the Car Damage Severity Dataset)
python train_damage_model.py

# 7. Start the backend (note: KMP flag needed due to a PyTorch/Scikit-learn OpenMP conflict on macOS)
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --reload

# 8. In a separate terminal, start the frontend
cd ../front_end
streamlit run app.py
```

Then open `http://localhost:8501`, submit a claim with a photo, and click "Process Claim with AI."

---

## 🐛 Real Engineering Challenges Solved

This project involved genuine debugging, not just following a tutorial:

- **Multiple Gemini API model deprecations mid-build** — `text-embedding-004` and `gemini-2.0-flash` were both retired by Google during development; migrated to `gemini-embedding-001` and `gemini-3.6-flash`.
- **LangChain package restructuring** — `text_splitter` moved to a standalone `langchain-text-splitters` package.
- **PyTorch/Scikit-learn OpenMP runtime conflict** — both libraries bundle their own copy of the OpenMP runtime; running both models in the same process caused a crash on macOS, resolved via the documented `KMP_DUPLICATE_LIB_OK=TRUE` workaround.
- **Orchestrator false-positive/false-negative bugs** — naive keyword matching on the RAG agent's free-text answers initially misclassified a hedged-but-covered answer as a denial, and separately misread an "I don't know" answer as confirmed coverage (because it restated the question using positive-sounding words). Fixed by tightening negative-phrase matching and prioritizing explicit uncertainty signals above all other phrase checks.

---

## 📁 Project Structure

```
ClaimIQ/
├── back_end/
│   ├── main.py                 # FastAPI app & all endpoints
│   ├── database.py             # SQLite schema & connection
│   ├── build_index.py          # Builds the policy document FAISS index
│   ├── rag_agent.py            # Policy RAG agent
│   ├── train_fraud_model.py    # Fraud model training script
│   ├── fraud_agent.py          # Fraud agent inference
│   ├── train_damage_model.py   # Damage model training script
│   ├── damage_agent.py         # Damage agent inference
│   └── orchestrator.py         # Rule-based decision engine
├── front_end/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── policy_docs/             # Source policy documents
│   ├── fraud_data/              # Fraud training CSV
│   └── Car Damage Severity Dataset/  # Damage training images
├── requirements.txt
└── .env.example
```

---

## 📌 Future Improvements

- Fully fine-tune the damage classifier rather than using a frozen backbone
- Replace keyword-based coverage decision logic with structured LLM output (forced JSON schema) for more robust denial/approval detection
- Expand the fraud input form to capture more of the model's full feature set from the user, rather than relying on defaults
- Add LangMem for session memory across repeat claimant interactions
- Add a reranking step to the RAG retrieval pipeline for improved accuracy on ambiguous questions
