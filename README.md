# 🔍 FAISS Semantic Search Engine

### *Efficient Semantic Retrieval using Sentence Transformers & FAISS*

---
<img width="1439" height="858" alt="Screenshot 2026-04-25 at 9 40 55 PM" src="https://github.com/user-attachments/assets/0928cf71-f20a-4305-8060-bed5aa96df3d" />


## 🧠 Abstract

This project implements a **Semantic Search Engine** that retrieves the most relevant results based on **meaning (semantic similarity)** rather than keyword matching. It leverages **Sentence Transformers** to generate dense vector embeddings and **FAISS (Facebook AI Similarity Search)** for fast nearest-neighbor search.

The system enables intelligent question-answer retrieval from unstructured text data, making it suitable for **chatbots, knowledge bases, and enterprise search systems**.

---

## 🚀 Overview

Traditional search relies on keyword matching, which fails to capture context. This project solves that by:

* Converting text into **vector embeddings**
* Using **vector similarity search** to retrieve relevant answers
* Providing a **real-time interactive UI** using Streamlit

---

## 🎯 Features

* 📄 Upload custom Q&A dataset
* 🧠 Semantic understanding using embeddings
* ⚡ Fast retrieval with FAISS indexing
* 🎛️ Adjustable number of results
* 💬 Natural language query input
* 🎨 Clean and modern UI

---

## 🏗️ System Architecture

```text id="arch_faiss"
User Query → Embedding Model → FAISS Index → Similarity Search → Ranked Results
```

---

## ⚙️ Tech Stack

* **Language:** Python
* **Libraries:**

  * Sentence Transformers
  * FAISS
  * NumPy
  * Streamlit
* **Model:** `paraphrase-MiniLM-L3-v2`

---

## 📂 Project Structure

```bash id="proj_faiss"
faiss-semantic-search-engine/
│
├── app.py                  # Main Streamlit app
├── Employee.txt            # Sample dataset (Q&A pairs)
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

### 1. Document Processing

* Input text file is split into Q&A pairs
* Each pair is treated as a searchable document

---

### 2. Embedding Generation

```python id="emb1"
model.encode(texts)
```

* Converts text into dense vectors
* Captures semantic meaning

---

### 3. FAISS Index Creation

```python id="faiss1"
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

* Stores vectors efficiently
* Enables fast similarity search

---

### 4. Query Retrieval

```python id="ret1"
distances, indices = index.search(query_embedding, top_k)
```

* Finds closest vectors
* Returns most relevant results

---

## 📊 Example Dataset

Sample Q&A from dataset:

* Q: *What is the office working time?*
  A: Office hours are from 9:00 AM to 6:00 PM

* Q: *Is paternity leave available?*
  A: Yes, 10 days of paternity leave is provided

---

## ▶️ Run the Application

### 1. Clone repo

```bash id="runf1"
git clone https://github.com/zebaAkther/faiss-semantic-search-engine.git
cd faiss-semantic-search-engine
```

---

### 2. Install dependencies

```bash id="runf2"
pip install -r requirements.txt
```

---

### 3. Run app

```bash id="runf3"
streamlit run app.py
```

---

### 4. Open browser

```text id="runf4"
http://localhost:8501
```

---

## 📸 Demo

### 🔹 Semantic Search UI

![Semantic Search](./assets/demo.png)

---

## 📈 Key Insights

* Semantic search outperforms keyword-based search
* Sentence embeddings capture contextual meaning
* FAISS enables efficient large-scale retrieval

---

## ⚠️ Limitations

* Uses L2 distance (can improve with cosine similarity)
* Works best with well-structured datasets
* No persistent storage (index recreated each run)

---

## 🔮 Future Enhancements

* 🔹 Integrate Pinecone for scalable vector storage
* 🔹 Add LLM-based answer generation (RAG)
* 🔹 Support PDF/CSV ingestion
* 🔹 Improve ranking with hybrid search

---

## 🧠 Learning Outcomes

* Vector embeddings and semantic similarity
* Efficient indexing using FAISS
* Building real-time ML applications
* Designing retrieval-based systems

---

## 👩‍💻 Author

**Zeba Akther**
🔗 GitHub: https://github.com/zebaAkther

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐!

---
