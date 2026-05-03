import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss                   
import numpy as np
import tempfile
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
.stTextInput > div > div > input {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
}
.stFileUploader {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
}
.result-box {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid #334155;
}
.score {
    color: #38bdf8;
    font-size: 18px;
    font-weight: bold;
}
.title-style {
    text-align: center;
    color: #38bdf8;
    font-size: 42px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL (LIGHTWEIGHT FIX)
# -----------------------------
@st.cache_resource
def load_model():
    # ✅ FIX: lighter model → avoids crash
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")


# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
def generate_embeddings(texts, model):
    embeddings = model.encode(texts)
    return np.array(embeddings).astype("float32")


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    documents = []

    for i in range(0, len(lines), 2):
        question = lines[i]
        answer = lines[i + 1] if i + 1 < len(lines) else ""
        documents.append(f"Q: {question}\nA: {answer}")

    return documents


# -----------------------------
# CREATE FAISS INDEX
# -----------------------------
def create_faiss_index(documents, model):
    embeddings = generate_embeddings(documents, model)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    return index


# -----------------------------
# RETRIEVE DOCUMENTS
# -----------------------------
def retrieve(query, model, index, documents, top_k=3):
    query_embedding = generate_embeddings([query], model)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "document": documents[idx],
            "score": float(distances[0][i])
        })

    return results


# -----------------------------
# MAIN APP
# -----------------------------
def main():

    st.markdown('<div class="title-style">🔍 Semantic Search Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">FAISS + Sentence Transformers</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    top_k = st.sidebar.slider("Number of Results", 1, 10, 3)

    # Upload file
    uploaded_file = st.file_uploader("📄 Upload Q&A text file", type=["txt"])

    if uploaded_file:

        temp_path = Path(tempfile.gettempdir()) / uploaded_file.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Loading model..."):
            model = load_model()

        documents = load_documents_from_file(temp_path)

        st.success(f"✅ Loaded {len(documents)} documents")

        with st.spinner("Creating FAISS index..."):
            index = create_faiss_index(documents, model)

        query = st.text_input("💬 Ask your question")

        if query:
            with st.spinner("Searching..."):
                results = retrieve(query, model, index, documents, top_k)

            st.markdown("## 🔎 Results")

            for r in results:
                st.markdown(f"""
                <div class="result-box">
                    <div class="score">Score: {r['score']:.4f}</div>
                    <br>
                    <div style="white-space: pre-wrap;">
                        {r['document']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("📌 Upload a file to start")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()