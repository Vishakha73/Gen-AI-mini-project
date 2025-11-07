!pip install sentence-transformers faiss-cpu PyPDF2 gradio

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re
import gradio as gr
import traceback
import html
import string

STOPWORDS = {
    "a","an","the","and","or","but","if","while","at","by","for","with","about","against",
    "between","into","through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more","most","other","some","such","no",
    "nor","not","only","own","same","so","than","too","very","can","will","just","should","could",
    "would","may","might","must","do","does","did","is","are","was","were","be","being","been","have",
    "has","had","i","you","he","she","it","we","they","them","their","our","us","me","my","mine","your"
}

QUESTION_WORDS = {"what","who","when","where","why","how","list","define","give","show","which","mention","summarize","tell"}

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

def split_text_by_sentences(text, max_length=400):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        # normalize whitespace
        sentence = " ".join(sentence.split())
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def normalize_text(s):
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))
    s = " ".join(s.split())
    return s

def tokenize_and_filter(s):
    s = normalize_text(s)
    tokens = [t for t in s.split() if t not in STOPWORDS and len(t)>1]
    return tokens

def looks_like_question(q):
    qn = normalize_text(q)
    # If contains any question word -> it's a question
    for w in QUESTION_WORDS:
        if re.search(r"\b" + re.escape(w) + r"\b", qn):
            return True
    # if ends with a question mark
    if q.strip().endswith("?"):
        return True
    # otherwise treat as keyword phrase
    return False

model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
chunks = []
embeddings = None

def process_pdf(pdf_file):
    """Extract text, chunk, build embeddings and FAISS index."""
    global index, chunks, embeddings

    if pdf_file is None:
        return "⚠️ Please upload a PDF file first."

    try:
        pdf_path = pdf_file.name
        text = extract_text_from_pdf(pdf_path)

        if not text:
            return "❌ No extractable text found. The PDF may be scanned images (use OCR)."

        chunks = split_text_by_sentences(text, max_length=400)
        if len(chunks) == 0:
            return "❌ No text chunks were produced from the PDF."

        # Create embeddings
        embeddings = model.encode(chunks, show_progress_bar=True)
        dim = embeddings.shape[1]

        # Build FAISS index
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        return f"✅ PDF processed successfully! Total chunks: {len(chunks)}"

    except Exception as e:
        return f"❌ Error processing PDF:\n{traceback.format_exc()}"

def keyword_search(query, top_k=3):
    """Return chunks that match the query by keyword rules."""
    global chunks
    if not chunks:
        return []

    q_norm = normalize_text(query)
    tokens = tokenize_and_filter(query)
    results = []

    # 1) Exact substring match (case-insensitive) - high priority
    for i, c in enumerate(chunks):
        if q_norm in normalize_text(c):
            results.append((i, "substring"))

    # 2) All-token match: chunk contains every token (order not necessary)
    if tokens:
        for i, c in enumerate(chunks):
            c_norm = normalize_text(c)
            has_all = True
            for t in tokens:
                if re.search(r"\b" + re.escape(t) + r"\b", c_norm) is None:
                    has_all = False
                    break
            if has_all:
                # avoid duplicate index if already in results
                if all(r[0] != i for r in results):
                    results.append((i, "token_all"))

    # 3) Partial token match (at least one token), rank by number of matched tokens
    partial = []
    for i, c in enumerate(chunks):
        c_norm = normalize_text(c)
        match_count = sum(1 for t in tokens if re.search(r"\b" + re.escape(t) + r"\b", c_norm))
        if match_count > 0:
            if all(r[0] != i for r in results):
                partial.append((i, match_count))
    # sort partial by descending match_count
    partial_sorted = sorted(partial, key=lambda x: -x[1])
    for i, mc in partial_sorted:
        if all(r[0] != i for r in results):
            results.append((i, "partial"))

    # Limit to top_k unique chunk indices
    out = []
    seen = set()
    for r in results:
        if r[0] not in seen:
            out.append((r[0], r[1]))
            seen.add(r[0])
        if len(out) >= top_k:
            break
    return out

def query_pdf(user_query, top_k=3, semantic_threshold=0.45):

    global index, chunks, embeddings
    if index is None or len(chunks) == 0:
        return "⚠️ Please upload and process a PDF first."

    if not user_query or not user_query.strip():
        return "⚠️ Please enter a question or keywords."

    try:
        # Normalize query
        q = user_query.strip()

        # 1️⃣ Encode query and search semantically
        query_vec = model.encode([q])
        D, I = index.search(np.array(query_vec), k=top_k)
        semantic_results = [(idx, 1 / (1 + dist)) for dist, idx in zip(D[0], I[0])]

        # 2️⃣ Get top result
        best_idx, best_score = semantic_results[0]
        best_chunk = chunks[best_idx].strip()

        # 3️⃣ If score too low, try keyword search
        if best_score < semantic_threshold:
            kw_matches = keyword_search(q, top_k=1)
            if kw_matches:
                best_idx, _ = kw_matches[0]
                best_chunk = chunks[best_idx].strip()
                method = "Keyword (Fallback)"
            else:
                # No match found at all
                return "❌ No relevant answer found for your query. Please try rephrasing or check the PDF content."
        else:
            method = "Semantic"

        # 4️⃣ Clean answer text
        cleaned = re.sub(r"\.{3,}", ".", best_chunk)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"(\n|\r)+", " ", cleaned)
        cleaned = re.sub(r"Page\s*\d+|Figure\s*\d+", "", cleaned, flags=re.IGNORECASE)

        # 5️⃣ If the cleaned answer is too short (e.g., 5 words), treat as no valid result
        if len(cleaned.split()) < 5:
            return "❌ No meaningful answer found for your query. Try using different words."

        # 6️⃣ Return formatted result
        return f"**Method Used:** {method}\n**Confidence Score:** {best_score:.2f}\n\n📄 **Answer:**\n{cleaned}"

    except Exception as e:
        import traceback
        return f"❌ Error while searching:\n{traceback.format_exc()}"

# ---------------- GRADIO FRONTEND ----------------
def make_demo():
    with gr.Blocks(title="Smart PDF Assistant") as demo:
        gr.Markdown(
            """
            # 🧠 Smart PDF Assistant
            Upload a PDF → Click **Process PDF** → Ask questions or type keywords.

            **Notes:**
            - Ask full questions (e.g., "What are the main topics discussed?") for best results.
            - Short phrases (e.g., "safety rules" or "project objectives") will also work using keyword matching.
            """
        )

        with gr.Row():
            pdf_input = gr.File(label="📂 Upload PDF (text-based)")
            process_btn = gr.Button("⚙️ Process PDF")

        process_output = gr.Textbox(label="Processing Status", interactive=False)

        with gr.Row():
            query_input = gr.Textbox(
                label="🔍 Ask a Question or Type Keywords",
                placeholder="e.g., What are the main points? OR safety rules"
            )
            query_btn = gr.Button("💬 Get Answer")

        query_output = gr.Textbox(label="📄 Answer", lines=12)

        # Connect buttons to functions
        process_btn.click(process_pdf, inputs=pdf_input, outputs=process_output)
        query_btn.click(query_pdf, inputs=query_input, outputs=query_output)

    return demo

demo = make_demo()
demo.launch(share=True)
