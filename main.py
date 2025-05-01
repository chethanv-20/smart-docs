import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import tempfile
import os
import json
import shutil
import datetime
from PIL import Image
import pytesseract
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.prompts import PromptTemplate

# Persistent storage setup
MEMORY_DIR = "memory"
NOTES_FILE = os.path.join(MEMORY_DIR, "notes.json")
SUMMARIES_FILE = os.path.join(MEMORY_DIR, "summaries.json")
ANALYTICS_FILE = os.path.join(MEMORY_DIR, "analytics.csv")
if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

# Notes handling
def load_notes():
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, "r") as f:
                notes = json.load(f)
            # Migrate old string-based notes to new dictionary format
            migrated_notes = []
            for note in notes:
                if isinstance(note, str):
                    migrated_notes.append({
                        "text": note,
                        "tags": [],
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                elif isinstance(note, dict) and "text" in note:
                    migrated_notes.append({
                        "text": note.get("text", ""),
                        "tags": note.get("tags", []),
                        "timestamp": note.get("timestamp", datetime.datetime.now().isoformat())
                    })
                else:
                    continue
            save_notes(migrated_notes)
            return migrated_notes
        except json.JSONDecodeError:
            st.error("Error reading notes file. Starting with empty notes.")
            return []
    return []

def save_notes(notes):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f)

# Summaries handling
def load_summaries():
    if os.path.exists(SUMMARIES_FILE):
        with open(SUMMARIES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_summaries(summaries):
    with open(SUMMARIES_FILE, "w") as f:
        json.dump(summaries, f)

# Analytics tracking
def log_analytics(event_type, details):
    timestamp = datetime.datetime.now().isoformat()
    df = pd.DataFrame([[timestamp, event_type, details]], columns=["timestamp", "event_type", "details"])
    if os.path.exists(ANALYTICS_FILE):
        df.to_csv(ANALYTICS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(ANALYTICS_FILE, index=False)

# PDF and Image processing
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vector_store(chunks, file_key):
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore_path = os.path.join(MEMORY_DIR, f"{file_key}_vectorstore")
    vectorstore.save_local(vectorstore_path)
    return vectorstore_path

def delete_vector_store(file_key):
    vectorstore_path = os.path.join(MEMORY_DIR, f"{file_key}_vectorstore")
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)

# Summarization
def generate_summary(documents):
    llm = Ollama(model="llama3.2")
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in 2-3 sentences, capturing the main ideas:\n\n{text}"
    )
    text = "\n".join([doc.page_content for doc in documents])
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    summaries = []
    for chunk in chunks:
        response = llm(prompt_template.format(text=chunk))
        summaries.append(response)
    return " ".join(summaries)

# Cached vectorstore loading
# Use cache_resource if available, else fallback to cache
cache_decorator = st.cache_resource if hasattr(st, 'cache_resource') else st.cache

@cache_decorator
def get_vectorstores():
    vectorstores = {}
    embeddings = OllamaEmbeddings(model="llama3.2")
    for folder in os.listdir(MEMORY_DIR):
        if folder.endswith("_vectorstore"):
            key = folder.replace("_vectorstore", "")
            vectorstores[key] = FAISS.load_local(
                os.path.join(MEMORY_DIR, folder), embeddings, allow_dangerous_deserialization=True
            )
    return vectorstores

# Notes as a vectorstore
@cache_decorator
def get_note_vectorstore(notes):
    note_texts = [note["text"] if isinstance(note, dict) else note for note in notes]
    if note_texts:
        embeddings = OllamaEmbeddings(model="llama3.2")
        return FAISS.from_texts(note_texts, embeddings)
    return None

# QA chain creation
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama3.2")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# Plot analytics
def plot_analytics():
    if os.path.exists(ANALYTICS_FILE):
        df = pd.read_csv(ANALYTICS_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x="date", hue="event_type")
        plt.title("Usage Analytics Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Advanced AI Assistant", layout="wide")
st.title("Personal Assistant with AI Memory")

notes = load_notes()
summaries = load_summaries()

# Sidebar for notes, PDFs, and analytics
with st.sidebar:
    st.header("🧠 AI Memory & Tools")

    # Add note
    st.subheader("Add Note")
    note = st.text_area("Add a note to memory:")
    tags = st.text_input("Tags (comma-separated):")
    if st.button("Store Note"):
        if note:
            note_entry = {"text": note, "tags": [t.strip() for t in tags.split(",") if t.strip()], "timestamp": datetime.datetime.now().isoformat()}
            notes.append(note_entry)
            save_notes(notes)
            log_analytics("note_added", f"Tags: {tags}")
            st.success("✅ Stored in memory!")
            st.rerun()

    # Upload Image
    st.subheader("Upload Image")
    uploaded_image = st.file_uploader("Upload an image for text extraction", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        extracted_text = extract_text_from_image(image)
        file_key = uploaded_image.name.split(".")[0]
        notes.append({"text": f"[Image Extracted Text] {extracted_text}", "tags": ["image"], "timestamp": datetime.datetime.now().isoformat()})
        save_notes(notes)
        log_analytics("image_processed", file_key)
        st.success(f"✅ Image '{file_key}' processed and text stored!")
        st.rerun()

    # List & manage notes
    if notes:
        st.subheader("Stored Notes:")
        all_tags = set()
        for note in notes:
            if isinstance(note, dict) and "tags" in note:
                all_tags.update(note["tags"])
        tag_filter = st.multiselect("Filter by tags:", list(all_tags))
        for i, note_entry in enumerate(notes):
            if not isinstance(note_entry, dict):
                continue
            if not tag_filter or any(tag in note_entry.get("tags", []) for tag in tag_filter):
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.write(f"{i+1}. {note_entry['text'][:100]}... (Tags: {', '.join(note_entry.get('tags', []))})")
                with col2:
                    if st.button("🗑️", key=f"delete_note_{i}"):
                        notes.pop(i)
                        save_notes(notes)
                        log_analytics("note_deleted", f"Note {i}")
                        st.rerun()

    # List & delete PDFs
    stored_pdfs = [f.replace("_vectorstore", "") for f in os.listdir(MEMORY_DIR) if f.endswith("_vectorstore")]
    if stored_pdfs:
        st.subheader("Stored PDFs:")
        for pdf in stored_pdfs:
            col1, col2, col3 = st.columns([0.7, 0.1, 0.1])
            with col1:
                st.write(f"- {pdf}")
            with col2:
                if st.button("📝", key=f"summary_pdf_{pdf}"):
                    st.session_state["show_summary"] = pdf
            with col3:
                if st.button("🗑️", key=f"delete_pdf_{pdf}"):
                    delete_vector_store(pdf)
                    if pdf in summaries:
                        del summaries[pdf]
                        save_summaries(summaries)
                    log_analytics("pdf_deleted", pdf)
                    st.rerun()

    # Analytics
    st.subheader("Usage Analytics")
    analytics_plot = plot_analytics()
    if analytics_plot:
        st.image(f"data:image/png;base64,{analytics_plot}")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("⏳ Processing PDF...")
    docs = load_pdf(tmp_path)
    chunks = split_documents(docs)
    file_key = uploaded_file.name.split(".pdf")[0]
    create_vector_store(chunks, file_key)
    
    summary = generate_summary(docs)
    summaries[file_key] = summary
    save_summaries(summaries)
    log_analytics("pdf_processed", file_key)
    
    st.success(f"✅ PDF '{file_key}' processed and stored with summary!")
    os.remove(tmp_path)

# Show summary if requested
if "show_summary" in st.session_state:
    pdf_key = st.session_state["show_summary"]
    if pdf_key in summaries:
        st.markdown(f"### 📝 Summary for {pdf_key}")
        st.write(summaries[pdf_key])
        if st.button("Close Summary"):
            del st.session_state["show_summary"]

# Load vectorstores & note vectorstore
vectorstores = get_vectorstores()
note_vectorstore = get_note_vectorstore(notes)

# Main Q&A with advanced query parsing
if vectorstores or notes:
    st.markdown("### 💬 Ask a Question")
    query = st.text_input("Enter your question about stored PDFs, notes, or images:")
    query_type = st.selectbox("Query Type:", ["General", "Summarization", "Fact-Checking"])
    
    if query:
        all_docs = []

        for vs in vectorstores.values():
            retriever = vs.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(query)
            all_docs.extend(docs)

        if note_vectorstore:
            note_retriever = note_vectorstore.as_retriever(search_kwargs={"k": 4})
            note_docs = note_retriever.get_relevant_documents(query)
            all_docs.extend(note_docs)

        if not all_docs:
            st.warning("⚠️ No relevant information found.")
        else:
            temp_vs = FAISS.from_documents(all_docs, OllamaEmbeddings(model="llama3.2"))
            qa_chain = create_qa_chain(temp_vs)

            if query_type == "Summarization":
                query = f"Provide a concise summary of the information related to: {query}"
            elif query_type == "Fact-Checking":
                query = f"Verify the accuracy of the following statement and provide evidence: {query}"

            answer = qa_chain({"query": query})
            st.markdown("### 🤖 Response:")
            st.write(answer["result"])

            with st.expander("📚 Source Chunks Used"):
                for doc in answer["source_documents"]:
                    st.markdown(f"> {doc.page_content[:500]}...")

            log_analytics("query_answered", f"Type: {query_type}, Query: {query[:50]}")
else:
    st.info("Please upload a PDF, image, or add a note to start querying.")