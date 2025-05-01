# Smart Docs

**Smart Docs** is an advanced AI-powered personal assistant designed to help users process, manage, and query information from multiple sources such as PDFs, images, and notes. It leverages powerful tools like **LangChain** and **FAISS** to enable efficient document understanding and retrieval.

---

## Features

- **PDF Processing**: Upload and extract text from PDFs and store them as vector embeddings for fast semantic retrieval.
- **Image OCR**: Extract text from image files using Tesseract OCR.
- **Notes Management**: Add, tag, and persistently store personal notes.
- **AI-Powered Q&A**: Ask questions about your documents, notes, or images and receive contextual answers.
- **Summarization**: Generate concise, meaningful summaries of long documents.
- **Analytics**: Visualize document usage patterns and insights with interactive charts.
- **Customizable Queries**: Choose between general queries, summarization, and fact-checking modes.

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Streamlit](https://streamlit.io/)
- [Pillow](https://python-pillow.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

##  Installation

### Prerequisites

Make sure you have **Python 3.8+** and **pip** installed.

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/chethanv-20/smart-docs.git
cd smart-docs

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run main.py
