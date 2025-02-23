# 📘 PDF Summarizer and Query Answering with RAG & LangChain

This project uses **Retrieval-Augmented Generation (RAG)** to summarize PDF documents and answer user queries based on the content. It leverages **LangChain**, **HuggingFace Embeddings**, and **Chroma** to create a powerful, context-aware document processing pipeline.

---

## 🚀 **Features**
- 📂 **PDF Loading:** Extract content from PDFs using `PyPDFLoader`.
- ✂️ **Text Splitting:** Split long documents into manageable chunks with `RecursiveCharacterTextSplitter`.
- 🧠 **Vector Embeddings:** Convert document chunks into vector embeddings using **HuggingFace's all-mpnet-base-v2** model.
- 🔍 **Vector Search with Chroma:** Store and query document chunks using **Chroma** as the vector store.
- 🗨️ **Question Answering:** Use a context-aware prompt to generate answers based on the most relevant document chunks.

---

## 🛠️ **Tech Stack**
- **Python**
- **LangChain**
- **Hugging Face Transformers**
- **Chroma**
- **PyPDFLoader**

---

## 📂 **Project Structure**
```
├── chroma_db/                     # Local directory to persist vector store
├── content/practice.pdf           # Sample PDF document
├── main.py                        # Main script to run the application
└── README.md                      # Project documentation
```

---

## ⚡ **Installation**
1. **Clone the repository:**
```bash
git clone https://github.com/palaklohade/pdf_summarizer.git
cd pdf_summarizer
```

2. **Install dependencies:**
```bash
!pip show chromadb
!pip install langchain
!pip install boto3
!pip install chromadb
!pip install pypdf
!pip install pytest
!pip check
!pip install --upgrade langchain transformers sentence-transformers chromadb
```

3. **Download the embedding model:**
```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
```

---

## 🟢 **Usage**

1. **Load and Split Documents:**
```python
from langchain.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document():
    loader = PyPDFLoader("/content/practice.pdf")
    return loader.load()

documents = load_document()
chunks = split_documents(documents)
```

2. **Create and Persist Vector Store:**
```python
from langchain.vectorstores import Chroma

def add_to_chroma(chunks):
    db = Chroma(persist_directory="./chroma_db", embedding_function=get_embedding_function())
    db.add_documents(chunks, ids=[str(i) for i in range(len(chunks))])
    db.persist()
    return db

db = add_to_chroma(chunks)
```

3. **Query the PDF Content:**
```python
query_text = "What is this document about?"
results = db.similarity_search_with_score(query_text, k=3)

context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

Prompt_Template = """
Answer the question based only on the following context:
{context}
___
Answer the question based on the above context: {question}
"""

from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(Prompt_Template)
prompt = prompt_template.format_prompt(context=context_text, question=query_text)
```

---

## 🎯 **Example Output**
```
Query: "What is this document about?"
Answer: "The document is a research paper on natural language processing and its applications."
```

---

## 📈 **Future Enhancements**
- 🟠 Add support for multiple PDF files.
- 🟡 Fine-tune the LLM for domain-specific queries.
- 🟢 Build a web interface for user interaction.

---

## 📄 **License**
This project is licensed under the MIT License.

---

## 👩‍💻 **Author**
**Palak Lohade**  
Third-year Computer Science Student  
Passionate about coding, web development, and AI! 🚀

---

## ⭐ **Contributions**
Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

