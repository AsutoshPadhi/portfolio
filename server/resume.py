from pathlib import Path

from fastmcp import FastMCP
from datetime import datetime
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")

load_dotenv()

# Cache loaded text so we only read the PDF once
_resume_text: str | None = None
# Vector store (FAISS), populated by index_and_store()
vector_store = None
# Path to your resume PDF (set via env or change default)
RESUME_PDF_PATH = Path(__file__).resolve().parent.parent / "static" / "resume.pdf"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant. Act as the person in the resume.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

parser = StrOutputParser()

# Built in warm_up() at app startup so first query doesn't build the chain
main_chain = None


def load_pdf_text(path: Path | str | None = None) -> str:
    """Load a PDF file and return its text content.

    Args:
        path: Path to the PDF. Defaults to RESUME_PDF_PATH.

    Returns:
        Extracted text from all pages, or empty string if file missing/invalid.
    """
    p = Path(path) if path else RESUME_PDF_PATH
    if not p.exists():
        raise FileNotFoundError(f"Resume PDF not found at: {p}")
    try:
        reader = PdfReader(p)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError(
                "PDF has no extractable text (might be scanned/images). "
                f"Path: {p}"
            )
        return text
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF at {p}: {e}") from e



def get_resume_text() -> str:
    """Return resume text from the PDF, loading it once and caching."""
    global _resume_text
    if _resume_text is None:
        loaded = load_pdf_text()
        # Only cache non-empty text so we don't permanently cache failure
        if loaded:
            _resume_text = loaded
        return loaded
    return _resume_text

def split_text() -> list[str]:
    """Split the text into chunks of 1000 characters."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(get_resume_text())
    return chunks

def index_and_store(chunks: list[str]) -> None:
    """Index and store the chunks in a vector database.

    First run may hang for a while: OpenAI embeddings use tiktoken, which
    downloads ~1MB from the internet on first use. Ensure you have network
    access and wait, or run once with: python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4')"
    to pre-cache the encoding.
    """
    global vector_store
    print("Loading embeddings (first time may download tiktoken data; requires internet)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")


def get_retriever():
    if vector_store is None:
        return "Vector store not loaded. Call index_and_store() first."
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def warm_up() -> None:
    """Load PDF, build FAISS index, and create main_chain (and retriever, parallel_chain).
    Call once at app startup so the chain is ready when the website is loaded.
    """
    global main_chain
    try:
        text = get_resume_text()
    except (FileNotFoundError, ValueError, RuntimeError):
        return
    if not text:
        return
    chunks = split_text()
    index_and_store(chunks)
    retriever = get_retriever()
    if isinstance(retriever, str):
        return
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(formatted_docs),
        "question": RunnablePassthrough(),
    })
    main_chain = parallel_chain | prompt | llm | parser


def answer_query(query: str) -> str:
    """Take the text input from the main page and return an answer.

    Args:
        query: The user's question (from the interaction input field).

    Returns:
        The answer to display in the left-panel-answers area.
    """
    if not query or not query.strip():
        return "Ask me anything about my experience, skills, or background."

    if main_chain is None:
        return "RAG is not ready. Ensure the resume PDF is at static/resume.pdf and the server ran warm_up at startup."

    try:
        return main_chain.invoke(query)
    except Exception as e:
        return f"Error answering: {e}"

def formatted_docs(retrieved_docs: list[str]) -> str:
    context_txt = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_txt


def main():
    try:
        chunks = split_text()
        index_and_store(chunks)
        retriever = get_retriever()

        # Create chain
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(formatted_docs),
            'question': RunnablePassthrough(),
        })

        main_chain = parallel_chain | prompt | llm | parser
        answer = main_chain.invoke("What is your previous company name?")
        print(answer)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()