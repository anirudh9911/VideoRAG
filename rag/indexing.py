import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# All FAISS indexes will be saved inside this folder
INDEX_DIR = "indexes"


def get_index_path(video_id: str) -> str:
    """
    Returns the folder path where a video's FAISS index is saved.
    e.g. indexes/abc123
    Each video gets its own subfolder so indexes don't overwrite each other.
    """
    return os.path.join(INDEX_DIR, video_id)


def index_exists(video_id: str) -> bool:
    """
    Checks if a FAISS index already exists on disk for this video.
    FAISS saves two files: index.faiss and index.pkl
    We check for the folder to confirm both exist.
    """
    return os.path.exists(get_index_path(video_id))


def build_and_save_index(video_id: str, chunks: list[Document]) -> FAISS:
    """
    Creates embeddings for all chunks and saves the FAISS index to disk.

    What are embeddings?
    Each chunk of text gets converted into a list of numbers (a vector)
    that captures its meaning. Similar chunks end up with similar vectors.
    This is what allows semantic search - finding chunks by meaning,
    not just keyword matching.

    We use text-embedding-3-small because:
    - It's cheap (cheaper than text-embedding-3-large)
    - It's fast
    - Quality is sufficient for RAG on conversational text
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # FAISS.from_documents does two things:
    # 1. Calls OpenAI API to embed every chunk
    # 2. Builds the FAISS index in memory
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Now save it to disk so we never embed this video again
    os.makedirs(INDEX_DIR, exist_ok=True)
    vector_store.save_local(get_index_path(video_id))

    print(f"Index saved for video: {video_id}")
    return vector_store


def load_index(video_id: str) -> FAISS:
    """
    Loads a previously saved FAISS index from disk.

    allow_dangerous_deserialization=True is required by LangChain
    because FAISS uses pickle (.pkl) files internally.
    Pickle files can execute arbitrary code if tampered with,
    so LangChain forces you to explicitly opt in.
    Since we are the ones saving these files, it is safe.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.load_local(
        get_index_path(video_id),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print(f"Index loaded from disk for video: {video_id}")
    return vector_store


def get_or_create_index(video_id: str, chunks: list[Document]) -> FAISS:
    """
    The main function the rest of the app calls.

    Logic:
    - If an index exists on disk for this video -> load it (free, fast)
    - If not -> build it by calling OpenAI and save it (costs API call, once)

    This means the first time you process a video it costs API credits.
    Every time after that it is free.
    """
    if index_exists(video_id):
        return load_index(video_id)

    return build_and_save_index(video_id, chunks)
