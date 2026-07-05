from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS


def format_docs(retrieved_docs):
    """
    Converts a list of Document objects into a single string
    that gets inserted into the prompt as context.

    We also include the timestamp so the LLM can cite it in the answer.
    e.g.
    [At 4:32] "Demis talked about AlphaFold solving protein folding..."
    [At 9:10] "He mentioned that DeepMind was founded in 2010..."
    """
    chunks = []
    for doc in retrieved_docs:
        start_seconds = doc.metadata.get("start_time", 0)

        # Convert seconds to mm:ss format for readability
        minutes = int(start_seconds // 60)
        seconds = int(start_seconds % 60)
        timestamp = f"{minutes}:{seconds:02d}"

        chunks.append(f"[At {timestamp}] {doc.page_content}")

    return "\n\n".join(chunks)


SUMMARY_KEYWORDS = ["summarize", "summary", "what is this video", "what is the video about", "overview", "explain the video", "tell me about the video"]


def is_summary_question(question: str) -> bool:
    """
    Detects if the user is asking for a broad summary of the video.
    Summary questions need more chunks than specific questions.
    """
    return any(keyword in question.lower() for keyword in SUMMARY_KEYWORDS)


def build_chain(vector_store: FAISS):
    """
    Builds and returns the full RAG chain.

    The chain flow:
    user question
        │
        ├──► retriever (finds relevant chunks from FAISS)
        │        │
        │        └──► format_docs (adds timestamps, joins into string)
        │
        ├──► RunnablePassthrough (passes the question through unchanged)
        │
        ▼
    prompt (assembles context + question into a full prompt)
        │
        ▼
    llm (sends prompt to GPT, gets answer back)
        │
        ▼
    parser (extracts the plain string from the LLM response object)
        │
        ▼
    final answer string
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Standard retriever for specific questions: fetches 4 diverse chunks
    # fetch_k=10 means: first fetch 10 candidates from FAISS
    # k=4 means: from those 10, pick the 4 most diverse ones
    # lambda_mult=0.7 means: 70% relevance, 30% diversity in the selection
    standard_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7},
    )

    # Summary retriever: fetches more chunks to cover the whole video
    summary_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.5},
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions about a YouTube video.
Use the transcript context provided below to answer the question.
Base your answer primarily on the context. If the context covers the topic, answer confidently.
Only say you don't know if the topic is genuinely absent from the context.
When referencing specific points, mention the timestamp (e.g. "At 4:32, ...").

Transcript context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    def route_retriever(question: str):
        """
        Picks the right retriever based on the question type.
        Summary questions get more chunks; specific questions get fewer.
        """
        if is_summary_question(question):
            return summary_retriever.invoke(question)
        return standard_retriever.invoke(question)

    # RunnableParallel runs two things at the same time:
    # 1. "context": routes to right retriever, then formats the docs
    # 2. "question": passes the question through unchanged
    # Both outputs are merged into a dict: {"context": "...", "question": "..."}
    # That dict is then passed to the prompt template
    parallel_step = RunnableParallel({
        "context": RunnableLambda(route_retriever) | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    # The full chain: parallel step → prompt → llm → string parser
    chain = parallel_step | prompt | llm | StrOutputParser()

    return chain
