from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor


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

    # Base MMR retriever for specific questions
    # fetch_k=10 means: first fetch 10 candidates from FAISS
    # k=4 means: from those 10, pick the 4 most diverse ones
    # lambda_mult=0.7 means: 70% relevance, 30% diversity in the selection
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7},
    )

    # Multi-query retriever wraps the base retriever.
    # When invoked with a question, it:
    #   1. Sends the question to the LLM and asks it to generate
    #      3 alternative versions of the question
    #   2. Runs a FAISS search for each of the 3 variants
    #   3. Merges and deduplicates all results
    # This means we search FAISS 3 times instead of once,
    # catching chunks that match different phrasings of the same question.
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    # Contextual compression wraps the multi-query retriever.
    # After multi-query fetches chunks, LLMChainExtractor reads each chunk
    # and extracts ONLY the sentences relevant to the question.
    # e.g. a 1000-char chunk becomes a 150-char extract.
    # This reduces noise in the prompt and lowers token usage.
    #
    # The retriever stack is now:
    # ContextualCompressionRetriever
    #   └──► MultiQueryRetriever        (3 query variants)
    #           └──► base_retriever     (MMR, k=4, fetch_k=10)
    compressor = LLMChainExtractor.from_llm(llm)
    standard_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever,
    )

    # Summary retriever: fetches more chunks to cover the whole video
    # We don't wrap this in MultiQueryRetriever because summary questions
    # already retrieve k=15 chunks — adding multi-query on top would be
    # too many tokens and too slow for a broad "what is this video about?" question
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

        For specific questions, we try compression first.
        If compression returns empty docs (100% reduction case), we fall back
        to uncompressed multi-query results so the LLM always has context.
        """
        if is_summary_question(question):
            return summary_retriever.invoke(question)

        # Try compressed retrieval first
        compressed_docs = standard_retriever.invoke(question)

        # Fallback: if compressor filtered everything out, use raw multi-query results
        if not compressed_docs:
            return multi_query_retriever.invoke(question)

        return compressed_docs

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
