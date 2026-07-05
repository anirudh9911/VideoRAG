import streamlit as st
from dotenv import load_dotenv

from rag.ingestion import ingest
from rag.indexing import get_or_create_index
from rag.pipeline import build_chain

load_dotenv()

st.set_page_config(page_title="VideoRAG", page_icon="🎬")
st.title("VideoRAG")
st.caption("Ask questions about any YouTube video")

# --- Session state initialization ---
# Streamlit reruns the entire script on every user interaction.
# st.session_state is a dictionary that persists across reruns.
# We use it to store the chain and chat history so they survive reruns.
if "chain" not in st.session_state:
    st.session_state.chain = None          # the RAG chain for the loaded video

if "video_id" not in st.session_state:
    st.session_state.video_id = None       # tracks which video is currently loaded

if "messages" not in st.session_state:
    st.session_state.messages = []         # chat history: list of {role, content} dicts


# --- Sidebar: video loader ---
# The sidebar is a separate panel on the left side of the screen.
# We put the video URL input here so it doesn't clutter the main chat area.
with st.sidebar:
    st.header("Load a Video")
    url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")

    load_button = st.button("Load Video", use_container_width=True)

    if load_button and url:
        with st.spinner("Fetching transcript and building index..."):
            try:
                # Step 1: extract video_id and get chunks from ingestion.py
                video_id, chunks = ingest(url)

                # Step 2: only reload if it's a different video than what's loaded
                # This avoids rebuilding the chain if the user clicks Load again
                # on the same video
                if video_id != st.session_state.video_id:
                    vector_store = get_or_create_index(video_id, chunks)
                    st.session_state.chain = build_chain(vector_store)
                    st.session_state.video_id = video_id
                    st.session_state.messages = []  # clear chat for new video

                st.success("Video loaded! Start asking questions.")

            except ValueError as e:
                st.error(str(e))

    # Show which video is currently loaded
    if st.session_state.video_id:
        st.info(f"Loaded: `{st.session_state.video_id}`")


# --- Main area: chat interface ---
# Display all previous messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# st.chat_input renders a fixed input box at the bottom of the screen
question = st.chat_input("Ask a question about the video...")

if question:
    # Don't allow questions if no video is loaded yet
    if st.session_state.chain is None:
        st.warning("Please load a YouTube video first using the sidebar.")
    else:
        # Show the user's question in the chat immediately
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get the answer from the RAG chain and display it
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
            st.write(answer)

        # Save the answer to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
