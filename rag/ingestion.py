from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a full YouTube URL.
    Handles formats like:
      - https://www.youtube.com/watch?v=abc123
      - https://youtu.be/abc123
    """
    parsed = urlparse(url)

    if parsed.netloc == "youtu.be":
        # youtu.be/abc123 → path is "/abc123"
        return parsed.path.lstrip("/")

    # youtube.com/watch?v=abc123 → query string has v=abc123
    query_params = parse_qs(parsed.query)
    video_id = query_params.get("v", [None])[0]

    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    return video_id


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetches the raw transcript from YouTube.
    Returns a list of dicts: [{text, start, duration}, ...]
    The 'start' field is the timestamp in seconds.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["en"])
        return transcript.to_raw_data()
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")


def build_chunks(transcript: list[dict]) -> list[Document]:
    """
    Converts raw transcript into LangChain Documents with metadata.

    Why metadata?
    Each chunk gets the timestamp of where it starts in the video.
    This lets us later say "this answer comes from 4:32 in the video."

    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraphs first, then sentences, then words.
    This keeps related sentences together better than splitting blindly.
    chunk_size=1000  → each chunk is ~1000 characters
    chunk_overlap=200 → chunks share 200 characters with the next one
                        so context isn't lost at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Join all transcript text into one string first
    # but track the start time of each snippet
    full_text = " ".join(snippet["text"] for snippet in transcript)

    # Create a single Document with the full text
    # We'll add per-chunk timestamps after splitting
    raw_docs = splitter.create_documents([full_text])

    # Now attach timestamps: find which transcript snippet
    # corresponds to the start character of each chunk
    chunks_with_metadata = []
    char_index = 0

    for i, doc in enumerate(raw_docs):
        # Estimate which snippet this chunk starts from
        # by tracking character position through the transcript
        running_chars = 0
        start_time = 0.0

        for snippet in transcript:
            running_chars += len(snippet["text"]) + 1  # +1 for the space
            if running_chars >= char_index:
                start_time = snippet["start"]
                break

        char_index += len(doc.page_content) - 200  # account for overlap

        chunks_with_metadata.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "start_time": round(start_time, 2),
                    "chunk_index": i,
                }
            )
        )

    return chunks_with_metadata


def ingest(url: str) -> tuple[str, list[Document]]:
    """
    Master function that orchestrates the full ingestion flow:
    URL → video_id → raw transcript → chunks with metadata

    Returns the video_id and the list of chunks.
    The video_id is returned so we can use it to name the saved index.
    """
    video_id = extract_video_id(url)
    transcript = fetch_transcript(video_id)
    chunks = build_chunks(transcript)
    return video_id, chunks
