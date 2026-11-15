import streamlit as st
import fitz
import faiss
import numpy as np
import json
from google import genai
from sentence_transformers import SentenceTransformer

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
# Load chat history from JSON file
def load_chat_history():
    try:
        with open("chat_history.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
# Save chat history to JSON file
def save_chat_history(history):
    with open("chat_history.json", "w") as file:
        json.dump(history, file, indent=4)
    
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
# Define the state of our Agentic RAG chatbot
class RAGState(TypedDict):
    question: str
    relevant_chunks: list
    max_score: float
    answer: str



# Node 2: Check retrieved information
def check_relevance(state: RAGState):

    if state["max_score"] >= 0.25:
        return "generate_answer"

    else:
        return "no_relevant_information"

# Node 3: Generate answer using Gemini
def generate_answer(state: RAGState):

    context = ""

    for chunk in state["relevant_chunks"]:

        context += (
            f"\nDocument: {chunk['file_name']}\n"
            f"Page: {chunk['page']}\n"
            f"{chunk['text']}\n"
        )

    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    prompt = f"""
    You are DocuMind AI, an intelligent document assistant.

    Answer the user's question using ONLY the context.

    Instructions:
    1. Use simple English.
    2. Give a clear answer.
    3. Do not invent information.
    4. Mention the source page numbers.

    Document Context:
    {context}

    User Question:
    {state['question']}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return {
        "answer": response.text or "No answer was generated."
    }


# Build the Agentic RAG workflow
def create_rag_graph(index, all_chunks, model):

    # Node 1: Retrieve relevant document chunks
    def retrieve_documents(state: RAGState):

        question = state["question"]

        question_embedding = model.encode(
            [question],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(question_embedding)

        number_of_results = min(5, index.ntotal)

        scores, results = index.search(
            question_embedding,
            number_of_results
        )

        relevant_chunks = []

        for rank, result in enumerate(results[0]):

            if result == -1:
                continue

            similarity = float(scores[0][rank])

            relevant_chunks.append({
                "file_name": all_chunks[result]["file_name"],
                "page": all_chunks[result]["page"],
                "text": all_chunks[result]["text"],
                "score": similarity
            })

        max_score = (
            max(chunk["score"] for chunk in relevant_chunks)
            if relevant_chunks else 0.0
        )

        return {
            "relevant_chunks": relevant_chunks,
            "max_score": max_score
        }

    graph_builder = StateGraph(RAGState)

    graph_builder.add_node(
        "retrieve",
        retrieve_documents
    )

    graph_builder.add_node(
        "generate_answer",
        generate_answer
    )

    graph_builder.add_edge(
        START,
        "retrieve"
    )

    graph_builder.add_conditional_edges(
        "retrieve",
        check_relevance,
        {
            "generate_answer": "generate_answer",
            "no_relevant_information": END
        }
    )

    graph_builder.add_edge(
        "generate_answer",
        END
    )

    return graph_builder.compile()

    # Add workflow nodes
    graph_builder.add_node(
        "retrieve",
        retrieve_documents
    )

    graph_builder.add_node(
        "generate_answer",
        generate_answer
    )

    # Connect the nodes
    graph_builder.add_edge(
        START,
        "retrieve"
    )

    # Add conditional decision
    graph_builder.add_conditional_edges(
        "retrieve",
        check_relevance,
        {
            "generate_answer": "generate_answer",
            "no_relevant_information": END
        }
    )

    graph_builder.add_edge(
        "generate_answer",
        END
    )

    return graph_builder.compile()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)
# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

st.title("📚 DocuMind AI")
st.subheader("Intelligent Educational Document Assistant")

st.write("Upload your college documents and explore their contents.")

st.divider()


# Function to divide text into smaller chunks
def split_text(text, chunk_size=1000, overlap=150):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# PDF upload section
st.header("📂 Upload Your Documents")

uploaded_files = st.file_uploader(
    "Choose your college PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF document(s) uploaded successfully!"
    )

    all_chunks = []

    # Process each uploaded PDF
    for uploaded_file in uploaded_files:

        document = fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        )

        st.write("Document:", uploaded_file.name)
        st.write("Total Pages:", len(document))

        # Extract text page by page
        for page_number, page in enumerate(document):

            page_text = page.get_text("text")

            with st.expander(
                f"{uploaded_file.name} - Page {page_number + 1}"
            ):

                if page_text.strip():

                    st.write(page_text)

                    page_chunks = split_text(page_text)

                    for chunk in page_chunks:

                        all_chunks.append({
                            "file_name": uploaded_file.name,
                            "page": page_number + 1,
                            "text": chunk
                        })

                else:
                    st.warning("No readable text found.")

        document.close()

    # Display chunking results
    st.divider()

    st.header("🧩 Text Chunking Results")

    st.success(
        f"Successfully created {len(all_chunks)} chunks!"
    )

    st.write("Each chunk contains a smaller section of your document.")

    for index, chunk in enumerate(all_chunks):

        with st.expander(f"Chunk {index + 1}"):

            st.write("Page:", chunk["page"])
            st.write(chunk["text"])

            # Generate embeddings
    st.divider()
    st.header("🧠 Embeddings and Vector Database")

    if all_chunks:

        with st.spinner("Loading AI embedding model..."):

            model = load_embedding_model()

        chunk_texts = [
            chunk["text"] for chunk in all_chunks
        ]

        with st.spinner("Generating embeddings..."):

            embeddings = model.encode(
                chunk_texts,
                convert_to_numpy=True
            ).astype("float32")

        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(embeddings)

        st.success(
            f"Successfully stored {index.ntotal} vectors in FAISS!"
        )

        st.write("Vector dimensions:", embeddings.shape[1])

        
        
        
        # Display previous conversations
        st.divider()

        if st.session_state.chat_history:
            st.subheader("🕘 Chat History")

            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                save_chat_history([])
                st.success("Chat history cleared!")
                st.rerun()

            for chat in st.session_state.chat_history:
                with st.expander(f"Q: {chat['question']}"):
                    st.write("**Answer:**")
                    st.write(chat["answer"])

                    st.write("**Sources:**")
                    for source in chat["sources"]:
                        st.caption(
                            f"{source['file_name']} - Page {source['page']}"
                        )

        # Document Summarization
        st.divider()
        st.header("📝 Document Summarization")

        document_names = sorted(
            set(chunk["file_name"] for chunk in all_chunks)
        )

        selected_document = st.selectbox(
            "Choose a PDF to summarize:",
            document_names
        )

        if st.button("Summarize Uploaded Documents"):
            try:
                with st.spinner("Gemini is summarizing..."):

                    document_text = ""

                    for chunk in all_chunks:
                        if chunk["file_name"] == selected_document:
                            document_text += (
                                f"\nPage: {chunk['page']}\n"
                                f"{chunk['text']}\n"
                            )

                    document_text = document_text[:30000]

                    prompt = f"""
                    Summarize this college document in simple English.

                    Include:
                    1. Introduction
                    2. Main Topics
                    3. Important Points
                    4. Conclusion

                    Document Content:
                    {document_text}
                    """

                    client = genai.Client(
                        api_key=st.secrets["GEMINI_API_KEY"]
                    )

                    response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=prompt
                    )

                st.subheader("📄 Document Summary")
                st.write(response.text)

                st.download_button(
                    label="⬇️ Download Summary",
                    data=response.text,
                    file_name="document_summary.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Summarization error: {e}")

        # Ask Your Document
        st.divider()
        st.header("💬 Ask Your Document")

        with st.form("question_form"):
            question = st.text_input(
                "Enter your question about the PDF:"
            )

            submitted = st.form_submit_button("Ask DocuMind AI")

        if submitted and question:
            try:
                with st.spinner("DocuMind AI is thinking..."):

                    rag_app = create_rag_graph(
                        index,
                        all_chunks,
                        model
                    )

                    result = rag_app.invoke({
                        "question": question,
                        "relevant_chunks": [],
                        "max_score": 0.0,
                        "answer": ""
                    })

                if result["relevant_chunks"] and result["max_score"] >= 0.25:

                    st.subheader("🔍 Relevant Document Sections")

                    for rank, chunk in enumerate(result["relevant_chunks"]):
                        st.write(
                            f"Result {rank + 1} | "
                            f"Similarity: {chunk['score']:.3f}"
                        )

                        st.write(
                            "Document:", chunk["file_name"],
                            "| Source Page:", chunk["page"]
                        )

                        st.info(chunk["text"])

                    st.divider()
                    st.subheader("🤖 DocuMind AI Answer")
                    st.write(result["answer"])

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": [
                            {
                                "file_name": chunk["file_name"],
                                "page": chunk["page"]
                            }
                            for chunk in result["relevant_chunks"]
                        ]
                    })

                    save_chat_history(st.session_state.chat_history)

                else:
                    st.warning(
                        "No sufficiently relevant information was found."
                    )

            except Exception as e:
                st.error(f"Agentic RAG error: {e}")

    else:
        st.info("Please upload a PDF document.")
