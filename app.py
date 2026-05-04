import streamlit as st
import time
from rag_pipeline import load_pdf, split_text, create_vector_store, create_qa_chain

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    st.title("Settings")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        st.success("PDF uploaded")

        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                docs = load_pdf(uploaded_file)
                chunks = split_text(docs)
                vectorstore = create_vector_store(chunks)
                qa_chain = create_qa_chain(vectorstore)

                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = qa_chain

            st.success(f"Indexed {len(chunks)} chunks")

    st.divider()

    st.subheader("System Info")
    if st.session_state.vectorstore:
        st.write("Vector DB: Ready")
    else:
        st.write("Vector DB: Not initialized")

st.title("Intelligent RAG Assistant")
st.caption("Ask questions from your document using Retrieval-Augmented Generation")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])


if prompt := st.chat_input("Ask a question about your document..."):

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.qa_chain:
            st.warning("Please upload and process a document first.")
        else:
            with st.spinner("Thinking..."):
                start = time.time()

                response = st.session_state.qa_chain(prompt)

                end = time.time()

                answer = response["result"]
                sources = response["source_documents"]

                clean_answer = answer.strip()

                st.markdown(clean_answer)
                st.caption(f"⏱ Response time: {round(end - start, 2)} sec")

                with st.expander("Sources"):
                    for i, doc in enumerate(sources):
                        cleaned = clean_text(doc.page_content[:400])
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(cleaned)

        if st.session_state.qa_chain:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": clean_answer
            })