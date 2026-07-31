import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

load_dotenv()

st.title("PDF ChatBot")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

question = st.text_input("Enter your question")
button = st.button("Ask")

if uploaded_file:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":2}
    )

    prompt = PromptTemplate(
        template="""
            You are a helpful assistant.

            Answer ONLY from the following context.

            Context:
            {context}

            Question:
            {question}

            If the answer is not present in the context, say:
            "I don't know."
            """,
        input_variables=["context", "question"]
    )

    model = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    parser = StrOutputParser()

    # Convert retrieved docs into text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | model
        | parser
    )

    if button:

        if question.strip():

            result = chain.invoke(question)

            st.subheader("Answer")
            st.write(result)

        else:
            st.warning("Please enter a question.")