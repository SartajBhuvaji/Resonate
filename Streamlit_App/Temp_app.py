import os

import streamlit as st
import dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# import pinecone
# from langchain_community.vectorstores import Pinecone
# from langchain_community.chat_models import ChatOpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

dotenv.load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

pinecone_index_name = "streamlit-langchain-pinecone-demo-index"
pinecone_index_dimension = 1536
pinecone_index_metric = "cosine"
cloud_type = "aws"
cloud_region = "us-west-2"


def doc_preprocessing():
    loader = DirectoryLoader(
        "data/", glob="**/*.pdf", show_progress=True  # only the PDFs
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs_split = text_splitter.split_documents(docs)
    return docs_split


@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    # Now do stuff
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=pinecone_index_dimension,
            metric=pinecone_index_metric,
            spec=ServerlessSpec(cloud=cloud_type, region=cloud_region),
        )

    docs_split = doc_preprocessing()
    doc_db = Pinecone.from_documents(
        docs_split, embeddings, index_name=pinecone_index_name
    )
    return doc_db


llm = ChatOpenAI()
doc_db = embedding_db()


def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result


def main():
    st.title(
        "Question and Answering App using Streamlit - Langchain - OpenAI - Pinecone"
    )

    text_input = st.text_input("Ask your query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)


if __name__ == "__main__":
    main()
