import os
import time

# import tempfile

import dotenv
import streamlit as st
from IPython.display import HTML, display
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from tqdm.auto import tqdm

from streamlit_app_common_functions import init_pinecone, load_transcript
from streamlit_app_aws_functions import *

# Mapping - PineCone DB to Organization Structure
# Pinecone Project      ==>==>==>==>==>    Organization
# Pinecone Index        ==>==>==>==>==>     Team
# Pinecone Namespace    ==>==>==>==>==>     Topic / Project


# # Loading Transcripts from CSV file
# file_name = "aws_parsed_transcript.csv"
# transcript = load_transcript(file_name)


def upsert_pinecone(pinecone_index, transcript, model_name, pinecone_namespace):
    # Initializing Embedding
    embed = OpenAIEmbeddings(model=model_name)

    batch_limit = 90
    transcript_texts = []
    metadata_records = []
    meeting_id = 1
    start_id = 0

    for i, record in tqdm(transcript.iterrows()):
        # Extracting and Preparing metadata fields for each row of transcript
        metadata = {
            "speaker": record["speaker_label"],
            "start_time": round(record["start_time"], 3),  # limit to 3 decimal places
            "meeting_id": meeting_id,
            "text": record["text"],
        }  # Storing the text in the metadata for now, later we'd need to decode it from vectors

        record_texts = record["text"]

        transcript_texts.append(record_texts)
        metadata_records.append(metadata)

        # if we've reached the batch limit, then index the batch
        if len(transcript_texts) >= batch_limit:
            ids = [
                str(i + 1) for i in range(start_id, (start_id + len(transcript_texts)))
            ]
            start_id += len(transcript_texts)
            embeds = embed.embed_documents(transcript_texts)

            pinecone_index.upsert(
                vectors=zip(ids, embeds, metadata_records), namespace=pinecone_namespace
            )
            transcript_texts = []
            metadata_records = []
            meeting_id += 1

    # add any remaining texts to the index
    if len(transcript_texts) > 0:
        ids = [str(i + 1) for i in range(start_id, (start_id + len(transcript_texts)))]

        embeds = embed.embed_documents(transcript_texts)
        pinecone_index.upsert(
            vectors=zip(ids, embeds, metadata_records), namespace=pinecone_namespace
        )

    time.sleep(5)
    # display(pinecone_index.describe_index_stats()) # Check index stats / data Freshness


# query = "What was talked regarding United States Congress?"
# pincone_response = pinecone_index.query(
#     vector=embed.embed_documents([query])[0],
#     # filter={
#     #     "meeting_id": {"$in":[1, 2]}
#     # },
#     namespace=pinecone_namespace,
#     top_k=10,
#     include_metadata=True,
# )

# display(pincone_response)


# delta = 5
# id = 60

# # build a window of size +- delta of all numbers around id
# window = [str(i) for i in range(id - delta, id + delta + 1)]

# fetch_response = pinecone_index.fetch(ids=window, namespace=pinecone_namespace)
# fetch_response


################################################################


def retrieval_answer(query):
    return True


def main():
    # Loading env variables
    dotenv.load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    # os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    aws_region_name = "us-east-2"
    aws_input_bucket = "resonate-input"
    aws_output_bucket = "resonate-output"
    aws_transcribe_job_name = "resonate-job"

    # Initializing Variables
    pinecone_index_name = "streamlit-langchain-pinecone-demo-index"
    pinecone_index_dimension = 1536
    pinecone_index_metric = "cosine"
    pinecone_cloud_type = "aws"
    pinecone_cloud_region = "us-west-2"
    pinecone_namespace = "meeting_topic"
    pinecone_embedding_model_name = "text-embedding-ada-002"

    uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav"])
    if uploaded_file:
        with st.spinner("Processing..."):
            file = uploaded_file.name
            with open(file, "wb") as f:
                f.write(uploaded_file.getbuffer())
                f.close()

            df_transcript = runner(
                file_name=file,
                input_bucket=aws_input_bucket,
                output_bucket=aws_output_bucket,
                transcribe_job_name=aws_transcribe_job_name,
                aws_access_key=aws_access_key,
                aws_secret_access_key=aws_secret_access_key,
                aws_region_name=aws_region_name,
            )

            st.success("File uploaded and transcribed successfully!")

            # Initializing Pinecone
            pinecone, pinecone_index = init_pinecone(
                pinecone_api_key,
                pinecone_index_name,
                pinecone_index_metric,
                pinecone_index_dimension,
                pinecone_cloud_type,
                pinecone_cloud_region,
            )
            st.success("Pinecone Initialized successfully!")
            upsert_pinecone(
                pinecone_index,
                transcript=df_transcript,
                model_name=pinecone_embedding_model_name,
                pinecone_namespace=pinecone_namespace,
            )
            st.success("Pinecone upsert completed  successfully!")

    # if uploaded_file is not None:
    #     st.title(
    #         "Question and Answering App using Streamlit - Langchain - OpenAI - Pinecone"
    #     )

    #     text_input = st.text_input("Ask your query...")
    #     if st.button("Ask Query"):
    #         if len(text_input) > 0:
    #             st.info("Your Query: " + text_input)
    #             answer = retrieval_answer(text_input)
    #             st.success(answer)


if __name__ == "__main__":
    main()
