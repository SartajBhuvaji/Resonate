import time

import pandas as pd
from IPython.display import HTML, display
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

display(HTML("<style>.container { width:100% !important; }</style>"))
# pd.set_option("display.max_rows", None)


# Loading Transcripts from CSV file
def load_transcript(transcript):
    # transcript = pd.read_csv(file_name)
    transcript.drop(["end_time"], axis=1, inplace=True)

    # Iterate through rows and update the text column
    for i in range(1, len(transcript)):
        if transcript.loc[i, "speaker_label"] == transcript.loc[i - 1, "speaker_label"]:
            transcript.loc[i - 1, "text"] += " " + transcript.loc[i, "text"]
            transcript.drop(index=i, inplace=True)

    # Resetting index after dropping rows
    transcript.reset_index(drop=True, inplace=True)
    # Display the updated dataframe
    # print("Transcript")
    # display(transcript)
    return transcript


# Initializing Pinecone
def init_pinecone(
    pinecone_api_key,
    pinecone_index_name,
    pinecone_index_metric,
    pinecone_index_dimension,
    pinecone_cloud_type,
    pinecone_cloud_region,
):
    pinecone = Pinecone(api_key=pinecone_api_key)

    pinecone_indexes_list = [
        index.get("name") for index in pinecone.list_indexes().get("indexes", [])
    ]

    # # Creating Index if it doesn't exist in Pinecone
    if pinecone_index_name not in pinecone_indexes_list:
        pinecone.create_index(
            name=pinecone_index_name,
            dimension=pinecone_index_dimension,
            metric=pinecone_index_metric,
            spec=ServerlessSpec(
                cloud=pinecone_cloud_type, region=pinecone_cloud_region
            ),
        )

        while not pinecone.describe_index(pinecone_index_name).status["ready"]:
            time.sleep(1)

    pinecone_index = pinecone.Index(pinecone_index_name)
    print("pinecone_index", pinecone_index)

    print(
        "pinecone_index.describe_index_stats()", pinecone_index.describe_index_stats()
    )
    return pinecone, pinecone_index


def upsert_pinecone(pinecone_index, transcript, model_name, pinecone_namespace=None):
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
