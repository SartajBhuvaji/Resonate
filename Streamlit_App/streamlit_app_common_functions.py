import time

import pandas as pd
from IPython.display import HTML, display
from pinecone import Pinecone, ServerlessSpec

display(HTML("<style>.container { width:100% !important; }</style>"))
# pd.set_option("display.max_rows", None)


# Loading Transcripts from CSV file
def load_transcript(file_name):
    transcript = pd.read_csv(file_name)
    transcript.drop(["end_time"], axis=1, inplace=True)

    # Iterate through rows and update the text column
    for i in range(1, len(transcript)):
        if transcript.loc[i, "speaker_label"] == transcript.loc[i - 1, "speaker_label"]:
            transcript.loc[i - 1, "text"] += " " + transcript.loc[i, "text"]
            transcript.drop(index=i, inplace=True)

    # Resetting index after dropping rows
    transcript.reset_index(drop=True, inplace=True)
    # Display the updated dataframe
    print("Transcript")
    display(transcript)
    return transcript


# Initializing Pinecone
def init_pinecone(
    PINECONE_API_KEY,
    pinecone_index_name,
    pinecone_index_metric,
    pinecone_index_dimension,
    pinecone_cloud_type,
    pinecone_cloud_region,
):
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

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
