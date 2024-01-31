# Mapping - PineCone DB to Organization Structure
# Pinecone Project      ==>==>==>==>==>    Organization
# Pinecone Index        ==>==>==>==>==>     Team
# Pinecone Namespace    ==>==>==>==>==>     Topic / Project

import os
import time

# import dotenv
import streamlit as st

from resonate_pinecone_functions import init_pinecone
from resonate_streamlit_functions import init_streamlit

# from resonate_streamlit_functions import (
#     add_meeting,
#     get_bot_response,
#     initialize_session_state,
#     show_keys_input,
# )

# from IPython.display import HTML, display


def main():
    # Initializing Variables
    aws_config = {
        "aws_region_name": "us-east-2",
        "aws_input_bucket": "resonate-input-jay",
        "aws_output_bucket": "resonate-output-jay",
        "aws_transcribe_job_name": "resonate-job-jay",
    }

    pinecone_config = {
        "pinecone_index_name": "resonate-meeting-index",
        "pinecone_index_dimension": 1536,
        "pinecone_index_metric": "cosine",
        "pinecone_cloud_type": "aws",
        "pinecone_cloud_region": "us-west-2",
        "pinecone_namespace": "meeting_topic",
        "pinecone_embedding_model_name": "text-embedding-ada-002",
    }

    init_streamlit(aws_config, pinecone_config)


if __name__ == "__main__":
    main()
