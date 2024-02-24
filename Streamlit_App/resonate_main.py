# Mapping - PineCone DB to Organization Structure
# Pinecone Project      ==>==>==>==>==>    Organization
# Pinecone Index        ==>==>==>==>==>     Team
# Pinecone Namespace    ==>==>==>==>==>     Topic / Project

# import os
# import time
from datetime import datetime

from resonate_streamlit_functions import init_streamlit


def main():

    current_timestamp = str.lower(datetime.now().strftime("%Y-%b-%d-%I-%M-%p"))

    # Initializing Variables
    aws_config = {
        "aws_region_name": "us-east-2",
        "aws_input_bucket": f"resonate-input-{str(current_timestamp)}",
        "aws_output_bucket": f"resonate-output-{str(current_timestamp)}",
        "aws_transcribe_job_name": f"resonate-job-{str(current_timestamp)}",
    }

    pinecone_config = {
        "pinecone_index_name": "resonate-meeting-index",
        "pinecone_index_dimension": 1536,
        "pinecone_index_metric": "cosine",
        "pinecone_cloud_type": "aws",
        "pinecone_cloud_region": "us-west-2",
        "pinecone_namespace": "meeting_topic",
        "pinecone_embedding_model_name": "text-embedding-3-large",
    }

    init_streamlit(aws_config, pinecone_config)


if __name__ == "__main__":
    main()
