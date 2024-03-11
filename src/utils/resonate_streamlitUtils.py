import moviepy.editor as mp
import json
from datetime import datetime, timedelta
from src.aws.resonate_aws_functions import resonate_aws_transcribe
import os

from src.pinecone.resonate_pinecone_functions import PineconeServerless
import time
import uuid
import pandas as pd


def convert_video_to_audio(video_path, audio_path):
    # Convert video file to audio file
    audio_clip = mp.VideoFileClip(video_path).audio
    audio_clip.write_audiofile(audio_path)


def transcript_text_editor_minutes_to_hhmmss(minutes):
    time_delta = timedelta(minutes=minutes)
    hhmmss_format = str(time_delta)
    return hhmmss_format


def load_json_config(json_file_path="./config/config.json"):
    # Use a context manager to ensure the file is properly closed after opening
    with open(json_file_path, "r") as file:
        # Load the JSON data
        data = json.load(file)
    return data


def aws_transcribe(file_name):

    json_config = load_json_config()
    current_timestamp = str.lower(datetime.now().strftime("%Y-%b-%d-%I-%M-%p"))

    json_config["AWS_INPUT_BUCKET"] += f"{str(current_timestamp)}"
    json_config["AWS_OUTPUT_BUCKET"] += f"{str(current_timestamp)}"
    json_config["AWS_TRANSCRIBE_JOB_NAME"] += f"{str(current_timestamp)}"

    print(json_config)

    try:
        rat = resonate_aws_transcribe()
        df = rat.runner(
            file_name=file_name,
            input_bucket=json_config["AWS_INPUT_BUCKET"],
            output_bucket=json_config["AWS_OUTPUT_BUCKET"],
            transcribe_job_name=json_config["AWS_TRANSCRIBE_JOB_NAME"],
            aws_access_key=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region_name=json_config["AWS_REGION"],
        )

        return df
    except Exception as e:
        return e


def pinecone_init_upsert(
    df_transcript: pd.DataFrame,
    meeting_title: str,
    meeting_summary: str,
    meeting_uuid: str,
):
    try:

        pinecone = PineconeServerless()
        pinecone.pinecone_upsert(
            df_transcript,
            # meeting_uuid=NULL,
            meeting_uuid=meeting_uuid,
            meeting_video_file=False,
            meeting_title=meeting_title,
            meeting_summary=meeting_summary,
        )
        time.sleep(5)
    except Exception as e:
        print("Error upserting transcript to Pinecone: ", e)
