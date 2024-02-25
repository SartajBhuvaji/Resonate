import moviepy.editor as mp
import json
from datetime import datetime, timedelta
from src.aws.resonate_aws_functions import resonate_aws_transcribe
import os
from src.pinecone.resonate_pinecone_functions import init_pinecone, upsert_pinecone


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

    json_config["INPUT_BUCKET"] += f"{str(current_timestamp)}"
    json_config["OUTPUT_BUCKET"] += f"{str(current_timestamp)}"
    json_config["AWS_TRANSCRIBE_JOB_NAME"] += f"{str(current_timestamp)}"

    print(json_config)

    try:
        rat = resonate_aws_transcribe()
        df = rat.runner(
            file_name=file_name,
            input_bucket=json_config["INPUT_BUCKET"],
            output_bucket=json_config["OUTPUT_BUCKET"],
            transcribe_job_name=json_config["AWS_TRANSCRIBE_JOB_NAME"],
            aws_access_key=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region_name=json_config["AWS_REGION"],
        )

        return df
    except Exception as e:
        return e


def pinecone_init_upsert(df_transcript):

    json_config = load_json_config()
    try:

        # Initializing Pinecone
        pinecone, pinecone_index = init_pinecone(
            os.getenv("PINECONE_API_KEY"),
            json_config["INDEX_NAME"],
            json_config["METRIC"],
            json_config["PINECONE_VECTOR_DIMENSION"],
            json_config["CLOUD_PROVIDER"],
            json_config["REGION"],
        )

    except Exception as e:
        print("Error initializing Pinecone: ", e)

    try:
        # Upserting transcript to Pinecone
        upsert_pinecone(
            pinecone_index,
            transcript=df_transcript,
            model_name=json_config["EMBEDDING_MODEL"],
            pinecone_namespace=json_config["NAMESPACE"],
        )
    except Exception as e:
        print("Error initializing Pinecone: ", e)
