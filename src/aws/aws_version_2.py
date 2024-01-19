# AWS utility functions for Resonate
# Author: Sartaj and Madhuroopa

# Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/transcribe/client/start_transcription_job.html

import boto3
import dotenv
import os
import time
import pandas as pd
import json
import webvtt
dotenv.load_dotenv()

def create_client(region_name: str = 'us-east-2') -> boto3.client:
    """
    Create and return an AWS Transcribe client with the specified or default AWS region.

    :param region_name: The AWS region where the Transcribe client will be created (default is 'us-east-2').
    :type region_name: str
    :return: An AWS Transcribe client and an S3 client.
    :rtype: boto3.clients
    """
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

    session = boto3.Session(
        aws_access_key_id = AWS_ACCESS_KEY,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
        region_name = region_name
        )
    return session.client('transcribe'), session.client('s3')



def create_s3_bucket(s3: boto3.client, bucket_name: str, region: str = 'us-east-2') -> bool:
    """
    Create an S3 bucket using the provided AWS S3 client if it doesn't exist.

    :param s3: The AWS S3 client used to interact with S3 services.
    :type s3: boto3.client
    :param bucket_name: The name of the S3 bucket.
    :type bucket_name: str
    :param region: The AWS region where the S3 bucket should be created (default is 'us-east-1').
    :type region: str
    :return: True if the S3 bucket is successfully created or already exists, else False.
    :rtype: bool
    """
    try:
        # Attempt to create the bucket with a specific location constraint
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': region}
        )
        print(f"S3 bucket '{bucket_name}' created successfully.")
        return True
    except s3.exceptions.BucketAlreadyExists as e:
        print(f"S3 bucket '{bucket_name}' already exists.")
        return True
    except Exception as e:
        print(f"Error creating S3 bucket '{bucket_name}': {e}")
        return False


def upload_to_s3(s3: boto3.client, file_path: str, bucket_name: str, object_name=None) -> str:
    """
    Upload a file to an S3 bucket using the provided AWS S3 client, and create the bucket if it doesn't exist.

    :param s3: The AWS S3 client used to interact with S3 services.
    :type s3: boto3.client
    :param file_path: The local path of the file to upload.
    :type file_path: str
    :param bucket_name: The name of the S3 bucket.
    :type bucket_name: str
    :param object_name: (Optional) The object name in the S3 bucket. If not specified, the file name will be used.
    :type object_name: str
    :return: The URI of the uploaded file (format: s3://bucket_name/object_name).
    :rtype: str
    """
    if object_name is None:
        object_name = file_path

    try:
        s3.upload_file(file_path, bucket_name, object_name)
        uri = f"s3://{bucket_name}/{object_name}"
        print(f"File '{file_path}' uploaded successfully to '{uri}'")
        return uri
    except Exception as e:
        print(f"Error uploading file '{file_path}' to '{bucket_name}/{object_name}': {e}")
        return ""


def download_from_s3(s3: boto3.client, object_name: str, bucket_name: str = 'resonate-output', local_directory: str = '.') -> bool:
    """
    Download a file from an S3 bucket to a local directory.

    :param s3: The AWS S3 client used to interact with S3 services.
    :type s3: boto3.client
    :param object_name: The object name in the S3 bucket.
    :type object_name: str
    :param bucket_name: The name of the S3 bucket.
    :type bucket_name: str
    :param local_directory: The local directory where the file should be saved (default is current directory).
    :type local_directory: str
    :return: True if the file was downloaded successfully, else False.
    :rtype: bool
    """
    local_file_json = f"{local_directory}/{object_name}.json"
    local_file_vtt = f"{local_directory}/{object_name}.vtt"

    try:
        # Download the file
        s3.download_file(bucket_name, object_name + '.json', local_file_json)
        s3.download_file(bucket_name, object_name + '.vtt', local_file_vtt)
        print(f"File '{object_name}' (JSON) downloaded successfully to '{local_file_json}'")
        print(f"File '{object_name}' (VTT) downloaded successfully to '{local_file_vtt}'")
        return True
    except Exception as e:
        print(f"Error downloading file '{object_name}' from '{bucket_name}': {e}")
        return False


def delete_from_s3(s3: boto3.client, bucket_name: str, object_name: str) -> bool:
    """
    Delete a file from an S3 bucket using the provided AWS S3 client.

    :param s3: The AWS S3 client used to interact with S3 services.
    :type s3: boto3.client
    :param bucket_name: The name of the S3 bucket.
    :type bucket_name: str
    :param object_name: The object name in the S3 bucket.
    :type object_name: str
    :return: True if the file was deleted successfully, else False.
    :rtype: bool
    """
    try:
        s3.delete_object(Bucket=bucket_name, Key=object_name)
        print(f"File '{object_name}' deleted successfully from '{bucket_name}'")
        return True
    except Exception as e:
        print(f"Error deleting file '{object_name}' from '{bucket_name}': {e}")
        return False


def delete_s3_bucket(s3, bucket_name):
    """
    Delete an S3 bucket along with its contents using the provided AWS S3 client.

    :param s3: The AWS S3 client used to interact with S3 services.
    :type s3: boto3.client
    :param bucket_name: The name of the S3 bucket.
    :type bucket_name: str
    :return: True if the S3 bucket and its contents were deleted successfully, else False.
    :rtype: bool
    """    
    try:
        # List all objects in the bucket
        objects = s3.list_objects(Bucket=bucket_name).get('Contents', [])
        
        # Delete each object in the bucket
        for obj in objects:
            s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
            print(f"Object '{obj['Key']}' deleted successfully from '{bucket_name}'")

        # Delete the bucket
        s3.delete_bucket(Bucket=bucket_name)
        print(f"S3 bucket '{bucket_name}' and its contents deleted successfully.")
        return True
    except Exception as e:
        print(f"Error deleting S3 bucket '{bucket_name}': {e}")
        return False  


def transcribe_audio(transcribe_client: boto3.client, uri: str, output_bucket: str, transcribe_job_name: str='job')-> dict:
    """
    Start a transcription job for audio stored in an S3 bucket using the AWS Transcribe service.

    :param s3: The AWS S3 client used to interact with S3 services.
    :type s3: boto3.client
    :param URI: The URI of the audio file in the S3 bucket.
    :type URI: str
    :return: The response from the Transcribe service containing information about the transcription job.
    :rtype: dict
    """
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName = transcribe_job_name,
        LanguageCode = 'en-US',
        MediaFormat = 'wav',
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 10,
            'ChannelIdentification': False,
        },  
        Media = {
            'MediaFileUri': uri
        },
        Subtitles={
            'Formats': ['vtt']
        },
        OutputBucketName=output_bucket,
    )
    return response

def combine_files(file_name: str):
    """
    Combines information from a JSON file and a WebVTT file into a CSV file.

    Parameters:
    - file_name (str): The base name of the files (without extensions).

    The function loads a JSON file containing speaker labels and a WebVTT file containing subtitles.
    It extracts relevant information, combines the data, and saves the result as a CSV file.

    Note: Update the file paths to the actual paths where your JSON and WebVTT files are stored.
    """

    # Load the JSON file
    with open(f'./{file_name}.json', 'r') as f:
        data = json.load(f)

    # Extract the relevant information from the JSON file
    segments = data['results']['speaker_labels']['segments']
    rows = []
    for segment in segments:
        start_time = float(segment['start_time']) / 60
        end_time = float(segment['end_time']) / 60
        speaker_label = segment['speaker_label']
        rows.append([start_time, end_time, speaker_label])
    df = pd.DataFrame(rows, columns=['start_time', 'end_time', 'speaker_label'])

    # Load the WebVTT file
    subtitles = webvtt.read(f'./{file_name}.vtt')

    # Initialize an empty list to store the captions
    data = []

    # Loop through the captions and extract the information
    for subtitle in subtitles:
        start_time = subtitle.start.split(':')
        end_time = subtitle.end.split(':')

        # Convert start and end time to minutes
        start_minutes = int(start_time[0]) * 60 + int(start_time[1]) + float(start_time[2]) / 60
        end_minutes = int(end_time[0]) * 60 + int(end_time[1]) + float(end_time[2]) / 60

        text = subtitle.text.strip()

        # Append the information to the data list
        data.append((start_minutes, end_minutes, text))

    # Create a pandas dataframe from the data list
    titles = pd.DataFrame(data, columns=['start_time', 'end_time', 'text'])

    # Merge the two tables based on start_time
    merged = pd.merge_asof(titles.sort_values('start_time'), df.sort_values('start_time'), on='start_time',direction='backward')

    # Drop rows with NaN values in the speaker_label column
    merged = merged.dropna(subset=['speaker_label'])

    # Rename the columns
    merged = merged[['start_time', 'end_time_x', 'speaker_label', 'text']]
    merged.columns = ['start_time', 'end_time', 'speaker_label', 'text']

    # Reset the index
    merged = merged.reset_index(drop=True)

    # Save the merged data as a CSV file ( the transcript)
    merged.to_csv(f'./{file_name}.csv')
    
    
def runner():

    transcribe_client, s3_client = create_client()
    input_bucket = 'resonate-input'
    output_bucket = 'resonate-output'  
    transcribe_job_name = 'job'
    file = 'test.wav'

    # Create S3 buckets
    print(create_s3_bucket(s3_client, input_bucket))
    print(create_s3_bucket(s3_client, output_bucket))

    URI = upload_to_s3(s3_client, file, input_bucket)
    transcribe_audio(transcribe_client, URI, output_bucket, transcribe_job_name=transcribe_job_name)
    time.sleep(10)

    # Check status of transcription job
    transcribe_client.list_transcription_jobs(Status='COMPLETED', JobNameContains='string')

    # Download transcription job output
    print(download_from_s3(s3_client, transcribe_job_name, output_bucket, local_directory='.'))
    
    ## combine the json and vtt results to cerate a transcript
    combine_files(transcribe_job_name)

    transcribe_client.delete_transcription_job(TranscriptionJobName = transcribe_job_name)
    transcribe_client.close()
    s3_client.close()