# AWS utility functions for Resonate
# Author: Sartaj and  Madhuroopa

# Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/transcribe/client/start_transcription_job.html

import boto3
import dotenv
import os
import time
import moviepy.editor as mp
dotenv.load_dotenv()
from firebase_admin import credentials, initialize_app, storage,db

## Todo add the needed installations in requirements.txt

def initialize_firebase()->bool:
    try:
        cred = credentials.Certificate("resonate.private_key.json")
        initialize_app(cred, {
            
            'databaseURL': f"{os.getenv('DATABASE_URL')}",
            'storageBucket': 
                f"{os.getenv('STORAGE_BUCKET')}"})
        print("Firebase initialization successful")
        return True
    except Exception as e:
        print(f"Firebase initialization failed: {e}")
        return False

def push_json_to_database(json_data, database_path)->bool:
    try:
        ref = db.reference(database_path)
        ref.set(json_data)
        print(f"JSON data pushed {database_path}' in Realtime Database")
        return True
    except Exception as e:
        print(f"Error pushing JSON data to database: {e}")
        return False
def upload_video_to_storage(local_path, storage_path)->str:
    try:
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        blob.upload_from_filename(local_path)

        # Opt this if we want to make public access from the URL
        blob.make_public()

        print(f"Video file uploaded to '{storage_path}' in Cloud Storage")
        print("Your file URL:", blob.public_url)
        return blob.public_url

    except Exception as e:
        print(f"Error uploading video file to Cloud Storage: {e}")
        return None
    
def video_to_audio(in_path, out_path) -> bool:
    """
    Convert video file to audio file.

    Parameters:
    - in_path (str): Path to the input video file.
    - out_path (str): Path to save the output audio file.

    Returns:
    - bool: True if conversion is successful, False otherwise.
    """
    try:
        video = mp.VideoFileClip(in_path)
        video.audio.write_audiofile(out_path)
       
        print("coverted the video to audio file")
        return True
    except Exception as e:
        print(f"error while converting the file {in_path}")
        return False
    


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


def runner():
    

    json_data= {
    "index": "langchain-retrieval-transcript",
    "namespace": "new_namespace_1",
    "last_meeting_no": 1,
    "last_conversation_no": 1,
    "unique_meeting_members": [
        "test2",
        "test1"
    ],
    "meetings": [
        {
            "meeting_no": 1,
            "meeting_date": "2024-01-24 13:23:38",
            "meeting_video_file": True,
            "meeting_members": [
                "test1",
                "test2"
            ]
            
        },
       
    ]
}

    
    video_file = 'video.mp4'
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpg', '.mpeg']
    file_extension = video_file.lower().split('.')[-1]
    print(file_extension)
    audio_extensions = ['.mp3', '.wav']
    file='test.wav'
    # Check if the extension is in the list of video extensions
    if "."+file_extension in video_extensions:
        
       print(video_to_audio(video_file,file))
        
    elif(file_extension not in audio_extensions):
        print("file format not accepted")
    initialize_firebase()
    database_path = '/meeting_metadata'
    print(push_json_to_database(json_data, database_path))

    uploaded_url = upload_video_to_storage(video_file, video_file)

    if uploaded_url:
        print(f"Video file uploaded successfully. Public URL: {uploaded_url}")
    transcribe_client, s3_client = create_client()
    input_bucket = 'resonate-input'
    output_bucket = 'resonate-output'  
    transcribe_job_name = 'job'
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
    transcribe_client.delete_transcription_job(TranscriptionJobName = transcribe_job_name)
    transcribe_client.close()
    s3_client.close()

    
if __name__ == "__main__":
    runner()
    