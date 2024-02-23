# Description: AWS utility functions for Resonate. This file contains the code to parse the AWS Transcribe output.
# Author: Sartaj and Madhuroopa

# Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/transcribe/client/start_transcription_job.html

import json
import os
import re
import time

import boto3
import dotenv
import pandas as pd
import webvtt
from IPython.display import HTML, display


def create_client(
    aws_access_key: str, aws_secret_access_key: str, aws_region_name: str = "us-east-2"
) -> tuple[boto3.client, boto3.client]:
    """
    Create and return AWS Transcribe and S3 clients with the specified or default AWS region.

    Parameters:
    - aws_access_key (str): AWS access key ID.
    - aws_secret_access_key (str): AWS secret access key.
    - aws_region_name (str): The AWS region where the clients will be created (default is 'us-east-2').

    Returns:
    - Tuple[boto3.client, boto3.client]: AWS Transcribe client and S3 client.

    This function creates an AWS Session with the provided AWS access key, secret access key, and
    region. It then returns AWS Transcribe and S3 clients.

    Example:
    >>> access_key = "your_access_key"
    >>> secret_key = "your_secret_key"
    >>> transcribe_client, s3_client = create_client(access_key, secret_key)
    """
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region_name,
    )
    return session.client("transcribe"), session.client("s3")


def create_s3_bucket(
    s3: boto3.client, bucket_name: str, aws_region_name: str = "us-east-2"
) -> bool:
    """
    Create an S3 bucket using the provided AWS S3 client if it doesn't exist.

    Parameters:
    - s3 (boto3.client): The AWS S3 client used to interact with S3 services.
    - bucket_name (str): The name of the S3 bucket.
    - aws_region_name (str): The AWS region where the S3 bucket should be created (default is 'us-east-2').

    Returns:
    - bool: True if the S3 bucket is successfully created or already exists, else False.

    This function attempts to create an S3 bucket with the specified name and region using the
    provided AWS S3 client. It handles the following cases:
    1. Successfully creates the bucket and returns True.
    2. If the bucket already exists, prints a message and returns True.
    3. If any other exception occurs during the creation process, prints an error message with the
        exception details and returns False.

    Example:
    >>> import boto3
    >>> s3_client = boto3.client('s3')
    >>> bucket_name = 'your_bucket_name'
    >>> result = create_s3_bucket(s3_client, bucket_name)
    >>> print(result)
    """
    try:
        # Attempt to create the bucket with a specific location constraint
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": aws_region_name},
        )
        print(f"S3 bucket '{bucket_name}' created successfully.")
        return True
    except s3.exceptions.BucketAlreadyExists:
        print(f"S3 bucket '{bucket_name}' already exists.")
        return True
    except Exception as e:
        print(f"Error creating S3 bucket '{bucket_name}': {e}")
        return False


def upload_to_s3(
    s3: boto3.client, file_path: str, bucket_name: str, object_name=None
) -> str:
    """
    Upload a file to an S3 bucket using the provided AWS S3 client, and create the bucket if it doesn't exist.

    Parameters:
    - s3 (boto3.client): The AWS S3 client used to interact with S3 services.
    - file_path (str): The local path of the file to upload.
    - bucket_name (str): The name of the S3 bucket.
    - object_name (Optional[str]): The object name in the S3 bucket. If not specified, the file name will be used.

    Returns:
    - str: The URI of the uploaded file (format: s3://bucket_name/object_name).

    Raises:
    - botocore.exceptions.NoCredentialsError: If AWS credentials are not available or valid.
    - botocore.exceptions.ParamValidationError: If the provided parameters are not valid.

    Example:
    ```python
    import boto3

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Specify local file path, S3 bucket name, and optional object name
    local_file_path = '/path/to/local/file.txt'
    s3_bucket_name = 'your-s3-bucket'
    s3_object_name = 'custom-object-name'

    # Upload the file to S3
    upload_to_s3(s3_client, local_file_path, s3_bucket_name, s3_object_name)
    ```

    Note:
    - This function uploads a file to an S3 bucket using the provided AWS S3 client.
    - If the object_name is not specified, the file name will be used as the object name.
    - The function prints success or error messages to the console.
    """
    if object_name is None:
        object_name = file_path

    try:
        # Attempt to upload the file to the specified S3 bucket and object
        s3.upload_file(file_path, bucket_name, object_name)

        # Construct the URI of the uploaded file
        uri = f"s3://{bucket_name}/{object_name}"

        # Print success message to the console
        print(f"File '{file_path}' uploaded successfully to '{uri}'")

        # Return the URI of the uploaded file
        return uri
    except Exception as e:
        # Print error message to the console in case of upload failure
        print(
            f"Error uploading file '{file_path}' to '{bucket_name}/{object_name}': {e}"
        )

        # Return an empty string to indicate upload failure
        return ""


def download_from_s3(
    s3: boto3.client,
    object_name: str,
    bucket_name: str = "resonate-output",
    local_directory: str = "Streamlit_App/data",
) -> bool:
    """
    Download a file from an S3 bucket to a local directory.

    Parameters:
    - s3 (boto3.client): The AWS S3 client used to interact with S3 services.
    - object_name (str): The object name in the S3 bucket.
    - bucket_name (str): The name of the S3 bucket (default is 'resonate-output').
    - local_directory (str): The local directory where the file should be saved (default is current directory).

    Returns:
    - bool: True if the file was downloaded successfully, else False.
    """
    local_file_json = f"{local_directory}/{object_name}.json"
    local_file_vtt = f"{local_directory}/{object_name}.vtt"

    try:
        # Download the file
        s3.download_file(bucket_name, object_name + ".json", local_file_json)
        s3.download_file(bucket_name, object_name + ".vtt", local_file_vtt)
        print(
            f"File '{object_name}' (JSON) downloaded successfully to '{local_file_json}'"
        )
        print(
            f"File '{object_name}' (VTT) downloaded successfully to '{local_file_vtt}'"
        )
        return True
    except Exception as e:
        print(f"Error downloading file '{object_name}' from '{bucket_name}': {e}")
        return False


def delete_from_s3(s3: boto3.client, bucket_name: str, object_name: str) -> bool:
    """
    Delete a file from an S3 bucket using the provided AWS S3 client.

    Parameters:
    - s3 (boto3.client): The AWS S3 client used to interact with S3 services.
    - bucket_name (str): The name of the S3 bucket.
    - object_name (str): The object name in the S3 bucket.

    Returns:
    - bool: True if the file was deleted successfully, else False.
    """
    try:
        s3.delete_object(Bucket=bucket_name, Key=object_name)
        print(f"File '{object_name}' deleted successfully from '{bucket_name}'")
        return True
    except Exception as e:
        print(f"Error deleting file '{object_name}' from '{bucket_name}': {e}")
        return False


def delete_s3_bucket(s3: boto3.client, bucket_name: str) -> bool:
    """
    Delete an S3 bucket along with its contents using the provided AWS S3 client.

    Parameters:
    - s3 (boto3.client): The AWS S3 client used to interact with S3 services.
    - bucket_name (str): The name of the S3 bucket.

    Returns:
    - bool: True if the S3 bucket and its contents were deleted successfully, else False.
    """
    try:
        # List all objects in the bucket
        objects = s3.list_objects(Bucket=bucket_name).get("Contents", [])

        # Delete each object in the bucket
        for obj in objects:
            s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
            print(f"Object '{obj['Key']}' deleted successfully from '{bucket_name}'")

        # Delete the bucket
        s3.delete_bucket(Bucket=bucket_name)
        print(f"S3 bucket '{bucket_name}' and its contents deleted successfully.")
        return True
    except Exception as e:
        print(f"Error deleting S3 bucket {bucket_name}: {e}")
        return False


def transcribe_audio(
    transcribe_client: boto3.client,
    uri: str,
    output_bucket: str,
    transcribe_job_name: str = "job",
) -> dict:
    """
    Start a transcription job for audio stored in an S3 bucket using the AWS Transcribe service.

    Parameters:
    - transcribe_client (boto3.client): The AWS Transcribe client used to interact with Transcribe services.
    - uri (str): The URI of the audio file in the S3 bucket.
    - output_bucket (str): The name of the S3 bucket where Transcribe will store the output.
    - transcribe_job_name (str): The name of the transcription job (default is 'job').

    Returns:
    - dict: The response from the Transcribe service containing information about the transcription job.
    """
    print("Calling AWS Transcribe Job...")
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=transcribe_job_name,
        LanguageCode="en-US",
        MediaFormat="wav",
        Settings={
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": 10,
            "ChannelIdentification": False,
        },
        Media={"MediaFileUri": uri},
        Subtitles={"Formats": ["vtt"]},
        OutputBucketName=output_bucket,
    )
    return response


def combine_files(file_name: str) -> pd.DataFrame:
    """
    Combines information from a JSON file and a WebVTT file into a CSV file.

    Parameters:
    - file_name (str): The base name of the files (without extensions).

    Returns:
    - pd.DataFrame: A DataFrame containing combined information from the JSON and WebVTT files.

    The function loads a JSON file containing speaker labels and a WebVTT file containing subtitles.
    It extracts relevant information, combines the data, and returns a DataFrame.

    Note: Update the file paths to the actual paths where your JSON and WebVTT files are stored.

    Example:
    ```
    combined_df = combine_files("example_file")
    ```

    # Explanation of the process:
    1. Load the JSON file containing speaker labels.
    2. Extract relevant information and create a DataFrame.
    3. Load the WebVTT file containing subtitles.
    4. Extract information from subtitles and create another DataFrame.
    5. Merge the two DataFrames based on start_time.
    6. Drop rows with NaN values in the speaker_label column.
    7. Rename columns for clarity.
    8. Reset the index of the final DataFrame.

    # Usage:
    - Ensure that the file paths in the function match the actual locations of your JSON and WebVTT files.

    # Note:
    - The returned DataFrame contains columns: 'start_time', 'end_time', 'speaker_label', and 'text'.
    """
    # Load the JSON file
    json_file_path = f"Streamlit_App/data/{file_name}.json"
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract the relevant information from the JSON file
    segments = data["results"]["speaker_labels"]["segments"]
    df = pd.DataFrame(segments)
    df["start_time"] = df["start_time"].astype(float) / 60
    df["end_time"] = df["end_time"].astype(float) / 60
    df = df.rename(
        columns={
            "start_time": "start_time",
            "end_time": "end_time",
            "speaker_label": "speaker_label",
        }
    )

    # Load the WebVTT file
    vtt_file_path = f"Streamlit_App/data/{file_name}.vtt"
    subtitles = webvtt.read(vtt_file_path)

    # Extract information from subtitles and create a DataFrame
    data = [
        (
            subtitle.start_in_seconds / 60,
            subtitle.end_in_seconds / 60,
            subtitle.text.strip(),
        )
        for subtitle in subtitles
    ]
    titles = pd.DataFrame(data, columns=["start_time", "end_time", "text"])

    # Merge the two DataFrames based on start_time
    transcript = pd.merge_asof(
        titles.sort_values("start_time"),
        df.sort_values("start_time"),
        on="start_time",
        direction="backward",
    )

    # Drop rows with NaN values in the speaker_label column
    transcript = transcript.dropna(subset=["speaker_label"])

    # Rename the columns
    transcript = transcript[["start_time", "end_time_x", "speaker_label", "text"]]
    transcript.columns = ["start_time", "end_time", "speaker_label", "text"]

    # Reset the index
    transcript = transcript.reset_index(drop=True)

    print("Combined transcript successfully!")
    return transcript


def aws_transcribe_parser(
    transcript_df: pd.DataFrame, output_filename: str
) -> pd.DataFrame:
    """
    Parses the AWS Transcribe output by cleaning duplicate texts and merging consecutive rows with
    the same speaker.

    Parameters:
    - transcript_df (pd.DataFrame): DataFrame containing AWS Transcribe output with columns 'text',
        'speaker_label', 'start_time', and 'end_time'.
    - output_filename (str): The base name for the output CSV file (without extension).

    Returns:
    - bool: True if the transcript is successfully cleaned and saved to a CSV file, else False.

    This function takes a DataFrame generated from AWS Transcribe output and performs the following
    operations:
    1. Removes unwanted characters (such as quotes and dashes) from the 'text' column.
    2. Identifies and removes duplicate consecutive rows with the same text and speaker.
    3. Merges consecutive rows with the same speaker, updating the 'end_time' accordingly.
    4. Creates a new DataFrame with columns 'speaker_label', 'start_time', 'end_time', and 'text',
        containing the cleaned and merged transcript data.
    5. Saves the cleaned transcript data to a CSV file with the specified output filename.

    Example:
    >>> import pandas as pd
    >>> data = {'text': ['Hello', 'Hello', 'How are you?', 'Fine, thank you.'],
    ...         'speaker_label': ['A', 'A', 'B', 'B'],
    ...         'start_time': [0.0, 1.5, 3.0, 4.0],
    ...         'end_time': [1.0, 3.0, 4.0, 6.0]}
    >>> transcript_df = pd.DataFrame(data)
    >>> cleaned_transcript = aws_transcribe_parser(transcript_df, "example_output")
    >>> print(cleaned_transcript)
    True
    """
    prev_text = None  # Initialize prev_text
    transcript_df["text"] = transcript_df["text"].apply(
        lambda x: re.sub(r"[\"\'\--]+", "", x)
    )

    for index, row in transcript_df.iterrows():
        if row["text"] == prev_text and row["speaker_label"] == prev_speaker:
            transcript_df.at[merge_start, "end_time"] = row["end_time"]
            transcript_df.drop(index, inplace=True)
        else:
            merge_start = index

        prev_text = row["text"]
        prev_speaker = row["speaker_label"]

    transcript_df["group"] = (
        transcript_df["speaker_label"] != transcript_df["speaker_label"].shift()
    ).cumsum()
    result_df = transcript_df.groupby(["group", "speaker_label"], as_index=False).agg(
        {"start_time": "first", "end_time": "last", "text": " ".join}
    )
    result_df = result_df.drop(columns=["group"])
    print("combining")
    result_df.to_csv(f"Streamlit_App/data/{output_filename}.csv", index=False)
    # print(f"Transcript saved to {output_filename}.csv")
    return result_df



def delete_local_temp_file(file_path: str) -> bool:
    """
    Delete a local temporary file specified by the file path.

    Parameters:
    - file_path (str): The path of the local temporary file to be deleted.

    Returns:
    - bool: True if the file was deleted successfully, False otherwise.

    This function attempts to delete a local temporary file using the provided file path. It
    handles the following cases:
    1. Successfully deletes the file and returns True.
    2. If the file is not found (FileNotFoundError), prints an error message and returns False.
    3. If any other exception occurs during the deletion process, prints an error message with the
        exception details and returns False.

    Example:
    >>> file_path = "path/to/your/temp/file.txt"
    >>> result = delete_local_temp_file(file_path)
    >>> print(result)
    """
    try:
        os.remove(file_path)
        print(f"Deleted local temp file: {file_path} - successfully")
        return True
    except FileNotFoundError:
        print(f"Error deleting file '{file_path}': File not found")
        return False
    except Exception as e:
        print(f"Error deleting file '{file_path}': {e}")
        return False


def runner(
    file_name: str,
    input_bucket: str,
    output_bucket: str,
    transcribe_job_name: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_region_name: str,
) -> None:
    """
    Run the transcription process for an audio file using AWS Transcribe.

    Parameters:
    - file_name (str): The base name of the audio file (without extensions).
    - input_bucket (str): The name of the input S3 bucket.
    - output_bucket (str): The name of the output S3 bucket.
    - transcribe_job_name (str): The name of the AWS Transcribe job.
    - aws_access_key (str): AWS access key ID.
    - aws_secret_access_key (str): AWS secret access key.
    - aws_region_name (str): The AWS region where the clients will be created.

    Returns:
    - None: This function does not return any value.

    This function orchestrates the transcription process using AWS Transcribe for an audio file.
    It performs the following steps:
    1. Creates AWS Transcribe and S3 clients.
    2. Defines input and output S3 buckets and transcribe job name.
    3. Deletes old S3 buckets and transcribe job if they exist.
    4. Creates new input and output S3 buckets.
    5. Uploads the audio file to the input S3 bucket.
    6. Initiates the transcription job using AWS Transcribe.
    7. Monitors the status of the transcription job until it is completed.
    8. Downloads the transcription job output from the output S3 bucket.
    9. Deletes S3 buckets and transcribe job after use.
    10. Closes the AWS Transcribe and S3 clients.
    11. Combines the JSON and VTT results to create a transcript CSV.
    12. Parses the transcript CSV using aws_transcribe_parser.
    13. Saves the parsed transcript as a CSV file.
    14. Deletes temporary local files.

    Example:
    >>> file_name = "test"
    >>> runner(file_name, "input_bucket", "output_bucket", "transcribe_job", "access_key", "secret_key", "region_name")
    """

    transcribe_client, s3_client = create_client(
        aws_access_key=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        aws_region_name=aws_region_name,
    )

    # print("transcribe_client : ", transcribe_client)
    # print("s3_client : ", s3_client)

    # Delete old S3 buckets and transcribe job
    try:
        print("Delete S3 Bucket : ", delete_s3_bucket(s3_client, input_bucket))
        print("Delete S3 Bucket : ", delete_s3_bucket(s3_client, output_bucket))
    except Exception:

        print("S3 bucket does not exist.")

    try:
        transcribe_client.delete_transcription_job(
            TranscriptionJobName=transcribe_job_name
        )
    except:
        print("Transcription Job does not exist.")

    # Create S3 buckets
    print(
        f"Create S3 Bucket {input_bucket} : ", create_s3_bucket(s3_client, input_bucket)
    )
    print(
        f"Create S3 Bucket {output_bucket} : ",
        create_s3_bucket(s3_client, output_bucket),
    )

    URI = upload_to_s3(s3_client, file_name, input_bucket)
    print("Upload completed now will initiate transcription job.")
    transcribe_audio(
        transcribe_client, URI, output_bucket, transcribe_job_name=transcribe_job_name
    )

    # Check status of transcription job
    while (
        transcribe_client.get_transcription_job(
            TranscriptionJobName=transcribe_job_name
        )["TranscriptionJob"]["TranscriptionJobStatus"]
        != "COMPLETED"
    ):
        time.sleep(3)

    # Download transcription job output
    print(
        "Download from S3 : ",
        download_from_s3(
            s3_client, transcribe_job_name, output_bucket
        ),
    )

    # Delete S3 buckets and transcribe job after use
    try:
        print("Delete S3 Bucket : ", delete_s3_bucket(s3_client, input_bucket))
        print("Delete S3 Bucket : ", delete_s3_bucket(s3_client, output_bucket))
    except:
        print("S3 bucket does not exist.")

    try:
        transcribe_client.delete_transcription_job(
            TranscriptionJobName=transcribe_job_name
        )
    except:
        print("Transcription Job does not exist.")

    # Close clients
    transcribe_client.close()
    s3_client.close()

    # combine the json and vtt results to create a transcript
    df_transcript_combined = combine_files(
        transcribe_job_name
    )  # transcribe_job_name is the name of the file that contains the json and vtt results
    df_transcript_combined_parsed = aws_transcribe_parser(
        transcript_df=df_transcript_combined, output_filename=transcribe_job_name
    )
    print("Transcript parsed successfully")

    # delete the temporary local files
    #########################    delete_local_temp_file(transcribe_job_name + ".json")
    #########################    delete_local_temp_file(transcribe_job_name + ".vtt")

    return df_transcript_combined_parsed


if __name__ == "__main__":
    dotenv.load_dotenv()
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    print(aws_access_key, aws_secret_access_key)
    aws_region_name = "us-east-2"
    file_name = "test.wav"
    input_bucket = "resonate-input-jay"
    output_bucket = "resonate-output-jay"
    transcribe_job_name = "resonate-job-jay"
    df = runner(
        file_name=file_name,
        input_bucket=input_bucket,
        output_bucket=output_bucket,
        transcribe_job_name=transcribe_job_name,
        aws_access_key=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        aws_region_name=aws_region_name,
    )
