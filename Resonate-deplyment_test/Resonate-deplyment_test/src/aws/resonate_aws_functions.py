# Description: AWS utility functions for Resonate. This file contains the code to parse the AWS Transcribe output.
# Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/transcribe/client/start_transcription_job.html

import json
import os
import re
import time
import boto3
import dotenv
import pandas as pd
import webvtt
from datetime import datetime
from IPython.display import HTML, display

class resonate_aws_transcribe:
    def create_client(
        self,
        aws_access_key: str,
        aws_secret_access_key: str,
        aws_region_name: str,
    ) -> tuple[boto3.client, boto3.client]:
        """
        Create and return AWS Transcribe and S3 clients with the specified AWS region.
        """
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
        )
        return session.client("transcribe"), session.client("s3")

    def create_s3_bucket(
        self, s3: boto3.client, bucket_name: str, aws_region_name: str
    ) -> bool:
        """
        Create an S3 bucket using the provided AWS S3 client if it doesn't exist.
        """
        try:
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
        self, s3: boto3.client, file_path: str, bucket_name: str, object_name=None
    ) -> str:
        """
        Upload the audio file to S3 bucket using the provided AWS S3 client.
        """
        if object_name is None:
            object_name = file_path

        try:
            s3.upload_file(file_path, bucket_name, object_name)
            uri = f"s3://{bucket_name}/{object_name}"
            print(f"File '{file_path}' uploaded successfully to '{uri}'")
            return uri

        except Exception as e:
            print(
                f"Error uploading file '{file_path}' to '{bucket_name}/{object_name}': {e}"
            )
            return ""

    def download_from_s3(
        self,
        s3: boto3.client,
        object_name: str,
        bucket_name: str,
        local_directory: str,
    ) -> bool:
        """
        Download the .json and .vtt files from an S3 bucket to a local directory.
        """
        local_file_json = f"{local_directory}/{object_name}.json"
        local_file_vtt = f"{local_directory}/{object_name}.vtt"

        try:
            s3.download_file(bucket_name, object_name + ".json", local_file_json)
            print(f"File '{object_name}' (JSON) downloaded successfully to '{local_file_json}'")

            s3.download_file(bucket_name, object_name + ".vtt", local_file_vtt)
            print(f"File '{object_name}' (VTT) downloaded successfully to '{local_file_vtt}'")
            return True
        except Exception as e:
            print(f"Error downloading file '{object_name}' from '{bucket_name}': {e}")
            return False

    def delete_from_s3(
        self, s3: boto3.client, bucket_name: str, object_name: str
    ) -> bool:
        """
        Delete the file from an S3 bucket using the provided AWS S3 client.
        """
        try:
            s3.delete_object(Bucket=bucket_name, Key=object_name)
            print(f"File '{object_name}' deleted successfully from '{bucket_name}'")
            return True
        except Exception as e:
            print(f"Error deleting file '{object_name}' from '{bucket_name}': {e}")
            return False

    def delete_s3_bucket(self, s3: boto3.client, bucket_name: str) -> bool:
        """
        Delete a S3 bucket along with its contents using the provided AWS S3 client.
        """
        try:
            objects = s3.list_objects(Bucket=bucket_name).get("Contents", [])
            for obj in objects:
                s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
                print(
                    f"Object '{obj['Key']}' deleted successfully from '{bucket_name}'"
                )

            s3.delete_bucket(Bucket=bucket_name)
            print(f"S3 bucket '{bucket_name}' and its contents deleted successfully.")
            return True
        except Exception as e:
            return e

    def transcribe_audio(
        self,
        transcribe_client: boto3.client,
        uri: str,
        output_bucket: str,
        transcribe_job_name: str = "job",
    ) -> dict:
        """
        Start a transcription job for audio stored in an S3 bucket using the AWS Transcribe service.
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

    def combine_files(self, file_name: str, local_directory: str) -> pd.DataFrame:
        """
        Combines information from a JSON file and a WebVTT file into a CSV file.
        """
        json_file_path = f"{local_directory}/{file_name}.json"
        with open(json_file_path, "r") as f:
            data = json.load(f)

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

        vtt_file_path = f"{local_directory}/{file_name}.vtt"
        subtitles = webvtt.read(vtt_file_path)

        data = [
            (
                subtitle.start_in_seconds / 60,
                subtitle.end_in_seconds / 60,
                subtitle.text.strip(),
            )
            for subtitle in subtitles
        ]
        titles = pd.DataFrame(data, columns=["start_time", "end_time", "text"])
        transcript = pd.merge_asof(
            titles.sort_values("start_time"),
            df.sort_values("start_time"),
            on="start_time",
            direction="backward",
        )

        transcript = transcript.dropna(subset=["speaker_label"])
        transcript = transcript[["start_time", "end_time_x", "speaker_label", "text"]]
        transcript.columns = ["start_time", "end_time", "speaker_label", "text"]

        # Reset the index
        transcript = transcript.reset_index(drop=True)

        print("Combined transcript successfully!")
        return transcript

    def aws_transcribe_parser(
        self, transcript_df: pd.DataFrame, output_filename: str
    ) -> pd.DataFrame:
        """
        Parses the AWS Transcribe output by cleaning duplicate texts and merging consecutive rows with
        the same speaker.
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
        result_df = transcript_df.groupby(
            ["group", "speaker_label"], as_index=False
        ).agg({"start_time": "first", "end_time": "last", "text": " ".join})
        result_df = result_df.drop(columns=["group"])

        result_df.to_csv(
            "./data/transcriptFiles/" + output_filename + ".csv", index=False
        )
        return result_df

    def delete_local_temp_file(self, tempFiles: str) -> bool:
        """
        Delete a local temporary file specified by the file path.
        """
        if os.path.exists("./data/tempFiles/" + tempFiles + ".json"):
            os.remove("./data/tempFiles/" + tempFiles + ".json")

        if os.path.exists("./data/tempFiles/" + tempFiles + ".vtt"):
            os.remove("./data/tempFiles/" + tempFiles + ".vtt")

    def runner(
        self,
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
        """
        transcribe_client, s3_client = self.create_client(
            aws_access_key=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
        )

        print("Transcribe_client created: ", transcribe_client)
        print("s3_client created: ", s3_client)

        # Create S3 buckets
        print(
            f"Create S3 Bucket {input_bucket} : ",
            self.create_s3_bucket(s3_client, input_bucket, aws_region_name),
        )
        print(
            f"Create S3 Bucket {output_bucket} : ",
            self.create_s3_bucket(s3_client, output_bucket, aws_region_name),
        )

        URI = self.upload_to_s3(
            s3_client, "./data/audioFiles/" + file_name, input_bucket
        )
        print("Upload completed now will initiate transcription job.")
        self.transcribe_audio(
            transcribe_client,
            URI,
            output_bucket,
            transcribe_job_name=transcribe_job_name,
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
            self.download_from_s3(
                s3_client,
                transcribe_job_name,
                output_bucket,
                local_directory="./data/tempFiles/",
            ),
        )

        print(
            "Delete S3 Bucket Input Bucket : ",
            self.delete_s3_bucket(s3_client, input_bucket),
        )
        print(
            "Delete S3 Bucket Output Bucket: ",
            self.delete_s3_bucket(s3_client, output_bucket),
        )

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
        df_transcript_combined = self.combine_files(
            transcribe_job_name, local_directory="./data/tempFiles/"
        )  
        df_transcript_combined_parsed = self.aws_transcribe_parser(
            transcript_df=df_transcript_combined, output_filename=transcribe_job_name
        )
        print("Transcript parsed successfully")

        self.delete_local_temp_file(tempFiles=transcribe_job_name)
        return df_transcript_combined_parsed


if __name__ == "__main__":
    dotenv.load_dotenv("./config/.env")

    current_timestamp = str.lower(datetime.now().strftime("%Y-%b-%d-%I-%M-%p"))

    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    print(aws_access_key, aws_secret_access_key)
    aws_region_name = "us-east-2"
    file_name = "test.wav"
    input_bucket = f"resonate-input-{str(current_timestamp)}"
    output_bucket = f"resonate-output-{str(current_timestamp)}"
    transcribe_job_name = f"resonate-job-{str(current_timestamp)}"

    rat = resonate_aws_transcribe()
    df = rat.runner(
        file_name=file_name,
        input_bucket=input_bucket,
        output_bucket=output_bucket,
        transcribe_job_name=transcribe_job_name,
        aws_access_key=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        aws_region_name=aws_region_name,
    )
    print(df)
