# Description: The code helps generate abstractive summary for the transcript data using BART model
# Reference: https://huggingface.co/vmarklynn/bart-large-cnn-samsum-acsi-ami-v2
# Reference : https://github.com/vmarklynn/parrot/blob/main/notebooks/summarizer.ipynb

import math
import pandas as pd
import os
import uuid
from transformers import pipeline

# Initialization of summarizer based on Bart
MODEL = 'vmarklynn/bart-large-cnn-samsum-acsi-ami-v2'
summarizer = pipeline("summarization", MODEL, truncation=True)

def format_text(text):
    '''
    Format the transcript data into a readable format
    '''
    try:
        formatted_data = [
            f"{row['speaker_label']}: {row['text']}" for _, row in text.iterrows()
        ]
        formatted_text = "\n".join([f"{line}" for line in formatted_data])
        return formatted_text
    except Exception as e:
        print(f"Error formatting text: {e}")
        return ""


def summarize_text(transcript):
    '''
    Summarize the text using the BART model
    '''
    try:
        text = format_text(transcript)

        print("\n\nSummarizing Text...")
        summary = summarizer(text)[0]["summary_text"]
        response = {"transcription": format_text, "summary": summary}
        return response
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return {}


def summarize_summary(summary_input):
    '''
    Summarize the summarized text using the BART model
    '''
    try:
        word_count = 1024 
        summary = summarizer(
            summary_input,
            min_length=math.ceil(int(word_count) * 0.1),
            max_length=math.ceil(int(word_count) * 0.25),
        )[0]["summary_text"]
        response = {"summary": summary}
        return response
    except Exception as e:
        print(f"Error summarizing summary: {e}")
        return {}


def append_summary_to_csv(summary_text):
    try:
        csv_filename = "./data/summaryFiles/abstract_summary_data.csv"
        meeting_uuid = str(uuid.uuid4())
        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename)
        else:
            df = pd.DataFrame(columns=["uuid", "text"])
        new_data = pd.DataFrame({"uuid": [meeting_uuid], "text": [summary_text]})
        df = pd.concat([df, new_data], ignore_index=True)

        df.to_csv(csv_filename, index=False)
        return meeting_uuid
    except Exception as e:
        print(f"Error appending summary to CSV: {e}")
        return False


def summarize_runner(transcript):
    try:
        transcript.drop(["end_time"], axis=1, inplace=True)
        summary_transcript = summarize_text(transcript)
        summarized_summary = summarize_summary(summary_transcript["summary"])
        final_summary = summarized_summary["summary"]
        meeting_uuid = append_summary_to_csv(final_summary)
    except Exception as e:
        print(f"Error in summarize_runner: {e}")
    return final_summary, meeting_uuid


if __name__ == "__main__":
    df = pd.read_csv("./data/transcriptFiles/Social_Media_-_Ruins_your_life.csv")
    summarize_runner(df)
