import math

import keybert
import pandas as pd
from transformers import pipeline
import datetime
import os
import uuid
# Initialization of summarizer based on Bart
summarizer = pipeline(
    "summarization", "vmarklynn/bart-large-cnn-samsum-acsi-ami-v2", truncation=True
)
#kw_model = keybert.KeyBERT(model="all-mpnet-base-v2")


def format_text(text):
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

    try:
        text = format_text(transcript)

        print("\n\nSummarizing Text...")
        summary = summarizer(text)[0]["summary_text"]
        print("\n", summary, "\n")
        response = {
            "transcription": format_text,
            "summary": summary
        }
        return response
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return {}


def summarize_summary(summary_input):

    try:
        word_count = 1024  # post_data.get('word_count-summ')

        print(
            "min: ",
            math.ceil(int(word_count) * 0.1),
            "max: ",
            math.ceil(int(word_count) * 0.25),
        )
        print("\n\nSummarizing again...")
        summary = summarizer(
            summary_input,
            min_length=math.ceil(int(word_count) * 0.1),
            max_length=math.ceil(int(word_count) * 0.25),
        )[0]["summary_text"]
        print("\n", summary, "\n")

        response = {"summary": summary}
        return response
    except Exception as e:
        print(f"Error summarizing summary: {e}")
        return {}


def append_summary_to_csv(summary_text, cluster=""):
    try:
        
        csv_filename = 'Streamlit_App/data/abstract_summary_data.csv'
        meeting_uuid=str(uuid.uuid4())
        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename)
        else:
            #df = pd.DataFrame(columns=['File_id', 'Text', 'cluster'])
            df = pd.DataFrame(columns=['meeting_uuid','text'])
        new_data = pd.DataFrame({'meeting_uuid': [meeting_uuid], 'text': [summary_text]})
        df = pd.concat([df, new_data], ignore_index=True)

        df.to_csv(csv_filename, index=False)
    except Exception as e:
        print(f"Error appending summary to CSV: {e}")


def summarize_runner(df):

    try:
        
        transcript=df
        transcript.drop(['end_time'], axis=1, inplace=True)
        summary_transcript = summarize_text(transcript)
        summarized_summary = summarize_summary(summary_transcript["summary"])
        final_summary = summarized_summary["summary"]
        append_summary_to_csv(final_summary)
        print(final_summary)
    except Exception as e:
        print(f"Error in summarize_runner: {e}")



#summarize_runner("C:/Users/madhu/Desktop/SeattleUniversity/capstone/resonate_madhu/Resonate/Notebooks/Streamlit_App/data/Discussion_on_Illegal_Migration_and_Border_Crisis_Bill.csv")