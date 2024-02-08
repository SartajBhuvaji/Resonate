import math

import keybert
import pandas as pd
from transformers import pipeline

# Initialization of summarizer based on Bart
summarizer = pipeline(
    "summarization", "vmarklynn/bart-large-cnn-samsum-acsi-ami-v2", truncation=True
)
kw_model = keybert.KeyBERT(model="all-mpnet-base-v2")


def format_text(text):
    formatted_data = [
        f"{row['speaker_label']}: {row['text']}" for _, row in text.iterrows()
    ]
    formatted_text = "\n".join([f"{line}" for line in formatted_data])
    return formatted_text


def summarize_text(transcript):

    text = format_text(transcript)

    print("\n\nSummarizing Text...")
    summary = summarizer(text)[0]["summary_text"]
    print("\n", summary, "\n")

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        highlight=False,
        top_n=5,
    )
    keywords_list_1 = list(dict(keywords).keys())
    print("1 gram keywords: ", keywords_list_1)

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(2, 2),
        stop_words="english",
        highlight=False,
        top_n=5,
    )
    keywords_list_2 = list(dict(keywords).keys())
    print("2 gram keywords: ", keywords_list_2)

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(3, 3),
        stop_words="english",
        highlight=False,
        top_n=5,
    )
    keywords_list_3 = list(dict(keywords).keys())
    print("3 gram keywords: ", keywords_list_3)

    response = {
        "transcription": format_text,
        "summary": summary,
        "keywords_list_1": keywords_list_1,
        "keywords_list_2": keywords_list_2,
        "keywords_list_3": keywords_list_3,
    }
    return response


def summarize_summary(summary_input):

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


def main(file_name):
    transcript = pd.read_csv(file_name, index_col=0)

    summary_transcript = summarize_text(transcript)
    # print(summary_transcript["summary"])
    summarized_summary = summarize_summary(summary_transcript["summary"])
    final_summary = summarized_summary["summary"]
    print(final_summary)


if __name__ == "__main__":
    main(file_name="data/Shark_Tank_US_Top_3_Products_For_Office_1.csv")
