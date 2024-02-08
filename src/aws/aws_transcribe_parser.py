# Description: This file contains the code to parse the AWS Transcribe output.
# Author: Sartaj and Madhuroopa

import json
import pandas as pd
import webvtt
import re

def combine_files(file_name: str) -> pd.DataFrame:
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

    segments = data['results']['speaker_labels']['segments']
    rows = []
    for segment in segments:
        start_time = float(segment['start_time']) / 60
        end_time = float(segment['end_time']) / 60
        speaker_label = segment['speaker_label']
        rows.append([start_time, end_time, speaker_label])
    df = pd.DataFrame(rows, columns=['start_time', 'end_time', 'speaker_label'])

    subtitles = webvtt.read(f'./{file_name}.vtt')
    data = []

    for subtitle in subtitles:
        start_time = subtitle.start.split(':')
        end_time = subtitle.end.split(':')

        start_minutes = int(start_time[0]) * 60 + int(start_time[1]) + float(start_time[2]) / 60
        end_minutes = int(end_time[0]) * 60 + int(end_time[1]) + float(end_time[2]) / 60
        text = subtitle.text.strip()
        data.append((start_minutes, end_minutes, text))

    titles = pd.DataFrame(data, columns=['start_time', 'end_time', 'text'])
    transcript = pd.merge_asof(titles.sort_values('start_time'), df.sort_values('start_time'), on='start_time',direction='backward')
    transcript = transcript.dropna(subset=['speaker_label'])
    transcript = transcript[['start_time', 'end_time_x', 'speaker_label', 'text']]
    transcript.columns = ['start_time', 'end_time', 'speaker_label', 'text']
    
    transcript = transcript.reset_index(drop=True)
    #merged.to_csv(f'./{file_name}.csv')
    return transcript


def aws_transcribe_parser(transcript: pd.DataFrame) -> None:
    """
    Parses the AWS Transcribe output by deleting duplicate texts, and merging consecutive rows with 
    the same speaker.
    """
    transcript['text'] = transcript['text'].apply(lambda x: re.sub(r"[\"\'\--]+", '', x))
    prev_text = ''
    prev_speaker = ''

    for index, row in transcript.iterrows():
        if row['text'] == prev_text and row['speaker_label'] == prev_speaker:
            transcript.at[merge_start, 'end_time'] = row['end_time']
            transcript.drop(index, inplace=True)
        else:
            merge_start = index

        prev_text = row['text']
        prev_speaker = row['speaker_label']

    transcript['group'] = (transcript['speaker_label'] != transcript['speaker_label'].shift()).cumsum()
    result_df = transcript.groupby(['group', 'speaker_label'], as_index=False).agg({'start_time': 'first', 'end_time': 'last', 'text': ' '.join})
    result_df = result_df.drop(columns=['group'])
    return result_df    


def runner():
    """
    Runs the program.
    """
    file_name = 'job1707326598'
    transcript = combine_files(file_name)
    transcript = aws_transcribe_parser(transcript)
    transcript.to_csv(f'./{file_name}.csv')

runner()