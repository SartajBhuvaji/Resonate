import os
from datetime import timedelta, datetime

import moviepy.editor as mp
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as ss
from src.clustering.resonate_clustering import *
from src.clustering.resonate_bert_summarizer import summarize_runner

from src.utils.streamlitUtils import (
    convert_video_to_audio,
    aws_transcribe,
    transcript_text_editor_minutes_to_hhmmss,
    pinecone_init_upsert,
)


def initialize_session_state():

    # Initialize API keys in session state if not present
    if "api_keys" not in ss:
        ss.api_keys = {}
        ss.api_keys["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
        ss.api_keys["pinecone_api_key"] = os.environ.get("PINECONE_API_KEY")
        ss.api_keys["aws_access_key"] = os.environ.get("AWS_ACCESS_KEY")
        ss.api_keys["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if "add_meeting" not in ss:
        ss.add_meeting = False

    # Initialize - Main Screen - Transcript Editor
    if "transcript_speaker_editor" not in ss:
        ss.transcript_speaker_editor = False

    if "transcript_text_editor" not in ss:
        ss.transcript_text_editor = False

    # # Initialize - Main Screen - chat history
    if "chat_history" not in ss:
        ss.chat_history = []

    # Initialize - Main Screen - User input and Chatbox
    if "user_input" not in ss:
        ss.user_input = ""
    if "chat_resonate" not in ss:
        ss.chat_resonate = False

    if "df_transcript_speaker" not in ss:
        ss.df_transcript_speaker = pd.DataFrame()

    if "df_transcript_text" not in ss:
        ss.df_transcript_text = pd.DataFrame()

    if "updated_df" not in ss:
        ss.updated_transcript_df_to_embed = pd.DataFrame()


# def get_bot_response(user_input):
#     # Replace this with your actual chatbot logic
#     return f"Chatbot: You said '{user_input}'"


def api_keys_input():
    with st.form("keys_input_form"):
        """API Keys Input:"""
        # Retrieve values from session state
        openai_api_key = st.text_input(
            "OpenAPI Key:",
            type="password",
            value=ss.api_keys["openai_api_key"],
        )
        pinecone_api_key = st.text_input(
            "Pinecone Key:",
            type="password",
            value=ss.api_keys["pinecone_api_key"],
        )
        aws_access_key = st.text_input(
            "AWS Access Key:",
            type="password",
            value=ss.api_keys["aws_access_key"],
        )
        aws_secret_access_key = st.text_input(
            "AWS Secret Access Key:",
            type="password",
            value=ss.api_keys["aws_secret_access_key"],
        )

        # Add a button to save the keys
        save_button = st.form_submit_button("Save API Keys")

        if save_button:
            ss.api_keys["openai_api_key"] = openai_api_key
            ss.api_keys["pinecone_api_key"] = pinecone_api_key
            ss.api_keys["aws_access_key"] = aws_access_key
            ss.api_keys["aws_secret_access_key"] = aws_secret_access_key

            os.environ["OPENAI_API_KEY"] = ss.api_keys["openai_api_key"]
            os.environ["PINECONE_API_KEY"] = ss.api_keys["pinecone_api_key"]
            os.environ["AWS_ACCESS_KEY"] = ss.api_keys["aws_access_key"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = ss.api_keys["aws_secret_access_key"]

            st.rerun()


def add_meeting():

    with st.form("add_meeting_form"):
        uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp4"])

        # Get user input
        meeting_name = st.text_input("Enter Meeting Name:")

        save_meeting_button = st.form_submit_button("Save Meeting")

        if save_meeting_button:
            if not meeting_name:
                st.warning("Please enter Meeting Name.")
            elif uploaded_file is None:
                st.warning("Please upload a meeting recording.")
            elif meeting_name and uploaded_file:
                with st.spinner("Processing..."):
                    file_name = uploaded_file.name.replace(" ", "_")

                    if file_name.endswith(".mp4") or file_name.endswith(".mpeg4"):
                        with open("data/videoFiles/" + file_name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            f.close()

                        # Convert video file to audio file
                        audio_path = "data/audioFiles/" + file_name[:-4] + ".wav"
                        convert_video_to_audio(
                            "data/videoFiles/" + file_name, audio_path
                        )

                        file_name = file_name[:-4] + ".wav"

                    elif file_name.endswith(".wav"):
                        with open("data/audioFiles/" + file_name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            f.close()

                    ss.df_transcript_speaker = aws_transcribe(file_name)
                    ss.transcript_speaker_editor = True


def transcript_speaker_editor():
    ss.add_meeting = False
    with st.form("transcript_speaker_editor_form"):
        st.write("Transcript Speaker Editor:")
        st.dataframe(ss.df_transcript_speaker)

        df = ss.df_transcript_speaker.copy(deep=True)

        # Create a list of unique speaker labels
        speaker_labels = df["speaker_label"].unique()

        # Create a dictionary to store the updated values
        updated_speaker_names = {}

        # Display text input boxes for each speaker label
        for speaker_label in speaker_labels:
            new_name = st.text_input(
                f"Edit speaker label '{speaker_label}'", speaker_label
            )
            updated_speaker_names[speaker_label] = new_name

        # Update the DataFrame with the new speaker label names
        for old_name, new_name in updated_speaker_names.items():
            df["speaker_label"] = df["speaker_label"].replace(old_name, new_name)

        update_speaker_button = st.form_submit_button("Update Speakers")

    if update_speaker_button and df is not None:
        ss.df_transcript_speaker = pd.DataFrame()
        ss.df_transcript_text = df.copy(deep=True)

        del df

        ss.transcript_text_editor = True
        ss.transcript_speaker_editor = False
        st.rerun()


# Function to update the text column
def transcript_text_editor_update_text(row_index, new_text):
    ss.updated_transcript_df_to_embed.at[row_index, "text"] = new_text


def transcript_text_editor():
    ss.transcript_speaker_editor = False
    st.write("Transcript Text Editor:")
    st.write(ss.df_transcript_text)

    df = ss.df_transcript_text.copy(deep=True)
    ss.updated_transcript_df_to_embed = df.copy(deep=True)

    # Convert start_time and end_time to HH:MM:SS format
    df["start_time"] = df["start_time"].apply(transcript_text_editor_minutes_to_hhmmss)
    df["end_time"] = df["end_time"].apply(transcript_text_editor_minutes_to_hhmmss)

    row_index = st.number_input(
        "Enter the row index:",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )
    new_text = st.text_area("Enter the new text:", df.at[row_index, "text"])

    update_text_button_inner = st.button("Update Text")

    if update_text_button_inner:
        transcript_text_editor_update_text(row_index, new_text)
        st.success("Text updated successfully!")

    # Display the updated dataframe
    st.header("Updated Dataframe")
    st.table(ss.updated_transcript_df_to_embed)

    update_text_button = st.button("Finish Transcript Editing")
    if update_text_button:
        ss.df_transcript_text = pd.DataFrame()
        summarize_runner(ss.updated_transcript_df_to_embed)
        create_Cluster(os.environ.get("OPENAI_API_KEY"))

        pinecone_init_upsert(ss.updated_transcript_df_to_embed)

        st.success("Pinecone upsert completed successfully!")

        st.rerun()


# def chat_resonate():
#     # Input box for user to enter queries as text
#     user_input = st.text_input("User Input:", value=ss.user_input)

#     # Send button to simulate sending user input
#     if st.button("Send") and user_input:
#         # Adding user input to chat history
#         ss.chat_history.append(f"User: {user_input}")

#         # Getting response from LLM and adding it to chatbot response of chat history
#         bot_response = get_bot_response(user_input)
#         ss.chat_history.append(bot_response)

#         # Clearing the user input field for next query
#         ss.user_input = ""

#     # Initializing chat history
#     st.subheader("Chat History")
#     for entry in ss.chat_history:
#         st.write(entry)


def init_streamlit():

    # Load environment variables from .env file
    load_dotenv("./config/.env")

    # Now you can access your environment variables using os.environ
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

    # Initializing session state for all Streamlit components
    initialize_session_state()

    # Set initial state of the sidebar
    if ss.api_keys["pinecone_api_key"] != "":
        st.set_page_config(initial_sidebar_state="collapsed")
    st.title("Resonate - Meeting Chatter")

    # Initializing sidebar and its components
    with st.sidebar:
        api_keys_input()

    # st.button("Add Meeting", on_click=add_meeting)

    if st.button("Add Meeting"):
        ss.add_meeting = not ss.add_meeting
        ss.transcript_speaker_editor = False
        ss.transcript_text_editor = False

    if ss.add_meeting:
        add_meeting()
    if ss.transcript_speaker_editor:
        transcript_speaker_editor()
    if ss.df_transcript_text is not None and ss.transcript_text_editor:
        transcript_text_editor()

    # try:

    # Initializing Pinecone
    # (
    #     ss.pinecone,
    #     ss.pinecone_index,
    # ) = init_pinecone(
    #     ss.pinecone_config["pinecone_api_key"],
    #     ss.pinecone_config["pinecone_index_name"],
    #     ss.pinecone_config["pinecone_index_metric"],
    #     ss.pinecone_config["pinecone_index_dimension"],
    #     ss.pinecone_config["pinecone_cloud_type"],
    #     ss.pinecone_config["pinecone_cloud_region"],
    # )

    # except Exception as e:
    #     print(e)

    # if ss.api_keys["pinecone_api_key"] != "":
    # if st.sidebar.button("Add Meeting"):
    #     ss.add_meeting = not ss.add_meeting
    #     if ss.api_keys_input == True:
    #         ss.api_keys_input = False

    # if ss.chat_resonate:
    #     chat_resonate()

    # if ss.api_keys["pinecone_api_key"] != "":
    #     if not ss.df_transcript_speaker.empty:
    #         ss.chat_resonate = False
    #         ss.transcript_text_editor = False
    #         transcript_speaker_editor()

    #     if not ss.df_transcript_text.empty:
    #         ss.chat_resonate = False
    #         ss.transcript_speaker_editor = False
    #         transcript_text_editor()

    #     if ss.df_transcript_text.empty and ss.df_transcript_speaker.empty:
    #         chat_resonate()


if __name__ == "__main__":
    init_streamlit()
