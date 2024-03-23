# Description: This file contains the main Streamlit application for the Resonate project.
# Run command: streamlit run app.py

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as ss
from streamlit_chat import message 
from src.clustering.resonate_bert_summarizer import summarize_runner
from src.clustering.resonate_clustering import Clustering
from src.langchain.resonate_langchain_functions import LangChain
from src.utils.resonate_streamlitUtils import (
    aws_transcribe,
    convert_video_to_audio,
    pinecone_init_upsert,
    transcript_text_editor_minutes_to_hhmmss,
)

def initialize_session_state():
    # Initialize API keys in session state if not present
    if "api_keys" not in ss:
        ss.api_keys = {}
        ss.api_keys["openai_api_key"] = None
        ss.api_keys["pinecone_api_key"] = None
        ss.api_keys["aws_access_key"] = None
        ss.api_keys["aws_secret_access_key"] = None

    if "api_key_set" not in ss:
        ss.api_key_set = False 
    if "add_meeting" not in ss:
        ss.add_meeting = False
    if "Clustering_obj" not in ss:
        ss.Clustering_obj = Clustering()
    # Initialize - Main Screen - Transcript Editor
    if "transcript_speaker_editor" not in ss:
        ss.transcript_speaker_editor = False
    if "transcript_text_editor" not in ss:
        ss.transcript_text_editor = False
    if "meeting_name" not in ss:
        ss.meeting_name = ""
    if "df_transcript_speaker" not in ss: 
        ss.df_transcript_speaker = pd.DataFrame()
    if "df_transcript_text" not in ss:
        ss.df_transcript_text = pd.DataFrame()
    if "updated_df" not in ss:
        ss.updated_transcript_df_to_embed = pd.DataFrame()
    if "chat_view" not in ss:
        ss.chat_view = True
    if "langchain_obj" not in ss and ss.api_key_set:
        ss.langchain_obj = LangChain()
    if "query" not in ss:
        ss.query = ""
    if "responses" not in ss:
        ss["responses"] = ["How can I assist you?"]
    if "requests" not in ss:
        ss["requests"] = []


def chat_view():
    st.header("Chat")
    response_container = st.container()
    textcontainer = st.container()
    with textcontainer:
        query = st.text_input(
            "Chat Here",
            placeholder="Message Resonate ... ",
            value=ss.query,
            key="query_input",
        )
        # Clear button
        if st.button("Clear"):
            ss.langchain_obj.conversation_bufw.memory.clear() # Clear conversation buffer
            ss.query = "" 
            ss.requests = []
            ss.responses = []
            ss["responses"] = ["How can I assist you?"]
            st.rerun()

        elif query:
            with st.spinner("typing..."):
                uuid_list = ss.Clustering_obj.uuid_for_query(query=query)
                print(f"Meeting Unique ID : {uuid_list}")
                response = ss.langchain_obj.chat(
                    query=query, in_filter=uuid_list, complete_db_flag=False
                )
                response = response["response"]
            ss.requests.append(query)
            ss.responses.append(response)
            ss.query = ""
    with response_container:
        if ss["responses"]:
            for i in range(len(ss["responses"])):
                message(ss["responses"][i], key=str(i))
                if i < len(ss["requests"]):
                    message(
                        ss["requests"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )


def api_keys_input():
    with st.form("keys_input_form"):
        # Retrieve values from session state
        openai_api_key = st.text_input(
            "OpenAPI Key:",
            type="password",
            value=ss.api_keys.get(
                "openai_api_key", ""
            ),  # Use default value if key is not present
        )
        pinecone_api_key = st.text_input(
            "Pinecone Key:",
            type="password",
            value=ss.api_keys.get(
                "pinecone_api_key", ""
            ),  # Use default value if key is not present
        )
        aws_access_key = st.text_input(
            "AWS Access Key:",
            type="password",
            value=ss.api_keys.get(
                "aws_access_key", ""
            ),  # Use default value if key is not present
        )
        aws_secret_access_key = st.text_input(
            "AWS Secret Access Key:",
            type="password",
            value=ss.api_keys.get(
                "aws_secret_access_key", ""
            ),  # Use default value if key is not present
        )
        # Add a button to save the keys
        save_button = st.form_submit_button("Save API Keys")
        if save_button:
            # Update session state with provided keys
            ss.api_keys["openai_api_key"] = openai_api_key
            ss.api_keys["pinecone_api_key"] = pinecone_api_key
            ss.api_keys["aws_access_key"] = aws_access_key
            ss.api_keys["aws_secret_access_key"] = aws_secret_access_key
            # Set environment variables only if the keys are not None
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = ss.api_keys["openai_api_key"]
            if pinecone_api_key:
                os.environ["PINECONE_API_KEY"] = ss.api_keys["pinecone_api_key"]
            if aws_access_key:
                os.environ["AWS_ACCESS_KEY"] = ss.api_keys["aws_access_key"]
            if aws_secret_access_key:
                os.environ["AWS_SECRET_ACCESS_KEY"] = ss.api_keys[
                    "aws_secret_access_key"
                ]

            ss.api_key_set = True
            print("API KEYS ARE: ", ss.api_keys)    
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
                        print("in video")
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
                        print("in audio")
                        with open("data/audioFiles/" + file_name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            f.close()
                    ss.df_transcript_speaker = aws_transcribe(file_name)
                    ss.meeting_name = meeting_name
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
    st.header("Updated Transcript")
    st.table(ss.updated_transcript_df_to_embed)
    update_text_button = st.button("Finish Transcript Editing")
    if update_text_button:
        with st.spinner("Uploading..."):
            ss.df_transcript_text = pd.DataFrame()
            meeting_summary, meeting_uuid = summarize_runner(
                ss.updated_transcript_df_to_embed
            )
            ss.Clustering_obj.create_Cluster()
            pinecone_init_upsert(
                ss.updated_transcript_df_to_embed,
                meeting_title=ss.meeting_name,
                meeting_summary=meeting_summary,
                meeting_uuid=meeting_uuid,
            )
            ss.meeting_name = "unnamed"
            st.success("Pinecone upsert completed successfully!")
            ss.transcript_text_editor = False
            ss.updated_transcript_df_to_embed = pd.DataFrame()
            ss.chat_view = True
            st.rerun()


def init_streamlit():
    initialize_session_state()
    if os.path.exists("./config/.env"):
        load_dotenv("./config/.env")

    else:
        print(".env file does not exist, API keys must be set manually.")

    # Set initial state of the sidebar
    st.set_page_config(
        initial_sidebar_state="collapsed",
        layout="wide",
        )
    st.title("RESONATE")

    # Initializing sidebar and its components
    with st.sidebar:
        api_keys_input()
    if st.button("Upload Meeting / Chat"):
        ss.add_meeting = not ss.add_meeting
        ss.chat_view = not ss.chat_view
        ss.transcript_speaker_editor = False
        ss.transcript_text_editor = False

    if not ss.api_key_set:
        st.header("Pre-requisites:")
        st.write("Please set the API keys to enable the chat view.")
        st.write("Please ensure that you have already run the 'pinecone_sample_dataloader.py'")

    if ss.add_meeting and ss.api_key_set:
        add_meeting()
    if ss.transcript_speaker_editor:
        transcript_speaker_editor()
    if ss.df_transcript_text is not None and ss.transcript_text_editor:
        transcript_text_editor()
    if ss.chat_view and ss.api_key_set:
        chat_view()  # Chat view



if __name__ == "__main__":
    # Please ensure you have data loaded in Pinecone before running the Streamlit app
    # Please refer https://github.com/SartajBhuvaji/Resonate/blob/master/init_one_time_utils/PREREQUISITE.md
    init_streamlit()

# Test questions:
# What was discussed about cyberbullying?
# What is one new feature planned for GitLab's code search?
# What is the goal of defining maintainability for the new diffs architecture?