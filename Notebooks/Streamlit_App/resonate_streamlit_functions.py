import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import moviepy.editor as mp
from datetime import timedelta
from resonate_aws_functions import *
from resonate_pinecone_functions import init_pinecone, upsert_pinecone
from resonate_bert_summarizer import *
from resonate_clustering import *


def initialize_session_state(aws_config, pinecone_config):
    # Initialize - Config
    if "pinecone_config" not in ss:
        ss.pinecone_config = pinecone_config
    if "pinecone_index" not in ss:
        ss.pinecone_index = None
    if "pinecone" not in ss:
        ss.pinecone = None

    if "aws_config" not in ss:
        ss.aws_config = aws_config

    # Initialize - Sidebar Components
    if "api_keys_input" not in ss:
        ss.api_keys_input = False
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

    # Initialize API keys in session state if not present
    if "api_keys" not in ss:
        ss.api_keys = {
            "openai_api_key": "",
            "pinecone_api_key": "",
            "aws_access_key": "",
            "aws_secret_access_key": "",
        }

    if "df_transcript_speaker" not in ss:
        ss.df_transcript_speaker = pd.DataFrame()

    if "df_transcript_text" not in ss:
        ss.df_transcript_text = pd.DataFrame()

    if "updated_df" not in ss:
        ss.updated_transcript_df_to_embed = pd.DataFrame()


def get_bot_response(user_input):
    # Replace this with your actual chatbot logic
    return f"Chatbot: You said '{user_input}'"


def api_keys_input():
    with st.form("keys_input_form"):
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

            ss.aws_config["aws_access_key"] = aws_access_key
            ss.aws_config["aws_secret_access_key"] = aws_secret_access_key

            ss.pinecone_config["pinecone_api_key"] = pinecone_api_key

            os.environ["OPENAI_API_KEY"] = ss.api_keys["openai_api_key"]
            os.environ["PINECONE_API_KEY"] = ss.api_keys["pinecone_api_key"]
            os.environ["AWS_ACCESS_KEY"] = ss.api_keys["aws_access_key"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = ss.api_keys["aws_secret_access_key"]

            ss.api_keys_input = False
            st.rerun()


def convert_video_to_audio(video_path, audio_path):
    # Convert video file to audio file
    audio_clip = mp.VideoFileClip(video_path).audio
    audio_clip.write_audiofile(audio_path)


def add_meeting():

    with st.form("add_meeting_form"):
        uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "mp4"])

        # Get user input
        meeting_name = st.text_input("Enter Meeting Name:")

        save_meeting_button = st.form_submit_button("Save Meeting")

        if save_meeting_button:
            if not meeting_name:
                st.warning("Please enter Meeting Name.")
            elif uploaded_file is None:
                st.warning("Please upload a meeting recording.")
            elif meeting_name and save_meeting_button and uploaded_file:
                with st.spinner("Processing..."):
                    file_name = uploaded_file.name.replace(" ", "_")
                    with open(file_name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        f.close()

                    if file_name.endswith(".mp4"):
                        # Convert video file to audio file
                        audio_path = file_name[:-4] + ".wav"
                        convert_video_to_audio(file_name, audio_path)

                        file_name = audio_path

                    # Your AWS transcription code here
                    ss.aws_config["aws_transcribe_job_name"] = file_name[:-4]

                    try:
                        ss.df_transcript_speaker = runner(
                            file_name=file_name,
                            input_bucket=ss.aws_config["aws_input_bucket"],
                            output_bucket=ss.aws_config["aws_output_bucket"],
                            transcribe_job_name=ss.aws_config[
                                "aws_transcribe_job_name"
                            ],
                            aws_access_key=ss.aws_config["aws_access_key"],
                            aws_secret_access_key=ss.aws_config[
                                "aws_secret_access_key"
                            ],
                            aws_region_name=ss.aws_config["aws_region_name"],
                        )
                        # ss.df_transcript_speaker = pd.read_csv(f"{meeting_name}.csv")
                        ss.df_transcript_speaker.to_csv(f"{file_name[:-4]}.csv")
                        #create_Cluster()
                        st.success("File uploaded and transcribed successfully!")
                        
                    except Exception:
                        st.warning("Please update valid AWS keys.")


def transcript_speaker_editor():
    with st.form("transcript_speaker_editor_form"):
        st.write("Transcript Speaker Editor:")
        st.write(ss.df_transcript_speaker)

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
        st.rerun()


def transcript_text_editor_minutes_to_hhmmss(minutes):
    time_delta = timedelta(minutes=minutes)
    hhmmss_format = str(time_delta)
    return hhmmss_format


# Function to update the text column
def transcript_text_editor_update_text(row_index, new_text):
    ss.updated_transcript_df_to_embed.at[row_index, "text"] = new_text


def transcript_text_editor():
    # with st.form("transcript_text_editor_form"):
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
        if not ss.updated_transcript_df_to_embed.empty:
            print("test")
            summarize_runner(ss.updated_transcript_df_to_embed)
            create_Cluster(ss.api_keys["openai_api_key"])
        upsert_pinecone(
            ss.pinecone_index,
            transcript=ss.updated_transcript_df_to_embed,
            model_name=ss.pinecone_config["pinecone_embedding_model_name"],
            pinecone_namespace=ss.pinecone_config["pinecone_namespace"],
        )
        st.success("Pinecone upsert completed successfully!")

        st.rerun()


def chat_resonate():
    # Input box for user to enter queries as text
    user_input = st.text_input("User Input:", value=ss.user_input)

    # Send button to simulate sending user input
    if st.button("Send") and user_input:
        # Adding user input to chat history
        ss.chat_history.append(f"User: {user_input}")

        # Getting response from LLM and adding it to chatbot response of chat history
        bot_response = get_bot_response(user_input)
        ss.chat_history.append(bot_response)

        # Clearing the user input field for next query
        ss.user_input = ""

    # Initializing chat history
    st.subheader("Chat History")
    for entry in ss.chat_history:
        st.write(entry)


def init_streamlit(aws_config, pinecone_config):
    # Initializing Components

    # Initializing session state for all Streamlit components
    initialize_session_state(aws_config=aws_config, pinecone_config=pinecone_config)

    # Set initial state of the sidebar
    if ss.api_keys["pinecone_api_key"] != "":
        st.set_page_config(initial_sidebar_state="collapsed")
    st.title("Resonate - Meeting Chatter")

    # Initializing sidebar and its components
    with st.sidebar:
        if st.sidebar.button("API Keys"):
            ss.api_keys_input = not ss.api_keys_input
            if ss.add_meeting == True:
                ss.add_meeting = False

        if ss.api_keys_input:
            try:
                api_keys_input()

                # Initializing Pinecone
                (
                    ss.pinecone,
                    ss.pinecone_index,
                ) = init_pinecone(
                    ss.pinecone_config["pinecone_api_key"],
                    ss.pinecone_config["pinecone_index_name"],
                    ss.pinecone_config["pinecone_index_metric"],
                    ss.pinecone_config["pinecone_index_dimension"],
                    ss.pinecone_config["pinecone_cloud_type"],
                    ss.pinecone_config["pinecone_cloud_region"],
                )

            except Exception as e:
                print(e)

        if ss.api_keys["pinecone_api_key"] != "":
            if st.sidebar.button("Add Meeting"):
                ss.add_meeting = not ss.add_meeting
                if ss.api_keys_input == True:
                    ss.api_keys_input = False

            if ss.add_meeting:
                add_meeting()

    if ss.api_keys["pinecone_api_key"] != "":
        if not ss.df_transcript_speaker.empty:
            ss.chat_resonate = False
            ss.transcript_text_editor = False
            transcript_speaker_editor()

        if not ss.df_transcript_text.empty:
            ss.chat_resonate = False
            ss.transcript_speaker_editor = False
            transcript_text_editor()

        if ss.df_transcript_text.empty and ss.df_transcript_speaker.empty:
            chat_resonate()
