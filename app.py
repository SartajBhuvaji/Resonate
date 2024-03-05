<<<<<<< HEAD
import os
from datetime import timedelta, datetime
from streamlit_chat import message  # streamlit_chat==0.0.2.2

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as ss

from src.clustering.resonate_clustering import Clustering
from src.clustering.resonate_bert_summarizer import summarize_runner
from src.langchain.resonate_langchain_functions import LangChain

from src.utils.resonate_streamlitUtils import (
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

    if "Clustering_obj" not in ss:
        ss.Clustering_obj = Clustering()

    # Initialize - Main Screen - Transcript Editor
    if "transcript_speaker_editor" not in ss:
        ss.transcript_speaker_editor = False

    if "transcript_text_editor" not in ss:
        ss.transcript_text_editor = False

    # # Initialize - Main Screen - chat history
    if "chat_history" not in ss:
        ss.chat_history = []

    if "meeting_name" not in ss:
        ss.meeting_name = ""

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


def view2(langchain_obj):
    st.header("Chat")

    if "responses" not in st.session_state:
        st.session_state["responses"] = ["How can I assist you?"]

    if "requests" not in st.session_state:
        st.session_state["requests"] = []

    response_container = st.container()
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("", key="input", placeholder="Message Resonate ... ")

        # clear button
        if st.button("Clear"):
            st.session_state.requests = []
            st.session_state.responses = []
            st.session_state["responses"] = ["How can I assist you?"]

        if query:
            with st.spinner("typing..."):
                uuid_list = ss.Clustering_obj.uuid_for_query(query=query)
                print(f"cluster labels: {uuid_list}")
                response = langchain_obj.chat(
                    query=query, in_filter=uuid_list, complete_db_flag=False
                )
                response = response["response"]

            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state["responses"]:

            for i in range(len(st.session_state["responses"])):
                message(st.session_state["responses"][i], key=str(i))
                if i < len(st.session_state["requests"]):
                    message(
                        st.session_state["requests"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )


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

            ss.meeting_name = ""

            st.success("Pinecone upsert completed successfully!")
            ss.transcript_text_editor = False
            ss.updated_transcript_df_to_embed = pd.DataFrame()
            st.rerun()


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

    langchain_obj = LangChain()

    # Display the selected view
    view2(langchain_obj)  # Chat view


if __name__ == "__main__":
    init_streamlit()


# How much is the compensation for the job?
# Whats the minumim age for shadowing?
# Why where the volunteer programs cancelled?
=======
import streamlit as st
import pandas as pd
from aws.aws import *
from aws.aws_transcribe_parser import *
import os


def aws_transcribe(uploaded_file):
    transcribe_client, s3_client = create_client()
    input_bucket = 'resonate-input'
    output_bucket = 'resonate-output'  
    transcribe_job_name = uploaded_file.name
    # file = 'test.wav'
    # print(create_s3_bucket(s3_client, input_bucket))
    # print(create_s3_bucket(s3_client, output_bucket))
    # save uploaded_file to local directory in directory temp_store/

    # initial_output_bucket_count = get_object_count(s3_client, output_bucket)
    file = uploaded_file.name
    with open(file, 'wb') as f:
        f.write(uploaded_file.getbuffer())
        f.close()
    try:
        URI = upload_to_s3(s3_client, file, input_bucket)
        transcribe_audio(transcribe_client, URI, output_bucket, transcribe_job_name=transcribe_job_name)

        response = transcribe_client.list_transcription_jobs(JobNameContains=transcribe_job_name, Status='IN_PROGRESS')
        while len(response['TranscriptionJobSummaries']) > 0:
            print("Transcription in progress")
            response = transcribe_client.list_transcription_jobs(JobNameContains=transcribe_job_name, Status='IN_PROGRESS')
            time.sleep(10)

        response = transcribe_client.list_transcription_jobs(JobNameContains=transcribe_job_name, Status='COMPLETED')
        if len(response['TranscriptionJobSummaries']) > 0:
            print("Transcription completed")
            transcribe_client.delete_transcription_job(TranscriptionJobName = transcribe_job_name)
            
        print(download_from_s3(s3_client, transcribe_job_name, output_bucket, local_directory='.'))    
        transcribe_client.close()
        s3_client.close()
        return transcribe_job_name
    
    except Exception as e:
        print("Error in transcribing audio ", e)
        transcribe_client.close()
        s3_client.close()
        return ''    


def upload_files():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        with st.spinner("Processing..."):
            st.success("File uploaded successfully!")
            #transcribe_job_name = aws_transcribe(uploaded_file)
            transcribe_job_name = 'mini_ami' ## TODO: Get file name

            if transcribe_job_name:
                print("parsing transcript")
                aws_parser_runner(transcribe_job_name)

                st.session_state.flag = True

            
            # if URI:
            #     st.success("Completed")

            # # Read the CSV file using pandas
            # df = pd.read_csv(uploaded_file)

            # # Store the DataFrame in session state
            # st.session_state.df = df

            # Display the top 5 rows in a text box
            #st.text_area("Top 5 Rows:", df.head().to_string(), height=200)

def initialize_session_state():
    # Initialize session state if not exists
    if "flag" not in st.session_state:
        st.session_state.flag = None

def view1():
    st.header("View 1")
    st.write("This is the content of View 1.")

    # Call the upload_files function in View 1
    upload_files()

def view2():
    st.header("View 2")
    st.write("This is the content of View 2.")

    # Access the stored DataFrame from session state
    flag = st.session_state.flag

    if flag:

        df = pd.read_csv('mini_ami.csv')
        df.drop(df.columns[0], axis=1, inplace=True)

        # display speaker names
        speakers = df['speaker_label'].unique()
        speaker_names = []
        for speaker in speakers:
            input_name = st.text_input(f"{speaker}:", key=f"{speaker}_input")
            if input_name == '':
                input_name = speaker
            speaker_names.append(input_name)

        # display transcript
        line = ''
        for row in df.iterrows():
            line += row[1][0] + " | " + str(row[1][1]) + " | " + str(row[1][2]) + " | " + row[1][3] + '\n'

        text_df = line     
        transcript_text = st.empty()
        transcript_text.text_area("Transcript", value=text_df, height=len(df) * 50)
        st.markdown(
            """
        <style>
        textarea {
            white-space: nowrap;
        }
            """,
            unsafe_allow_html=True,
        )

        # Button to update transcript
        if st.button("Update Transcript"):
            # Use the entered speaker names to update the df
            for idx, speaker in enumerate(speakers):
                df.loc[df['speaker_label'] == speaker, 'speaker_label'] = speaker_names[idx]

            # Update transcript with modified speaker names
            updated_line = ''
            for row in df.iterrows():
                updated_line += row[1][0] + " | " + str(row[1][1]) + " | " + str(row[1][2]) + " | " + row[1][3] + '\n'
            transcript_text.text_area("Transcript", value = updated_line,  height=len(df) * 50)

    else:
        st.warning("Please upload a file in View 1 to populate the DataFrame.")

def main():
    st.set_page_config(page_title="Resonate ", page_icon=":link:")
    st.header("Resonate ")

    initialize_session_state()  # Initialize session state

    # TOGGLE SIDEBAR TO ACCEPT API KEYS
    with st.sidebar:
        OPENAI_API_KEY = st.sidebar.text_input("Enter OpenAI API Key", type="password")
        HUGGINGFACEHUB_API_KEY = st.sidebar.text_input("Enter Hugging Face API Key", type="password")

        if not OPENAI_API_KEY or not HUGGINGFACEHUB_API_KEY:
            st.sidebar.error("Please enter your API keys")
            st.stop()

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["HUGGINGFACEHUB_API_KEY"] = HUGGINGFACEHUB_API_KEY

        # Toggle free run
        global FREE_RUN
        FREE_RUN = st.sidebar.checkbox("Free run", value=False)
        ###

    # Button to toggle between views
    view_toggle_button = st.button("Toggle View")
    view_state = st.session_state.get("view_state", False)

    if view_toggle_button:
        view_state = not view_state
        st.session_state.view_state = view_state

    # Display the selected view
    if view_state:
        view2() # Using this to upload files
    else:
        view1() # Using this for Chat Interface

if __name__ == "__main__":
    main()
>>>>>>> f2f32a388ad50769da2a656d52f82e657c7949d3
