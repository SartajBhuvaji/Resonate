import streamlit as st
import pandas as pd

from resonate_aws_functions import *
from resonate_pinecone_functions import init_pinecone, upsert_pinecone


def initialize_session_state():
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "api_keys_input" not in st.session_state:
        st.session_state.api_keys_input = False
    if "add_meeting" not in st.session_state:
        st.session_state.add_meeting = False
    if "chat_resonate" not in st.session_state:
        st.session_state.chat_resonate = False

    # Initialize API keys in session state if not present
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {
            "openai_api_key": "",
            "pinecone_api_key": "",
            "aws_access_key": "",
            "aws_secret_access_key": "",
        }

    if "df_transcript" not in st.session_state:
        st.session_state.df_transcript = pd.DataFrame()


def get_bot_response(user_input):
    # Replace this with your actual chatbot logic
    return f"Chatbot: You said '{user_input}'"


def api_keys_input():
    with st.form("keys_input_form"):
        # Retrieve values from session state
        openai_api_key = st.text_input(
            "OpenAPI Key:",
            type="password",
            value=st.session_state.api_keys["openai_api_key"],
        )
        pinecone_api_key = st.text_input(
            "Pinecone Key:",
            type="password",
            value=st.session_state.api_keys["pinecone_api_key"],
        )
        aws_access_key = st.text_input(
            "AWS Access Key:",
            type="password",
            value=st.session_state.api_keys["aws_access_key"],
        )
        aws_secret_access_key = st.text_input(
            "AWS Secret Access Key:",
            type="password",
            value=st.session_state.api_keys["aws_secret_access_key"],
        )

        # Add a button to save the keys
        save_button = st.form_submit_button("Save API Keys")

        if save_button:
            st.session_state.api_keys["openai_api_key"] = openai_api_key
            st.session_state.api_keys["pinecone_api_key"] = pinecone_api_key
            st.session_state.api_keys["aws_access_key"] = aws_access_key
            st.session_state.api_keys["aws_secret_access_key"] = aws_secret_access_key

            os.environ["OPENAI_API_KEY"] = st.session_state.api_keys["openai_api_key"]
            os.environ["PINECONE_API_KEY"] = st.session_state.api_keys[
                "pinecone_api_key"
            ]
            os.environ["AWS_ACCESS_KEY"] = st.session_state.api_keys["aws_access_key"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = st.session_state.api_keys[
                "aws_secret_access_key"
            ]

            st.session_state.api_keys_input = False
            st.rerun()


def add_meeting(aws_config, pinecone_config, pinecone_index):
    aws_config["aws_access_key"] = st.session_state.api_keys["aws_access_key"]
    aws_config["aws_secret_access_key"] = st.session_state.api_keys[
        "aws_secret_access_key"
    ]

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
                    file = uploaded_file.name
                    with open(file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        f.close()

                    st.session_state.df_transcript = runner(
                        file_name=file,
                        input_bucket=aws_config["aws_input_bucket"],
                        output_bucket=aws_config["aws_output_bucket"],
                        transcribe_job_name=aws_config["aws_transcribe_job_name"],
                        aws_access_key=aws_config["aws_access_key"],
                        aws_secret_access_key=aws_config["aws_secret_access_key"],
                        aws_region_name=aws_config["aws_region_name"],
                    )

                    # df_transcript.to_csv(f"{meeting_name}.csv", index=False)
                    # st.session_state.df_transcript = pd.read_csv(f"{meeting_name}.csv")

                    st.success("File uploaded and transcribed successfully!")

                    # upsert_pinecone(
                    #     pinecone_index,
                    #     transcript=df_transcript,
                    #     model_name=pinecone_config["pinecone_embedding_model_name"],
                    #     pinecone_namespace=pinecone_config["pinecone_namespace"],
                    # )
                    # st.success("Pinecone upsert completed successfully!")


def chat_resonate():
    # Input box for user to enter queries as text
    user_input = st.text_input("User Input:", value=st.session_state.user_input)

    # Send button to simulate sending user input
    if st.button("Send") and user_input:
        # Adding user input to chat history
        st.session_state.chat_history.append(f"User: {user_input}")

        # Getting response from LLM and adding it to chatbot response of chat history
        bot_response = get_bot_response(user_input)
        st.session_state.chat_history.append(bot_response)

        # Clearing the user input field for next query
        st.session_state.user_input = ""

    # Initializing chat history
    st.subheader("Chat History")
    for entry in st.session_state.chat_history:
        st.write(entry)


def init_streamlit(aws_config, pinecone_config):
    # Initializing Components

    # Initializing Pinecone
    pinecone_index = None
    pinecone = None

    # Set initial state of the sidebar
    # st.set_page_config(initial_sidebar_state="collapsed")
    st.set_page_config()
    st.title("Resonate - Meeting Chatter")

    # Initializing session state for all Streamlit components
    initialize_session_state()

    # Initializing sidebar and its components
    with st.sidebar:
        if st.sidebar.button("API Keys"):
            st.session_state.api_keys_input = not st.session_state.api_keys_input
            if st.session_state.add_meeting == True:
                st.session_state.add_meeting = False

        if st.session_state.api_keys_input:
            try:
                api_keys_input()

                # Initializing Pinecone
                pinecone, pinecone_index = init_pinecone(
                    pinecone_config["pinecone_api_key"],
                    pinecone_config["pinecone_index_name"],
                    pinecone_config["pinecone_index_metric"],
                    pinecone_config["pinecone_index_dimension"],
                    pinecone_config["pinecone_cloud_type"],
                    pinecone_config["pinecone_cloud_region"],
                )

            except Exception as e:
                print(e)

        if st.session_state.api_keys["pinecone_api_key"] != "":
            if st.sidebar.button("Add Meeting"):
                st.session_state.add_meeting = not st.session_state.add_meeting
                if st.session_state.api_keys_input == True:
                    st.session_state.api_keys_input = False

            if st.session_state.add_meeting:
                add_meeting(aws_config, pinecone_config, pinecone_index)

    if st.session_state.api_keys["pinecone_api_key"] != "":
        if not st.session_state.df_transcript.empty:
            st.dataframe(st.session_state.df_transcript)

            # Allow users to edit the 'text' column
            st.write("\n\nEdit 'text' column:")
            for index, row in st.session_state.df_transcript.iterrows():
                new_text = st.text_input(
                    f"Edit 'text' for row {index + 1}", row["text"]
                )
                st.session_state.df_transcript.at[index, "text"] = new_text

            # Display the updated DataFrame
            st.write("\n\nUpdated DataFrame:")
            st.write(st.session_state.df_transcript)

        chat_resonate()
