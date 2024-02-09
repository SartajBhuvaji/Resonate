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

    file = uploaded_file.name
    with open(file, 'wb') as f:
        f.write(uploaded_file.getbuffer())
        f.close()
    try:
        URI = upload_to_s3(s3_client, file, input_bucket)

        # transcribe audio
        transcribe_audio(transcribe_client, URI, output_bucket, transcribe_job_name=transcribe_job_name)

        # Check status of transcription job
        while (
            transcribe_client.get_transcription_job(
                TranscriptionJobName=transcribe_job_name
            )["TranscriptionJob"]["TranscriptionJobStatus"] != "COMPLETED"):
            time.sleep(3)
    
        # Download transcript to loacl store from S3
        print(download_from_s3(s3_client, transcribe_job_name, output_bucket, local_directory='.'))    
        

        # Delete S3 buckets and transcribe job after use
        try:
            print("Delete S3 Bucket : ", delete_s3_bucket(s3_client, input_bucket))
            print("Delete S3 Bucket : ", delete_s3_bucket(s3_client, output_bucket))
        except:
            print(f"S3 bucket does not exist.")

        try:
            transcribe_client.delete_transcription_job(
                TranscriptionJobName=transcribe_job_name
            )
        except:
            print(f"Transcription Job does not exist.")

        # Close clients
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

            transcribe_job_name = aws_transcribe(uploaded_file)

            # if transcribe_job_name:
            #     # called from aws_transcribe_parser.py
            #     transcript_df = aws_parser_runner(transcribe_job_name)

            #     st.session_state.new_file_uploaded = True

            #     if 





def view1():
    st.header("View 1")
    st.write("This is the content of View 1.")

    upload_files() 


def view2():
    st.header("View 2")
    st.write("This is the content of View 2.")

   

def runner():
    with st.sidebar:
        OPENAI_API_KEY = st.sidebar.text_input("Enter OpenAI API Key", type="password")
        HUGGINGFACEHUB_API_KEY = st.sidebar.text_input("Enter Hugging Face API Key", type="password")

        if not OPENAI_API_KEY or not HUGGINGFACEHUB_API_KEY:
            st.sidebar.error("Please enter your API keys")
            st.stop()

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["HUGGINGFACEHUB_API_KEY"] = HUGGINGFACEHUB_API_KEY

        # store the API keys in session state
        st.session_state.OPENAI_API_KEY = OPENAI_API_KEY
        st.session_state.HUGGINGFACEHUB_API_KEY = HUGGINGFACEHUB_API_KEY

    # Button to toggle between views
    view_toggle_button = st.button("Toggle View")
    view_state = st.session_state.get("view_state", False)

    if view_toggle_button:
        view_state = not view_state
        st.session_state.view_state = view_state


    if view_state:
        view2() 
    else:
        view1()





def initialize_session_state():
    # Initialize session state if not exists
    if "new_file_uploaded" not in st.session_state:
        st.session_state.new_file_uploaded = False


def main():
    st.set_page_config(page_title="Resonate ", page_icon=":link:")
    st.header("Resonate ")
    initialize_session_state()
    

if __name__ == "__main__":
    main()