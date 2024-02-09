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
