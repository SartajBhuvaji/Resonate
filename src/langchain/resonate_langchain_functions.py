#Description: This file contains the LangChain class which is used to query the pinecone and chatbot

import os
import json
from src.pinecone.resonate_pinecone_functions import PineconeServerless
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (
    ConversationSummaryBufferMemory,
)
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

def load_json_config(json_file_path="./config/config.json"):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data


class LangChain:
    def __init__(self):

        json_config = load_json_config()
        # check if the .env file exists using os
        if os.path.exists("./config/.env"):
            load_dotenv("./config/.env")

        self.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        print("PINECONE_API_KEY: ", self.PINECONE_API_KEY)
        if self.PINECONE_API_KEY and self.OPENAI_API_KEY :
            self.pinecone = PineconeServerless()
            print("INSIDE PINECONE")
            self.llm_temperature = json_config["LC_LLM_TEMPERATURE"]
            self.llm_model = json_config["LC_LLM_MODEL"]
            self.llm_summary_max_token_limit = json_config[
                "LC_LLM_SUMMARY_MAX_TOKEN_LIMIT"
            ]
            self.llm_conv_buffer_memory_window = json_config[
                "LC_CONV_BUFFER_MEMORY_WINDOW"
            ]

            self.llm = ChatOpenAI(
                temperature=self.llm_temperature,
                model_name=self.llm_model,
                streaming=False,
            )

            self.conversation_bufw = ConversationChain(
                llm=self.llm,
                memory=ConversationSummaryBufferMemory(
                    llm=self.llm, max_token_limit=self.llm_summary_max_token_limit
                ),
            )

    def prompt(self, query, context):

        system_template = SystemMessagePromptTemplate.from_template(
            "As a helpful assistant, your task is to provide concise and relevant answers based on the given context, which consists of a transcript excerpt from a meeting. The format of the context is as follows:"
            "\nConversations in meeting: <meeting_title>, <meeting_date>"
            "\nStart Time - Speaker: Text"
            "\nYou may receive multiple meeting transcripts, if you feel your response requires reference to multiple meetings, feel free to mention <meeting_title> in your response."
            "Your responses should strictly adhere to the provided context. If you cannot find an answer within the given context, you may ask the user for more information. Ensure clarity by referencing the meeting_title, meeting_date, and speaker as needed."
            "Your responses should be succinct and directly address the user query using the context provided. Avoid discussing any information beyond the given context or using external sources."
            "Skip unnecessary phrases like 'Based on the context provided' and focus solely on answering the users query. No need for greetings or farewells."
            "\nContext:\n"
            "{context}"
        )

        human_template = HumanMessagePromptTemplate.from_template("\nUser Query: {input}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_template, human_template]
        )
        chat_prompt_value = chat_prompt.format_prompt(context=context, input=query)
        return chat_prompt_value.to_messages()


    def query_chatbot(self, query, context):
        '''
        Query the chatbot with the given query and context
        '''
        self.messages = self.prompt(query, context)
        #print("Complete msg passed to LLM: \n"self.messages)
        #print("CONTEXT: \n", self.messages)
        resp = self.conversation_bufw(self.messages)
        return resp

    def parse_conversations(self, conversations) -> str:
        ''''
        Parse the conversations and return the data in a readable format
        '''
        '''
        Format:
        Conversations in meeting: <meeting_title1>
        Start Time - Speaker: Text
        Start Time - Speaker: Text
        .
        .
        \n\n
        Conversations in meeting: <meeting_title2>
        Start Time - Speaker: Text
        Start Time - Speaker: Text
        .
        .
        '''
        data = []
        for cluster_id, cluster_df in conversations.items():
            with open(f"./data/jsonMetaDataFiles/{cluster_id}.json") as f:
                meeting_data = json.load(f)
                meeting_title = meeting_data["meeting_title"]
                meeting_date = datetime.strptime(meeting_data["meeting_date"], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                data.append(f"Conversations in meeting: '{meeting_title}', meeting_date: {meeting_date} :")
                for i, row in cluster_df.iterrows():
                    data.append(
                        f"{row['start_time']} - {row['speaker']}: {row['text']}"
                    )
                data.append("\n\n")
        data = "\n".join(data)
        return data

    def chat(self, query, in_filter: list[str] = [], complete_db_flag: bool = False):
        '''
        Primary chat function to query the pinecone and chatbot
        If in_filter is provided, the query will be filtered based on the in_filter
        If complete_db_flag is True, the query will be searched in the complete database
        '''
        # if "summary" in query:
        #     pass # Future implementation, using semantic routing

        self.pinecone.query_pinecone(query, in_filter, complete_db_flag)
        conversation = self.pinecone.query_delta_conversations()
        # print("Conversation: ", conversation)
        context = self.parse_conversations(conversation)
        print("Context: ", context)
        response = self.query_chatbot(query, context)
        return response

    def count_tokens(self, chain, query):
        with get_openai_callback() as callback:
            response = chain(query)
            print(f"Call Back:  {callback}")
        return response
