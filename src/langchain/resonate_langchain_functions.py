from src.pinecone.resonate_pinecone_functions import PineconeServerless
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# from langchain_community.chat_models import ChatCohere
from langchain.chains.conversation.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)

# from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_community.callbacks import get_openai_callback

# from langchain.callbacks import get_openai_callback
import os
import json

from dotenv import load_dotenv


def load_json_config(json_file_path="./config/config.json"):
    # Use a context manager to ensure the file is properly closed after opening
    with open(json_file_path, "r") as file:
        # Load the JSON data
        data = json.load(file)

    return data


class LangChain:
    def __init__(self):

        json_config = load_json_config()
        # Load environment variables from .env file
        load_dotenv("./config/.env")

        self.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        self.pinecone = PineconeServerless()
        self.llm_temperature = json_config["LC_LLM_TEMPERATURE"]
        self.llm_model = json_config["LC_LLM_MODEL"]
        self.llm_summary_max_token_limit = json_config["LC_LLM_SUMMARY_MAX_TOKEN_LIMIT"]
        self.llm_conv_buffer_memory_window = json_config["LC_CONV_BUFFER_MEMORY_WINDOW"]

        self.llm = ChatOpenAI(
            temperature=self.llm_temperature, model_name=self.llm_model, streaming=False
        )

        self.conversation_bufw = ConversationChain(
            llm=self.llm,
            memory=ConversationSummaryBufferMemory(
                llm=self.llm, max_token_limit=self.llm_summary_max_token_limit
            ),
        )

    def prompt(self, query, context):
        system_template = SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant."
            "You are provided with a context below. You are expected to answer the user query based on the context below."
            "The context provided is a part of transcript of a meeting, in the format:"
            "Conversations in meeting: <meeting_title>"
            "Start Time - Speaker: Text \n"
            "You will respond using the context below only. If you cannot find an answer from the below context, you can ask for more information."
            "You answers should be concise and relevant to the context."
            "You can mention the meeting_title in your response if you want to refer to the meeting."
            "You are not allowed to talk about anything else other than the context below."
            "You cannot use any external information other than the context below."
            "No need to greet or say goodbye. Just answer the user query based on the context below."
            "You can also skip mentioning phrases such as : Based on the context provided. Instead simply answer the user query based on the context below.\n\n"
            "Context:\n"
            "{context}"
        )
        # system_template = SystemMessagePromptTemplate.from_template(
        #     'You are a helpful assistant.'
        #     'You will answer the user query based on the context below.'
        #     'You are also provided with the chat history of the user query and the response. You can use this information to answer the user query as well'
        #     'Context: \n'
        #     '{context}'
        # )

        human_template = HumanMessagePromptTemplate.from_template(
            " \nUser Query: {input}"
        )
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_template, human_template]
        )

        chat_prompt_value = chat_prompt.format_prompt(context=context, input=query)
        print(chat_prompt_value)
        return chat_prompt_value.to_messages()

    def query_chatbot(self, query, context):
        self.messages = self.prompt(query, context)
        resp = self.conversation_bufw(self.messages)
        print(resp)
        return resp

    def parse_conversations(self, conversations) -> str:
        data = []
        for cluster_id, cluster_df in conversations.items():
            with open(f"./data/jsonMetaDataFiles/{cluster_id}.json") as f:
                meeting_data = json.load(f)
                meeting_title = meeting_data["meeting_title"]
                data.append(f"Conversations in meeting '{meeting_title}':")
                for i, row in cluster_df.iterrows():
                    data.append(
                        f"{row['start_time']} - {row['speaker']}: {row['text']}"
                    )
                data.append("\n\n")
        data = "\n".join(data)
        return data

    def chat(self, query, in_filter: list[str] = [], complete_db_flag: bool = True):
        if "summary" in query:
            pass
        self.pinecone.query_pinecone(query, in_filter, complete_db_flag)
        conversation = self.pinecone.query_delta_conversations()
        context = self.parse_conversations(conversation)
        # print(context)
        response = self.query_chatbot(query, context)
        return response

    def count_tokens(self, chain, query):
        with get_openai_callback() as callback:
            response = chain(query)
            print(f"Call Back:  {callback}")
            return response
