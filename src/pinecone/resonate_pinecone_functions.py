# Description: Pinecone Serverless Class for Resonate
# Reference: https://www.pinecone.io/docs/

import datetime
import uuid
import json
import os
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

def load_json_config(json_file_path="./config/config.json"):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data


class PineconeServerless:
    def __init__(self) -> None:
        print("Pinecone Serverless Initializing")
        json_config = load_json_config()
        load_dotenv("./config/.env")

        self.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if self.PINECONE_API_KEY is not None:
            self.pinecone = Pinecone(api_key=self.PINECONE_API_KEY)
        self._init_config(json_config)
        self.meeting_title = None
        self.base_data_path = "./data/jsonMetaDataFiles/"
        self.master_json_file = f"{self.base_data_path}{self.master_json_filename}.json"
        self._create_master_json()
        self._create_index()
        self.response = None
        print("Pinecone Serverless Initialized")

    def _init_config(self, json_config) -> None:
        for key, value in json_config.items():
            setattr(self, key.lower(), value)

    def check_index_already_exists(self) -> bool:
        return self.pinecone_index_name in self.pinecone.list_indexes()

    def _get_vector_embedder(self):
        if self.embedding_provider == "OpenAI":
            return OpenAIEmbeddings(model=self.embedding_model_name)
        else:
            raise ValueError("Invalid Embedding Model")

    def _get_index(self):
        return self.pinecone.Index(self.pinecone_index_name)

    def _create_index(self) -> None:
        '''
        Creates a new index in Pinecone if it does not exist
        '''
        pinecone_indexes_list = [
            index.get("name")
            for index in self.pinecone.list_indexes().get("indexes", [])]

        if self.pinecone_index_name not in pinecone_indexes_list:
            try:
                self.pinecone.create_index(
                    name=self.pinecone_index_name,
                    metric=self.pinecone_metric,
                    dimension=self.pinecone_vector_dimension,
                    spec=ServerlessSpec(
                        cloud=self.pinecone_cloud_provider,
                        region=self.pinecone_region,
                        # pod_type="p1.x1", # Future use
                    ),
                )

                while not self.pinecone.describe_index(self.pinecone_index_name).status["ready"]:
                    time.sleep(5)

            except Exception as e:
                print("Index creation failed: ", e)

    def describe_index_stats(self) -> dict:
        try:
            index = self._get_index()
            return index.describe_index_stats()
        except Exception as e:
            print("Index does not exist: ", e)
            return {}

    def _delete_index(self) -> None:
        try:
            self.pinecone.delete_index(self.pinecone_index_name)
        except Exception as e:
            print("Index does not exist: ", e)

    def _create_master_json(self) -> None:
        '''
        Check if the master json file exists, if not, create it
        '''
        os.makedirs(os.path.dirname(self.base_data_path), exist_ok=True)
        if not os.path.exists(self.master_json_file):
            with open(self.master_json_file, "w") as file:
                data = {
                    "index": self.pinecone_index_name,
                    "namespace": self.pinecone_namespace,
                    "last_conversation_no": 0,
                    "meeting_uuids": [],
                    "meetings": [],
                }

                with open(self.master_json_file, "w") as f:
                    json.dump(data, f, indent=4)

                print(f"Created {self.master_json_file}")

    def _update_master_json(
        self,
        meeting_uuid: str,
        meeting_title: str,
        last_conversation_no: int,
        meeting_video_file: bool,
        time_stamp: str,
    ) -> dict:
        '''
        Updates the master json file with the new meeting details
        '''
        with open(self.master_json_file, "r+") as f:
            data = json.load(f)
            data["meeting_uuids"] = list(set(data["meeting_uuids"] + [meeting_uuid]))
            data["last_conversation_no"] = last_conversation_no
            data["meetings"].append(
                {
                    "meeting_uuid": meeting_uuid,
                    "meeting_title": meeting_title,
                    "meeting_date": time_stamp,
                    "meeting_video_file": meeting_video_file,
                }
            )
            return data

    def _get_meeting_members(self, transcript: pd.DataFrame) -> list[str]:
        return list(transcript["speaker_label"].unique())

    def _create_new_meeting_json(
        self,
        meeting_uuid: str,
        meeting_title: str,
        last_conversation_no: int,
        meeting_members: list[str],
        meeting_video_file: bool,
        time_stamp: str,
        meeting_summary: str,
    ) -> dict:
        '''
        Creates a new json file for the meeting details
        '''
        data = {
            "index": self.pinecone_index_name,
            "namespace": self.pinecone_namespace,
            "meeting_title": meeting_title,
            "meeting_uuid": meeting_uuid,
            "meeting_date": time_stamp,
            "last_conversation_no": last_conversation_no,
            "meeting_video_file": meeting_video_file,
            "meeting_members": meeting_members,
            "meeting_summary": meeting_summary,
        }

        meeting_details_file = os.path.join(self.base_data_path, f"{meeting_uuid}.json")
        with open(meeting_details_file, "w") as f:
            json.dump(data, f, indent=4)

    def _get_last_conversation_no(self) -> list[str]:

        with open(self.master_json_file, "r") as f:
            data = json.load(f)

            return data["last_conversation_no"]

    def _set_new_meeting_json(
        self,
        meeting_uuid: str,
        meeting_title: str,
        last_conversation_no: str,
        meeting_members: list[str],
        meeting_video_file: bool,
        meeting_summary: str,
    ) -> dict:
        '''
        Updates the master json file with the new meeting details
        '''
        time_stamp = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        self._create_new_meeting_json(
            meeting_uuid,
            meeting_title,
            last_conversation_no,
            meeting_members,
            meeting_video_file,
            time_stamp,
            meeting_summary,
        )
        data = self._update_master_json(
            meeting_uuid,
            meeting_title,
            last_conversation_no,
            meeting_video_file,
            time_stamp,
        )

        with open(self.master_json_file, "w") as f:
            json.dump(data, f, indent=4)

    def _convert_to_hr_min_sec(self, time_in_minutes) -> str:
        # Hr:Min:Sec
        hours = int(time_in_minutes // 60)
        minutes = int(time_in_minutes % 60)
        seconds = int((time_in_minutes - int(time_in_minutes)) * 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def pinecone_upsert(
        self,
        transcript: pd.DataFrame,
        meeting_uuid: str = "",
        meeting_video_file: bool = False,
        meeting_title: str = "Unnamed",
        meeting_summary: str = "",
    ) -> None:
        """
        Upserts the transcript into Pinecone
        """
        print("Upserting transcript into Pinecone...")
        texts = []
        metadatas = []

        last_conversation_no = self._get_last_conversation_no()
        last_conversation_no = int(last_conversation_no) 

        embed = self._get_vector_embedder()
        meeting_members = self._get_meeting_members(transcript)
        index = self._get_index()

        for _, record in transcript.iterrows():
            start_time = self._convert_to_hr_min_sec(record["start_time"])

            metadata = {
                "speaker": record["speaker_label"],
                "start_time": start_time,
                "text": record["text"],
                "meeting_uuid": meeting_uuid,
            }
            texts.append(record["text"])
            metadatas.append(metadata)

            if len(texts) >= self.pinecone_upsert_batch_limit:
                ids = list(
                    map(
                        lambda i: str(i + 1),
                        range(last_conversation_no, last_conversation_no + len(texts)),
                    )
                )
                last_conversation_no += len(texts)
                embeds = embed.embed_documents(texts)

                try:
                    index.upsert(
                        vectors=zip(ids, embeds, metadatas),
                        namespace=self.pinecone_namespace,
                    )
                except Exception as e:
                    print("Error upserting into Pinecone: ", e)
                texts = []
                metadatas = []

        # Upsert the remaining texts
        if len(texts) > 0:
            ids = list(
                map(
                    lambda i: str(i + 1),
                    range(last_conversation_no, last_conversation_no + len(texts)),
                )
            )
            last_conversation_no += len(texts)
            embeds = embed.embed_documents(texts)

            try:
                index.upsert(
                    vectors=zip(ids, embeds, metadatas),
                    namespace=self.pinecone_namespace,
                )
            except Exception as e:
                print("Error upserting into Pinecone: ", e)

        self._set_new_meeting_json(
            meeting_uuid,
            meeting_title,
            last_conversation_no,
            meeting_members,
            meeting_video_file,
            meeting_summary,
        )

        print("Upserted transcript into Pinecone")

    def _extract_id_from_response(self, response: list) -> list[int]:
        if response:
            return list(int(match["id"]) for match in response["matches"])
        return []

    def query_pinecone(
        self, query: str, in_filter: list[str] = [], complete_db_flag: bool = False
    ) -> list:
        """
        Queries Pinecone for the given query, where in_filter is the list of meeting_uuids to filter the query 
        and if complete_db_flag is True, the entire database is queried
        """
        # for using without clustering, complete_db_flag to True
        try:
            index = self._get_index()
            embed = self._get_vector_embedder()

            filter = None if complete_db_flag else {"meeting_uuid": {"$in": in_filter}}

            self.response = index.query(
                vector=embed.embed_documents([query])[0],
                namespace=self.pinecone_namespace,
                top_k=self.pinecone_top_k_results,
                include_metadata=True,
                filter=filter,
            )
            return self.response
        except Exception as e:
            print("Error querying Pinecone: ", e)
        return []


    def query_delta_conversations(self) -> pd.DataFrame:
        """
        Queries Pinecone for the given query and returns the delta conversations (conversation window around the query result)
        """
        ids = self._extract_id_from_response(self.response)
        last_conversation_no = self._get_last_conversation_no()
        index = self._get_index()
        conversation = {}

        for id in ids:
            left = (
                id - self.pinecone_delta_window
                if id - self.pinecone_delta_window > 0
                else 1
            )
            right = (
                id + self.pinecone_delta_window
                if id + self.pinecone_delta_window <= last_conversation_no
                else last_conversation_no
            )
            window = [str(i) for i in range(left, right + 1)]
            try:
                # print("Fetch window: ", window)
                print("Contextual Window Conversation IDs: ", window)
                fetch_response = index.fetch(
                    ids=window, namespace=self.pinecone_namespace
                )
                conversation[id] = fetch_response
            except Exception as e:
                print("Error fetching from Pinecone for id:", id, "Error:", e)
                continue
        # print('conversation length: ', len(conversation))
        return self._parse_fetch_conversations(conversation)


    def _parse_fetch_conversations(self, conversation)-> dict:
        '''
        Parses the conversation dictionary and returns a grouped_dfs
        '''
        data_rows = []
        for primary_hit_id, primary_hit_data in conversation.items():
            for _, vector_data in primary_hit_data["vectors"].items():
                id = vector_data["id"]
                meeting_uuid = vector_data["metadata"]["meeting_uuid"]
                speaker = vector_data["metadata"]["speaker"]
                start_time = vector_data["metadata"]["start_time"]
                text = vector_data["metadata"]["text"]

                data_rows.append(
                    (primary_hit_id, id, meeting_uuid, speaker, start_time, text)
                )

        columns = ["primary_id", "id", "meeting_uuid", "speaker", "start_time", "text"]
        delta_conversation_df = pd.DataFrame(data_rows, columns=columns)
        delta_conversation_df = delta_conversation_df.sort_values(by=["id"])
        delta_conversation_df = delta_conversation_df.drop_duplicates(subset=["id"])

        # creating separate df for rows with same meeting_cluster_id
        grouped_dfs = {
            group_name: group.reset_index(drop=True, inplace=False)
            for group_name, group in delta_conversation_df.groupby("meeting_uuid")
        }
        # return delta_conversation_df
        return grouped_dfs


if __name__ == "__main__":
    pinecone = PineconeServerless()
    print(pinecone.describe_index_stats())

    for i in range(1, 3):
        print(i)
        transcript = pd.read_csv(f"./data/transcriptFiles/healthcare_{i}.csv")
        transcript.dropna(inplace=True)
        pinecone.pinecone_upsert(
            transcript,
            meeting_uuid=str(uuid.uuid4()),
            meeting_video_file=False,
            meeting_title=f"Healthcare Meeting {i}",
            meeting_summary=f"Healthcare Meeting Summary Meeting {i}",
        )
        time.sleep(5)
    print(pinecone.describe_index_stats())

    query = "I am one of the directors in Wappingers Central School District."
    response1 = pinecone.query_pinecone(query, "", True)
    print(response1)
