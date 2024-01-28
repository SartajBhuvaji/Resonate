import pandas as pd
import dotenv
import json
import datetime
import os
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
import time
dotenv.load_dotenv()
from tqdm.autonotebook import tqdm

# PENDING : Move these to a config file
INDEX_NAME = 'langchain-retrieval-transcript'
PINECONE_VECTOR_DIMENSION = 1536
PINECONE_UPSERT_BATCH_LIMIT = 90
PINECONE_TOP_K_RESULTS = 2
DELTA = 2
CLOUD_PROVIDER = 'aws'
REGION = 'us-west-2'
METRIC = 'cosine'

EMBEDDING = 'OpenAI'
EMBEDDING_MODEL = 'text-embedding-ada-002'

class PineconeServerless:
    def __init__(self, namespace: str = 'default_namespace') -> None:
        PINECONE_API_KEY = os.getenv('PINECONE_SERVERLESS_API_KEY') or 'PINECONE_SERVERLESS_API_KEY'
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
        self.index_name = INDEX_NAME
        self.namespace = namespace
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY)


    def check_index_already_exists(self) -> bool:
        return self.index_name in self.pinecone.list_indexes()


    def _get_index(self):
        return self.pinecone.Index(self.index_name)
    

    def _create_index(self, INDEX_NAME: str) -> None:
        try:
            self.pinecone.create_index(
                name=INDEX_NAME,
                metric=METRIC,
                dimension=PINECONE_VECTOR_DIMENSION,
            
                spec=ServerlessSpec(
                    cloud=CLOUD_PROVIDER, 
                    region=REGION,
                    # pod_type="p1.x1",
                ) 
            )    

            while not self.pinecone.describe_index(self.index_name).status['ready']:
                time.sleep(5)

        except Exception as e:
            print('Index creation failed: ', e)      


    def describe_index_stats(self) -> dict:
        try:
            index = self._get_index()
            return index.describe_index_stats()
        except Exception as e:
            print('Index does not exist: ', e)
            return {}

    
    def _delete_index(self, index_name: str) -> None:
        try:
            self.pinecone.delete_index(index_name)
        except Exception as e:
            print('Index does not exist: ', e)


    def _set_new_meeting_json(self, namespace: str, last_conversation_no: str, meeting_video_file: bool,meeting_members: list[str]) -> dict:
        data = {
                "index": INDEX_NAME,
                "namespace": namespace,
                "last_meeting_no": 1,
                "last_conversation_no": last_conversation_no,
                "unique_meeting_members": meeting_members,
                "meetings": [
                    {
                        "meeting_no": 1,
                        "meeting_last_conversation_no": last_conversation_no,
                        "meeting_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "meeting_video_file": meeting_video_file,
                        "meeting_members": meeting_members,
                    },
                ]
            }
        return data
    

    def _append_meeting_details(self,meeting_details_file: str, last_meeting_no: int, last_conversation_no: int, meeting_video_file: bool, meeting_members: list[str]) -> dict:
        with open(meeting_details_file, 'r') as f:
            data = json.load(f)
            data['last_meeting_no'] = last_meeting_no + 1
            data['last_conversation_no'] = last_conversation_no 
            unique_meeting_members = set(data.get('unique_meeting_members', []))
            unique_meeting_members.update(meeting_members)
            data['unique_meeting_members'] = list(unique_meeting_members)

            data['meetings'].append(
                {
                    "meeting_no": last_meeting_no + 1,
                    "meeting_last_conversation_no": last_conversation_no,
                    "meeting_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "meeting_video_file": meeting_video_file,
                    "meeting_members": meeting_members,
                    
                }
            )
            return data    


    def _get_meeting_details(self, namespace: str, base_data_path: str) -> str:
        meeting_details_file = os.path.join(base_data_path, f'{namespace}.json')  
        if not os.path.exists(meeting_details_file):
           print('Namespace does not exist in JSON Store')
           return 1, 0
        
        with open(meeting_details_file, 'r') as f:
            data = json.load(f)
            return data['last_meeting_no'] + 1, data['last_conversation_no']
        

    def _set_meeting_details(self, namespace: str, last_meeting_no: int, last_conversation_no: int, meeting_video_file: bool, meeting_members: list[str], base_data_path: str ) -> None:
        # PENDING : Update the meeting details in the Firebase database

        if not os.path.exists(base_data_path):
            os.makedirs(base_data_path)

        meeting_details_file = os.path.join(base_data_path, f'{namespace}.json')  

        if not os.path.exists(meeting_details_file):
            data = self._set_new_meeting_json(namespace, last_conversation_no,meeting_video_file, meeting_members)
        else:
            data = self._append_meeting_details(meeting_details_file, last_meeting_no, last_conversation_no, meeting_video_file, meeting_members)

        with open(meeting_details_file, 'w') as f:
            json.dump(data, f, indent=4)

    def _get_meeting_members(self, transcript: pd.DataFrame) -> list[str]:
        return list(transcript['speaker_label'].unique())
    
        
    def _get_vector_embedder(self, EMBEDDING: str = 'OpenAI'):
        if EMBEDDING == 'OpenAI':
            return OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=self.OPENAI_API_KEY)
        
    
    # PENDING 3 : get_entire_namespace_data 
    def get_entire_namespace_data(self, namespace: str) -> pd.DataFrame: ##################
        try:
            index = self._get_index()
            self.response = index.fetch_all(namespace=namespace) # does not work
            return pd.DataFrame(self.response['vectors']).T
        except Exception as e:
            print('Error fetching from Pinecone: ', e)
            return pd.DataFrame(columns=['id', 'meeting_id', 'speaker', 'start_time', 'text'])    


    def pinecone_upsert(self, transcript: pd.DataFrame, meeting_video_file: bool=False) -> None:
        texts = []
        metadatas = []

        base_data_path = os.path.join(os.getcwd(), '../../','bin/data/', INDEX_NAME)
        meeting_no, last_conversation_no = self._get_meeting_details(self.namespace, base_data_path) 

        meeting_members = self._get_meeting_members(transcript) 
        embed = self._get_vector_embedder(EMBEDDING)
        index = self._get_index()

        for _ , record in transcript.iterrows():
            metadata = {
                'speaker': record['speaker_label'],
                'start_time': round(record['start_time'], 4), # fix a time format
                'meeting_id': meeting_no,
                'text': record['text'], 
            }

            texts.append(record['text']) 
            metadatas.append(metadata)

            if len(texts) >= PINECONE_UPSERT_BATCH_LIMIT:
                ids = list(map(lambda i: str(i+1), range(last_conversation_no, last_conversation_no + len(texts))))
                last_conversation_no += len(texts)
                ids = [meeting_no] * len(texts)
                embeds = embed.embed_documents(texts)
                try:
                    index.upsert(vectors=zip(ids, embeds, metadatas), namespace=self.namespace)
                except Exception as e:
                    print('Error upserting into Pinecone: ', e)    
                texts = []
                metadatas = []

        if len(texts) > 0:
            ids = list(map(lambda i: str(i+1), range(last_conversation_no, last_conversation_no + len(texts))))
            last_conversation_no += len(texts)
            embeds = embed.embed_documents(texts)
            try:
                index.upsert(vectors=zip(ids, embeds, metadatas), namespace=self.namespace)
            except Exception as e:
                print('Error upserting into Pinecone: ', e)

        self._set_meeting_details(self.namespace, meeting_no, last_conversation_no, meeting_video_file, meeting_members, base_data_path)     


    def query_pinecone(self, query: str, namespace: str) -> list:
        try:
            index = self._get_index()
            embed = self._get_vector_embedder(EMBEDDING)
            self.response = index.query(
                vector= embed.embed_documents([query])[0],
                namespace=namespace, 
                top_k=PINECONE_TOP_K_RESULTS,
                include_metadata=True,
                # filter={"meeting_id": {"$in":[1, 2]}},
            )
            return self.response
        except Exception as e:
            print('Error querying Pinecone: ', e)
            return []
        

    def _extract_id_from_response(self, response: list) -> list[int]:
        return list(int(match['id']) for match in response['matches'])


    # PENDING 2 : query_every_namespace returns all data from all indexes
    def query_every_namespace(self, query:str) -> list: ################## ##################
        index_info = self.describe_index_stats()
        all_namespaces = index_info['namespaces']
        print('All namespaces: ', all_namespaces)
        self.response = []
        for namespace in all_namespaces:
            try:
                namespace_response = self.query_pinecone(query, namespace=namespace)
                self.response.append(namespace_response)
            except Exception as e:
                print('Error querying Pinecone namespace: ', namespace, 'Error:' , e)
                continue

        return self.response

    def query_delta_conversations(self, namespace: str) -> pd.DataFrame:
        ids = self._extract_id_from_response(self.response)
        base_data_path = os.path.join(os.getcwd(), '../../','bin/data/', INDEX_NAME)
        _ , last_conversation_no = self._get_meeting_details(self.namespace, base_data_path) 

        index = self._get_index()
        conversation = {}
        print("Fetching conversation IDs: ", ids)
        for id in ids:
            left = id - DELTA if id - DELTA > 0 else 1
            right = id + DELTA if id + DELTA <= last_conversation_no else last_conversation_no
            window = [str(i) for i in range(left, right+1)]
            print("Fetching ID in window size: ", window)
            try:
                fetch_response = index.fetch(ids=window,namespace=namespace) 
                conversation[id] = fetch_response 

            except Exception as e:
                print('Error fetching from Pinecone for id: ', id, " Error: ", e)    
                continue

        return self._parse_query_delta_conversations(conversation)


    def _parse_query_delta_conversations(self, conversation) -> pd.DataFrame:  
        df = pd.DataFrame(columns=['primary_id', 'id', 'meeting_id', 'speaker', 'start_time', 'text'])
        for primary_id, values in conversation.items():
            for id_, details in values['vectors'].items():
                meeting_id = details['metadata']['meeting_id']
                start_time = details['metadata']['start_time']
                speaker = details['metadata']['speaker']
                text = details['metadata']['text']

                data = {'primary_id': primary_id, 'id': id_, 
                        'meeting_id': meeting_id, 'speaker': speaker,
                        'start_time': start_time, 'text': text}

                df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)     

        return df
