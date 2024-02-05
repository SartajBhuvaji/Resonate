import pandas as pd
import dotenv
import json
import datetime
import os
import uuid
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
import time
dotenv.load_dotenv()

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

NAMESPACE = 'default_namespace'
master_json_file = 'master_meeting_details'

class PineconeServerless:
    def __init__(self) -> None:
        PINECONE_API_KEY = os.getenv('PINECONE_SERVERLESS_API_KEY') or 'PINECONE_SERVERLESS_API_KEY'
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
        self.index_name = INDEX_NAME
        self.meeting_title = None
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
        self.base_data_path = os.path.join(os.getcwd(), '../../','bin/data/', NAMESPACE)
        self.master_json_file = os.path.join(self.base_data_path, master_json_file)
        self.response = None

    def check_index_already_exists(self) -> bool:
        return self.index_name in self.pinecone.list_indexes()

    def _get_vector_embedder(self, EMBEDDING: str = 'OpenAI'):
        if EMBEDDING == 'OpenAI':
            return OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=self.OPENAI_API_KEY)

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


    def _create_master_json(self) -> dict:

        data = {
                "index": INDEX_NAME,
                "namespace": NAMESPACE,
                "last_conversation_no": 0,
                "meetings" :[]
        }
        if not os.path.exists(self.base_data_path):
            os.makedirs(self.base_data_path)

        meeting_details_file = os.path.join(self.base_data_path, f'{master_json_file}.json') 
        with open(meeting_details_file, 'w') as f:
            json.dump(data, f, indent=4)


    def _update_master_json(self, meeting_uuid:str, meeting_title:str, last_conversation_no:int,
                               meeting_members:list[str], meeting_video_file:bool, time_stamp:str ) -> dict:
    
        meeting_details_file = os.path.join(self.base_data_path, f'{master_json_file}.json')
        with open(meeting_details_file, 'r+') as f:
            data = json.load(f)
            print("MASTER JSON: LOADED ", data['last_conversation_no'])

            data['last_conversation_no'] = last_conversation_no 
            data['meetings'].append(
                {
                        "meeting_uuid" : meeting_uuid,
                        "meeting_title" : meeting_title,
                        "meeting_date" : time_stamp,
                        "meeting_video_file" : meeting_video_file,
                        "meeting_members" : meeting_members,
                        "meeting_summary" : None
                }
            )
            print("UPDATED MASTER JSON: ", data['last_conversation_no'] )
            return data
               
    def _get_meeting_members(self, transcript: pd.DataFrame) -> list[str]:
        return list(transcript['speaker_label'].unique())

    def _create_new_meeting_json(self, meeting_uuid:str, meeting_title:str, last_conversation_no:int,
                                  meeting_members:list[str], meeting_video_file:bool, time_stamp:str) -> dict:
        data = {
                "index": INDEX_NAME,
                "namespace": NAMESPACE,

                "meeting_title" : meeting_title,
                "meeting_uuid" : meeting_uuid,
                "meeting_date" : time_stamp,

                "last_conversation_no": last_conversation_no,
                "meeting_video_file": meeting_video_file,
                "meeting_members": meeting_members,
                "meeting_summary" : None,
        } 

        meeting_details_file = os.path.join(self.base_data_path,f'{meeting_uuid}.json') 
        with open(meeting_details_file, 'w') as f:
            json.dump(data, f, indent=4)

    def _get_last_conversation_no(self) -> list[str]:   

        meeting_details_file = os.path.join(self.base_data_path, f'{master_json_file}.json')
        with open(meeting_details_file, 'r') as f:
            data = json.load(f)
            print('last_conversation_no fetched from master json: ', data['last_conversation_no'])
            return data['last_conversation_no']

    def _set_new_meeting_json(self, meeting_uuid: str, meeting_title: str, last_conversation_no: str,
                               meeting_members: list[str], meeting_video_file: bool) -> dict:
        
        time_stamp = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        self._create_new_meeting_json(meeting_uuid, meeting_title, last_conversation_no, meeting_members, meeting_video_file, time_stamp)
        data = self._update_master_json(meeting_uuid, meeting_title, last_conversation_no, meeting_members, meeting_video_file, time_stamp)   

        meeting_details_file = os.path.join(self.base_data_path, f'{master_json_file}.json')
        with open(meeting_details_file, 'w') as f:
            json.dump(data, f, indent=4)

    def pinecone_upsert(self, transcript: pd.DataFrame, meeting_video_file: bool=False, meeting_title: str = 'Unnamed') -> None:
        '''
        Upserts the transcript into Pinecone
        '''
        texts = []
        metadatas = []
        
        last_conversation_no = self._get_last_conversation_no()
        print('last_conversation_no: ', last_conversation_no)
        last_conversation_no = int(last_conversation_no) #+ 1
        
        embed = self._get_vector_embedder(EMBEDDING)
        meeting_members = self._get_meeting_members(transcript)
        meeting_uuid = str(uuid.uuid4())
        index = self._get_index()

        for _ , record in transcript.iterrows():
            metadata = {
                'speaker': record['speaker_label'],
                'start_time': round(record['start_time'], 4), # fix a time format
                # 'meeting_no': meeting_no,
                'text': record['text'], 
                'meeting_uuid': meeting_uuid
            }        
            texts.append(record['text']) 
            metadatas.append(metadata)

            if len(texts) >= PINECONE_UPSERT_BATCH_LIMIT:
                ids = list(map(lambda i: str(i+1), range(last_conversation_no, last_conversation_no + len(texts))))
                print('ids: ', ids)
                last_conversation_no += len(texts)
                embeds = embed.embed_documents(texts)
                try:
                    index.upsert(vectors=zip(ids, embeds, metadatas), namespace=NAMESPACE)
                except Exception as e:
                    print('Error upserting into Pinecone: ', e)    
                texts = []
                metadatas = []

        if len(texts) > 0:
            ids = list(map(lambda i: str(i+1), range(last_conversation_no, last_conversation_no + len(texts))))
            last_conversation_no += len(texts)
            print('ids: ', ids)
            embeds = embed.embed_documents(texts)
            try:
                index.upsert(vectors=zip(ids, embeds, metadatas), namespace=NAMESPACE)
            except Exception as e:
                print('Error upserting into Pinecone: ', e)

        print("Sending last_conversation_no to update main " ,last_conversation_no)
        self._set_new_meeting_json(meeting_uuid, meeting_title, last_conversation_no, meeting_members, meeting_video_file)  


    def _extract_id_from_response(self, response: list) -> list[int]:
        return list(int(match['id']) for match in response['matches'])    


    def query_pinecone(self, query: str, in_filter: list[str]=[]) -> list:
        '''
        Queries Pinecone for the given query
        '''
        try:
            index = self._get_index()
            embed = self._get_vector_embedder(EMBEDDING)
            self.response = index.query(
                vector= embed.embed_documents([query])[0],
                namespace=NAMESPACE, 
                top_k=PINECONE_TOP_K_RESULTS,
                include_metadata=True,
                filter={"meeting_uuid": {"$in": in_filter}},
            )
            return self.response
        except Exception as e:
            print('Error querying Pinecone: ', e)
        return []
        

    def query_delta_conversations(self) -> pd.DataFrame: 
        '''
        Queries Pinecone for the given query and returns the delta conversations
        '''
        ids = self._extract_id_from_response(self.response)
        last_conversation_no = self._get_last_conversation_no()
        index = self._get_index()
        conversation = {}

        for id in ids: 
            left = id - DELTA if id - DELTA > 0 else 1
            right = id + DELTA if id + DELTA <= last_conversation_no else last_conversation_no
            window = [str(i) for i in range(left, right + 1)]    
            try:
                fetch_response = index.fetch(ids=window, namespace=NAMESPACE)
                conversation[id] = fetch_response
            except Exception as e:
                print('Error fetching from Pinecone for id:', id, "Error:", e)
                continue

        print('conversation length: ', len(conversation))
        return self._parse_fetch_conversations(conversation)

    def _parse_fetch_conversations(self, conversation) -> pd.DataFrame:  
        data_rows = []
        for primary_hit_id, primary_hit_data in conversation.items():
            for _ , vector_data in primary_hit_data['vectors'].items():
                id = vector_data['id']
                meeting_uuid = vector_data['metadata']['meeting_uuid']
                speaker = vector_data['metadata']['speaker']
                start_time = vector_data['metadata']['start_time']
                text = vector_data['metadata']['text']
                
                data_rows.append((primary_hit_id, id, meeting_uuid, speaker, start_time, text))

        columns = ['primary_id', 'id', 'meeting_uuid', 'speaker', 'start_time', 'text']
        delta_conversation_df = pd.DataFrame(data_rows, columns=columns)
        delta_conversation_df = delta_conversation_df.sort_values(by=['id'])
        print('LENGTH delta_conversation_df: ', len(delta_conversation_df))
        delta_conversation_df = delta_conversation_df.drop_duplicates(subset=['id'])
        return delta_conversation_df
