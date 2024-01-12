import pinecone
import dotenv
import os
import time
import pandas as pd

class pinecone_db:
    def __init__(self) -> None:
        dotenv.load_dotenv()
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
        
        if pinecone.is_initialized():
            raise RuntimeError('Pinecone is already initialized')
        
        try:
            self.pinecone.init(api_key= PINECONE_API_KEY, environment= PINECONE_ENVIRONMENT)
        except Exception as e:
            print('Error initializing Pinecone: ', e)


    def create_index(self, index_name: str, metric: str, dimension: int) -> None:
        if index_name in self.pinecone.list_indexes():
            print('Index already exists')
            return
        
        try:
            self.pinecone.create_index(name=index_name, metric=metric, shards=1, dimension=dimension)            
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)

        except Exception as e:
            print('Index already exists: ', e) 


    def list_indexes(self) -> list:
        return self.pinecone.list_indexes()


    def describe_index_stats(self, index_name: str) -> dict:
        try:
            index = pinecone.Index(index_name)
            return index.describe_index_stats()
        except Exception as e:
            print('Index does not exist: ', e)
            return {}     


    def delete_index(self, index_name: str) -> None:
        try:
            self.pinecone.delete_index(index_name)
        except Exception as e:
            print('Index does not exist: ', e)
