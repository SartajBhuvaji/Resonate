1. .env File

You can get the required API keys from:

- [OPENAI_API_KEY](https://platform.openai.com/api-keys)
- [PINECONE_API_KEY](https://app.pinecone.io/)
- [AWS_ACCESS_KEY](https://console.aws.amazon.com/)
- [AWS_SECRET_ACCESS_KEY](https://console.aws.amazon.com/) 



2. config.json
This file contains the configuration values required for the complete application. Please go through the necessary documentation before making changes to the config.

#Pinecone
PINECONE_INDEX_NAME: An index is the highest-level organizational unit of vector data in Pinecone and must have a name.
PINECONE_VECTOR_DIMENSION: Vector dimension of the embedding model.
PINECONE_UPSERT_BATCH_LIMIT: Represents how muuch transcript rows are parallely inserted into Pinecone Serverless.
PINECONE_TOP_K_RESULTS: Represents the number of results Pinecone fetches for a query.
PINECONE_DELTA_WINDOW: Represents the +/- conversation window size fetched for TOP_K results.
PINECONE_CLOUD_PROVIDER: Cloud provider for Pinecone DB.
PINECONE_REGION: Pinecone Cloud provider region.
PINECONE_METRIC: Represents the distance metric used by Pinecone to calculate similarity.
PINECONE_NAMESPACE: A logical sepration inside the Pinecone Index.

EMBEDDING_PROVIDER: Embedding model provider to convert text to vectors.
EMBEDDING_MODEL_NAME: Name of embedding model provided by the provider.

MASTER_JSON_FILENAME: Name of master file that collects metadata.

#AWS
AWS_INPUT_BUCKET: AWS input bucket to store the audio file for [AWS Transcribe](https://aws.amazon.com/transcribe/)
AWS_OUTPUT_BUCKET: AWS output bucket that collects transcribed files.
AWS_REGION: AWS region used.
AWS_TRANSCRIBE_JOB_NAME: Default name for transcribe job.

#LangChain
LC_LLM_TEMPERATURE: Temprature value for the Large Language Model.
LC_CONV_BUFFER_MEMORY_WINDOW: Represents Conversation memoory window limit.
LC_LLM_SUMMARY_MAX_TOKEN_LIMIT: Represents the number of max tokens allowed for summary in the memory buffer.
LC_LLM_MODEL: Large Language Model used for inference.


# Documentation:
- [AWS Transcribe Docs](https://docs.aws.amazon.com/transcribe/)
- [Pinecone Docs](https://docs.pinecone.io/docs/overview)
- [Open AI Docs](https://platform.openai.com/docs/introduction)
- [Langchain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Gemini Docs](https://ai.google.dev/docs)
- [Mistral Docs](https://docs.mistral.ai/)
- [Anthropic AI Docs](https://docs.anthropic.com/claude/docs/intro-to-claude)