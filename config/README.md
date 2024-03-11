# Project Configuration

## .env File

This file contains the necessary API keys required for the application to function properly. Obtain the API keys from the following sources:

- [OPENAI_API_KEY](https://platform.openai.com/api-keys)
- [PINECONE_API_KEY](https://app.pinecone.io/)
- [AWS_ACCESS_KEY](https://console.aws.amazon.com/)
- [AWS_SECRET_ACCESS_KEY](https://console.aws.amazon.com/)

## config.json

This JSON file holds crucial configuration values for the entire application. Please refer to the documentation before modifying any configurations.

### Pinecone Configuration

- **PINECONE_INDEX_NAME**: The name of the index, the highest-level organizational unit of vector data in Pinecone.
- **PINECONE_VECTOR_DIMENSION**: Dimensionality of the embedding model's vectors.
- **PINECONE_UPSERT_BATCH_LIMIT**: Number of transcript rows inserted into Pinecone Serverless in parallel.
- **PINECONE_TOP_K_RESULTS**: Number of results fetched by Pinecone for a query.
- **PINECONE_DELTA_WINDOW**: Conversation window size fetched for TOP_K results.
- **PINECONE_CLOUD_PROVIDER**: Cloud provider for Pinecone DB.
- **PINECONE_REGION**: Region of the Pinecone Cloud provider.
- **PINECONE_METRIC**: Distance metric used by Pinecone to calculate similarity.
- **PINECONE_NAMESPACE**: Logical separation inside the Pinecone Index.

### Embedding Provider Configuration

- **EMBEDDING_PROVIDER**: Provider of the embedding model for text-to-vector conversion.
- **EMBEDDING_MODEL_NAME**: Name of the embedding model provided by the provider.

### AWS Configuration

- **AWS_INPUT_BUCKET**: Bucket for storing audio files for AWS Transcribe.
- **AWS_OUTPUT_BUCKET**: Bucket collecting transcribed files.
- **AWS_REGION**: AWS region in use.
- **AWS_TRANSCRIBE_JOB_NAME**: Default name for Transcribe job.

### LangChain Configuration

- **LC_LLM_TEMPERATURE**: Temperature value for the Large Language Model.
- **LC_CONV_BUFFER_MEMORY_WINDOW**: Conversation memory window limit. (Future Use)
- **LC_LLM_SUMMARY_MAX_TOKEN_LIMIT**: Maximum tokens allowed for summary in the memory buffer.
- **LC_LLM_MODEL**: Large Language Model used for inference.

