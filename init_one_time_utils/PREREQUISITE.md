# PRE-REQUISITE

- Getting data into pinecone

## Why?
- Resonate uses Meanshift algorithm to create clusters and creates relevant UUIDs which helps system to identify meeting transcripts stored in Pinecone Vector Database. So, if someone wants to use Resonate system, they will require summaries of atleast 10 meetings stored in abstract_summary.csv file and their relevant meetings transcripts to be stored in Pinecone with their UUIDs. But as this requirement might not be possible to achieve for everyone. We have provided pinecone_sample_data_loader.py file to do the same. If you want dont have the 10 meetins ready, you can go to `src\clustering\resonate_clustering.py` and change def graph_filter( .. nn= YOUR_VALUE).