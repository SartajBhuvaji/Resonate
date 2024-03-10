# Description: Using Facebook's Faiss library to perform semantic search according to the query
# Reference: https://deepnote.com/blog/semantic-search-using-faiss-and-mpnet

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import faiss

class SemanticEmbedding:

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embedding(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().numpy()


class FaissForQuerySearch:

    def __init__(self, model, dim=768):
        self.index = faiss.IndexFlatIP(dim)
        # Maintaining the document data
        self.doc_map = dict()
        self.model = model
        self.ctr = 0
        self.uuid = []
        self.labels = []

    def search_query(self, query, k=1):
        D, I = self.index.search(self.model.get_embedding(query), k)
        return [
            self.labels[idx] for idx, score in zip(I[0], D[0]) if idx in self.doc_map
        ]

    def add_summary(self, document_text, id, predicted_label):
        self.index.add((self.model.get_embedding(document_text)))  # index
        self.uuid.append(id)  # appending the uuid
        self.labels.append(predicted_label)  # appending the predicted label
        self.doc_map[self.ctr] = document_text  # store the original document text
        self.ctr += 1
