from typing import TypeVar, Union, List
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

class CustomEmbedding(Embeddings):
    Embeddable = Union[Document]
    D = TypeVar("D", bound=Embeddable, contravariant=True)

    def __init__(self):
        self.sentence_transformer_model = SentenceTransformer(model_name_or_path='intfloat/multilingual-e5-large-instruct', device='cuda')

    def __call__(self, input: D):
        embedding = self.sentence_transformer_model.encode(str(input), convert_to_tensor=True, normalize_embeddings=True)
        return [float(x) for x in embedding]

    def get_text_embedding(self, input: D):
        embedding = self.sentence_transformer_model.encode(str(input), convert_to_tensor=True, normalize_embeddings=True)
        return [float(x) for x in embedding]

    def embed_documents(self, documents: List[Document]):
        embeddings = []
        for doc in documents:
            embedding = self.sentence_transformer_model.encode(str(doc), convert_to_tensor=True, normalize_embeddings=True)
            embeddings.append([float(x) for x in embedding])
        return embeddings

    def embed_query(self, query: str):
        embedding = self.sentence_transformer_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        return [float(x) for x in embedding]
