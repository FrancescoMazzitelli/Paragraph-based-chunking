from typing import TypeVar, Union
from langchain_community.embeddings import HuggingFaceEmbeddings as HFE
from langchain.docstore.document import Document

class CustomEmbedding:
    Embeddable = Union[Document]
    D = TypeVar("D", bound=Embeddable, contravariant=True)

    def __init__(self):
        self.hugging_face_model = HFE()

    def __call__(self, input: D):
        embedding = self.hugging_face_model.embed_query(str(input))
        return [float(x) for x in embedding]

    def get_text_embedding(self, input: D):
        embedding = self.hugging_face_model.embed_query(str(input))
        return [float(x) for x in embedding]
