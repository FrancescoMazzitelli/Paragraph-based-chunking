import os
import re
import shutil
import pymupdf as fitz
from unidecode import unidecode
import chromadb
import uuid
import pandas as pd

import nltk
nltk.download('punkt')

from chromadb.config import Settings
from customembedding import CustomEmbedding
from langchain.storage import InMemoryStore
# from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration

from langchain.schema.document import Document

os.environ["TOKENIZERS_PARALLELISM"] = "false"


CHROMA_DB_DIRECTORY='RAGNewLine/db'
DOCUMENT_SOURCE_DIRECTORY='Documents'

TARGET_SOURCE_CHUNKS=4
CHUNK_SIZE=800
CHUNK_OVERLAP=50
HIDE_SOURCE_DOCUMENTS=False

class MyKnowledgeBase:
    def __init__(self, path: str) -> None:
        files = os.listdir(CHROMA_DB_DIRECTORY)
        for f in files:
            db_path = CHROMA_DB_DIRECTORY + '/' + f
            if os.path.isfile(db_path):
                os.remove(db_path)
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)

        self.path = path
        self.embedder = CustomEmbedding()
        self.vectorstore = Chroma(
            collection_name="paragraphs", 
            embedding_function=HuggingFaceEmbeddings(),
            persist_directory=CHROMA_DB_DIRECTORY
        )
        self.store = InMemoryStore()
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

    def load_pdfs(self):
        loader = DirectoryLoader(
            self.path
        )
        loaded_pdfs = loader.load()
        return loaded_pdfs

    def split_documents(
        self,
        loaded_docs,
    ):
        splitter = RecursiveCharacterTextSplitter(
            separators = ['\n\n']
        )
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs

    def convert_document_to_embeddings(
        self, chunked_docs
    ):

        ids = [str(uuid.uuid1()) for _ in range(len(chunked_docs))]
        retriever = ParentDocumentRetriever(
            vectorstore = self.vectorstore,
            docstore = self.store,
            child_splitter = self.child_splitter,
            search_kwargs={"k": TARGET_SOURCE_CHUNKS}
        )

        retriever.add_documents(chunked_docs)

        print(f"Number of parent chunks  is: {len(list(self.store.yield_keys()))}")

        print(f"Number of child chunks is: {len(retriever.vectorstore.get()['ids'])}")

        self.retriever = retriever
        

    def initiate_document_injetion_pipeline(self):
        loaded_pdfs = self.load_pdfs()
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)
        
        print("=> PDF loading and chunking done.")

        self.convert_document_to_embeddings(chunked_documents)

        print("=> Vector db initialised and created.")
        print("=> All done")