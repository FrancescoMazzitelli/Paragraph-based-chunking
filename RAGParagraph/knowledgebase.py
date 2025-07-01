import os
import shutil
import re
import fitz as fitz
from unidecode import unidecode
import uuid
import pandas as pd
import torch
torch.cuda.empty_cache()
from customembedding import CustomEmbedding
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.schema.document import Document

from collections import Counter

CHROMA_DB_DIRECTORY='RAG1/db'
TARGET_SOURCE_CHUNKS=1
CHUNK_SIZE=250

class MyKnowledgeBase:
    def __init__(self, path) -> None:
        files = os.listdir(CHROMA_DB_DIRECTORY)
        for f in files:
            db_path = CHROMA_DB_DIRECTORY + '/' + f
            if os.path.isfile(db_path):
                os.remove(db_path)
            if os.path.isdir(db_path):
                shutil.rmtree(db_path)
            
        self.path = path
        #self.db = chromadb.PersistentClient(path=CHROMA_DB_DIRECTORY)
        self.vectorstore = Chroma(
            collection_name="paragraphs", 
            embedding_function=CustomEmbedding(),
            persist_directory=CHROMA_DB_DIRECTORY
        )
        self.store = InMemoryStore()
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
        self.retriever = None
        
    def load_pdfs(self):
        paths = []
        if(os.path.exists(self.path)):
            for filename in os.listdir(self.path):
                filepath = os.path.join(self.path, filename)
                if os.path.isfile(filepath):
                    paths.append(filepath)
        return paths
    
    def extract_filtered_blocks(self, page, center_range=(0.4, 0.6), bottom_threshold=0.85):
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]

        center_fonts = []
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    y_center = (span["bbox"][1] + span["bbox"][3]) / 2
                    if page_height * center_range[0] <= y_center <= page_height * center_range[1]:
                        center_fonts.append((round(span["size"], 1), span["font"]))

        if not center_fonts:
            return blocks

        font_counter = Counter(center_fonts)
        font_mode_size, font_mode_name = font_counter.most_common(1)[0][0]

        filtered_blocks = []
        for block in blocks:
            if block["type"] != 0:
                filtered_blocks.append(block)
                continue

            filtered_lines = []
            for line in block["lines"]:
                filtered_spans = []
                for span in line["spans"]:
                    y_center = (span["bbox"][1] + span["bbox"][3]) / 2
                    span_font = span["font"]
                    span_size = round(span["size"], 1)

                    if (y_center > page_height * bottom_threshold):
                        if (span_size != font_mode_size or span_font != font_mode_name):
                            continue

                    filtered_spans.append(span)

                if filtered_spans:
                    filtered_lines.append({'spans': filtered_spans})

            if filtered_lines:
                filtered_blocks.append({'type': 0, 'lines': filtered_lines})

        return filtered_blocks


    def split_documents(self, paths):
        info = []
        for file in paths:
            doc = fitz.open(file)
            if doc.get_toc():
                pages_to_delete = sorted(set([entry[2] - 1 for entry in doc.get_toc()]))
                for page in reversed(pages_to_delete):
                    doc.delete_page(page)

            block_dict = {}
            page_num = 1
            for page in doc:
                filtered_blocks = self.extract_filtered_blocks(page)
                block_dict[page_num] = filtered_blocks
                page_num += 1

            info.append(block_dict)
        return info

    def convert_docs_in_dataframe(self, info_dicts):
        dfs = []

        for block_dict in info_dicts:
            rows = []
            for page_num, blocks in block_dict.items():
                for block in blocks:
                    if block['type'] == 0:
                        for line in block['lines']:
                            for span in line['spans']:
                                xmin, ymin, xmax, ymax = list(span['bbox'])
                                font_size = span['size']
                                text = unidecode(span['text'])
                                span_font = span['font']
                                is_upper = False
                                is_bold = False
                                if "bold" in span_font.lower():
                                    is_bold = True
                                if re.sub("[\\(\\[].*?[\\)\\]]", "", text).isupper():
                                    is_upper = True
                                rows.append((xmin, ymin, xmax, ymax, text, is_upper, is_bold, span_font, font_size))
            span_df = pd.DataFrame(rows, columns=['xmin','ymin','xmax','ymax', 'text', 'is_upper','is_bold','span_font', 'font_size'])
            dfs.append(span_df)
        
        return dfs  

    def analyze_dataframes(self, dfs):
        paragraphs = []

        bullet_symbols = ('•', '-', '▪', '*', '‣', '◦', '‧', '–', '→')
        numbered_bullet_pattern = re.compile(r'^([0-9]+|[a-zA-Z]+)[\.\)](?:\s|$)')
        dotted_line_pattern = re.compile(r'\.{5,}')
        dashed_line_pattern = re.compile(r'\-{5,}')
        undescore_line_pattern = re.compile(r'\_{5,}')

        for df in dfs:
            font_size_threshold = df['font_size'].mode().iloc[0]
            doc_font = df['span_font'].mode().iloc[0]

            paragraph = ''
            pending_bullet = None

            for _, row in df.iterrows():
                text = unidecode(row['text']) if row['text'] else ''
                stripped_text = text.strip()
                current_font_size = row['font_size']
                current_font = row['span_font']

                is_summary = (
                    dotted_line_pattern.search(stripped_text) or
                    dashed_line_pattern.search(stripped_text) or 
                    undescore_line_pattern.search(stripped_text))
                
                if is_summary:
                    continue

                is_bullet_point = (
                    stripped_text.startswith(bullet_symbols) or
                    numbered_bullet_pattern.match(stripped_text)
                )

                is_title = (
                    not is_bullet_point and
                    current_font_size > font_size_threshold + 1 and
                    len(stripped_text) < 60 
                )

                if is_title:
                    if paragraph.strip():
                        if len(paragraph.split()) < 100:
                            paragraph += '\n' + stripped_text
                            continue
                        else:
                            paragraphs.append(paragraph.strip())
                            paragraph = ''
                    paragraph = stripped_text + '\n'
                else:
                    if pending_bullet:
                        paragraph += '\n' + pending_bullet + ' ' + stripped_text
                        pending_bullet = None
                    elif is_bullet_point and len(stripped_text) <= 2:
                        pending_bullet = stripped_text
                    elif is_bullet_point:
                        paragraph += '\n' + stripped_text
                    elif stripped_text:
                        paragraph += ' ' + stripped_text

            if paragraph.strip():
                paragraphs.append(paragraph.strip())

        return [p for p in paragraphs if p.strip()]


    def extract_keywords(self, text):
        string = ''
        self.keywords_extractor.extract_keywords_from_text(text)
        extracted = self.keywords_extractor.get_ranked_phrases()
        string = string.join(extracted)
        embedding = self.embedder.get_text_embedding(string)
        return embedding

    def convert_document_to_embeddings(self, chunked_docs):
        embeddings = []
        ids = [str(uuid.uuid1()) for _ in range(len(chunked_docs))]
        retriever = ParentDocumentRetriever(
            vectorstore = self.vectorstore,
            docstore = self.store,
            child_splitter = self.child_splitter,
            search_kwargs={"k": TARGET_SOURCE_CHUNKS}
        )

        documents = []
        for p in chunked_docs:
            doc = Document(page_content=p)
            documents.append(doc)
        retriever.add_documents(documents)

        print(f"Number of parent chunks  is: {len(list(self.store.yield_keys()))}")
        print(f"Number of child chunks is: {len(retriever.vectorstore.get()['ids'])}")

        self.retriever = retriever

    def initiate_document_injetion_pipeline(self):
        loaded_pdfs = self.load_pdfs()
        chunked_documents = self.split_documents(paths=loaded_pdfs)
        dfs = self.convert_docs_in_dataframe(chunked_documents)
        splitted_paragraphs = self.analyze_dataframes(dfs)
        self.convert_document_to_embeddings(splitted_paragraphs)
        
        print("==> PDF loading and chunking done.")
        print("==> Vector db initialised and created.")
        print("==> All done")