import os
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
import json

from RAG1 import knowledgebase as kb1
from RAGFixedSize import knowledgebase as kb2
from RAGNewLine import knowledgebase as kb3

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig, PretrainedConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_community.llms import HuggingFacePipeline

from langchain_community.llms import LlamaCpp
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
    
DOCUMENT_SOURCE_DIRECTORY = 'Documents'

model_id="Phi-3.5-mini-ITA"

llm = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    do_sample=True,
    temperature=0.000001
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

df = pd.read_csv("Questions.csv")
questions = df['questions']

features = Features({
    "question": Value("string"),
    "answer": Value("string"),
    "contexts": Sequence(Value("string")),
})

print("Inizializzando kb_custom...")
kb_custom = kb1.MyKnowledgeBase(path=DOCUMENT_SOURCE_DIRECTORY)
kb_custom.initiate_document_injetion_pipeline()

pipe = pipeline("text-generation", model=llm, tokenizer=tokenizer)

# RAG WITH CUSTOM PARAGRAPH SEGMENTATION
answers = []
contexts = []

for query in questions:
    if query == 'exit':
        break

    context = kb_custom.retriever.invoke(query)
 
    template = f"""Utilizzando le seguenti informazioni: {context}
    dai una risposta esauriente alla domanda: {query}. Rispondi solo alla domanda posta, la risposta deve essere concisa e pertinente alla domanda. 
    Se la risposta non può essere dedotta dal contesto, non fornire una risposta.
    """
    messages = [{"role": "user", "content": template}]
    response = "\""

    outputs = pipe(messages, max_new_tokens=5000, do_sample=True, temperature=0.000001)
    
    assistant_answer = None
    entry = outputs[0]['generated_text']
    for message in entry:
        if isinstance(message, dict):
            if 'role' in message:
                if message['role'] == 'assistant':
                    assistant_answer = message['content']
    response += assistant_answer
    response += "\""
    answers.append(response)

    retrieved_docs = context
    print(retrieved_docs)
    
    context_list = [doc.page_content for doc in retrieved_docs]
    contexts.append(context_list)


data = []

for q, a, c in zip(questions.to_list(), answers, contexts):
    data.append({
        "question": q,
        "answer": a,
        "contexts": c
    })

with open("DatasetCustom.json", "w") as f:
    json.dump(data, f)

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
}

dataset1 = Dataset.from_dict(data, features=features)
dataset1.to_csv("DatasetCustom.csv", index=False)


# RAG WITH FIXED SIZE SEGMENTATION
print("Inizializzando kb_fixed_size...")
kb_fixed_size = kb2.MyKnowledgeBase(path=DOCUMENT_SOURCE_DIRECTORY)
kb_fixed_size.initiate_document_injetion_pipeline()


answers = []
contexts = []

for query in questions:
    if query == 'exit':
        break

    context = kb_fixed_size.retriever.invoke(query)
    template = f"""Utilizzando le seguenti informazioni: {context}
    dai una risposta esauriente alla domanda: {query}. Rispondi solo alla domanda posta, la risposta deve essere concisa e pertinente alla domanda. 
    Se la risposta non può essere dedotta dal contesto, non fornire una risposta.
    """
    messages = [{"role": "user", "content": template}]
    response = "\""

    outputs = pipe(messages, max_new_tokens=5000, do_sample=True, temperature=0.000001)
    
    assistant_answer = None
    entry = outputs[0]['generated_text']
    for message in entry:
        if isinstance(message, dict):
            if 'role' in message:
                if message['role'] == 'assistant':
                    assistant_answer = message['content']
    response += assistant_answer
    response += "\""
    answers.append(response)

    retrieved_docs = context
    print(retrieved_docs)
    
    context_list = [doc.page_content for doc in retrieved_docs]
    contexts.append(context_list)


data = []

for q, a, c in zip(questions.to_list(), answers, contexts):
    data.append({
        "question": q,
        "answer": a,
        "contexts": c
    })

with open("DatasetFixedSize.json", "w") as f:
    json.dump(data, f)

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
}

dataset2 = Dataset.from_dict(data, features=features)
dataset2.to_csv("DatasetFixedSize.csv", index=False)

# RAG WITH NEW LINE PARAGRAPH SEGMENTATION
print("Inizializzando kb_new_line...")
kb_new_line = kb3.MyKnowledgeBase(path=DOCUMENT_SOURCE_DIRECTORY)
kb_new_line.initiate_document_injetion_pipeline()

answers = []
contexts = []

for query in questions:
    if query == 'exit':
        break
    
    context = kb_new_line.retriever.invoke(query)
    template = f"""Utilizzando le seguenti informazioni: {context}
    dai una risposta esauriente alla domanda: {query}. Rispondi solo alla domanda posta, la risposta deve essere concisa e pertinente alla domanda. 
    Se la risposta non può essere dedotta dal contesto, non fornire una risposta.
    """
    messages = [{"role": "user", "content": template}]
    response = "\""

    outputs = pipe(messages, max_new_tokens=5000, do_sample=True, temperature=0.000001)
    
    assistant_answer = None
    entry = outputs[0]['generated_text']
    for message in entry:
        if isinstance(message, dict):
            if 'role' in message:
                if message['role'] == 'assistant':
                    assistant_answer = message['content']
    response += assistant_answer
    response += "\""
    answers.append(response)

    retrieved_docs = context
    print(retrieved_docs)
    
    context_list = [doc.page_content for doc in retrieved_docs]
    contexts.append(context_list)


data = []

for q, a, c in zip(questions.to_list(), answers, contexts):
    data.append({
        "question": q,
        "answer": a,
        "contexts": c
    })

with open("DatasetNewLine.json", "w") as f:
    json.dump(data, f)

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
}

dataset3 = Dataset.from_dict(data, features=features)
dataset3.to_csv("DatasetNewLine.csv", index=False)