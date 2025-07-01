import pandas as pd
from unidecode import unidecode

# Elenco dei file CSV
csv_files = ['DatasetCustom.csv', 'DatasetFixedSize.csv', 'DatasetNewLine.csv']

questions = []
df_oracle = pd.read_csv('Oracle.csv')
oracles = df_oracle['oracle'].tolist()
composed_questions = []
answers_dict = { }

for file in csv_files:
    df = pd.read_csv(file)
    
    if len(questions) == 0:
        questions = df['question'].tolist()
        for question, oracle in zip(questions, oracles):
            composed = question + "\nRisposta corretta:\n" + oracle
            composed_questions.append(composed)
    
    answers = df['answer'].tolist()

    for q, answer in zip(composed_questions, answers):
        key = unidecode(q)
        if key not in answers_dict:
            answers_dict[key] = []
        answers_dict[key].append(unidecode(answer))

df_final = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in answers_dict.items()]))

df_final.to_excel('output.xlsx', index=False)

print("File Excel creato con successo!")
