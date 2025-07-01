import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = False

response_df = pd.read_excel("interview_answers.xlsx")


num_questions = len(response_df.columns)
users = len(response_df)

rag_paragraph = 0
rag_fixed_size = 0
rag_new_line = 0
not_compatible = 0
questions = []

columns = response_df.columns
i = 1
df_questions = pd.DataFrame(columns=['question', 'paragraph', 'fixed size', 'new line', 'not comparable'])
for column in columns:
    entries = response_df[column]
    string = f"q {i}"
    i += 1

    paragraph = 0
    fixed_size = 0
    new_line = 0
    incompatible = 0
    for entry in entries:
        entry = str(entry)
        if 'Opzione 1' in entry:
            rag_paragraph +=1
            paragraph += 1
        if 'Opzione 2' in entry:
            rag_fixed_size +=1
            fixed_size += 1
        if 'Opzione 3' in entry:
            rag_new_line +=1
            new_line += 1
        if 'Nessuna delle precedenti' in entry:
            not_compatible +=1
            incompatible += 1

    new_row = pd.DataFrame({
        'question': [string],
        'paragraph': [paragraph],
        'fixed size': [fixed_size],
        'new line': [new_line],
        'not comparable': [incompatible]
    })

    df_questions = pd.concat([df_questions, new_row], ignore_index=True)

data = {
    'Chunking strategy': ['paragraph', 'fixed size', 'new line', 'not comparable'],
    'Count': [rag_paragraph, rag_fixed_size, rag_new_line, not_compatible]
}
plot_df = pd.DataFrame.from_dict(data)
fig = px.bar(plot_df, x='Count', y='Chunking strategy', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(
    font=dict(
        family="Arial",
        size=14,
        color="black"
    )
)
# fig.show()

chunking_strategy_order = ['paragraph', 'fixed size', 'new line', 'not comparable']
fig = px.pie(plot_df, 
             values='Count', 
             names='Chunking strategy', 
             color_discrete_sequence=px.colors.sequential.RdBu,
             category_orders={'Chunking strategy': chunking_strategy_order})

fig.update_layout(
    legend=dict(
        title=dict(
            text='Chunking strategy',
            font=dict(
                size=20
            )
        ),
        font=dict(
            size=18
        ),
        orientation='h',
        yanchor="bottom",
        xanchor="center", 
        x=0.5, 
        y=1.02 
    )
)
# fig.show()
fig.write_image("piechartOfFeedbacksAllQuestions.pdf")

df_melted = df_questions.melt(id_vars='question', 
                              value_vars=['paragraph', 'fixed size', 'new line', 'not comparable'], 
                              var_name='Chunking strategy', 
                              value_name='Count')

fig = px.bar(df_melted, 
             x='Count', 
             y='question', 
             color='Chunking strategy', 
             labels={'question': 'Questions', 'Count': 'Count'},
             color_discrete_sequence=px.colors.sequential.RdBu,
             orientation='h'
             )

fig.update_layout(
    height=1600,
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    xaxis=dict(
        showgrid=False, 
        gridcolor='lightgrey', 
        title='Count',
        zeroline=False,
        #tickangle=-90
    ),
    yaxis=dict(
        showgrid=False,
        gridcolor='lightgrey', 
        title='Questions',
        zeroline=False, 
        categoryorder='array',
        categoryarray=[question for question in df_melted['question'].unique()][::-1]
    ),
    font=dict(
        family="Arial",
        size=20,
        color="black"
    ),
    barmode='group', 
    bargap=0.1,
    #bargroupgap=0.5,
    legend=dict(
        title=dict(
            text='Chunking strategy',
            font=dict(
                size=22
            )
        ),
        font=dict(
            size=20
        ),
        orientation='h',
        yanchor="bottom",
        xanchor="center", 
        x=0.5, 
        y=1.02 
    )
)
# fig.show()
fig.write_image("feedbackOverQuestions.pdf")

################################################################################################################################################

df_users = pd.DataFrame(columns=['user', 'paragraph', 'fixed size', 'new line', 'not comparable'])

paragraph = 0
fixed_size = 0
new_line = 0
incompatible = 0
for user in response_df.index:
    rag_paragraph = 0
    rag_fixed_size = 0
    rag_new_line = 0
    not_compatible = 0

    for entry in response_df.loc[user]:
        entry = str(entry)  
        if 'Opzione 1' in entry:
            rag_paragraph += 1
            paragraph += 1
        if 'Opzione 2' in entry:
            rag_fixed_size += 1
            fixed_size += 1
        if 'Opzione 3' in entry:
            rag_new_line += 1
            new_line += 1
        if 'Nessuna delle precedenti' in entry:
            not_compatible += 1
            incompatible += 1

    new_row = pd.DataFrame({
        'user': [user],
        'paragraph': [rag_paragraph],
        'fixed size': [rag_fixed_size],
        'new line': [rag_new_line],
        'not comparable': [not_compatible]
    })

    df_users = pd.concat([df_users, new_row], ignore_index=True)

df_melted_users = df_users.melt(id_vars='user', 
                                value_vars=['paragraph', 'fixed size', 'new line', 'not comparable'], 
                                var_name='Chunking strategy', 
                                value_name='Count')

df_melted_users['user'] = df_melted_users['user'].astype(str)
df_melted_users['user'] = df_melted_users['user'].str.replace('user ', '') 
df_melted_users['user'] = df_melted_users['user'].astype(int) + 1  

df_melted_users['user'] = 'u ' + df_melted_users['user'].astype(str)

fig = px.bar(df_melted_users, 
             x='Count', 
             y='user', 
             color='Chunking strategy', 
             labels={'user': 'Users', 'Count': 'Count'},
             color_discrete_sequence=px.colors.sequential.RdBu,
             orientation='h')

fig.update_layout(
    height=1200,
    plot_bgcolor='rgba(255, 255, 255, 1)',
    xaxis=dict(
        showgrid=False, 
        gridcolor='lightgrey',
        title='Count',
        zeroline=False,
        #tickangle=-90
    ),
    yaxis=dict(
        showgrid=False,  
        gridcolor='lightgrey',  
        title='Users',
        zeroline=False, 
        categoryorder='array',
        categoryarray=[question for question in df_melted_users['user'].unique()][::-1]
    ),
    font=dict(
        family="Arial",
        size=20,
        color="black"
    ),
    barmode='group',  
    bargap=0.1,
    #bargroupgap=0.1,
    legend=dict(
        title=dict(
            text='Chunking strategy',
            font=dict(
                size=22
            )
        ),
        font=dict(
            size=20
        ),
        orientation='h',
        yanchor="bottom",
        xanchor="center", 
        x=0.5, 
        y=1.02 
    )
)

# fig.show()
fig.write_image("feedbackOverUsers.pdf")

fig = px.box(df_melted_users, 
             x='Chunking strategy', 
             y='Count', 
             color='Chunking strategy', 
             labels={'Chunking strategy': 'Chunking Strategy', 'Count': 'Count'},
             color_discrete_sequence=px.colors.sequential.RdBu)

# Aggiorna il layout
fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 1)',
    xaxis=dict(
        title='Chunking Strategy',
        showgrid=True,
        gridcolor='lightgrey',
        zeroline=False,
        tickangle=-90
    ),
    yaxis=dict(
        title='Count',
        showgrid=True,
        gridcolor='lightgrey',
        zeroline=False,
    ),
    font=dict(
        family="Arial",
        size=20,
        color="black"
    ),
    legend=dict(
        title=dict(
            text='Chunking strategy',
            font=dict(
                size=22
            )
        ),
        font=dict(
            size=20
        ),
        orientation='h',
        yanchor="bottom",
        xanchor="center", 
        x=0.5, 
        y=1.02 
    )
)

# Mostra il grafico
# fig.show()
fig.write_image("boxplotOfStrategies.pdf")

# print(rag_paragraph)
# print(rag_fixed_size)
# print(rag_new_line)

print("###################### -- RAW -- ######################")
total = num_questions * users
percentage_paragraph = paragraph / total
print("RAG paragragraph: " + str(percentage_paragraph * 100) + "%")
percentage_fixed = fixed_size / total
print("RAG fixed size: " + str(percentage_fixed * 100) + "%")
percentage_newline = new_line / total
print("RAG paragragraph: " + str(percentage_newline * 100) + "%")
print()

columns_to_drop_partial = [
    ""
]

columns_to_drop = [col for col in response_df.columns if any(part in col for part in columns_to_drop_partial)]
response_df = response_df.drop(columns=columns_to_drop, axis=1)

# print(f"Colonne eliminate: {columns_to_drop}")
# print(response_df.head())

# print(response_df.head())

rag_paragraph = 0
rag_fixed_size = 0
rag_new_line = 0

columns = response_df.columns
i = 1
df_questions = pd.DataFrame(columns=['question', 'paragraph', 'fixed size', 'new line', 'not comparable'])
for column in columns:
    entries = response_df[column]
    string = f"q {i}"
    i += 1

    paragraph = 0
    fixed_size = 0
    new_line = 0
    incompatible = 0
    for entry in entries:
        entry = str(entry)
        if 'Opzione 1' in entry:
            rag_paragraph +=1
            paragraph += 1
        if 'Opzione 2' in entry:
            rag_fixed_size +=1
            fixed_size += 1
        if 'Opzione 3' in entry:
            rag_new_line +=1
            new_line += 1
        if 'Nessuna delle precedenti' in entry:
            not_compatible +=1
            incompatible += 1

    new_row = pd.DataFrame({
        'question': [string],
        'paragraph': [paragraph],
        'fixed size': [fixed_size],
        'new line': [new_line],
        'not comparable': [incompatible]
    })

    df_questions = pd.concat([df_questions, new_row], ignore_index=True)
# print(rag_paragraph)
# print(rag_fixed_size)
# print(rag_new_line)

data = {
    'Chunking strategy': ['paragraph', 'fixed size', 'new line', 'not comparable'],
    'Count': [rag_paragraph, rag_fixed_size, rag_new_line, not_compatible]
}
plot_df = pd.DataFrame.from_dict(data)
fig = px.bar(plot_df, x='Chunking strategy', y='Count', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(
    font=dict(
        family="Arial",
        size=16,
        color="black"
    )
)
# fig.show()

fig = px.pie(plot_df, values='Count', names='Chunking strategy', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(
    legend=dict(
        title=dict(
            text='Chunking strategy',
            font=dict(
                size=20
            )
        ),
        font=dict(
            size=18
        ),
        orientation='h',
        yanchor="bottom",
        xanchor="center", 
        x=0.5, 
        y=1.02 
    )
)
# fig.show()
fig.write_image("pieChartWithoutIncopatileResponses.pdf")

df_melted = df_questions.melt(id_vars='question', 
                              value_vars=['paragraph', 'fixed size', 'new line', 'not comparable'], 
                              var_name='Chunking strategy', 
                              value_name='Count')

fig = px.bar(df_melted, 
             x='Count', 
             y='question', 
             color='Chunking strategy', 
             labels={'question': 'Questions', 'Count': 'Count'},
             color_discrete_sequence=px.colors.sequential.RdBu,
             orientation='h')

fig.update_layout(
    height=1000,
    plot_bgcolor='rgba(255, 255, 255, 1)', 
    xaxis=dict(
        showgrid=False, 
        gridcolor='lightgrey', 
        title='Count',
        zeroline=False,
        #tickangle=-90
    ),
    yaxis=dict(
        showgrid=False,
        gridcolor='lightgrey', 
        title='Questions',
        zeroline=False, 
        categoryorder='array',
        categoryarray=[question for question in df_melted['question'].unique()][::-1]
    ),
    font=dict(
        family="Arial",
        size=20,
        color="black"
    ),
    barmode='group', 
    bargap=0.1,
    #bargroupgap=0.1,
    legend=dict(
        title=dict(
            text='Chunking strategy',
            font=dict(
                size=22
            )
        ),
        font=dict(
            size=20
        ),
        orientation='h',
        yanchor="bottom",
        xanchor="center", 
        x=0.5, 
        y=1.02 
    )
)
# fig.show()
fig.write_image("feedbackOverQuestionsWithoutIncompatibleResponse.pdf")

print("###################### -- SENZA LE RISPOSTE CONSIDERATE INCOMPATIBILI SU 31 (TUTTE LE DOMANDE) -- ######################")
total = num_questions * users
percentage_paragraph = rag_paragraph / total
print("RAG paragragraph: " + str(percentage_paragraph * 100) + "%")
percentage_fixed = rag_fixed_size / total
print("RAG fixed size: " + str(percentage_fixed * 100) + "%")
percentage_newline = rag_new_line / total
print("RAG paragragraph: " + str(percentage_newline * 100) + "%")
print()
print("###################### -- SENZA LE RISPOSTE CONSIDERATE INCOMPATIBILI SU 24 (SOLO LE DOMANDE COMPATIBILI)-- ######################")
total = len(response_df.columns) * users
percentage_paragraph = rag_paragraph / total
print("RAG paragragraph: " + str(percentage_paragraph * 100) + "%")
percentage_fixed = rag_fixed_size / total
print("RAG fixed size: " + str(percentage_fixed * 100) + "%")
percentage_newline = rag_new_line / total
print("RAG new line: " + str(percentage_newline * 100) + "%")