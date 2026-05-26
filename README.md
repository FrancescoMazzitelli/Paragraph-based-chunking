# Paragraph-Based Chunking for Retrieval-Augmented Generation

A comparative evaluation of **three text chunking strategies** for Retrieval-Augmented Generation (RAG) systems on complex Italian PDF documents. The project implements a novel **paragraph-based segmentation** approach that leverages visual and layout analysis of PDFs, and compares it against fixed-size and newline-based chunking methods using both automated NLP metrics and human-feedback analysis.

## Overview

Retrieval-Augmented Generation (RAG) is a powerful paradigm that enhances LLM outputs by retrieving relevant information from a knowledge base. However, the quality of a RAG system heavily depends on **how documents are split into chunks** before embedding and indexing. Poor chunking, such as cutting sentences mid-thought or mixing unrelated content, degrades retrieval quality and, consequently, the generated answers.

This project addresses the problem by introducing a **visually-aware paragraph segmentation** technique that analyzes PDF layout (font sizes, styles, positions, bullet points) to extract semantically coherent paragraphs. It then benchmarks this approach against two baselines in a controlled experiment using Italian technical documents and the Phi-3.5-mini-ITA language model.

## Chunking Strategies

| Strategy | Description | Implementation |
|---|---|---|
| **Paragraph-based** (proposed) | Visual/layout analysis: detects titles via font size, filters page numbers/footers, identifies bullet points and numbered lists, skips decorative separators. Paragraphs are built by grouping semantically related spans based on typographic heuristics. | `RAGParagraph/knowledgebase.py` |
| **Fixed-size** (baseline) | Naive fixed-token window: `RecursiveCharacterTextSplitter` with `chunk_size=800` characters and `chunk_overlap=50`. | `RAGFixedSize/knowledgebase.py` |
| **Newline-based** (baseline) | Splits at double-newline (`\n\n`) boundaries using `RecursiveCharacterTextSplitter` with no size limit. | `RAGNewLine/knowledgebase.py` |

## Pipeline

```
PDF Documents
     │
     ├──► Paragraph-Based Chunking ──┐
     ├──► Fixed-Size Chunking  ──────┤
     └──► Newline-Based Chunking ────┘
                     │
                     ▼
          Embedding (multilingual-e5-large-instruct)
                     │
                     ▼
          ChromaDB Vector Store
                     │
                     ▼
      ┌──────────────────────────────┐
      │  Retrieve (ParentDocument)   │
      │  Generate (Phi-3.5-mini-ITA) │
      └──────────────────────────────┘
                     │
                     ▼
      ┌──────────────────────────────┐
      │  Evaluation                  │
      │  ├─ BERTScore                │
      │  ├─ METEOR                   │
      │  ├─ BLEU                     │
      │  ├─ ROUGE                    │
      │  └─ Human survey analysis    │
      └──────────────────────────────┘
```

## Architecture

### Core Components

- **`customembedding.py`**, Wraps the `intfloat/multilingual-e5-large-instruct` SentenceTransformer model as a LangChain-compatible `Embeddings` class, running on CUDA with normalized embeddings.

- **`evaluation_chain.py`**, The main orchestration script. For each of the three chunking strategies it:
  1. Initializes the knowledge base and injects all PDF documents
  2. Iterates over a list of questions from `Questions.csv`
  3. Retrieves the most relevant chunks via `ParentDocumentRetriever`
  4. Generates answers using `microsoft/Phi-3.5-mini-ITA` (an Italian-tuned 3.8B parameter LLM)
  5. Saves results as CSV and JSON datasets

- **`RAGParagraph/knowledgebase.py`**, The proposed paragraph-based chunking strategy. Key methods:
  - `extract_filtered_blocks()`, Identifies the body font by analyzing the central portion of each page, then filters out headers/footers/page numbers based on font mismatch in the bottom margin.
  - `convert_docs_in_dataframe()`, Converts extracted text spans into a structured DataFrame with spatial (bbox), typographic (font, size, bold), and semantic (is_upper, is_bullet) features.
  - `analyze_dataframes()`, The core paragraph segmentation algorithm using heuristics:
    - **Titles**: font size > mode + 1 and length < 60 chars
    - **Bullet points**: `•`, `-`, `▪`, `*` and numbered patterns (`1.`, `a)`, etc.)
    - **Separators**: dotted/dashed/underscored lines are skipped
    - **Short bullets** (≤2 chars): pended and merged with following text
  - `convert_document_to_embeddings()`, Creates a `ParentDocumentRetriever` with parent chunks (paragraphs) stored in an `InMemoryStore` and child chunks (250-char windows) in ChromaDB.

- **`Postprocessing/evaluation_metrics.py`**, Computes automated NLP metrics comparing RAG answers against oracle (ground-truth) answers:
  - **BERTScore** (F1) using `distilbert-base-multilingual-cased`
  - **METEOR**, **BLEU**, **ROUGE** via HuggingFace `evaluate`

- **`Postprocessing/preprocessing_files.py`**, Merges all three RAG output CSVs into a single Excel file for side-by-side human review.

- **`Postprocessing/interview_analysis.py`**, Analyzes human preference survey data and generates Plotly visualizations (pie charts, bar charts, box plots) saved as PDFs.

### Key Design Decisions

- **ParentDocumentRetriever**: Parent chunks (paragraphs) are stored in memory while child chunks (smaller windows) are indexed in the vector store. This allows retrieving full paragraphs even when only a fragment matches the query embedding.
- **Custom embeddings**: `intfloat/multilingual-e5-large-instruct` provides strong multilingual performance for Italian text.
- **PDF parsing**: PyMuPDF (`fitz`) gives direct access to per-span text geometry and font metadata, enabling the visual layout analysis.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Paragraph-based-chunking.git
cd Paragraph-based-chunking

# Install dependencies
pip install -r requirements.txt
```

Additional NLTK data downloads are triggered automatically in the code (stopwords).

### Hardware Requirements

- CUDA-capable GPU with at least 8 GB VRAM (recommended: 16 GB+)
  - The Phi-3.5-mini-ITA model (~3.8B parameters) and the embedding model both run on GPU
- Tested on a single NVIDIA GPU

## Usage

### 1. Prepare Input Data

Place your PDF documents in a `Documents/` directory at the project root:

```
Documents/
├── document1.pdf
├── document2.pdf
└── ...
```

Create `Questions.csv` with a column named `questions`:

```csv
questions
What is the main contribution of this work?
How does the proposed method compare to baselines?
...
```

Create `Oracle.csv` with a column named `oracle` containing ground-truth answers:

```csv
oracle
The main contribution is...
The method outperforms baselines by...
...
```

Optionally create `Cleaned_Oracle.csv` (preprocessed oracle answers for metric evaluation).

### 2. Run the Full Evaluation

```bash
python evaluation_chain.py
```

This processes all PDFs through all three chunking strategies, generates answers, and produces:
- `DatasetCustom.csv` / `DatasetCustom.json` (paragraph-based)
- `DatasetFixedSize.csv` / `DatasetFixedSize.json` (fixed-size)
- `DatasetNewLine.csv` / `DatasetNewLine.json` (newline-based)

### 3. Automated Metrics

```bash
python Postprocessing/evaluation_metrics.py
```

Outputs BERTScore, METEOR, BLEU, and ROUGE scores for each strategy vs. oracle.

### 4. Human Evaluation Preparation

```bash
python Postprocessing/preprocessing_files.py
```

Creates `output.xlsx`, a single spreadsheet with questions, oracle answers, and all three RAG answers side by side.

### 5. Survey Analysis

Place survey results in `interview_answers.xlsx` and run:

```bash
python Postprocessing/interview_analysis.py
```

Generates PDF charts: preference distribution, per-question breakdown, per-user breakdown, and box plots.

## Project Structure

```
├── customembedding.py                 # Custom embedding model wrapper
├── evaluation_chain.py                # Main RAG evaluation pipeline
├── requirements.txt                   # Python dependencies
│
├── RAGParagraph/                      # Paragraph-based chunking
│   ├── knowledgebase.py               #   Visual layout analysis & segmentation
│   └── db/                            #   ChromaDB persistent storage
│
├── RAGFixedSize/                      # Fixed-size chunking (baseline)
│   ├── knowledgebase.py               #   800-char windows, 50 overlap
│   └── db/                            #   ChromaDB persistent storage
│
├── RAGNewLine/                        # Newline-based chunking (baseline)
│   ├── knowledgebase.py               #   Splits on double newlines
│   └── db/                            #   ChromaDB persistent storage
│
├── Postprocessing/
│   ├── evaluation_metrics.py          # BERTScore, METEOR, BLEU, ROUGE
│   ├── interview_analysis.py          # Human survey analysis + Plotly charts
│   └── preprocessing_files.py         # Merge outputs for human review
│
├── Documents/                         # Input PDFs (you create this)
├── Questions.csv                      # Input questions (you create this)
├── Oracle.csv                         # Ground-truth answers (you create this)
│
├── DatasetCustom.csv/.json            # Output: paragraph-based answers
├── DatasetFixedSize.csv/.json         # Output: fixed-size answers
├── DatasetNewLine.csv/.json           # Output: newline-based answers
└── output.xlsx                        # Output: merged for human eval
```

## Evaluation Methodology

### Automated Metrics

- **BERTScore**, Embedding-based precision, recall, and F1 using `distilbert-base-multilingual-cased`. Captures semantic similarity beyond exact n-gram overlap.
- **METEOR**, Harmonic mean of unigram precision and recall with synonym matching.
- **BLEU**, N-gram precision up to 4-grams, with brevity penalty.
- **ROUGE**, Recall-oriented measure based on n-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L).

### Human Evaluation

Survey participants evaluated the three RAG-generated answers (presented blinded) for each question, selecting the best one or "none of the above." Results are analyzed as:
- Overall preference distribution (pie chart)
- Per-question preference breakdown (stacked bar)
- Per-user preference breakdown
- Distribution of preferences per strategy (box plot)

## References

### Paper 1, Journal of Visual Language and Computing (JVLC)

> Mazzitelli, F. C., & Zimeo, E. (2025). **Paragraph-based chunking: improving Retrieval-Augmented Generation for complex documents**. *Journal of Visual Language and Computing*, 2025(2), 17–25.  
> [https://doi.org/10.18293/jvlc2025-072](https://doi.org/10.18293/jvlc2025-072)

This paper presents the full methodology and experimental results of the paragraph-based chunking approach, including detailed analysis of the visual layout heuristics, the experimental setup with Italian technical documents, and the comparative evaluation against fixed-size and newline-based strategies.

### Paper 2, International Conference on Distributed Multimedia Systems (DMS 2025)

> Mazzitelli, F. C., & Zimeo, E. (2025). **Empowering RAG for complex documents with paragraph-based chunking (S)**. *Proceedings of the 31st International Conference on Distributed Multimedia Systems*, 2025, 59–65.  
> [https://doi.org/10.18293/dms2025-033](https://doi.org/10.18293/dms2025-033)

This short paper introduces the core idea of paragraph-based chunking for RAG, presenting the preliminary results and the motivation behind using visual document analysis to improve retrieval quality in complex, multi-column PDF documents.

## Citation

If you use this code or methodology in your research, please cite both papers:

```bibtex
@article{mazzitelli2025paragraph,
  author    = {Mazzitelli, Francesco Claudio and Zimeo, Eugenio},
  title     = {Paragraph-based chunking: improving Retrieval-Augmented Generation for complex documents},
  journal   = {Journal of Visual Language and Computing},
  year      = {2025},
  volume    = {2025},
  number    = {2},
  pages     = {17--25},
  doi       = {10.18293/jvlc2025-072}
}

@inproceedings{mazzitelli2025empowering,
  author    = {Mazzitelli, Francesco Claudio and Zimeo, Eugenio},
  title     = {Empowering RAG for complex documents with paragraph-based chunking},
  booktitle = {Proceedings of the 31st International Conference on Distributed Multimedia Systems},
  year      = {2025},
  pages     = {59--65},
  doi       = {10.18293/dms2025-033}
}
```
