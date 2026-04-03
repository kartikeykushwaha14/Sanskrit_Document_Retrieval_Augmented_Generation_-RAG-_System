# Sanskrit RAG System - User Guide

## 🎯 Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
pip install flask numpy python-docx --break-system-packages
```

### Step 2: Test the System
```bash
python demo.py
```

### Step 3: Start Web Interface
```bash
python app.py
```

### Step 4: Access the Application
Open your browser: `http://localhost:5000`

---

## 📖 Detailed Usage Guide

### Running the Demo Script

The demo script (`demo.py`) demonstrates the full capabilities of the RAG system:

```bash
python demo.py
```

**What it does:**
1. Loads the Sanskrit document
2. Processes and chunks the text
3. Builds the TF-IDF index
4. Runs 6 test queries (Sanskrit and English)
5. Displays detailed results
6. Shows system statistics

**Expected Output:**
- Document processing status
- Number of chunks created
- Query results with relevance scores
- Retrieved context previews
- System statistics (vocabulary size, etc.)

---

### Using the Web Interface

#### Starting the Server

```bash
python app.py
```

**Console Output:**
```
Loading existing RAG index...
✓ RAG system loaded successfully

======================================================================
🚀 Starting Sanskrit RAG Web Interface
======================================================================

Access the application at: http://localhost:5000
```

#### Using the Interface

1. **Query Input Box**: Enter your question in Sanskrit (Devanagari) or English
2. **Context Selection**: Choose how many relevant contexts to retrieve (2, 3, or 5)
3. **Search Button**: Click to submit your query
4. **Results Display**: View the generated answer and retrieved contexts

#### Features

**Example Queries Section:**
- Click any example query to auto-fill the input box
- Includes both Sanskrit and English examples

**System Statistics Panel:**
- Real-time display of indexed documents
- Vocabulary size
- Number of chunks

**Retrieved Contexts:**
- Each context shows:
  - Source document name
  - Similarity score (0.0 to 1.0)
  - Full text of the retrieved chunk

---

### Programmatic Usage

#### Basic Example

```python
from sanskrit_rag_system import SanskritRAGSystem

# Initialize
rag = SanskritRAGSystem(chunk_size=400, overlap=50)

# Load documents
rag.ingest_documents(['path/to/document.docx'])

# Build index
rag.build_index()

# Query
result = rag.query("कालीदासः कः आसीत्?", top_k=3)

# Access results
print(result['answer'])
print(f"Found {result['num_contexts']} relevant contexts")
```

#### Advanced Example with Saving

```python
from sanskrit_rag_system import SanskritRAGSystem

# Initialize and process
rag = SanskritRAGSystem(chunk_size=400, overlap=50)
rag.ingest_documents(['doc1.txt', 'doc2.docx'])
rag.build_index()

# Save for future use
rag.save_index('my_sanskrit_corpus.pkl')

# Later, load and use
rag_loaded = SanskritRAGSystem()
rag_loaded.load_index('my_sanskrit_corpus.pkl')

# Query
result = rag_loaded.query("What is the moral of the story?")
```

---

## 🔍 Query Examples

### Sanskrit Queries (Devanagari)

1. **Story Content**
   ```
   मूर्खभृत्यस्य कथा किम् अस्ति?
   (What is the story of the foolish servant?)
   ```

2. **Character Information**
   ```
   कालीदासः कः आसीत्?
   (Who was Kalidasa?)
   ```

3. **Plot Elements**
   ```
   घण्टाकर्णः कः?
   (Who is Ghantakarna?)
   ```

4. **Events**
   ```
   वृद्धा किम् अकरोत्?
   (What did the old woman do?)
   ```

### English Queries

1. **General Questions**
   ```
   What is the story about the old woman?
   Tell me about King Bhoj
   What happened to the foolish servant?
   ```

2. **Thematic Questions**
   ```
   What lesson does the story teach?
   What is the moral of the story?
   How did Kalidasa show his cleverness?
   ```

3. **Character Questions**
   ```
   Who was Shankhanaad?
   Tell me about Govardhan Das
   Who solved the problem in the kingdom?
   ```

---

## 📊 Understanding Results

### Result Structure

```python
{
    'question': 'User query',
    'answer': 'Generated answer text',
    'retrieved_contexts': [
        {
            'text': 'Full context text',
            'score': 0.1234,  # Similarity score
            'source': 'document_name.docx',
            'chunk_id': 5
        },
        # ... more contexts
    ],
    'num_contexts': 3
}
```

### Similarity Scores

- **Range**: 0.0 to 1.0
- **Interpretation**:
  - `0.0 - 0.1`: Low relevance
  - `0.1 - 0.3`: Moderate relevance
  - `0.3 - 0.5`: Good relevance
  - `0.5+`: High relevance

**Note**: The exact score depends on query and document characteristics. Even lower scores can contain useful information.

---

## ⚙️ Configuration Options

### Chunking Parameters

```python
SanskritRAGSystem(
    chunk_size=400,  # Characters per chunk (default: 400)
    overlap=50       # Overlap between chunks (default: 50)
)
```

**Guidelines:**
- **Smaller chunks** (200-300): Better precision, more granular retrieval
- **Larger chunks** (500-600): Better context, may include irrelevant info
- **Overlap** (30-100): Ensures important info isn't split across chunks

### Retrieval Parameters

```python
result = rag.query(
    question="Your question",
    top_k=3,          # Number of contexts (default: 3)
    verbose=True      # Print detailed info (default: True)
)
```

**top_k Guidelines:**
- `top_k=1`: Fastest, may miss relevant info
- `top_k=3`: Balanced (recommended)
- `top_k=5`: Comprehensive, but may include noise

---

## 📝 Adding New Documents

### Single Document

```python
rag = SanskritRAGSystem()
rag.ingest_documents(['new_document.txt'])
rag.build_index()
```

### Multiple Documents

```python
documents = [
    'document1.txt',
    'document2.docx',
    'document3.txt'
]

rag = SanskritRAGSystem()
rag.ingest_documents(documents)
rag.build_index()
rag.save_index('multi_doc_index.pkl')
```

### Updating Index

**Note**: To add new documents to an existing index, you need to rebuild:

```python
# Load existing
rag = SanskritRAGSystem()
rag.load_index('old_index.pkl')

# Add new documents
new_docs = ['new_doc.txt']

# Combine with existing raw documents
all_docs = rag.raw_documents + [rag.load_document(d) for d in new_docs]

# Rebuild completely
rag = SanskritRAGSystem()
rag.ingest_documents(all_docs)
rag.build_index()
rag.save_index('updated_index.pkl')
```

---

---

## 📚 Example Workflows

### Workflow 1: Research Assistant

```python
# Setup
rag = SanskritRAGSystem()
rag.ingest_documents(['sanskrit_texts/'])
rag.build_index()

# Research questions
questions = [
    "कालीदासस्य कृतयः काः सन्ति?",
    "What are the main themes?",
    "भोजराजस्य दरबारे के विद्वांसः आसन्?"
]

# Get answers
for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    result = rag.query(q, verbose=False)
    print(f"A: {result['answer']}")
```

### Workflow 2: Content Summarization

```python
# Get overview of document
result = rag.query("What are the main stories?", top_k=5)

# Extract all contexts
all_contexts = [ctx['text'] for ctx in result['retrieved_contexts']]
summary = "\n\n".join(all_contexts[:3])

print("Document Summary:")
print(summary)
```

### Workflow 3: Comparative Analysis

```python
characters = ["कालीदासः", "शंखनादः", "वृद्धा"]

for char in characters:
    query = f"{char} कः आसीत्?"
    result = rag.query(query, verbose=False)
    print(f"\n{char}: {result['answer'][:200]}...")
```

---

## 🆘 Support

### Getting Help

1. **Read the documentation**: Start with README.md
2. **Check examples**: Review the demo.py script
3. **Test with examples**: Use provided example queries
4. **Review code comments**: Code is well-documented

### Common Questions

**Q: Can I use PDF files?**
A: Currently supports .txt and .docx. PDF support can be added with pdfplumber or PyPDF2.

**Q: How accurate is the system?**
A: Accuracy depends on document quality and query specificity. TF-IDF provides good baseline performance.

**Q: Can it handle mixed scripts?**
A: Yes! It processes both Devanagari and Latin scripts simultaneously.

**Q: Is GPU required?**
A: No! System runs entirely on CPU using NumPy for computations.

---

For technical details, see README.md  
For code reference, see sanskrit_rag_system.py
