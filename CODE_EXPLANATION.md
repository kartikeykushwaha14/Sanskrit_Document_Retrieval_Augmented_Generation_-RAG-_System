# Sanskrit RAG System - Complete Code Explanation

## 📚 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Flow](#architecture-flow)
3. [Class-by-Class Breakdown](#class-by-class-breakdown)
4. [Training Process](#training-process)
5. [Query Process](#query-process)
6. [Code Walkthrough with Examples](#code-walkthrough-with-examples)

---

## 🎯 System Overview

### What is RAG?
**RAG (Retrieval-Augmented Generation)** is a technique that combines:
1. **Retrieval**: Finding relevant information from a document database
2. **Augmentation**: Adding retrieved information to the query context
3. **Generation**: Creating an answer based on the augmented context

### Our Implementation
```
Input Document → Process → Chunk → Vectorize → Index (Training)
User Query → Vectorize → Search → Retrieve → Generate Answer (Inference)
```

---

## 🏗️ Architecture Flow

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
└─────────────────────────────────────────────────────────────┘

1. Load Document (.docx/.txt)
        ↓
2. Clean Text (remove noise, normalize)
        ↓
3. Chunk Text (split into 400-char pieces)
        ↓
4. Tokenize (extract words)
        ↓
5. Build Vocabulary (unique words)
        ↓
6. Calculate TF-IDF (importance scores)
        ↓
7. Create Vectors (numerical representation)
        ↓
8. Store Index (save for future use)

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PHASE                               │
└─────────────────────────────────────────────────────────────┘

1. User enters query
        ↓
2. Tokenize query
        ↓
3. Vectorize query (same as documents)
        ↓
4. Calculate similarity with all chunks
        ↓
5. Rank chunks by similarity
        ↓
6. Select top-K chunks
        ↓
7. Extract relevant sentences
        ↓
8. Generate final answer
```

---

## 📦 Class-by-Class Breakdown

### Class 1: DocumentChunk

**Purpose**: Store information about each text chunk

```python
@dataclass
class DocumentChunk:
    chunk_id: int        # Unique identifier (0, 1, 2, ...)
    text: str            # Actual text content
    source: str          # Source file name
    start_pos: int       # Starting position in original document
    end_pos: int         # Ending position in original document
    metadata: Dict       # Additional info (chunk_index, etc.)
```

**Example**:
```python
chunk = DocumentChunk(
    chunk_id=0,
    text="कालीदासः कविः आसीत्।",
    source="Rag-docs.docx",
    start_pos=0,
    end_pos=50,
    metadata={'chunk_index': 0}
)
```

**Why needed?**: Keeps track of where each chunk came from, useful for citing sources.

---

### Class 2: SanskritTextProcessor

**Purpose**: Clean and split Sanskrit text into manageable chunks

#### Function 1: `__init__(self)`

```python
def __init__(self):
    self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
    self.danda = '।'          # Single danda
    self.double_danda = '॥'   # Double danda
```

**What it does**:
- Sets up pattern to detect Devanagari script (Unicode range U+0900 to U+097F)
- Defines Sanskrit punctuation marks

**Why needed**: To identify Sanskrit text and handle it differently from English

---

#### Function 2: `is_devanagari(self, text: str) -> bool`

```python
def is_devanagari(self, text: str) -> bool:
    return bool(self.devanagari_pattern.search(text))
```

**What it does**: Checks if text contains Devanagari characters

**Example**:
```python
processor = SanskritTextProcessor()
print(processor.is_devanagari("Hello"))           # False
print(processor.is_devanagari("कालीदासः"))        # True
print(processor.is_devanagari("Hello कालीदासः"))  # True
```

**Why needed**: To decide which chunking strategy to use

---

#### Function 3: `clean_text(self, text: str) -> str`

```python
def clean_text(self, text: str) -> str:
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Keep only: Devanagari, English, numbers, basic punctuation
    text = re.sub(r'[^\u0900-\u097F\s।॥a-zA-Z0-9.,;:()\-"\'?!]', '', text)
    return text.strip()
```

**What it does**:
1. Replaces multiple spaces with single space
2. Removes special characters (keeps Sanskrit, English, punctuation)
3. Trims leading/trailing spaces

**Example**:
```python
dirty = "कालीदासः    कविः   @#$  आसीत्।"
clean = processor.clean_text(dirty)
# Result: "कालीदासः कविः आसीत्।"
```

**Why needed**: Clean data = better retrieval accuracy

---

#### Function 4: `split_by_shloka(self, text: str) -> List[str]`

```python
def split_by_shloka(self, text: str) -> List[str]:
    chunks = []
    verses = re.split(r'॥', text)  # Split by double danda
    
    for verse in verses:
        if '।' in verse:
            sub_verses = re.split(r'।', verse)  # Split by single danda
            for sv in sub_verses:
                if sv.strip():
                    chunks.append(sv.strip())
        else:
            chunks.append(verse.strip())
    
    return chunks
```

**What it does**:
1. Splits text by `॥` (end of verse)
2. Further splits by `।` (half verse)
3. Returns list of verse chunks

**Example**:
```python
text = "कालीदासः कविः आसीत् । भोजराजस्य दरबारे अस्ति ॥ सः चतुरः आसीत् ।"
chunks = processor.split_by_shloka(text)
# Result: [
#   "कालीदासः कविः आसीत्",
#   "भोजराजस्य दरबारे अस्ति",
#   "सः चतुरः आसीत्"
# ]
```

**Why needed**: Respects natural Sanskrit text boundaries

---

#### Function 5: `chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]`

```python
def chunk_text(self, text: str, chunk_size=400, overlap=50):
    text = self.clean_text(text)
    
    # Strategy 1: Verse-based (for Sanskrit)
    if self.is_devanagari(text):
        verses = self.split_by_shloka(text)
        chunks = []
        current_chunk = ""
        
        for verse in verses:
            if len(current_chunk) + len(verse) < chunk_size:
                current_chunk += verse + " । "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = verse + " । "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    # Strategy 2: Sliding window (fallback)
    # ... (creates overlapping chunks)
```

**What it does**:
1. **For Sanskrit**: Combines verses until reaching chunk_size
2. **For English**: Uses sliding window with overlap

**Visual Example**:
```
Original text: [A B C D E F G H I J] (10 units)
chunk_size = 4, overlap = 1

Chunks:
[A B C D]
    [D E F G]
        [G H I J]
```

**Why overlap?**: Prevents losing context at chunk boundaries

**Example**:
```python
text = "First sentence. Second sentence. Third sentence. Fourth sentence."
chunks = processor.chunk_text(text, chunk_size=30, overlap=10)
# Result: Multiple overlapping chunks of ~30 characters
```

**Why needed**: Makes documents searchable in smaller pieces

---

### Class 3: TFIDFRetriever

**Purpose**: Convert text to numbers and find similar documents

#### Key Concepts

**TF (Term Frequency)**:
```
TF = (Number of times term appears in document) / (Total terms in document)
```

**IDF (Inverse Document Frequency)**:
```
IDF = log((Total documents + 1) / (Documents containing term + 1)) + 1
```

**TF-IDF Score**:
```
TF-IDF = TF × IDF
```

**Intuition**: 
- Common words (like "the", "is") have low IDF → low importance
- Rare words (like "कालीदासः") have high IDF → high importance

---

#### Function 1: `tokenize(self, text: str) -> List[str]`

```python
def tokenize(self, text: str) -> List[str]:
    # Extract Devanagari words
    devanagari_words = re.findall(r'[\u0900-\u097F]+', text)
    # Extract English words
    latin_words = re.findall(r'[a-zA-Z]+', text.lower())
    
    return devanagari_words + latin_words
```

**What it does**: Extracts all words from text (both scripts)

**Example**:
```python
text = "कालीदासः was a clever poet"
tokens = retriever.tokenize(text)
# Result: ["कालीदासः", "was", "a", "clever", "poet"]
```

**Why needed**: Converts text to list of words for analysis

---

#### Function 2: `compute_tf(self, tokens: List[str]) -> Dict[str, float]`

```python
def compute_tf(self, tokens: List[str]) -> Dict[str, float]:
    tf = Counter(tokens)
    total = len(tokens)
    return {token: count / total for token, count in tf.items()}
```

**What it does**: Calculates frequency of each word

**Example**:
```python
tokens = ["कालीदासः", "कविः", "कालीदासः", "चतुरः"]
tf = retriever.compute_tf(tokens)
# Result: {
#   "कालीदासः": 0.5,   # appears 2/4 times
#   "कविः": 0.25,       # appears 1/4 times
#   "चतुरः": 0.25        # appears 1/4 times
# }
```

**Why needed**: Measures word importance in a document

---

#### Function 3: `compute_idf(self, documents: List[List[str]]) -> Dict[str, float]`

```python
def compute_idf(self, documents):
    num_docs = len(documents)
    doc_freq = defaultdict(int)
    
    for doc_tokens in documents:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    
    idf = {}
    for token, freq in doc_freq.items():
        idf[token] = math.log((num_docs + 1) / (freq + 1)) + 1
    
    return idf
```

**What it does**: Calculates rarity of each word across all documents

**Example**:
```python
docs = [
    ["कालीदासः", "कविः"],
    ["कविः", "चतुरः"],
    ["भोजराजः", "राजा"]
]
idf = retriever.compute_idf(docs)
# Result:
# "कविः": low (appears in 2/3 docs)
# "कालीदासः": high (appears in 1/3 docs)
# "चतुरः": high (appears in 1/3 docs)
```

**Why needed**: Gives higher weight to rare/unique words

---

#### Function 4: `fit(self, documents: List[str])`

```python
def fit(self, documents: List[str]):
    # Step 1: Tokenize all documents
    tokenized_docs = [self.tokenize(doc) for doc in documents]
    
    # Step 2: Build vocabulary (all unique words)
    all_tokens = set()
    for tokens in tokenized_docs:
        all_tokens.update(tokens)
    self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
    
    # Step 3: Compute IDF scores
    self.idf_scores = self.compute_idf(tokenized_docs)
    
    # Step 4: Create TF-IDF vectors for all documents
    self.doc_vectors = []
    for tokens in tokenized_docs:
        vector = self.vectorize(tokens)
        self.doc_vectors.append(vector)
    
    self.documents = documents
    self.is_fitted = True
```

**What it does**: The TRAINING step - learns from all documents

**Step-by-step process**:
```
INPUT: ["Doc1 text", "Doc2 text", "Doc3 text"]

Step 1: Tokenize
→ [["doc1", "text"], ["doc2", "text"], ["doc3", "text"]]

Step 2: Build vocabulary
→ {"doc1": 0, "doc2": 1, "doc3": 2, "text": 3}

Step 3: Calculate IDF
→ {"text": 1.28, "doc1": 2.09, "doc2": 2.09, "doc3": 2.09}

Step 4: Create vectors
Doc1: [TF-IDF(doc1), 0, 0, TF-IDF(text)]
Doc2: [0, TF-IDF(doc2), 0, TF-IDF(text)]
Doc3: [0, 0, TF-IDF(doc3), TF-IDF(text)]
```

**Why needed**: Prepares system to understand and search documents

---

#### Function 5: `vectorize(self, tokens: List[str]) -> np.ndarray`

```python
def vectorize(self, tokens: List[str]) -> np.ndarray:
    vector = np.zeros(len(self.vocabulary))
    tf = self.compute_tf(tokens)
    
    for token, tf_score in tf.items():
        if token in self.vocabulary:
            idx = self.vocabulary[token]
            idf_score = self.idf_scores.get(token, 0)
            vector[idx] = tf_score * idf_score
    
    # Normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector
```

**What it does**: Converts text to a numerical vector

**Example**:
```
Vocabulary: {"कालीदासः": 0, "कविः": 1, "चतुरः": 2}
Text: "कालीदासः कविः"

Step 1: Calculate TF
→ {"कालीदासः": 0.5, "कविः": 0.5}

Step 2: Get IDF
→ {"कालीदासः": 1.5, "कविः": 1.2}

Step 3: Calculate TF-IDF
→ [0.5×1.5, 0.5×1.2, 0] = [0.75, 0.60, 0]

Step 4: Normalize
→ [0.78, 0.62, 0]
```

**Why needed**: Computers can't understand text, only numbers

---

#### Function 6: `cosine_similarity(self, vec1, vec2) -> float`

```python
def cosine_similarity(self, vec1, vec2):
    return np.dot(vec1, vec2)
```

**What it does**: Measures similarity between two vectors

**Visual Explanation**:
```
Vector1: [1, 0]  Vector2: [1, 0]  → Similarity: 1.0 (identical)
Vector1: [1, 0]  Vector2: [0, 1]  → Similarity: 0.0 (perpendicular)
Vector1: [1, 0]  Vector2: [-1, 0] → Similarity: -1.0 (opposite)
```

**Why needed**: To find which documents are most similar to query

---

#### Function 7: `retrieve(self, query: str, top_k: int) -> List`

```python
def retrieve(self, query: str, top_k=3):
    # Step 1: Convert query to vector
    query_tokens = self.tokenize(query)
    query_vector = self.vectorize(query_tokens)
    
    # Step 2: Calculate similarity with all documents
    similarities = []
    for idx, doc_vector in enumerate(self.doc_vectors):
        sim = self.cosine_similarity(query_vector, doc_vector)
        similarities.append((idx, sim, self.documents[idx]))
    
    # Step 3: Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Step 4: Return top-k results
    return similarities[:top_k]
```

**What it does**: Finds most relevant documents for a query

**Step-by-step**:
```
INPUT: Query = "कालीदासः कः?"

Step 1: Vectorize query
→ [0.8, 0.3, 0.1, ...]

Step 2: Compare with all documents
Doc1 vector: [0.7, 0.4, 0.2, ...] → Similarity: 0.89
Doc2 vector: [0.1, 0.9, 0.0, ...] → Similarity: 0.35
Doc3 vector: [0.8, 0.2, 0.3, ...] → Similarity: 0.75

Step 3: Sort
→ [(Doc1, 0.89), (Doc3, 0.75), (Doc2, 0.35)]

Step 4: Return top-3
→ [Doc1, Doc3, Doc2]
```

**Why needed**: Core retrieval functionality of RAG

---

### Class 4: SanskritQAGenerator

**Purpose**: Generate answers from retrieved contexts

#### Function 1: `detect_question_type(self, query: str) -> str`

```python
def detect_question_type(self, query: str):
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['who', 'कः', 'का']):
        return 'who'
    elif any(word in query_lower for word in ['what', 'किम्']):
        return 'what'
    # ... more question types
```

**What it does**: Identifies the type of question

**Example**:
```python
detect_question_type("कालीदासः कः आसीत्?")  → "who"
detect_question_type("किम् अभवत्?")          → "what"
detect_question_type("कुत्र आसीत्?")         → "where"
```

**Why needed**: Different question types need different answer formats

---

#### Function 2: `extract_relevant_sentences(self, context, query, max_sentences=3)`

```python
def extract_relevant_sentences(self, context, query, max_sentences=3):
    # Step 1: Split into sentences
    sentences = re.split(r'[।॥.!?]', context)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Step 2: Get query terms
    query_tokens = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+', query.lower()))
    
    # Step 3: Score each sentence
    scored_sentences = []
    for sentence in sentences:
        sentence_tokens = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+', sentence.lower()))
        overlap = len(query_tokens & sentence_tokens)
        if overlap > 0:
            scored_sentences.append((sentence, overlap))
    
    # Step 4: Sort and return top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sent for sent, _ in scored_sentences[:max_sentences]]
```

**What it does**: Finds sentences most relevant to the query

**Example**:
```
Query: "कालीदासः कः?"
Context: "भोजराजः राजा आसीत्। कालीदासः कविः आसीत्। सः चतुरः आसीत्।"

Step 1: Split sentences
→ ["भोजराजः राजा आसीत्", "कालीदासः कविः आसीत्", "सः चतुरः आसीत्"]

Step 2: Query terms
→ {"कालीदासः", "कः"}

Step 3: Score
Sentence 1: overlap = 0
Sentence 2: overlap = 1 (contains "कालीदासः")
Sentence 3: overlap = 0

Step 4: Return
→ ["कालीदासः कविः आसीत्"]
```

**Why needed**: Extracts most relevant parts from long contexts

---

#### Function 3: `generate_answer(self, query, retrieved_contexts)`

```python
def generate_answer(self, query, retrieved_contexts):
    if not retrieved_contexts:
        return "मम ज्ञाने एतस्य उत्तरम् नास्ति।"
    
    # Combine top contexts
    combined_context = "\\n\\n".join([ctx for _, _, ctx in retrieved_contexts[:2]])
    
    # Extract relevant sentences
    relevant_sentences = self.extract_relevant_sentences(combined_context, query)
    
    if not relevant_sentences:
        # Fallback: return part of top context
        top_context = retrieved_contexts[0][2]
        return top_context[:300] + "..."
    
    # Combine sentences
    answer = " । ".join(relevant_sentences)
    return answer
```

**What it does**: Creates final answer from retrieved information

**Process**:
```
INPUT:
Query: "कालीदासः कः?"
Contexts: [
  (idx=5, score=0.17, text="...कालीदासः कविः आसीत्..."),
  (idx=12, score=0.15, text="...सः चतुरः आसीत्...")
]

Step 1: Combine contexts
→ "...कालीदासः कविः आसीत्......सः चतुरः आसीत्..."

Step 2: Extract relevant sentences
→ ["कालीदासः कविः आसीत्", "सः चतुरः आसीत्"]

Step 3: Combine
→ "कालीदासः कविः आसीत् । सः चतुरः आसीत्"

OUTPUT: "कालीदासः कविः आसीत् । सः चतुरः आसीत्"
```

**Why needed**: Generates coherent answer from multiple sources

---

### Class 5: SanskritRAGSystem

**Purpose**: Main orchestrator - combines all components

#### Function 1: `__init__(self, chunk_size, overlap)`

```python
def __init__(self, chunk_size=400, overlap=50):
    self.processor = SanskritTextProcessor()
    self.retriever = TFIDFRetriever()
    self.generator = SanskritQAGenerator()
    
    self.chunk_size = chunk_size
    self.overlap = overlap
    
    self.chunks = []
    self.raw_documents = []
```

**What it does**: Initializes all components

**Why needed**: Sets up the RAG pipeline

---

#### Function 2: `load_document(self, file_path: str) -> str`

```python
def load_document(self, file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.docx'):
        from docx import Document
        doc = Document(file_path)
        return '\\n'.join([para.text for para in doc.paragraphs])
```

**What it does**: Reads document from file

**Example**:
```python
text = rag.load_document('/path/to/Rag-docs.docx')
# Result: Full document text as string
```

**Why needed**: Gets raw text from files

---

#### Function 3: `ingest_documents(self, file_paths: List[str])`

```python
def ingest_documents(self, file_paths):
    all_chunks = []
    chunk_id = 0
    
    for file_path in file_paths:
        # Load document
        doc_text = self.load_document(file_path)
        self.raw_documents.append(doc_text)
        
        # Chunk document
        chunks = self.processor.chunk_text(doc_text, self.chunk_size, self.overlap)
        
        # Create DocumentChunk objects
        for i, chunk_text in enumerate(chunks):
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=os.path.basename(file_path),
                start_pos=i * (self.chunk_size - self.overlap),
                end_pos=i * (self.chunk_size - self.overlap) + len(chunk_text),
                metadata={'chunk_index': i}
            )
            all_chunks.append(chunk)
            chunk_id += 1
    
    self.chunks = all_chunks
```

**What it does**: STEP 1 of training - loads and chunks documents

**Process**:
```
INPUT: ["Rag-docs.docx"]

For each file:
  1. Load: Read full text
  2. Chunk: Split into 400-char pieces
  3. Wrap: Create DocumentChunk objects
  4. Store: Save to self.chunks

OUTPUT: 26 DocumentChunk objects stored
```

**Why needed**: Prepares documents for indexing

---

#### Function 4: `build_index(self)`

```python
def build_index(self):
    if not self.chunks:
        raise ValueError("No documents ingested")
    
    # Extract chunk texts
    chunk_texts = [chunk.text for chunk in self.chunks]
    
    # Fit retriever
    self.retriever.fit(chunk_texts)
```

**What it does**: STEP 2 of training - builds search index

**Process**:
```
INPUT: 26 chunks

1. Extract texts from chunks
   → ["chunk1 text", "chunk2 text", ..., "chunk26 text"]

2. Call retriever.fit()
   → Tokenize all chunks
   → Build vocabulary (695 unique words)
   → Calculate IDF scores
   → Create TF-IDF vectors

OUTPUT: Trained retriever ready for queries
```

**Why needed**: Creates searchable index

---

#### Function 5: `query(self, question, top_k, verbose)`

```python
def query(self, question, top_k=3, verbose=True):
    # Step 1: Retrieve relevant contexts
    retrieved = self.retriever.retrieve(question, top_k=top_k)
    
    # Step 2: Generate answer
    answer = self.generator.generate_answer(question, retrieved)
    
    # Step 3: Format result
    return {
        'question': question,
        'answer': answer,
        'retrieved_contexts': [
            {
                'text': text,
                'score': float(score),
                'source': self.chunks[doc_idx].source,
                'chunk_id': self.chunks[doc_idx].chunk_id
            }
            for doc_idx, score, text in retrieved
        ],
        'num_contexts': len(retrieved)
    }
```

**What it does**: INFERENCE - answers user queries

**Complete Flow**:
```
INPUT: "कालीदासः कः आसीत्?"

Step 1: RETRIEVAL
  1a. Tokenize query → ["कालीदासः", "कः", "आसीत्"]
  1b. Vectorize query → [0.78, 0.12, 0.05, ...]
  1c. Calculate similarity with all 26 chunks
  1d. Sort by similarity
  1e. Get top-3 chunks
  
  Result: [
    (chunk_11, score=0.17, "...कालीदासः कविः..."),
    (chunk_22, score=0.16, "...सः चतुरः..."),
    (chunk_15, score=0.09, "...देवः...")
  ]

Step 2: GENERATION
  2a. Combine top contexts
  2b. Extract relevant sentences
  2c. Create answer
  
  Result: "कालीदासः कविः आसीत् । सः चतुरः आसीत्"

Step 3: FORMAT
  Package everything nicely
  
OUTPUT: {
  'question': "कालीदासः कः आसीत्?",
  'answer': "कालीदासः कविः आसीत् । सः चतुरः आसीत्",
  'retrieved_contexts': [...],
  'num_contexts': 3
}
```

**Why needed**: The main user-facing function

---

#### Function 6: `save_index(self, save_path)` & `load_index(self, load_path)`

```python
def save_index(self, save_path):
    data = {
        'chunks': [asdict(chunk) for chunk in self.chunks],
        'retriever_vocab': self.retriever.vocabulary,
        'retriever_idf': self.retriever.idf_scores,
        'retriever_doc_vectors': [vec.tolist() for vec in self.retriever.doc_vectors],
        # ... more data
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def load_index(self, load_path):
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    # Restore all components
    self.chunks = [DocumentChunk(**chunk) for chunk in data['chunks']]
    self.retriever.vocabulary = data['retriever_vocab']
    # ... restore more data
```

**What it does**: Saves/loads trained model

**Why needed**: Avoid retraining every time

---

## 🎓 Training Process (Input to Output)

### Phase 1: Document Ingestion

```
USER ACTION: rag.ingest_documents(['Rag-docs.docx'])

INTERNAL PROCESS:
┌────────────────────────────────────────┐
│ 1. Read file                            │
│    Input: Rag-docs.docx                 │
│    Output: Full text string (~6500 words)│
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 2. Clean text                           │
│    Remove: Extra spaces, special chars  │
│    Keep: Devanagari, English, punct.    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 3. Chunk text                           │
│    Strategy: Verse-aware (।, ॥)        │
│    Size: 400 chars, overlap: 50        │
│    Output: 26 chunks                    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 4. Create DocumentChunk objects         │
│    Each chunk gets:                     │
│    - Unique ID (0-25)                   │
│    - Source file name                   │
│    - Position info                      │
│    - Metadata                           │
└────────────────────────────────────────┘

RESULT: 26 DocumentChunk objects stored in memory
```

### Phase 2: Index Building

```
USER ACTION: rag.build_index()

INTERNAL PROCESS:
┌────────────────────────────────────────┐
│ 1. Extract chunk texts                  │
│    Input: 26 DocumentChunk objects      │
│    Output: 26 text strings              │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 2. Tokenize all chunks                  │
│    Chunk 0: ["मूर्खभृत्यस्य", ...]    │
│    Chunk 1: ["गोवर्धनदासः", ...]      │
│    ...                                  │
│    Chunk 25: ["गन्तुं", "इति", ...]    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 3. Build vocabulary                     │
│    Collect all unique words             │
│    Result: 695 unique tokens            │
│    Example: {                           │
│      "कालीदासः": 0,                     │
│      "कविः": 1,                         │
│      ...                                │
│      "clever": 694                      │
│    }                                    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 4. Calculate IDF scores                 │
│    For each word:                       │
│    IDF = log((26+1)/(doc_count+1))+1   │
│    Example:                             │
│    "कालीदासः": 2.3 (rare)               │
│    "अस्ति": 1.5 (common)                │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 5. Create TF-IDF vectors                │
│    Each chunk → 695-dimensional vector  │
│    Chunk 0: [0.12, 0.00, 0.34, ...]    │
│    Chunk 1: [0.00, 0.23, 0.00, ...]    │
│    ...                                  │
│    Chunk 25: [0.05, 0.00, 0.18, ...]   │
└────────────────────────────────────────┘

RESULT: Search index ready!
```

### Phase 3: Saving Model

```
USER ACTION: rag.save_index('sanskrit_rag_index.pkl')

INTERNAL PROCESS:
┌────────────────────────────────────────┐
│ Pickle all important data:              │
│ - 26 chunks                             │
│ - Vocabulary (695 words)                │
│ - IDF scores                            │
│ - 26 TF-IDF vectors                     │
│ - Configuration                         │
└────────────────────────────────────────┘

RESULT: File saved (~500 KB)
```

---

## 🔍 Query Process (Input to Output)

### Complete Query Flow

```
USER INPUT: "कालीदासः कः आसीत्?"

STEP 1: TOKENIZATION
┌────────────────────────────────────────┐
│ Input: "कालीदासः कः आसीत्?"             │
│ Process: Extract words                  │
│ Output: ["कालीदासः", "कः", "आसीत्"]     │
└────────────────────────────────────────┘

STEP 2: VECTORIZATION
┌────────────────────────────────────────┐
│ Calculate TF:                           │
│   कालीदासः: 1/3 = 0.33                 │
│   कः: 1/3 = 0.33                        │
│   आसीत्: 1/3 = 0.33                     │
│                                         │
│ Multiply by IDF:                        │
│   कालीदासः: 0.33 × 2.3 = 0.76          │
│   कः: 0.33 × 1.8 = 0.59                 │
│   आसीत्: 0.33 × 1.5 = 0.50              │
│                                         │
│ Create vector:                          │
│   [0.76, 0.59, 0.50, 0, 0, ..., 0]     │
│   (695 dimensions)                      │
│                                         │
│ Normalize:                              │
│   [0.68, 0.53, 0.45, 0, 0, ..., 0]     │
└────────────────────────────────────────┘

STEP 3: SIMILARITY CALCULATION
┌────────────────────────────────────────┐
│ Compare query with all 26 chunks:       │
│                                         │
│ Chunk 0 vs Query:                       │
│   [0.12, 0.34, ...] · [0.68, 0.53, ...] │
│   = 0.0826                              │
│                                         │
│ Chunk 11 vs Query:                      │
│   [0.45, 0.38, ...] · [0.68, 0.53, ...] │
│   = 0.1700  ← Highest!                  │
│                                         │
│ Chunk 22 vs Query:                      │
│   [0.42, 0.35, ...] · [0.68, 0.53, ...] │
│   = 0.1678                              │
│                                         │
│ ... (all 26 chunks)                     │
└────────────────────────────────────────┘

STEP 4: RANKING
┌────────────────────────────────────────┐
│ Sort by similarity:                     │
│   1. Chunk 11: 0.1700                   │
│   2. Chunk 22: 0.1678                   │
│   3. Chunk 15: 0.0963                   │
│   4. Chunk 0: 0.0826                    │
│   ... (rest)                            │
└────────────────────────────────────────┘

STEP 5: RETRIEVAL
┌────────────────────────────────────────┐
│ Select top-3:                           │
│                                         │
│ Context 1 (Chunk 11, score=0.17):      │
│ "न खलु वक्तुं अशक्नुवन् केऽपि विद्वानाः│
│  यत् जानन्ति तत् काव्यं इति । अतः    │
│  अप्राप्नोत् कविः लक्षरुप्यकाणि ।     │
│  चतुरः खलु कालीदासः । ..."            │
│                                         │
│ Context 2 (Chunk 22, score=0.16):      │
│ "न खलु जानाति पण्डितः यत् कालीदासः एव│
│  सः । पालखीं स्कन्दयोः वहन् निर्गतः   │
│  कालीदासः पण्डितेन सह । ..."          │
│                                         │
│ Context 3 (Chunk 15, score=0.09):      │
│ "सः प्रतिदिने भक्त्या देवस्य प्रार्थनाम्│
│  करोति । ..."                          │
└────────────────────────────────────────┘

STEP 6: ANSWER GENERATION
┌────────────────────────────────────────┐
│ Combine contexts 1 & 2                  │
│    ↓                                    │
│ Split into sentences                    │
│    ↓                                    │
│ Score sentences by query overlap:       │
│   "चतुरः खलु कालीदासः": 1 match        │
│   "कालीदासः एव सः": 1 match            │
│   "भक्त्या देवस्य": 0 matches          │
│    ↓                                    │
│ Select top sentences                    │
│    ↓                                    │
│ Join with "।"                           │
└────────────────────────────────────────┘

FINAL OUTPUT:
{
  "question": "कालीदासः कः आसीत्?",
  "answer": "चतुरः खलु कालीदासः । न खलु जानाति पण्डितः यत् कालीदासः एव सः",
  "retrieved_contexts": [
    {
      "text": "न खलु वक्तुं...",
      "score": 0.1700,
      "source": "Rag-docs.docx",
      "chunk_id": 11
    },
    {
      "text": "न खलु जानाति...",
      "score": 0.1678,
      "source": "Rag-docs.docx",
      "chunk_id": 22
    },
    {
      "text": "सः प्रतिदिने...",
      "score": 0.0963,
      "source": "Rag-docs.docx",
      "chunk_id": 15
    }
  ],
  "num_contexts": 3
}
```

---

## 📊 How Training Actually Works

### Mathematical Perspective

**Training** creates this mapping:
```
Text Space → Number Space

"कालीदासः कविः आसीत्" → [0.45, 0.32, 0.18, 0, 0, ...]
```

**Why?** Because:
1. Computers can't compare text directly
2. Computers CAN compare numbers using math
3. Similar texts → Similar numbers (vectors)

### What Gets "Learned"?

The system learns:
1. **Vocabulary**: Which words exist
2. **Word Importance**: Which words are rare/common
3. **Document Representations**: How to represent each chunk as numbers

### What Does NOT Get Learned?

- Grammar rules
- Word meanings
- Context understanding
- Language structure

**It's just statistical pattern matching!**

---

## 🎯 Key Insights

### Why TF-IDF Works

**Example**:
```
Query: "कालीदासः"
Document 1: "कालीदासः कालीदासः कालीदासः" (word repeated)
Document 2: "कालीदासः कविः आसीत्" (more context)

Without IDF:
  Doc1 scores higher (more repetition)

With IDF:
  Doc2 scores higher (better content)
```

IDF prevents matching on word frequency alone.

### Why Chunking Matters

**Without chunking**:
```
Document: 6500 words → 1 giant vector
Problem: Too broad, hard to find specific info
```

**With chunking**:
```
Document: 6500 words → 26 small vectors
Benefit: Precise matching to specific parts
```

### Why Overlap Helps

**Without overlap**:
```
Chunk 1: "...कालीदासः कविः"
Chunk 2: "आसीत् । सः चतुरः..."
Problem: Important phrase split across chunks
```

**With overlap**:
```
Chunk 1: "...कालीदासः कविः आसीत्..."
Chunk 2: "...कविः आसीत् । सः चतुरः..."
Benefit: Complete phrases in both chunks
```

---

## 🔄 Complete System Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                          │
└─────────────────────────────────────────────────────────────┘

Document File
     ↓
load_document() ───────→ Raw Text String
     ↓
clean_text() ──────────→ Cleaned Text
     ↓
chunk_text() ──────────→ 26 Text Chunks
     ↓
Create DocumentChunk ──→ 26 DocumentChunk Objects
     ↓
tokenize() ────────────→ List of Words per Chunk
     ↓
build vocabulary ──────→ 695 Unique Words
     ↓
calculate IDF ─────────→ Importance Score per Word
     ↓
vectorize() ───────────→ 26 Vectors (695-dim each)
     ↓
save_index() ──────────→ Saved Model File


┌─────────────────────────────────────────────────────────────┐
│                      QUERY PHASE                             │
└─────────────────────────────────────────────────────────────┘

User Query: "कालीदासः कः?"
     ↓
tokenize() ────────────→ ["कालीदासः", "कः"]
     ↓
vectorize() ───────────→ Query Vector [0.68, 0.53, ...]
     ↓
cosine_similarity() ───→ Compare with all 26 vectors
     ↓
sort by score ─────────→ Ranked list
     ↓
retrieve top-3 ────────→ 3 Most Similar Chunks
     ↓
extract_sentences() ───→ Relevant Sentences
     ↓
join sentences ────────→ Final Answer
     ↓
format_result() ───────→ JSON Response
     ↓
Return to User
```

---

## 💡 Summary

### What Happens During Training?

1. **Read** documents
2. **Clean** text
3. **Split** into chunks
4. **Count** word frequencies
5. **Calculate** word importance
6. **Convert** text to numbers
7. **Save** the model

### What Happens During Query?

1. **Receive** user question
2. **Convert** question to numbers
3. **Compare** with all chunks
4. **Rank** by similarity
5. **Select** top matches
6. **Extract** relevant parts
7. **Return** answer

### Core Algorithm: TF-IDF

**TF (Term Frequency)**: How often does a word appear?
**IDF (Inverse Document Frequency)**: How unique is the word?
**TF-IDF**: Balance between frequency and uniqueness

### Why It Works?

Similar questions and answers contain similar words, which produce similar number vectors, which have high cosine similarity!

---

**End of Explanation**

This RAG system uses classical information retrieval techniques (TF-IDF) rather than deep learning, making it:
- ✅ Fast (CPU-only)
- ✅ Interpretable (you can see why it retrieved each chunk)
- ✅ Lightweight (no large models)
- ⚠️ Limited (no true understanding, just pattern matching)

For better results, you could upgrade to:
- Neural embeddings (BERT, SentenceTransformers)
- Large language models (GPT, Claude)
- Fine-tuned models for Sanskrit
