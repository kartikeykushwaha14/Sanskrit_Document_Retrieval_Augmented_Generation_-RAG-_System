"""
Complete Sanskrit Document Retrieval-Augmented Generation (RAG) System
CPU-only implementation with TF-IDF retrieval and rule-based generation
Author: Sanskrit RAG System
Date: January 2026
"""

import os
import re
import json
import pickle
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import math


@dataclass
class DocumentChunk:
    """Represents a chunk of Sanskrit document with metadata"""
    chunk_id: int
    text: str
    source: str
    start_pos: int
    end_pos: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SanskritTextProcessor:
    """
    Handles Sanskrit text preprocessing and chunking
    Supports Devanagari script processing
    """
    
    def __init__(self):
        # Devanagari Unicode range: U+0900 to U+097F
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        # Sanskrit punctuation marks
        self.danda = '।'  # Single danda
        self.double_danda = '॥'  # Double danda
        
    def is_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari script"""
        return bool(self.devanagari_pattern.search(text))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Sanskrit text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Devanagari and basic punctuation
        # Keep: Devanagari, English letters, numbers, basic punctuation, Sanskrit punctuation
        text = re.sub(r'[^\u0900-\u097F\s।॥a-zA-Z0-9.,;:()\-"\'?!]', '', text)
        return text.strip()
    
    def split_by_shloka(self, text: str) -> List[str]:
        """
        Split text by Sanskrit verse markers
        । (danda) - marks half verse or clause
        ॥ (double danda) - marks end of verse
        """
        chunks = []
        
        # First split by double danda (॥) - end of verse
        verses = re.split(r'॥', text)
        
        for verse in verses:
            if not verse.strip():
                continue
            
            # Check if verse contains single danda
            if '।' in verse:
                # Split by single danda for sub-verses
                sub_verses = re.split(r'।', verse)
                for sv in sub_verses:
                    if sv.strip():
                        chunks.append(sv.strip())
            else:
                chunks.append(verse.strip())
        
        return chunks
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """
        Chunk text with overlap for better context preservation
        
        Args:
            text: Input Sanskrit text
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        text = self.clean_text(text)
        
        # If text contains Sanskrit, try verse-based chunking first
        if self.is_devanagari(text):
            verses = self.split_by_shloka(text)
            
            if verses:
                chunks = []
                current_chunk = ""
                
                for verse in verses:
                    # If adding this verse doesn't exceed chunk_size, add it
                    if len(current_chunk) + len(verse) + 3 < chunk_size:  # +3 for " । "
                        current_chunk += verse + " । "
                    else:
                        # Save current chunk and start new one
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = verse + " । "
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        # Fallback: Sliding window chunking with sentence boundary detection
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Try to break at sentence boundary if not at end
            if end < text_len:
                # Look for sentence endings in order of preference
                boundaries = [
                    text.rfind('॥', start, end),
                    text.rfind('।', start, end),
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                ]
                
                # Use the latest boundary found that's reasonable
                best_boundary = -1
                for boundary in boundaries:
                    if boundary > start + chunk_size // 2:  # At least half the chunk
                        best_boundary = boundary
                        break
                
                if best_boundary != -1:
                    end = best_boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
        
        return chunks


class TFIDFRetriever:
    """
    TF-IDF based document retriever optimized for Sanskrit text
    CPU-only implementation
    """
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.doc_vectors = []
        self.documents = []
        self.is_fitted = False
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        Handles both Devanagari and romanized text
        """
        # Extract Devanagari words
        devanagari_words = re.findall(r'[\u0900-\u097F]+', text)
        # Extract English/romanized words
        latin_words = re.findall(r'[a-zA-Z]+', text.lower())
        
        return devanagari_words + latin_words
    
    def compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        tf = Counter(tokens)
        total = len(tokens)
        return {token: count / total for token, count in tf.items()}
    
    def compute_idf(self, documents: List[List[str]]) -> Dict[str, float]:
        """Compute inverse document frequency"""
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
    
    def fit(self, documents: List[str]):
        """
        Build vocabulary and compute IDF scores
        
        Args:
            documents: List of document texts
        """
        # Tokenize all documents
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        
        # Build vocabulary
        all_tokens = set()
        for tokens in tokenized_docs:
            all_tokens.update(tokens)
        
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        
        # Compute IDF scores
        self.idf_scores = self.compute_idf(tokenized_docs)
        
        # Compute TF-IDF vectors for all documents
        self.doc_vectors = []
        for tokens in tokenized_docs:
            vector = self.vectorize(tokens)
            self.doc_vectors.append(vector)
        
        self.documents = documents
        self.is_fitted = True
        
        print(f"✓ Fitted retriever with {len(documents)} documents")
        print(f"✓ Vocabulary size: {len(self.vocabulary)}")
    
    def vectorize(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to TF-IDF vector"""
        vector = np.zeros(len(self.vocabulary))
        tf = self.compute_tf(tokens)
        
        for token, tf_score in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf_score = self.idf_scores.get(token, 0)
                vector[idx] = tf_score * idf_score
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        return dot_product
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Retrieve most relevant documents for query
        
        Args:
            query: Query string
            top_k: Number of top documents to retrieve
            
        Returns:
            List of (doc_index, similarity_score, document_text) tuples
        """
        if not self.is_fitted:
            raise ValueError("Retriever must be fitted before retrieval")
        
        # Vectorize query
        query_tokens = self.tokenize(query)
        query_vector = self.vectorize(query_tokens)
        
        # Compute similarities
        similarities = []
        for idx, doc_vector in enumerate(self.doc_vectors):
            sim = self.cosine_similarity(query_vector, doc_vector)
            similarities.append((idx, sim, self.documents[idx]))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class SanskritQAGenerator:
    """
    Question answering generator for Sanskrit text
    Uses template-based and extractive approaches (CPU-only)
    """
    
    def __init__(self):
        self.context_templates = {
            'definition': 'Based on the context, {}',
            'explanation': 'According to the text, {}',
            'story': 'The narrative describes that {}',
            'general': '{}'
        }
    
    def detect_question_type(self, query: str) -> str:
        """Detect type of question asked"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['who', 'कः', 'का', 'के']):
            return 'who'
        elif any(word in query_lower for word in ['what', 'किम्', 'क्या']):
            return 'what'
        elif any(word in query_lower for word in ['where', 'कुत्र', 'कहाँ']):
            return 'where'
        elif any(word in query_lower for word in ['when', 'कदा', 'कब']):
            return 'when'
        elif any(word in query_lower for word in ['why', 'किमर्थम्', 'क्यों']):
            return 'why'
        elif any(word in query_lower for word in ['how', 'कथम्', 'कैसे']):
            return 'how'
        else:
            return 'general'
    
    def extract_relevant_sentences(self, context: str, query: str, max_sentences: int = 3) -> List[str]:
        """Extract most relevant sentences from context"""
        # Split into sentences
        sentences = re.split(r'[।॥.!?]', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Tokenize query
        query_tokens = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+', query.lower()))
        
        # Score sentences by word overlap
        scored_sentences = []
        for sentence in sentences:
            sentence_tokens = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+', sentence.lower()))
            overlap = len(query_tokens & sentence_tokens)
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        return [sent for sent, _ in scored_sentences[:max_sentences]]
    
    def generate_answer(self, query: str, retrieved_contexts: List[Tuple[int, float, str]]) -> str:
        """
        Generate answer from retrieved contexts
        
        Args:
            query: User query
            retrieved_contexts: List of (idx, score, context) tuples
            
        Returns:
            Generated answer
        """
        if not retrieved_contexts:
            return "मम ज्ञाने एतस्य उत्तरम् नास्ति । (I don't have information about this in my knowledge base.)"
        
        # Combine top contexts
        combined_context = "\n\n".join([ctx for _, _, ctx in retrieved_contexts[:2]])
        
        # Extract relevant sentences
        relevant_sentences = self.extract_relevant_sentences(combined_context, query)
        
        if not relevant_sentences:
            # Return a portion of the top context
            top_context = retrieved_contexts[0][2]
            # Return first 300 characters
            return top_context[:300] + "..."
        
        # Combine relevant sentences
        answer = " । ".join(relevant_sentences)
        
        return answer


class SanskritRAGSystem:
    """
    Complete Sanskrit RAG System
    Integrates document processing, retrieval, and generation
    """
    
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.processor = SanskritTextProcessor()
        self.retriever = TFIDFRetriever()
        self.generator = SanskritQAGenerator()
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.chunks = []
        self.raw_documents = []
        
        print("=" * 60)
        print("Sanskrit RAG System Initialized")
        print("=" * 60)
    
    def load_document(self, file_path: str) -> str:
        """Load document from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Handle different file types
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.endswith('.docx'):
            # Import here to avoid dependency issues if not needed
            try:
                from docx import Document
                doc = Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                print("Warning: python-docx not available. Reading as text...")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        else:
            # Try reading as text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    def ingest_documents(self, file_paths: List[str]):
        """
        Ingest and process multiple documents
        
        Args:
            file_paths: List of document file paths
        """
        print("\n📚 Ingesting Documents...")
        print("-" * 60)
        
        all_chunks = []
        chunk_id = 0
        
        for file_path in file_paths:
            print(f"Processing: {os.path.basename(file_path)}")
            
            # Load document
            doc_text = self.load_document(file_path)
            self.raw_documents.append(doc_text)
            
            # Chunk document
            chunks = self.processor.chunk_text(doc_text, self.chunk_size, self.overlap)
            
            print(f"  → Created {len(chunks)} chunks")
            
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
        
        print(f"\n✓ Total chunks created: {len(self.chunks)}")
        print(f"✓ Total documents: {len(file_paths)}")
    
    def build_index(self):
        """Build retrieval index from ingested documents"""
        print("\n🔍 Building Retrieval Index...")
        print("-" * 60)
        
        if not self.chunks:
            raise ValueError("No documents ingested. Call ingest_documents() first.")
        
        # Extract chunk texts
        chunk_texts = [chunk.text for chunk in self.chunks]
        
        # Fit retriever
        self.retriever.fit(chunk_texts)
        
        print("✓ Index built successfully")
    
    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of contexts to retrieve
            verbose: Print detailed information
            
        Returns:
            Dictionary with answer and metadata
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"Query: {question}")
            print("=" * 60)
        
        # Retrieve relevant contexts
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        if verbose:
            print(f"\n📖 Retrieved {len(retrieved)} relevant contexts:")
            print("-" * 60)
            for idx, (doc_idx, score, text) in enumerate(retrieved, 1):
                print(f"\n[Context {idx}] Similarity: {score:.4f}")
                print(f"Source: {self.chunks[doc_idx].source}")
                print(f"Preview: {text[:200]}...")
        
        # Generate answer
        answer = self.generator.generate_answer(question, retrieved)
        
        if verbose:
            print("\n" + "=" * 60)
            print("💡 Generated Answer:")
            print("=" * 60)
            print(answer)
            print("=" * 60)
        
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
    
    def save_index(self, save_path: str):
        """Save the RAG system to disk"""
        data = {
            'chunks': [asdict(chunk) for chunk in self.chunks],
            'retriever_vocab': self.retriever.vocabulary,
            'retriever_idf': self.retriever.idf_scores,
            'retriever_doc_vectors': [vec.tolist() for vec in self.retriever.doc_vectors],
            'retriever_documents': self.retriever.documents,
            'raw_documents': self.raw_documents,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ System saved to: {save_path}")
    
    def load_index(self, load_path: str):
        """Load a saved RAG system from disk"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore chunks
        self.chunks = [
            DocumentChunk(**{k: v for k, v in chunk.items() if k != 'embedding'})
            for chunk in data['chunks']
        ]
        
        # Restore retriever
        self.retriever.vocabulary = data['retriever_vocab']
        self.retriever.idf_scores = data['retriever_idf']
        self.retriever.doc_vectors = [np.array(vec) for vec in data['retriever_doc_vectors']]
        self.retriever.documents = data['retriever_documents']
        self.retriever.is_fitted = True
        
        # Restore other attributes
        self.raw_documents = data['raw_documents']
        self.chunk_size = data['chunk_size']
        self.overlap = data['overlap']
        
        print(f"✓ System loaded from: {load_path}")
        print(f"✓ Loaded {len(self.chunks)} chunks")


def main():
    """Main function demonstrating the RAG system"""
    
    # Initialize RAG system
    rag = SanskritRAGSystem(chunk_size=400, overlap=50)
    
    # Ingest documents
    documents = ['data/Rag-docs.docx']
    rag.ingest_documents(documents)
    
    # Build index
    rag.build_index()
    
    # Save the system
    rag.save_index('sanskrit_rag_index.pkl')
    
    # Example queries
    print("\n\n" + "🎯 DEMO QUERIES" + "\n" + "=" * 60)
    
    queries = [
        "मूर्खभृत्यस्य कथा किम् अस्ति?",  # What is the story of the foolish servant?
        "कालीदासः कः आसीत्?",  # Who was Kalidasa?
        "घण्टाकर्णः कः?",  # Who is Ghantakarna?
        "What is the story about the old woman?",
        "Tell me about Bhoj Raja"
    ]
    
    for query in queries:
        result = rag.query(query, top_k=3, verbose=True)
        print("\n")


if __name__ == "__main__":
    main()
