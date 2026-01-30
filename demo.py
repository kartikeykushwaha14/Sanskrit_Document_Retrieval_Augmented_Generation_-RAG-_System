"""
Standalone Demo Script for Sanskrit RAG System
Tests the system with various queries and displays results
"""

import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sanskrit_rag_system import SanskritRAGSystem


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_result(result, query_num):
    """Print query results in a formatted way"""
    print(f"\n{'─' * 70}")
    print(f"Query #{query_num}: {result['question']}")
    print(f"{'─' * 70}")
    
    print(f"\n💡 ANSWER:")
    print(f"{'─' * 70}")
    print(result['answer'])
    
    print(f"\n📚 RETRIEVED CONTEXTS ({result['num_contexts']}):")
    print(f"{'─' * 70}")
    
    for i, ctx in enumerate(result['retrieved_contexts'], 1):
        print(f"\n[Context {i}]")
        print(f"  Source: {ctx['source']}")
        print(f"  Similarity Score: {ctx['score']:.4f}")
        print(f"  Chunk ID: {ctx['chunk_id']}")
        print(f"\n  Text Preview:")
        preview = ctx['text'][:300] + "..." if len(ctx['text']) > 300 else ctx['text']
        print(f"  {preview}")


def main():
    """Main demo function"""
    
    print_section("🕉️  SANSKRIT RAG SYSTEM - DEMO")
    
    print("Initializing Sanskrit RAG System...")
    rag = SanskritRAGSystem(chunk_size=400, overlap=50)
    
    print_section("📚 STEP 1: DOCUMENT INGESTION")
    
    # Check if document exists
    doc_path = 'data/Rag-docs.docx'
    if not os.path.exists(doc_path):
        print(f"❌ Error: Document not found at {doc_path}")
        return
    
    print(f"Loading document: {doc_path}")
    rag.ingest_documents([doc_path])
    
    print_section("🔍 STEP 2: BUILDING INDEX")
    rag.build_index()
    
    print_section("💾 STEP 3: SAVING MODEL")
    save_path = 'sanskrit_rag_demo.pkl'
    rag.save_index(save_path)
    print(f"✓ Model saved to: {save_path}")
    
    print_section("🎯 STEP 4: TESTING QUERIES")
    
    # Define test queries
    test_queries = [
        {
            'query': 'मूर्खभृत्यस्य कथा किम् अस्ति?',
            'description': 'Sanskrit: What is the story of the foolish servant?'
        },
        {
            'query': 'कालीदासः कः आसीत्?',
            'description': 'Sanskrit: Who was Kalidasa?'
        },
        {
            'query': 'घण्टाकर्णः कः?',
            'description': 'Sanskrit: Who is Ghantakarna?'
        },
        {
            'query': 'What is the story about the old woman?',
            'description': 'English: Story about old woman'
        },
        {
            'query': 'Tell me about King Bhoj',
            'description': 'English: Information about Bhoj Raja'
        },
        {
            'query': 'What lesson does the story teach?',
            'description': 'English: Moral/lesson from stories'
        }
    ]
    
    # Execute queries
    for i, test in enumerate(test_queries, 1):
        print(f"\n\n{'═' * 70}")
        print(f"TEST QUERY #{i}: {test['description']}")
        print(f"{'═' * 70}")
        
        result = rag.query(test['query'], top_k=3, verbose=False)
        print_result(result, i)
    
    print_section("📊 STEP 5: SYSTEM STATISTICS")
    
    print(f"Total Chunks:      {len(rag.chunks)}")
    print(f"Total Documents:   {len(rag.raw_documents)}")
    print(f"Vocabulary Size:   {len(rag.retriever.vocabulary)}")
    print(f"Chunk Size:        {rag.chunk_size} characters")
    print(f"Chunk Overlap:     {rag.overlap} characters")
    
    # Document distribution
    print(f"\nDocument Distribution:")
    doc_counts = {}
    for chunk in rag.chunks:
        doc_counts[chunk.source] = doc_counts.get(chunk.source, 0) + 1
    
    for doc, count in doc_counts.items():
        print(f"  • {doc}: {count} chunks")
    
    # Top tokens
    print(f"\nTop 10 Vocabulary Terms:")
    sorted_vocab = sorted(rag.retriever.vocabulary.items(), 
                         key=lambda x: rag.retriever.idf_scores.get(x[0], 0), 
                         reverse=True)[:10]
    for i, (term, _) in enumerate(sorted_vocab, 1):
        idf = rag.retriever.idf_scores.get(term, 0)
        print(f"  {i}. {term} (IDF: {idf:.4f})")
    
    print_section("✅ DEMO COMPLETED SUCCESSFULLY")
    
    print("The Sanskrit RAG system is ready to use!")
    print("\nNext steps:")
    print("  1. Run the web interface: python app.py")
    print("  2. Access at http://localhost:5000")
    print("  3. Try your own queries!")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
