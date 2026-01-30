"""
Flask Web Application for Sanskrit RAG System
Provides a user-friendly web interface for querying Sanskrit documents
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Import the RAG system
from sanskrit_rag_system import SanskritRAGSystem

app = Flask(__name__)

# Global RAG system instance
rag_system = None

def initialize_rag():
    """Initialize or load the RAG system"""
    global rag_system
    
    index_path = 'sanskrit_rag_index.pkl'
    
    if os.path.exists(index_path):
        print("Loading existing RAG index...")
        rag_system = SanskritRAGSystem()
        rag_system.load_index(index_path)
        print("✓ RAG system loaded successfully")
    else:
        print("Creating new RAG system...")
        rag_system = SanskritRAGSystem(chunk_size=400, overlap=50)
        
        # Ingest document
        documents = ['data/Rag-docs.docx']
        rag_system.ingest_documents(documents)
        
        # Build index
        rag_system.build_index()
        
        # Save for future use
        rag_system.save_index(index_path)
        print("✓ RAG system created and saved")


@app.route('/')
def home():
    """Home page with query interface"""
    return render_template('templates/index.html')


@app.route('/query', methods=['POST'])
def query():
    """Handle query requests"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        top_k = int(data.get('top_k', 3))
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Please provide a question'
            })
        
        # Query the RAG system
        result = rag_system.query(question, top_k=top_k, verbose=False)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/stats')
def stats():
    """Get system statistics"""
    try:
        return jsonify({
            'success': True,
            'stats': {
                'total_chunks': len(rag_system.chunks),
                'total_documents': len(rag_system.raw_documents),
                'vocabulary_size': len(rag_system.retriever.vocabulary),
                'chunk_size': rag_system.chunk_size,
                'overlap': rag_system.overlap
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/documents')
def documents():
    """Get list of indexed documents"""
    try:
        doc_info = []
        seen_sources = set()
        
        for chunk in rag_system.chunks:
            if chunk.source not in seen_sources:
                seen_sources.add(chunk.source)
                doc_info.append({
                    'source': chunk.source,
                    'num_chunks': sum(1 for c in rag_system.chunks if c.source == chunk.source)
                })
        
        return jsonify({
            'success': True,
            'documents': doc_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    # Initialize the RAG system
    initialize_rag()
    
    # Run Flask app
    print("\n" + "=" * 60)
    print("🚀 Starting Sanskrit RAG Web Interface")
    print("=" * 60)
    print("\nAccess the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
