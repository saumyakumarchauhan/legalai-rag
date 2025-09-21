from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import tempfile
import time
from werkzeug.utils import secure_filename
import threading
from datetime import datetime
import logging

# Import your RAG pipeline functions
from rag import (
    load_document,
    chunk_text,
    load_index,
    embed_texts,
    search_external_index,
    search_document_chunks,
    ask_gemini_enhanced,
    run_document_verifier,
    run_brief_mode,
    SentenceTransformer,
    torch
)

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Your RAG configuration
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K_EXTERNAL = 3
TOP_K_DOCUMENT = 2

# Global variables for loaded models and index
embedding_model = None
external_index = None
external_meta = None
external_available = False
model_loaded = False

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_models():
    """Initialize the embedding model and FAISS index on startup."""
    global embedding_model, external_index, external_meta, external_available, model_loaded
    try:
        logger.info(f"🔄 Loading embedding model: {MODEL_NAME}")
        embedding_model = SentenceTransformer(MODEL_NAME)
       
        if torch.cuda.is_available():
            embedding_model = embedding_model.to(torch.device("cuda"))
            logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("⚠ GPU not found, using CPU")
       
        model_loaded = True
       
        # Load external FAISS index
        try:
            external_index, external_meta = load_index()
            external_available = True
            logger.info("✅ External FAISS index loaded successfully")
        except Exception as e:
            logger.warning(f"⚠ Could not load external FAISS index: {e}")
            external_available = False
           
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        model_loaded = False

# API Routes
@app.route('/')
def serve_frontend():
    """Serve the main frontend page."""
    return send_from_directory('.', 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        'status': 'ready' if model_loaded else 'loading',
        'models_loaded': model_loaded,
        'external_index_available': external_available,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'index_entries': len(external_meta['ids']) if external_available else 0
    })

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Handle document upload and processing."""
    try:
        if not model_loaded:
            return jsonify({'error': 'Models not loaded yet. Please wait.'}), 503
           
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
       
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
       
        # Validate file type
        allowed_extensions = {'.pdf', '.doc', '.docx', '.txt', '.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(file.filename)[1].lower()
       
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {file_ext} not supported'}), 400
       
        # Save uploaded file with timestamp
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
       
        logger.info(f"📁 Saving file: {filename}")
        file.save(file_path)
       
        # Process document using your RAG pipeline
        try:
            start_time = time.time()
           
            logger.info("📄 Loading document...")
            document_text = load_document(file_path)
           
            logger.info("✂ Chunking document...")
            chunks = chunk_text(document_text)
           
            processing_time = time.time() - start_time
           
            logger.info(f"✅ Document processed: {len(document_text)} chars, {len(chunks)} chunks")
           
            # Store processed data in app context
            if not hasattr(app, 'document_cache'):
                app.document_cache = {}
               
            app.document_cache[timestamp] = {
                'text': document_text,
                'chunks': chunks,
                'filename': file.filename,
                'file_path': file_path,
                'upload_time': datetime.now().isoformat()
            }
           
            # Create response
            response_data = {
                'success': True,
                'file_id': timestamp,
                'filename': file.filename,
                'file_size': os.path.getsize(file_path),
                'character_count': len(document_text),
                'chunk_count': len(chunks),
                'processing_time': round(processing_time, 2)
            }
           
            return jsonify(response_data)
           
        except Exception as e:
            # Clean up file on processing error
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"❌ Document processing error: {e}")
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500
           
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_question():
    """Analyze a question against uploaded document and legal knowledge base."""
    try:
        if not model_loaded:
            return jsonify({'error': 'Models not loaded yet. Please wait.'}), 503
           
        data = request.get_json()
       
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question'}), 400
       
        question = data['question'].strip()
        file_id = data.get('file_id', None)
       
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
       
        # Get document data from cache if file_id is provided
        document_results = []
        if file_id:
            document_cache = getattr(app, 'document_cache', {})
            if file_id not in document_cache:
                return jsonify({'error': 'Document not found. Please re-upload.'}), 404
           
            doc_data = document_cache[file_id]
            chunks = doc_data['chunks']
           
            logger.info("📄 Searching document chunks...")
            document_results = search_document_chunks(chunks, embedding_model, question, TOP_K_DOCUMENT)
       
        logger.info(f"🔎 Analyzing question: {question[:100]}...")
        start_time = time.time()
       
        # Search external index
        external_results = []
        if external_available:
            logger.info("🔍 Searching external legal knowledge base...")
            external_results = search_external_index(external_index, external_meta, embedding_model, question, TOP_K_EXTERNAL)
       
        # Generate response
        logger.info("🤖 Generating AI response...")
        response = ask_gemini_enhanced(question, external_results, document_results)
       
        analysis_time = time.time() - start_time
       
        logger.info(f"✅ Analysis complete in {analysis_time:.2f}s")
       
        return jsonify({
            'success': True,
            'response': response,
            'external_sources': len(external_results),
            'document_sources': len(document_results),
            'analysis_time': round(analysis_time, 2),
            'external_scores': [r['score'] for r in external_results],
            'document_scores': [r['score'] for r in document_results]
        })
       
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def general_chat():
    """Handle general chat questions without a document."""
    try:
        if not model_loaded:
            return jsonify({'error': 'Models not loaded yet. Please wait.'}), 503
           
        data = request.get_json()
       
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question'}), 400
       
        question = data['question'].strip()
       
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
       
        logger.info(f"💬 General chat question: {question[:100]}...")
        start_time = time.time()
       
        # Search external index
        external_results = []
        if external_available:
            logger.info("🔍 Searching external legal knowledge base...")
            external_results = search_external_index(external_index, external_meta, embedding_model, question, TOP_K_EXTERNAL)
       
        # Generate response
        logger.info("🤖 Generating AI response...")
        response = ask_gemini_enhanced(question, external_results, [])
       
        analysis_time = time.time() - start_time
       
        logger.info(f"✅ Chat complete in {analysis_time:.2f}s")
       
        return jsonify({
            'success': True,
            'response': response,
            'external_sources': len(external_results),
            'analysis_time': round(analysis_time, 2),
            'external_scores': [r['score'] for r in external_results]
        })
       
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

# NEW: Document Briefing Endpoint
@app.route('/api/brief', methods=['POST'])
def brief_document():
    """Generate structured document briefing."""
    try:
        if not model_loaded:
            return jsonify({'error': 'Models not loaded yet. Please wait.'}), 503
           
        data = request.get_json()
        file_id = data.get('file_id')
       
        if not file_id:
            return jsonify({'error': 'Missing file_id'}), 400
       
        # Get document data from cache
        document_cache = getattr(app, 'document_cache', {})
        if file_id not in document_cache:
            return jsonify({'error': 'Document not found. Please re-upload.'}), 404
           
        doc_data = document_cache[file_id]
        document_text = doc_data['text']
       
        if not document_text.strip():
            return jsonify({'error': 'Document appears to be empty'}), 400
       
        logger.info(f"📋 Generating document brief for: {doc_data['filename']}")
        start_time = time.time()
       
        # Generate structured brief using your existing function
        brief_result = run_brief_mode("Brief this legal document with key details", document_text)
       
        processing_time = time.time() - start_time
       
        if not brief_result:
            return jsonify({'error': 'Failed to generate document brief'}), 500
       
        logger.info(f"✅ Document brief generated in {processing_time:.2f}s")
       
        return jsonify({
            'success': True,
            'brief': brief_result,
            'processing_time': round(processing_time, 2),
            'document_info': {
                'filename': doc_data['filename'],
                'character_count': len(document_text),
                'upload_time': doc_data['upload_time']
            }
        })
       
    except Exception as e:
        logger.error(f"❌ Brief generation error: {e}")
        return jsonify({'error': f'Failed to generate brief: {str(e)}'}), 500

# NEW: Document Verification Endpoint
@app.route('/api/verify', methods=['POST'])
def verify_document():
    """Run document verification analysis."""
    try:
        if not model_loaded:
            return jsonify({'error': 'Models not loaded yet. Please wait.'}), 503
           
        data = request.get_json()
        file_id = data.get('file_id')
       
        if not file_id:
            return jsonify({'error': 'Missing file_id'}), 400
       
        # Get document data from cache
        document_cache = getattr(app, 'document_cache', {})
        if file_id not in document_cache:
            return jsonify({'error': 'Document not found. Please re-upload.'}), 404
           
        doc_data = document_cache[file_id]
        document_text = doc_data['text']
       
        logger.info(f"🔍 Running document verification for: {doc_data['filename']}")
        start_time = time.time()
       
        # Run document verification using your existing function
        if external_available:
            verification_results = run_document_verifier(document_text, external_index, external_meta, embedding_model)
        else:
            # Fallback to rule-based checking only
            from rag import run_document_verifier_rules
            rules, score = run_document_verifier_rules(document_text)
            verification_results = {
                'sufficiency_score': score,
                'rule_checklist': rules,
                'chunks': []
            }
       
        processing_time = time.time() - start_time
       
        logger.info(f"✅ Document verification complete in {processing_time:.2f}s")
       
        # Format the results for frontend consumption
        formatted_results = {
            'sufficiency_score': verification_results['sufficiency_score'],
            'total_checks': len(verification_results['rule_checklist']),
            'rule_checklist': verification_results['rule_checklist'],
            'chunks_analyzed': len(verification_results['chunks']),
            'similar_cases_found': sum(len(chunk.get('similar_cases', [])) for chunk in verification_results['chunks']),
            'processing_time': round(processing_time, 2),
            'document_info': {
                'filename': doc_data['filename'],
                'character_count': len(document_text),
                'upload_time': doc_data['upload_time']
            }
        }
       
        # Add assessment level
        score = verification_results['sufficiency_score']
        total = len(verification_results['rule_checklist'])
        
        if score == total:
            formatted_results['assessment'] = 'complete'
            formatted_results['assessment_text'] = 'Document appears to be a complete legal document'
        elif score >= total * 0.75:
            formatted_results['assessment'] = 'mostly_complete'
            formatted_results['assessment_text'] = 'Document appears mostly complete but may be missing key elements'
        elif score >= total * 0.5:
            formatted_results['assessment'] = 'incomplete'
            formatted_results['assessment_text'] = 'Document may be incomplete or informal'
        else:
            formatted_results['assessment'] = 'inadequate'
            formatted_results['assessment_text'] = 'Document appears incomplete or not a formal legal document'
       
        return jsonify({
            'success': True,
            'verification': formatted_results
        })
       
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return jsonify({'error': f'Failed to verify document: {str(e)}'}), 500

@app.route('/api/document/<file_id>', methods=['GET'])
def get_document_info(file_id):
    """Get information about an uploaded document."""
    try:
        document_cache = getattr(app, 'document_cache', {})
       
        if file_id not in document_cache:
            return jsonify({'error': 'Document not found'}), 404
           
        doc_data = document_cache[file_id]
       
        return jsonify({
            'success': True,
            'filename': doc_data['filename'],
            'character_count': len(doc_data['text']),
            'chunk_count': len(doc_data['chunks']),
            'upload_time': doc_data['upload_time']
        })
       
    except Exception as e:
        logger.error(f"❌ Document info error: {e}")
        return jsonify({'error': f'Failed to get document info: {str(e)}'}), 500

@app.route('/api/clear/<file_id>', methods=['DELETE'])
def clear_document(file_id):
    """Clear uploaded document from memory and disk."""
    try:
        document_cache = getattr(app, 'document_cache', {})
       
        if file_id in document_cache:
            doc_data = document_cache[file_id]
           
            # Remove file from disk
            if 'file_path' in doc_data and os.path.exists(doc_data['file_path']):
                os.remove(doc_data['file_path'])
                logger.info(f"🗑 Deleted file: {doc_data['file_path']}")
           
            # Remove from cache
            del document_cache[file_id]
           
        return jsonify({'success': True, 'message': 'Document cleared'})
       
    except Exception as e:
        logger.error(f"❌ Clear error: {e}")
        return jsonify({'error': f'Failed to clear document: {str(e)}'}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents (for debugging)."""
    try:
        document_cache = getattr(app, 'document_cache', {})
       
        documents = []
        for file_id, doc_data in document_cache.items():
            documents.append({
                'file_id': file_id,
                'filename': doc_data['filename'],
                'upload_time': doc_data['upload_time'],
                'character_count': len(doc_data['text']),
                'chunk_count': len(doc_data['chunks'])
            })
       
        return jsonify({
            'success': True,
            'documents': documents,
            'total_count': len(documents)
        })
       
    except Exception as e:
        logger.error(f"❌ List documents error: {e}")
        return jsonify({'error': f'Failed to list documents: {str(e)}'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# Cleanup function
def cleanup_uploads():
    """Clean up old uploaded files on shutdown."""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                if os.path.isfile(file_path):
                    # Delete files older than 1 hour
                    if time.time() - os.path.getctime(file_path) > 3600:
                        os.remove(file_path)
                        logger.info(f"🗑 Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

# Initialize models in a separate thread
def initialize_async():
    """Initialize models asynchronously."""
    initialize_models()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0
    })

if __name__ == '__main__':
    # Record start time
    app.start_time = time.time()
    
    # Initialize models on startup
    print("🚀 Starting LegalAI Backend Server...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🤖 Model: {MODEL_NAME}")
    
    # Start model initialization in background
    init_thread = threading.Thread(target=initialize_async, daemon=True)
    init_thread.start()
    
    try:
        app.run(
            host='0.0.0.0',
            port=7860,
            debug=True,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")
        cleanup_uploads()
    finally:
        cleanup_uploads()