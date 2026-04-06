"""
Complete Production-Ready Sanskrit RAG Web Application
- Integrated with actual RAG system (sanskrit_rag_system.py)
- Credit-based system with payment gateway
- Razorpay integration for credit purchases
- FIXED: Duplicate document prevention
- FIXED: Additive index building (new docs added to existing index)
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from google import genai
from datetime import datetime, timedelta
import os
import secrets
import sys
from dotenv import load_dotenv
import requests

# Import the RAG system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sanskrit_rag_system_new import SanskritRAGSystem

app = Flask(__name__)

# ============================================
# CONFIGURATION
# ============================================
load_dotenv()

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///sanskrit_rag.db')
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'docx'}

app.config['RAZORPAY_KEY_ID'] = os.environ.get('RAZORPAY_KEY_ID', 'rzp_test_YOUR_KEY')
app.config['RAZORPAY_KEY_SECRET'] = os.environ.get('RAZORPAY_KEY_SECRET', 'YOUR_SECRET')

db = SQLAlchemy(app)
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# -------------------------
# Gemini Client
# -------------------------
client = genai.Client(api_key="GEMINI_API_KEY")


# ============================================
# CREDIT & PRICING CONSTANTS
# ============================================
SIGNUP_CREDITS = 20
QUERY_COST = 3
DAILY_FREE_CREDITS = 10

PRICING_PLANS = {
    'basic':   {'credits': 50,  'price': 99,  'name': 'Basic'},
    'pro':     {'credits': 150, 'price': 249, 'name': 'Pro'},
    'premium': {'credits': 500, 'price': 749, 'name': 'Premium'}
}

# ============================================
# INITIALIZE RAG SYSTEM (GLOBAL)
# ============================================
rag_system = None
INDEX_PATH = 'sanskrit_rag_index.pkl'

def get_rag_system():
    """Initialize or get the RAG system"""
    global rag_system

    if rag_system is None:
        print("🔧 Initializing Sanskrit RAG System...")
        rag_system = SanskritRAGSystem(chunk_size=400, overlap=50)

        if os.path.exists(INDEX_PATH):
            try:
                rag_system.load_index(INDEX_PATH)
                print("✓ Loaded existing RAG index")
            except Exception as e:
                print(f"⚠️  Could not load index: {e}")

    return rag_system


# ============================================
# DATABASE MODELS
# ============================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    credits = db.Column(db.Integer, default=SIGNUP_CREDITS)
    last_credit_refresh = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    queries = db.relationship('QueryHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    transactions = db.relationship('Transaction', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def refresh_daily_credits(self):
        now = datetime.utcnow()
        if self.last_credit_refresh and (now - self.last_credit_refresh) >= timedelta(hours=24):
            self.credits += DAILY_FREE_CREDITS
            self.last_credit_refresh = now
            db.session.commit()
            return True
        return False

    def has_credits(self, cost=QUERY_COST):
        return self.credits >= cost

    def deduct_credits(self, cost=QUERY_COST):
        if self.has_credits(cost):
            self.credits -= cost
            db.session.commit()
            return True
        return False

    def add_credits(self, amount):
        self.credits += amount
        db.session.commit()


class QueryHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query_text = db.Column(db.Text, nullable=False)
    answer_text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    credits_used = db.Column(db.Integer, default=QUERY_COST)
    num_contexts = db.Column(db.Integer, default=3)


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), unique=True, nullable=False)   # ← unique constraint
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    chunks_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='ready')


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    order_id = db.Column(db.String(100), nullable=False)
    payment_id = db.Column(db.String(100))
    plan = db.Column(db.String(50), nullable=False)
    credits = db.Column(db.Integer, nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='pending')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# ============================================
# HELPER FUNCTIONS
# ============================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ============================================
# ROUTES - PUBLIC
# ============================================
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('workspace'))
    return redirect(url_for('landing'))


@app.route('/landing')
def landing():
    return render_template('landing.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()

        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 400

        user = User(username=username, email=email, credits=SIGNUP_CREDITS)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Registration successful! You received {SIGNUP_CREDITS} free credits!'
        })

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            user.refresh_daily_credits()
            session['user_id'] = user.id
            session['username'] = user.username
            return jsonify({'success': True, 'message': 'Login successful!'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))


# ============================================
# ROUTES - AUTHENTICATED
# ============================================
@app.route('/workspace')
@login_required
def workspace():
    user = User.query.get(session['user_id'])
    credits_added = user.refresh_daily_credits()

    documents = Document.query.order_by(Document.upload_date.desc()).limit(10).all()
    recent_queries = QueryHistory.query.filter_by(user_id=user.id)\
        .order_by(QueryHistory.timestamp.desc()).limit(5).all()

    total_queries = QueryHistory.query.filter_by(user_id=user.id).count()

    return render_template('workspace.html',
                           user=user,
                           documents=documents,
                           recent_queries=recent_queries,
                           credits_added=credits_added,
                           total_queries=total_queries,
                           query_cost=QUERY_COST,
                           daily_free=DAILY_FREE_CREDITS)


@app.route('/pricing')
@login_required
def pricing():
    user = User.query.get(session['user_id'])
    return render_template('pricing.html',
                           user=user,
                           plans=PRICING_PLANS,
                           razorpay_key=app.config['RAZORPAY_KEY_ID'])


# ============================================
# UPLOAD DOCUMENT  ← KEY FIX: additive indexing + duplicate prevention
# ============================================
# @app.route('/api/upload_document', methods=['POST'])
# @login_required
# def upload_document():
#     if 'file' not in request.files:
#         return jsonify({'success': False, 'message': 'No file uploaded'}), 400
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({'success': False, 'message': 'No file selected'}), 400
#
#     if not allowed_file(file.filename):
#         return jsonify({'success': False, 'message': 'Invalid file type. Use .txt or .docx'}), 400
#
#     filename = secure_filename(file.filename)
#
#     # ── DUPLICATE CHECK ──────────────────────────────────────────────────────
#     existing = Document.query.filter_by(filename=filename).first()
#     if existing:
#         return jsonify({
#             'success': False,
#             'message': f'"{filename}" has already been uploaded on '
#                        f'{existing.upload_date.strftime("%b %d, %Y")}. '
#                        f'Please rename the file if you want to upload a new version.',
#             'duplicate': True,
#             'existing_doc': {
#                 'filename': existing.filename,
#                 'upload_date': existing.upload_date.strftime('%b %d, %Y'),
#                 'chunks_count': existing.chunks_count
#             }
#         }), 409
#
#     try:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#
#         # ── ADDITIVE INDEX BUILD ─────────────────────────────────────────────
#         # get_rag_system() already loads the saved index on first call,
#         # so rag_system.chunks already contains ALL previously indexed chunks.
#         # We only ingest the NEW file, then rebuild the FAISS index over
#         # the combined chunk list, and save.
#         rag = get_rag_system()
#
#         chunks_before = len(rag.chunks)          # how many existed already
#         rag.ingest_documents([filepath])          # appends new chunks
#         rag.build_index()                         # rebuilds FAISS over ALL chunks
#         rag.save_index(INDEX_PATH)                # persists everything
#
#         new_chunks = len(rag.chunks) - chunks_before
#
#         # ── DATABASE RECORD ──────────────────────────────────────────────────
#         doc = Document(
#             filename=filename,
#             chunks_count=new_chunks,
#             status='ready'
#         )
#         db.session.add(doc)
#         db.session.commit()
#
#         return jsonify({
#             'success': True,
#             'message': f'"{filename}" uploaded and indexed successfully!',
#             'doc_id': doc.id,
#             'new_chunks': new_chunks,
#             'total_chunks': len(rag.chunks)
#         })
#
#     except Exception as e:
#         return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/upload_document', methods=['POST'])
@login_required
def upload_document():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type. Use .txt or .docx'}), 400

    filename = secure_filename(file.filename)

    # ── DUPLICATE CHECK ──────────────────────────────────────────────────────
    existing = Document.query.filter_by(filename=filename).first()
    if existing:
        return jsonify({
            'success': False,
            'message': f'"{filename}" was already uploaded on {existing.upload_date.strftime("%b %d, %Y")}. '
                       f'Please rename the file if you want to upload a new version.',
            'duplicate': True,
            'existing_doc': {
                'filename': existing.filename,
                'upload_date': existing.upload_date.strftime('%b %d, %Y'),
                'chunks_count': existing.chunks_count
            }
        }), 409

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ── INCREMENTAL INDEX BUILD ─────────────────────────────────────────
        # get_rag_system() loads the existing index on first call.
        # We now use append=True to ADD new chunks to existing ones.
        rag = get_rag_system()

        chunks_before = len(rag.chunks)  # Count before adding new document

        # Ingest new document (append=True adds to existing chunks)
        rag.ingest_documents([filepath], append=True)

        # Rebuild index over ALL chunks (old + new)
        rag.build_index()

        # Save the complete index
        rag.save_index(INDEX_PATH)

        new_chunks = len(rag.chunks) - chunks_before

        # ── DATABASE RECORD ──────────────────────────────────────────────────
        doc = Document(
            filename=filename,
            chunks_count=new_chunks,
            status='ready'
        )
        db.session.add(doc)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'"{filename}" uploaded and indexed successfully!',
            'doc_id': doc.id,
            'new_chunks': new_chunks,
            'total_chunks': len(rag.chunks),
            'total_documents': len(rag.raw_documents)
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


# ============================================
# QUERY
# ============================================
@app.route('/api/query', methods=['POST'])
@login_required
def query():
    user = User.query.get(session['user_id'])

    if not user.has_credits(QUERY_COST):
        return jsonify({
            'success': False,
            'message': f'Insufficient credits! You need {QUERY_COST} credits. You have {user.credits}.',
            'credits': user.credits
        }), 403

    data = request.get_json()
    query_text = data.get('query', '').strip()

    if not query_text:
        return jsonify({'success': False, 'message': 'Query cannot be empty'}), 400

    try:
        user.deduct_credits(QUERY_COST)

        rag = get_rag_system()
        result = rag.query(query_text, top_k=3, verbose=False)

        answer = result.get('answer', 'No answer generated')
        contexts = result.get('retrieved_contexts', [])

        # === Gemini refinement step ===
        # refined_answer = refine_with_gemini(answer,query_text)
        refined_answer = refine_with_gemini(answer, contexts, query_text)

        history = QueryHistory(
            user_id=user.id,
            query_text=query_text,
            answer_text=refined_answer,
            credits_used=QUERY_COST,
            num_contexts=len(contexts)
        )
        db.session.add(history)
        db.session.commit()

        return jsonify({
            'success': True,
            'answer': refined_answer,
            'contexts': contexts,
            'credits_remaining': user.credits,
            'credits_used': QUERY_COST
        })

    except Exception as e:
        user.credits += QUERY_COST
        db.session.commit()

        return jsonify({
            'success': False,
            'message': f'Error processing query: {str(e)}',
            'credits_remaining': user.credits
        }), 500

# def refine_with_gemini(answer: str, contexts: list, user_query: str) -> str:
#     """
#     Refines Sanskrit text fragments and retrieved contexts into a concise Sanskrit answer
#     with an English translation inline.
#     """
#
#     # Join all retrieved contexts into a single string
#     context_text = "\n".join([c.get("text", "") for c in contexts])
#
#     prompt = f"""
# You are an expert Sanskrit scholar. The user asked: {user_query}
#
# The initial answer is:
# {answer}
#
# The retrieved contexts from documents are:
# {context_text}
#
# Instructions:
# 1. Refine the answer into a **concise, clear Sanskrit sentence or two** that directly answers the question.
# 2. Include **English translation immediately in parentheses** after the Sanskrit.
# 3. Do NOT use bullet points, brackets, notes, or extra commentary.
# 4. Output only the **final refined Sanskrit sentence(s) with translation**.
#
# Example format:
# Sanskrit answer. (Translated: English answer)
# """
#
#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt
#     )
#
#     return response.text.strip()

def refine_with_gemini(answer: str, contexts: list, user_query: str) -> str:
    """
    Refines text fragments and retrieved contexts into a concise answer.
    - If the query is in Sanskrit, output Sanskrit + English translation.
    - If the query is in English, output only English.
    """

    # Join all retrieved contexts into a single string
    context_text = "\n".join([c.get("text", "") for c in contexts])

    # Detect if the question is in Sanskrit (basic check for Devanagari characters)
    is_sanskrit = any("\u0900" <= c <= "\u097F" for c in user_query)

    if is_sanskrit:
        prompt = f"""
You are an expert Sanskrit scholar. The user asked: {user_query}

The initial answer is:
{answer}

The retrieved contexts from documents are:
{context_text}

Instructions:
1. Refine the answer into a **concise, clear Sanskrit sentence or two** that directly answers the question.
2. Include **English translation immediately in parentheses** after the Sanskrit.
3. Do NOT use bullet points, brackets, notes, or extra commentary.
4. Output only the **final refined Sanskrit sentence(s) with translation**.
"""
    else:
        prompt = f"""
You are an expert in the topic. The user asked: {user_query}

The initial answer is:
{answer}

The retrieved contexts from documents are:
{context_text}

Instructions:
1. Refine the answer into a **concise, clear English sentence or two** that directly answers the question.
2. Do NOT output Sanskrit or any translation note.
3. Do NOT use bullet points, brackets, notes, or extra commentary.
4. Output only the **final English answer**.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


# ============================================
# PAYMENT ROUTES
# ============================================
@app.route('/api/create_order', methods=['POST'])
@login_required
def create_order():
    data = request.get_json()
    plan = data.get('plan')

    if plan not in PRICING_PLANS:
        return jsonify({'success': False, 'message': 'Invalid plan'}), 400

    try:
        import razorpay
        client = razorpay.Client(auth=(app.config['RAZORPAY_KEY_ID'], app.config['RAZORPAY_KEY_SECRET']))

        plan_info = PRICING_PLANS[plan]
        amount = plan_info['price'] * 100

        order = client.order.create({
            'amount': amount,
            'currency': 'INR',
            'payment_capture': 1
        })

        transaction = Transaction(
            user_id=session['user_id'],
            order_id=order['id'],
            plan=plan,
            credits=plan_info['credits'],
            amount=amount,
            status='pending'
        )
        db.session.add(transaction)
        db.session.commit()

        return jsonify({
            'success': True,
            'order_id': order['id'],
            'amount': amount,
            'currency': 'INR',
            'key': app.config['RAZORPAY_KEY_ID']
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/verify_payment', methods=['POST'])
@login_required
def verify_payment():
    data = request.get_json()

    try:
        import razorpay
        client = razorpay.Client(auth=(app.config['RAZORPAY_KEY_ID'], app.config['RAZORPAY_KEY_SECRET']))

        client.utility.verify_payment_signature(data)

        transaction = Transaction.query.filter_by(order_id=data['razorpay_order_id']).first()

        if transaction:
            transaction.payment_id = data['razorpay_payment_id']
            transaction.status = 'success'

            user = User.query.get(session['user_id'])
            user.add_credits(transaction.credits)
            db.session.commit()

            return jsonify({
                'success': True,
                'message': f'{transaction.credits} credits added successfully!',
                'credits': user.credits
            })

        return jsonify({'success': False, 'message': 'Transaction not found'}), 404

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/user_info')
@login_required
def user_info():
    user = User.query.get(session['user_id'])
    user.refresh_daily_credits()

    total_queries = QueryHistory.query.filter_by(user_id=user.id).count()

    return jsonify({
        'username': user.username,
        'email': user.email,
        'credits': user.credits,
        'total_queries': total_queries,
        'member_since': user.created_at.strftime('%B %Y'),
        'query_cost': QUERY_COST,
        'daily_free': DAILY_FREE_CREDITS
    })


# ============================================
# HEALTH CHECK & ERROR HANDLERS
# ============================================
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'Sanskrit RAG System'}), 200


@app.errorhandler(404)
def not_found(e):
    return redirect(url_for('landing'))


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'

    print("\n" + "="*60)
    print("  SANSKRIT RAG SYSTEM - PRODUCTION READY")
    print("="*60)
    print(f"\n🚀 Server starting on port {port}")
    print(f"💳 Credit System:")
    print(f"   - Signup bonus: {SIGNUP_CREDITS} credits")
    print(f"   - Per query cost: {QUERY_COST} credits")
    print(f"   - Daily free: {DAILY_FREE_CREDITS} credits")
    print(f"💰 Pricing Plans:")
    for key, plan in PRICING_PLANS.items():
        print(f"   - {plan['name']}: ₹{plan['price']} = {plan['credits']} credits")
    print("\n" + "="*60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
