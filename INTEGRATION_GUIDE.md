# Integration Guide: Combining Old & New Code

## 🎯 Goal
Integrate the modern web UI with your existing Sanskrit RAG system (`sanskrit_rag_ollama_only.py`)

## 📋 Step-by-Step Integration

### Step 1: Understand What You Have

**OLD CODE (Basic):**
- `app.py` - Simple Flask app with basic query interface
- `sanskrit_rag_system.py` - Original RAG without LLM
- `templates/index.html` - Basic single-page interface

**NEW CODE (Modern):**
- `app_integrated.py` - Full-stack app with authentication
- Beautiful UI templates (landing, login, dashboard, etc.)
- Database for users and history

**BEST CODE (With LLM):**
- `sanskrit_rag_ollama_only.py` - RAG system with Ollama LLM

---

### Step 2: File Organization

Create this folder structure:

```
sanskrit_rag_project/
├── app_integrated.py          # Main application (NEW - already created)
├── sanskrit_rag_ollama_only.py # Your RAG system (KEEP THIS)
├── templates/                  # HTML files
│   ├── landing.html           # Already created
│   ├── login.html             # Already created
│   ├── register.html          # Need to create
│   ├── dashboard.html         # Need to create
│   ├── query.html             # Need to create
│   ├── history.html           # Need to create
│   ├── documents.html         # Need to create
│   └── profile.html           # Need to create
├── static/                     # CSS, JS, images
│   ├── css/
│   └── js/
├── uploads/                    # Uploaded documents
└── sanskrit_rag.db            # SQLite database (auto-created)
```

---

### Step 3: Modify `app_integrated.py` to Use Your RAG System

**Find this section** (around line 49):

```python
def get_rag_system():
    """Get or initialize the RAG system"""
    global rag_system
    
    if rag_system is None:
        # ===== REPLACE THIS SECTION =====
        from sanskrit_rag_ollama_only import SanskritRAGSystem
        
        rag_system = SanskritRAGSystem(
            chunk_size=400,
            overlap=50,
            model_name="llama3.2"  # or "gemma2:2b"
        )
        
        # Load existing index if available
        index_path = 'sanskrit_rag_index.pkl'
        if os.path.exists(index_path):
            print("📂 Loading existing RAG index...")
            rag_system.load_index(index_path)
        else:
            print("⚠️  No index found. Upload documents to create one.")
    
    return rag_system
```

---

### Step 4: Copy Your Templates

**You already have:**
- ✅ `templates/landing.html`
- ✅ `templates/login.html`

**You need to create** (I'll provide code):
1. `register.html`
2. `dashboard.html`
3. `query.html`
4. `history.html`
5. `documents.html`
6. `profile.html`

---

### Step 5: Create Missing Templates

#### **templates/query.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query - Sanskrit RAG</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Poppins', sans-serif; background: #f5f7fa; }
        .sidebar {
            position: fixed; left: 0; top: 0; width: 260px; height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px 20px; color: white;
        }
        .sidebar .logo { font-size: 24px; font-weight: 700; margin-bottom: 40px; }
        .nav-menu a {
            display: block; padding: 15px 20px; color: white; text-decoration: none;
            border-radius: 10px; margin-bottom: 10px; transition: all 0.3s ease;
        }
        .nav-menu a:hover, .nav-menu a.active { background: rgba(255, 255, 255, 0.2); }
        .nav-menu i { margin-right: 10px; width: 20px; }
        .main-content { margin-left: 260px; padding: 30px; }
        .query-container {
            max-width: 900px; margin: 0 auto; background: white;
            border-radius: 20px; padding: 40px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }
        .query-input { width: 100%; padding: 20px; font-size: 16px; border: 2px solid #e0e0e0;
            border-radius: 15px; margin-bottom: 20px; font-family: 'Poppins', sans-serif;
            resize: vertical; min-height: 120px;
        }
        .query-input:focus { outline: none; border-color: #667eea; }
        .btn-query {
            padding: 15px 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 10px; font-size: 16px;
            font-weight: 600; cursor: pointer; transition: all 0.3s ease;
        }
        .btn-query:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3); }
        .answer-box {
            margin-top: 30px; padding: 30px; background: #f8f9fa;
            border-radius: 15px; border-left: 4px solid #667eea; display: none;
        }
        .answer-box h3 { margin-bottom: 15px; color: #333; }
        .answer-text { line-height: 1.8; color: #555; font-size: 16px; }
        .sources { margin-top: 20px; padding-top: 20px; border-top: 1px solid #e0e0e0; }
        .source-item {
            background: white; padding: 15px; margin-bottom: 10px;
            border-radius: 10px; font-size: 14px;
        }
        .loading { display: none; text-align: center; padding: 20px; }
        .loading i { font-size: 32px; color: #667eea; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="logo"><i class="fas fa-book-open"></i> Sanskrit RAG</div>
        <nav class="nav-menu">
            <a href="/dashboard"><i class="fas fa-home"></i> Dashboard</a>
            <a href="/query" class="active"><i class="fas fa-search"></i> Query</a>
            <a href="/history"><i class="fas fa-history"></i> History</a>
            <a href="/documents"><i class="fas fa-file"></i> Documents</a>
            <a href="/profile"><i class="fas fa-user"></i> Profile</a>
            <a href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
        </nav>
    </div>

    <div class="main-content">
        <div class="query-container">
            <h1>Ask a Question</h1>
            <p style="color: #777; margin-bottom: 30px;">Enter your query in Sanskrit (Devanagari) or English</p>
            
            <textarea class="query-input" id="queryInput" placeholder="Enter your question here... (e.g., कालीदासः कः आसीत्?)"></textarea>
            
            <button class="btn-query" onclick="submitQuery()">
                <i class="fas fa-search"></i> Search
            </button>
            
            <div class="loading" id="loading">
                <i class="fas fa-spinner"></i>
                <p>Processing your query...</p>
            </div>
            
            <div class="answer-box" id="answerBox">
                <h3><i class="fas fa-lightbulb"></i> Answer:</h3>
                <div class="answer-text" id="answerText"></div>
                
                <div class="sources" id="sources" style="display: none;">
                    <h4>Sources:</h4>
                    <div id="sourcesList"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function submitQuery() {
            const query = document.getElementById('queryInput').value.trim();
            
            if (!query) {
                alert('Please enter a question');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('answerBox').style.display = 'none';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                if (data.success) {
                    // Show answer
                    document.getElementById('answerText').textContent = data.answer;
                    document.getElementById('answerBox').style.display = 'block';
                    
                    // Show sources if available
                    if (data.sources && data.sources.length > 0) {
                        const sourcesList = document.getElementById('sourcesList');
                        sourcesList.innerHTML = '';
                        
                        data.sources.forEach((source, idx) => {
                            const sourceDiv = document.createElement('div');
                            sourceDiv.className = 'source-item';
                            sourceDiv.innerHTML = `
                                <strong>Source ${idx + 1}</strong> (Score: ${source.score.toFixed(4)})<br>
                                ${source.text}
                            `;
                            sourcesList.appendChild(sourceDiv);
                        });
                        
                        document.getElementById('sources').style.display = 'block';
                    }
                } else {
                    alert('Error: ' + data.message);
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('An error occurred. Please try again.');
                console.error(error);
            }
        }
        
        // Allow Enter key to submit (Shift+Enter for new line)
        document.getElementById('queryInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitQuery();
            }
        });
    </script>
</body>
</html>
```

---

### Step 6: Install Dependencies

```bash
pip install flask flask-sqlalchemy werkzeug requests --break-system-packages
```

---

### Step 7: Run the Integrated App

```bash
# Make sure Ollama is running
ollama list

# Run the integrated app
python app_integrated.py
```

---

### Step 8: Test the Integration

1. **Go to**: `http://localhost:5000`
2. **Register** a new account
3. **Login**
4. **Upload a document** (Documents page)
5. **Query the system** (Query page)
6. **View history** (History page)

---

## 🔄 How Integration Works

### Data Flow:

```
User Query → Web UI → app_integrated.py → sanskrit_rag_ollama_only.py → Ollama → Answer → Web UI
```

### Key Integration Points:

**1. Query Processing** (Line 200 in `app_integrated.py`):
```python
rag = get_rag_system()
result = rag.query(query_text, top_k=3, verbose=False)
```

**2. Document Upload** (Line 255 in `app_integrated.py`):
```python
rag.ingest_documents([filepath])
rag.build_index()
```

**3. Index Persistence**:
```python
# Save after ingesting documents
rag.save_index('sanskrit_rag_index.pkl')

# Load on startup
rag.load_index('sanskrit_rag_index.pkl')
```

---

## 📝 What Changed?

### Old `app.py` vs New `app_integrated.py`:

| Feature | Old | New |
|---------|-----|-----|
| Authentication | ❌ None | ✅ Full system |
| Database | ❌ None | ✅ SQLite |
| UI | Basic HTML | Modern glassmorphism |
| History | ❌ None | ✅ Full tracking |
| Document Upload | ❌ None | ✅ Yes |
| User Profiles | ❌ None | ✅ Yes |

---

## 🎯 Quick Migration Checklist

- [ ] Copy `sanskrit_rag_ollama_only.py` to project folder
- [ ] Create templates folder
- [ ] Copy landing.html and login.html
- [ ] Create remaining templates (register, dashboard, query, history, documents, profile)
- [ ] Update `get_rag_system()` function in `app_integrated.py`
- [ ] Install dependencies
- [ ] Run Ollama
- [ ] Run `app_integrated.py`
- [ ] Test registration
- [ ] Test login
- [ ] Upload a document
- [ ] Test query

---

## 🚨 Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'sanskrit_rag_ollama_only'"
**Solution**: Make sure `sanskrit_rag_ollama_only.py` is in the same folder as `app_integrated.py`

### Issue 2: "Ollama not available"
**Solution**: 
```bash
ollama pull llama3.2
ollama list  # Verify it's installed
```

### Issue 3: Database errors
**Solution**: Delete `sanskrit_rag.db` and restart the app

### Issue 4: "Template not found"
**Solution**: Make sure all templates are in the `templates/` folder

---

## 🎉 You're Done!

You now have a **fully integrated modern web application** with:
- ✅ Beautiful UI
- ✅ User authentication
- ✅ Your existing RAG system
- ✅ Ollama LLM integration
- ✅ Query history
- ✅ Document management

Enjoy your upgraded Sanskrit RAG system! 🚀
