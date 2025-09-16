import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import requests
import json
import anthropic
from typing import List, Dict, Any


load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Initialize Anthropic client
try:
    claude_client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    print("Claude client initialized successfully")
except Exception as e:
    print(f"Warning: Claude client not initialized: {e}")
    claude_client = None

# Cache for file contents and folder structure
file_content_cache = {}
folder_structure_cache = {}

# Dummy user model
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

users = {"admin": User(1, "admin", "password123")}

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == int(user_id):
            return user
    return None

def get_all_files_and_folders(http_server_url, max_files=50):
    """Get all files and explore folder structure"""
    try:
        # First, get all items (files and folders)
        response = requests.post(f"{http_server_url}/call_tool", json={
            "name": "search_gdrive_files",
            "arguments": {
                "query": "",  # Empty query gets all supported files
                "max_results": max_files
            }
        })
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                files = result.get("data", [])
                print(f"Found {len(files)} files in Google Drive")
                return files
        
    except Exception as e:
        print(f"Error getting all files: {e}")
    
    return []

def get_comprehensive_file_list(http_server_url):
    """Get a comprehensive list of all accessible files"""
    all_files = []
    
    # Try different queries to catch more files
    search_queries = [
        "",  # All supported files
        "pdf",  # PDFs specifically
        "doc",  # Documents
        "sheet", # Spreadsheets
        "csv",  # CSV files
        "project",  # Project-related files
        "plan",  # Planning documents
        "report",  # Reports
        "data"   # Data files
    ]
    
    seen_file_ids = set()
    
    for query in search_queries:
        try:
            response = requests.post(f"{http_server_url}/call_tool", json={
                "name": "search_gdrive_files",
                "arguments": {
                    "query": query,
                    "max_results": 30
                }
            })
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    files = result.get("data", [])
                    for file in files:
                        if file["id"] not in seen_file_ids:
                            all_files.append(file)
                            seen_file_ids.add(file["id"])
        except Exception as e:
            print(f"Error searching with query '{query}': {e}")
            continue
    
    print(f"Comprehensive search found {len(all_files)} unique files")
    return all_files

def read_file_content(http_server_url, file_id, file_name):
    """Read and extract content from a specific file with caching"""
    
    # Check cache first
    if file_id in file_content_cache:
        print(f"Using cached content for: {file_name}")
        return file_content_cache[file_id]
    
    try:
        response = requests.post(f"{http_server_url}/call_tool", json={
            "name": "read_gdrive_file",
            "arguments": {"file_id": file_id}
        })
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                file_data = result.get("data", {})
                # Cache the result
                file_content_cache[file_id] = file_data
                print(f"Successfully read and cached: {file_name}")
                return file_data
            else:
                print(f"Failed to read file {file_name}: {result.get('error', 'Unknown error')}")
        else:
            print(f"HTTP error reading file {file_name}: {response.status_code}")
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
    
    return None

def extract_text_content(file_data, max_length=8000):
    """Extract readable text from file data with better formatting"""
    if not file_data or "content" not in file_data:
        return ""
    
    content = file_data["content"]
    file_type = content.get("type", "unknown")
    
    text = ""
    
    if file_type == "pdf":
        text = content.get("content", "")
    elif file_type in ["csv", "excel"]:
        # Better formatting for spreadsheet data
        data = content.get("data", [])
        columns = content.get("columns", [])
        shape = content.get("shape", [0, 0])
        
        text += f"Spreadsheet Data ({shape[0]} rows, {shape[1]} columns)\n"
        text += "="*50 + "\n"
        
        if columns:
            text += f"Columns: {', '.join(columns)}\n\n"
        
        if data:
            text += "Data Sample:\n"
            for i, row in enumerate(data[:15]):  # More rows for better context
                row_items = []
                for k, v in row.items():
                    if v is not None and str(v).strip():
                        row_items.append(f"{k}: {v}")
                if row_items:
                    text += f"Row {i+1}: {', '.join(row_items)}\n"
        
        # Include summary statistics if available
        summary = content.get("summary", {})
        if summary:
            text += "\nStatistical Summary:\n"
            for col, stats in summary.items():
                if isinstance(stats, dict):
                    text += f"{col}: mean={stats.get('mean', 'N/A')}, max={stats.get('max', 'N/A')}, min={stats.get('min', 'N/A')}\n"
                    
    elif file_type in ["docx", "txt"]:
        text = content.get("content", "")
    elif file_type == "pptx":
        slides = content.get("slides", [])
        text += f"PowerPoint Presentation ({len(slides)} slides)\n"
        text += "="*50 + "\n"
        for slide in slides:
            text += f"Slide {slide.get('slide', '')}: {slide.get('content', '')}\n\n"
    elif file_type == "json":
        text = content.get("content", "")
    elif file_type == "xml":
        preview = content.get("preview", [])
        text = "\n".join(preview)
    
    # Clean and limit the text
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length] + "\n\n[Content truncated due to length...]"
    
    return text

def gather_relevant_content(http_server_url, user_query, max_files=20):
    """Gather content from files with comprehensive search"""
    
    print(f"Gathering content for query: '{user_query}'")
    
    # Get comprehensive file list
    all_files = get_comprehensive_file_list(http_server_url)
    
    if not all_files:
        return {"success": False, "error": "No files found in Google Drive"}
    
    print(f"Processing up to {min(len(all_files), max_files)} files")
    
    # Collect content from files
    file_contents = []
    processed_count = 0
    
    for file_info in all_files[:max_files]:  # Limit processing for performance
        file_id = file_info["id"]
        file_name = file_info["name"]
        file_type = file_info.get("mimeType", "unknown")
        
        print(f"Processing: {file_name} ({file_type})")
        
        # Read file content
        file_data = read_file_content(http_server_url, file_id, file_name)
        
        if not file_data:
            continue
        
        # Extract text content
        text_content = extract_text_content(file_data)
        
        if text_content.strip():
            file_contents.append({
                "file_name": file_name,
                "content": text_content,
                "file_type": file_data.get("content", {}).get("type", "unknown"),
                "mime_type": file_type,
                "file_id": file_id
            })
            processed_count += 1
    
    return {
        "success": True,
        "file_contents": file_contents,
        "total_files_available": len(all_files),
        "total_files_processed": processed_count
    }

def generate_intelligent_response(user_query, file_contents):
    """Use Claude API to generate intelligent response based on file contents"""
    
    if not claude_client:
        return "Claude API service not available. Please check your ANTHROPIC_API_KEY."
    
    # Prepare context from file contents
    context = "=== DOCUMENT LIBRARY ===\n\n"
    
    for i, file_info in enumerate(file_contents[:15], 1):  # Claude can handle more files
        context += f"DOCUMENT {i}: {file_info['file_name']}\n"
        context += f"Type: {file_info['file_type'].upper()} ({file_info.get('mime_type', 'unknown')})\n"
        context += f"Content:\n{file_info['content'][:4000]}\n"  # More content per file
        context += "\n" + "="*60 + "\n\n"
    
    # Enhanced prompt for Claude
    prompt = f"""You are a professional document analyst and business intelligence assistant. You have access to a comprehensive document library containing project files, reports, spreadsheets, and other business documents.

USER QUESTION: {user_query}

DOCUMENT LIBRARY:
{context}

Please provide a detailed, professional response to the user's question based on the documents provided. Follow these guidelines:

1. Answer the specific question directly and comprehensively
2. Include relevant details, numbers, names, dates, and specific information from the documents
3. Cite which documents contain the information (use document names)
4. If the question asks for lists, provide complete, well-formatted lists
5. For data analysis requests, include relevant statistics and insights
6. If information spans multiple documents, synthesize it coherently
7. Use proper business formatting (bullet points, numbered lists, sections) when appropriate
8. If the question cannot be fully answered from the provided documents, explain what information is available and what might be missing
9. Be thorough but concise - provide complete answers without unnecessary repetition

Response:"""

    try:
        # Call Claude API
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Latest Claude model
            max_tokens=2000,  # Claude has much higher limits
            temperature=0.1,  # Low temperature for accuracy
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return f"I encountered an error while analyzing the documents: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    departments = [
        {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
        {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
        {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
        {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
        {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
        {"name": "Rancho Cordova", "key": "ranchocordova", "icon": "ranchocordova.jpeg"},
        {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
        {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
        {"name": "Office of Energy Infrastructure", "key": "energy", "icon": "energy.jpeg"},
    ]
    return render_template("dashboard.html", departments=departments)

@app.route("/ftb")
@login_required
def ftb():
    return render_template("ftb.html")

@app.route("/oops")
@login_required
def oops():
    return render_template("oops.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/insights")
@login_required
def insights():
    return render_template("insights.html")

@app.route("/qa_chat", methods=["GET"])
@login_required
def qa_page():
    return render_template("qa_chat.html")



# Replace your qa_endpoint with this debug version to identify the issue
from claude_processor import ClaudeLikeDocumentProcessor

# Global processor instance to maintain session across requests
document_processor = None

@app.route("/qa_api", methods=["POST"])
@login_required
def qa_endpoint():
    global document_processor
    
    try:
        query = request.json.get("query")
        
        if not query or not query.strip():
            return jsonify({"answer": "Please provide a valid question."})
        
        print(f"\n=== ITERATIVE CLAUDE-LIKE PROCESSING ===")
        print(f"User query: '{query}'")
        
        http_server_url = "http://localhost:8000"
        
        if not claude_client:
            return jsonify({"answer": "Claude API service not available."})
        
        # Initialize processor if not exists (maintains session)
        if document_processor is None:
            document_processor = ClaudeLikeDocumentProcessor(http_server_url, claude_client)
            print("New session started")
        else:
            print(f"Continuing session - {len(document_processor.session_context['files_mentioned'])} files in context")
        
        # Process query iteratively with session memory
        response = document_processor.process_query_iteratively(query)
        
        return jsonify({"answer": response})
    
    except Exception as e:
        print(f"Error in iterative processing: {e}")
        return jsonify({"answer": f"System error: {str(e)}"})

# Add endpoint to clear session
@app.route("/clear_session", methods=["POST"])
@login_required  
def clear_session():
    global document_processor
    document_processor = None
    return jsonify({"status": "Session cleared successfully"})

# Add endpoint to view session context
@app.route("/session_context", methods=["GET"])
@login_required
def view_session_context():
    global document_processor
    
    if document_processor is None:
        return jsonify({"status": "No active session"})
    
    return jsonify({
        "session_active": True,
        "files_mentioned": len(document_processor.session_context["files_mentioned"]),
        "topics_discussed": document_processor.session_context["topics_discussed"],
        "session_summary": document_processor.session_context["session_summary"],
        "conversation_length": len(document_processor.conversation_history)
    })

    

# Add a simple cache inspection endpoint
@app.route("/inspect_cache")
@login_required
def inspect_cache():
    cache_info = {
        "cache_size": len(file_content_cache),
        "cached_file_ids": list(file_content_cache.keys()),
    }
    return jsonify(cache_info)



# Add a manual file discovery test endpoint
@app.route("/test_discovery")
@login_required 
def test_discovery():
    http_server_url = "http://54.172.238.47:8000"
    
    try:
        response = requests.post(f"{http_server_url}/call_tool", json={
            "name": "search_gdrive_files",
            "arguments": {
                "query": "",
                "max_results": 50
            }
        })
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                files = result.get("data", [])
                return jsonify({
                    "status": "success",
                    "total_files_found": len(files),
                    "sample_files": [f["name"] for f in files[:10]]
                })
        
        return jsonify({"status": "error", "message": "Failed to get files"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

    

# Enhanced cache clearing and debugging endpoints
@app.route("/clear_cache", methods=["POST"])
@login_required
def clear_cache():
    global file_content_cache, folder_structure_cache
    file_content_cache = {}
    folder_structure_cache = {}
    return jsonify({"status": "All caches cleared"})

@app.route("/cache_status")
@login_required
def cache_status():
    return jsonify({
        "cached_files": len(file_content_cache),
        "cached_folders": len(folder_structure_cache),
        "cache_keys": list(file_content_cache.keys())[:10]  # First 10 for debugging
    })

@app.route("/modules/<dept>")
@login_required
def modules(dept):
    icons = {
        "ftb": "ftb.jpeg", "dmv": "dmv.jpeg", "sanjose": "sanjose.jpeg",
        "edd": "edd.jpeg", "fiscal": "fiscal.jpeg", "ranchocordova": "ranchocordova.jpeg",
        "calpers": "calpers.jpeg", "cdfa": "cdfa.jpeg", "energy": "energy.jpeg",
    }
    display_names = {
        "ftb": "Franchise Tax Board (FTB)", "dmv": "Dept of Motor Vehicles",
        "sanjose": "City of San Jose", "edd": "Employment Development Dept",
        "fiscal": "Fi$cal", "ranchocordova": "Rancho Cordova",
        "calpers": "CalPERS", "cdfa": "CDFA", "energy": "Office of Energy Infrastructure",
    }
    
    modules_list = [
        {"name": "Enterprise QA", "icon": "qa.png", "route": "qa_page"},
        {"name": "Workflow", "icon": "workflow.png", "route": "oops"},
        {"name": "Transaction", "icon": "transaction.png", "route": "oops"},
        {"name": "Insights", "icon": "insights.png", "route": "insights"},
        {"name": "Data Management", "icon": "datamanagement.png", "route": "oops"},
        {"name": "Voice Agent", "icon": "voiceagent.png", "route": "oops"}
    ]
    
    company_icon = icons.get(dept, "default.jpeg")
    company_name = display_names.get(dept, "Department")
    
    return render_template("modules.html", company_icon=company_icon, 
                         company_name=company_name, modules=modules_list)

@app.route("/debug")
def debug():
    return f"""
    <h1>Enhanced Debug Info</h1>
    <p>Current working directory: {os.getcwd()}</p>
    <p>GROQ_API_KEY set: {'GROQ_API_KEY' in os.environ}</p>
    <p>Cached files: {len(file_content_cache)}</p>
    <p>Cached folders: {len(folder_structure_cache)}</p>
    <p>HTTP Server Status: <a href="http://localhost:8000/health">Check Health</a></p>
    """

if __name__ == "__main__":
    app.run(debug=True)

