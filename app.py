import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import requests
import json
import anthropic
from typing import List, Dict, Any
from claude_processor import ClaudeLikeDocumentProcessor
from visual_analyst import VisualizationAnalyst
from mcp_logic import generate_table_description,detect_visualization_intent,discover_tools,parse_user_query,call_tool_with_sql,generate_llm_response,generate_visualization,call_mcp_tool


load_dotenv()


class ProjectDocumentProcessor(ClaudeLikeDocumentProcessor):
    def __init__(self, http_server_url, claude_client):
        super().__init__(http_server_url, claude_client)
        # You'll need to find your actual project management folder ID from Google Drive
        self.project_folder_id = "1_RuIezT1KN8miQ_3_167rkCXP9ZoZq35"
        
    def _search_files(self, args):
        """Override to search with project-specific terms"""
        query = args.get("query", "")
        max_results = args.get("max_results", 20)
        
        # Try project-specific search terms
        project_queries = []
        if query:
            project_queries = [
                f"{query} project",
                f"{query} management", 
                f"project {query}",
                query  # Original query as fallback
            ]
        else:
            project_queries = ["project", "plan", "milestone", "timeline"]
        
        all_files = []
        seen_ids = set()
        
        for search_query in project_queries:
            response = requests.post(f"{self.http_server_url}/call_tool", json={
                "name": "search_gdrive_files",
                "arguments": {
                    "query": search_query,
                    "max_results": max_results
                }
            })
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    files_found = result.get("data", [])
                    # Filter for project-related files and avoid duplicates
                    for f in files_found:
                        if f["id"] not in seen_ids:
                            file_name_lower = f.get('name', '').lower()
                            # Check if file is project-related
                            if any(keyword in file_name_lower 
                                 for keyword in ['project', 'plan', 'milestone', 'timeline', 'budget', 
                                               'resource', 'task', 'deliverable', 'roadmap']):
                                all_files.append(f)
                                seen_ids.add(f["id"])
            
            # Stop if we have enough files
            if len(all_files) >= max_results:
                break
        
        return {
            "success": True,
            "files": [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "type": f.get("mimeType", "unknown"),
                    "size": f.get("size", "unknown"),
                    "modified": f.get("modifiedTime", "unknown")
                }
                for f in all_files[:max_results]
            ],
            "total_found": len(all_files),
            "search_location": "project_management_files"
        }


# Operations-specific document processor class
class OperationsDocumentProcessor(ClaudeLikeDocumentProcessor):
    def __init__(self, http_server_url, claude_client):
        super().__init__(http_server_url, claude_client)
        self.operations_folder_id = "116wKTVLaOkK6QRyozG6cx9SBloxUWTro"
        
    def _search_files(self, args):
        """Override to search only in operations folder"""
        query = args.get("query", "")
        max_results = args.get("max_results", 20)
        
        response = requests.post(f"{self.http_server_url}/call_tool", json={
            "name": "search_gdrive_files",
            "arguments": {
                "query": f"{query} operations",
                "max_results": max_results
            }
        })
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                files_found = result.get("data", [])
                operations_files = [f for f in files_found 
                                  if 'operations' in f.get('name', '').lower() 
                                  or 'operational' in f.get('name', '').lower()]
                
                return {
                    "success": True,
                    "files": [
                        {
                            "id": f["id"],
                            "name": f["name"],
                            "type": f.get("mimeType", "unknown"),
                            "size": f.get("size", "unknown"),
                            "modified": f.get("modifiedTime", "unknown")
                        }
                        for f in operations_files
                    ],
                    "total_found": len(operations_files),
                    "search_location": "operations_folder"
                }
        
        return {"success": False, "error": "Failed to search operations folder"}



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


def extract_viz_data_from_response(response_text, query):
    """Extract real numerical data from your files for visualization"""
    import re
    
    viz_data = {}
    
    # Extract different types of data based on patterns in your files
    
    # Project budgets/costs (dollar amounts)
    budget_matches = re.findall(r'budget[:\s]*\$?([\d,]+\.?\d*)', response_text, re.IGNORECASE)
    cost_matches = re.findall(r'cost[:\s]*\$?([\d,]+\.?\d*)', response_text, re.IGNORECASE)
    
    # Employee efficiency scores (decimals out of 5)
    efficiency_matches = re.findall(r'efficiency[:\s]*([\d\.]+)', response_text, re.IGNORECASE)
    score_matches = re.findall(r'score[:\s]*([\d\.]+)', response_text, re.IGNORECASE)
    
    # Availability percentages
    availability_matches = re.findall(r'availability[:\s]*([\d\.]+)%?', response_text, re.IGNORECASE)
    percent_matches = re.findall(r'([\d\.]+)%', response_text)
    
    # Project benefits
    benefit_matches = re.findall(r'benefit[:\s]*\$?([\d,]+\.?\d*)', response_text, re.IGNORECASE)
    
    # Convert to numbers
    def clean_numbers(matches):
        return [float(m.replace(',', '')) for m in matches if m]
    
    budgets = clean_numbers(budget_matches + cost_matches)
    efficiencies = clean_numbers(efficiency_matches + score_matches)
    availabilities = clean_numbers(availability_matches)
    benefits = clean_numbers(benefit_matches)
    percentages = clean_numbers(percent_matches)
    
    # Create visualizations based on what data we found
    
    # Budget/Cost visualization
    if budgets and ('budget' in query.lower() or 'cost' in query.lower()):
        if len(budgets) >= 2:
            viz_data['budgetComparison'] = {
                'Current Budget': budgets[0],
                'Projected Cost': budgets[1] if len(budgets) > 1 else budgets[0] * 1.1,
                'Benefits': benefits[0] if benefits else budgets[0] * 0.8
            }
    
    # Employee efficiency scores
    if efficiencies and ('efficiency' in query.lower() or 'score' in query.lower()):
        viz_data['efficiencyScores'] = {}
        for i, score in enumerate(efficiencies[:5]):  # Max 5 scores
            viz_data['efficiencyScores'][f'Employee {i+1}'] = score
    
    # Availability percentages
    if availabilities and 'availability' in query.lower():
        viz_data['availabilityRates'] = {}
        departments = ['IT', 'Operations', 'Finance', 'HR', 'Marketing']
        for i, avail in enumerate(availabilities[:5]):
            dept = departments[i] if i < len(departments) else f'Department {i+1}'
            viz_data['availabilityRates'][dept] = avail
    
    # General percentage breakdown
    if percentages and len(percentages) >= 2:
        viz_data['percentageBreakdown'] = {}
        categories = ['Category A', 'Category B', 'Category C', 'Category D']
        for i, pct in enumerate(percentages[:4]):
            cat = categories[i] if i < len(categories) else f'Category {i+1}'
            viz_data['percentageBreakdown'][cat] = pct
    
    return viz_data if viz_data else None



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


def generate_intelligent_response(user_query, file_contents):
    """Use Claude API to generate intelligent response with better formatting"""
    
    if not claude_client:
        return "Claude API service not available. Please check your ANTHROPIC_API_KEY."
    
    # Prepare context from file contents
    context = "=== DOCUMENT LIBRARY ===\n\n"
    
    for i, file_info in enumerate(file_contents[:15], 1):
        context += f"DOCUMENT {i}: {file_info['file_name']}\n"
        context += f"Type: {file_info['file_type'].upper()}\n"
        context += f"Content:\n{file_info['content'][:4000]}\n"
        context += "\n" + "="*60 + "\n\n"
    
    # Use enhanced prompt
    enhanced_prompt = enhance_claude_prompt(user_query, context)

    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": enhanced_prompt}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return f"I encountered an error while analyzing the documents: {str(e)}"
    


def enhance_claude_prompt(user_query, context):
    """
    Enhance the prompt to encourage proper formatting
    """
    
    enhanced_prompt = f"""You are a professional document analyst. When presenting information, use proper formatting:

- Use markdown tables for tabular data
- Use bullet points for lists
- Use headers (##, ###) for sections
- Use **bold** for important information
- Use code blocks for any technical data

USER QUESTION: {user_query}

DOCUMENT CONTEXT:
{context}

Please provide a well-formatted response that makes good use of tables, headers, and lists where appropriate. If you need to present data in rows and columns, always use markdown table format like:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

Response:"""

    return enhanced_prompt

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for("categories"))  # Changed from "dashboard"
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")


@app.route("/categories")
@login_required
def categories():
    return render_template("categories.html")


@app.route("/dashboard/<category>")
@login_required
def dashboard(category):
    # Define ALL departments for each category (9 total)
    category_departments = {
        "public_services": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {"name": "Office of Energy Infrastructure", "key": "energy", "icon": "energy.jpeg"},
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
            {"name": "Rancho Cordova", "key": "ranchocordova", "icon": "ranchocordova.jpeg"},
        ],
        "energy": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {"name": "Office of Energy Infrastructure", "key": "energy", "icon": "energy.jpeg"},
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
            {"name": "Rancho Cordova", "key": "ranchocordova", "icon": "ranchocordova.jpeg"},
        ],
        "health": [
            {"name": "FTB", "key": "ftb", "icon": "ftb.jpeg"},
            {"name": "Dept of Motor Vehicles", "key": "dmv", "icon": "dmv.jpeg"},
            {"name": "City of San Jose", "key": "sanjose", "icon": "sanjose.jpeg"},
            {"name": "Employment Development Dept", "key": "edd", "icon": "edd.jpeg"},
            {"name": "CalPERS", "key": "calpers", "icon": "calpers.jpeg"},
            {"name": "CDFA", "key": "cdfa", "icon": "cdfa.jpeg"},
            {"name": "Office of Energy Infrastructure", "key": "energy", "icon": "energy.jpeg"},
            {"name": "Fi$cal", "key": "fiscal", "icon": "fiscal.jpeg"},
            {"name": "Rancho Cordova", "key": "ranchocordova", "icon": "ranchocordova.jpeg"},
        ]
    }
    
    departments = category_departments.get(category, [])
    category_names = {
        "public_services": "Public Services",
        "energy": "Energy", 
        "health": "Health"
    }
    
    return render_template("dashboard.html", 
                         departments=departments, 
                         category_name=category_names.get(category, "Unknown"))

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

@app.route("/insights_api", methods=["POST"])
@login_required
def insights_api():
    try:
        query = request.json.get("query")
        if not query:
            return jsonify({"answer": "Please enter a valid query."})
        
        mcp_server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
        
        # Discover tools and parse query (keep existing code...)
        available_tools = discover_tools(mcp_server_url)
        if not available_tools:
            available_tools = {
                "Bigquery_Customer": "Customer data queries",
                "Cloud_SQL_Product": "Product catalog queries", 
                "SAP_Hana_Sales": "Sales transaction queries",
                "Oracle_CustomerFeedback": "Customer feedback queries",
                "amazon_redshift_CustomerCallLog": "Customer service call logs"
            }
        
        parsed = parse_user_query(query, available_tools)
        if "error" in parsed:
            return jsonify({"answer": parsed["error"]})
        
        tool = parsed.get("tool")
        sql = parsed.get("sql")
        
        if not tool or not sql:
            return jsonify({"answer": "Could not parse your query. Please try rephrasing."})
        
        # Call MCP tool
        data = call_tool_with_sql(tool, sql, mcp_server_url)
        
        answer = None
        html_table = None
        
        # Generate response (keep existing code for answer and html_table...)
        if data.get("rows") and len(data["rows"]) > 0:
            try:
                import pandas as pd
                df = pd.DataFrame(data["rows"])
                answer = generate_table_description(df, data, "read", tool, query)
                html_table = df.head(50).to_html(
                    classes='table table-striped table-hover',
                    table_id='data-table',
                    escape=False,
                    index=False,
                    max_rows=50
                )
            except Exception as e:
                print(f"Error processing DataFrame: {e}")
                answer = f"Successfully retrieved {data.get('row_count', 0)} records from the database."
                html_table = "<div class='alert alert-info'>Data retrieved successfully.</div>"
        else:
            answer = generate_llm_response(data, "read", tool, query)
            html_table = "<div class='alert alert-warning'>No data found for your query.</div>"
        
        # NEW: Use the same visualization approach as project/operations
        viz_data = None
        if data.get("rows"):
            try:
                viz_data = extract_insights_viz_data(data, query, tool)
                print(f"Viz data keys: {list(viz_data.keys()) if viz_data else 'None'}")
            except Exception as e:
                print(f"Error in extract_insights_viz_data: {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify({
            "answer": answer or "Query processed successfully.",
            "html_table": html_table,
            "visualization": viz_data,  # Now returns simple data objects like project/operations
            "has_visualization": viz_data is not None
        })
        
    except Exception as e:
        print(f"Error in insights_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": f"System error: {str(e)}"})





project_processor = None
operations_processor = None


@app.route("/qa_projects")
@login_required
def qa_projects():
    """Project Management QA Chat"""
    return render_template("qa_chat_projects.html")

@app.route("/qa_operations")
@login_required
def qa_operations():
    """Operations QA Chat"""
    return render_template("qa_chat_operations.html")


@app.route("/qa_api", methods=["POST"])
@login_required
def qa_api():
    """Handle QA requests for project management using ClaudeDocumentProcessor"""
    global project_processor
    
    try:
        query = request.json.get("query")
        
        if not query or not query.strip():
            return jsonify({"answer": "Please provide a valid question."})
        
        print(f"\n=== PROJECT MANAGEMENT QA ENDPOINT (CLAUDE PROCESSOR) ===")
        print(f"User query: '{query}'")
        
        http_server_url = "http://0.0.0.0:8000"
        
        if not claude_client:
            return jsonify({"answer": "Claude API service not available."})
        
        # Initialize project processor if not exists - but use the ORIGINAL ClaudeDocumentProcessor
        if project_processor is None:
            from claude_processor import ClaudeLikeDocumentProcessor
            project_processor = ClaudeLikeDocumentProcessor(http_server_url, claude_client)
            print("New project management session started with ClaudeDocumentProcessor")
        else:
            print(f"Continuing project session - {len(project_processor.session_context['files_mentioned'])} files in context")
        
        # Process query iteratively using the original ClaudeDocumentProcessor
        print("Processing query with ClaudeDocumentProcessor...")
        response = project_processor.process_query_iteratively(query)
        
        # Extract visualization data from the response
        viz_data = extract_viz_data_from_response(response, query)
        
        if viz_data:
            print(f"Found visualization data: {list(viz_data.keys())}")
        else:
            print("No visualization data found")
        
        return jsonify({
            "answer": response,  # Full detailed response from ClaudeDocumentProcessor
            "visualization": viz_data,
            "has_visualization": viz_data is not None,
            "source": "claude_document_processor"
        })
        
    except Exception as e:
        print(f"Error in project management processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": f"System error: {str(e)}"})


@app.route("/qa_operations_api", methods=["POST"])
@login_required
def qa_operations_endpoint():
    """API endpoint for operations QA with operations-specific search"""
    global operations_processor
    
    try:
        query = request.json.get("query")
        
        if not query or not query.strip():
            return jsonify({"answer": "Please provide a valid question."})
        
        print(f"\n=== OPERATIONS QA ENDPOINT ===")
        print(f"User query: '{query}'")
        
        http_server_url = "http://0.0.0.0:8000"
        
        if not claude_client:
            return jsonify({"answer": "Claude API service not available."})
        
        # Initialize operations processor if not exists
        if operations_processor is None:
            operations_processor = OperationsDocumentProcessor(http_server_url, claude_client)
            print("New operations session started")
        else:
            print(f"Continuing operations session - {len(operations_processor.session_context['files_mentioned'])} files in context")
        
        # Process query iteratively using operations-specific processor
        response = operations_processor.process_query_iteratively(query)
        
        # Only attempt to extract REAL data from the response - no fallbacks
        print("Attempting to extract visualization data from actual response...")
        viz_data = extract_viz_data_from_response(response, query)
        
        if viz_data:
            print(f"Found real data for visualization: {list(viz_data.keys())}")
        else:
            print("No real data found for visualization")
        
        return jsonify({
            "answer": response,
            "visualization": viz_data,
            "has_visualization": viz_data is not None,
            "source": "operations"
        })
        
    except Exception as e:
        print(f"Error in operations processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": f"System error: {str(e)}"})



def extract_insights_viz_data(data, query, tool):
    """Extract visualization data from MCP response data"""
    if not data.get("rows") or len(data["rows"]) == 0:
        return None
    
    import pandas as pd
    df = pd.DataFrame(data["rows"])
    viz_data = {}
    
    # Car data specific visualizations
    if 'CarData' in tool and 'Selling_Price' in df.columns:
        # Top 10 cars by price
        if 'Car_Name' in df.columns:
            top_cars = df.nlargest(10, 'Selling_Price')
            viz_data['topCarPrices'] = top_cars.set_index('Car_Name')['Selling_Price'].to_dict()
        
        # Fuel type distribution (if available)
        if 'Fuel_Type' in df.columns:
            fuel_counts = df['Fuel_Type'].value_counts().head(5)
            viz_data['fuelTypeDistribution'] = fuel_counts.to_dict()
            
        # Year distribution
        if 'Year' in df.columns:
            df['Year_Only'] = pd.to_datetime(df['Year']).dt.year
            year_counts = df['Year_Only'].value_counts().head(8).sort_index()
            viz_data['yearDistribution'] = year_counts.to_dict()
    
    # Sales data visualizations
    elif 'Sales' in tool:
        if 'TotalAmount' in df.columns and 'CustomerID' in df.columns:
            top_customers = df.groupby('CustomerID')['TotalAmount'].sum().nlargest(10)
            viz_data['topCustomerSales'] = top_customers.to_dict()
        
        if 'SaleDate' in df.columns and 'TotalAmount' in df.columns:
            df['SaleDate'] = pd.to_datetime(df['SaleDate'])
            monthly_sales = df.groupby(df['SaleDate'].dt.to_period('M'))['TotalAmount'].sum()
            viz_data['monthlySales'] = {str(k): v for k, v in monthly_sales.to_dict().items()}
    
    # Customer data visualizations
    elif 'Customer' in tool:
        if 'JoinDate' in df.columns:
            df['JoinDate'] = pd.to_datetime(df['JoinDate'])
            monthly_joins = df.groupby(df['JoinDate'].dt.to_period('M')).size()
            viz_data['customerGrowth'] = {str(k): v for k, v in monthly_joins.to_dict().items()}
    
    # Product data visualizations
    elif 'Product' in tool:
        if 'Category' in df.columns:
            category_counts = df['Category'].value_counts().head(8)
            viz_data['productCategories'] = category_counts.to_dict()
        
        if 'Price' in df.columns and 'ProductName' in df.columns:
            top_priced = df.nlargest(10, 'Price')
            viz_data['topProductPrices'] = top_priced.set_index('ProductName')['Price'].to_dict()
    
    # Feedback data visualizations  
    elif 'Feedback' in tool:
        if 'Score' in df.columns:
            score_dist = df['Score'].value_counts()
            viz_data['feedbackScores'] = score_dist.to_dict()
    
    # Generic numeric column visualization
    else:
        # Find numeric columns for generic visualization
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            first_numeric = numeric_cols[0]
            if len(df) <= 20:  # Small dataset
                if len(df.columns) > 1:
                    label_col = df.columns[0] if df.columns[0] != first_numeric else df.columns[1]
                    viz_data['genericData'] = df.head(10).set_index(label_col)[first_numeric].to_dict()
            else:  # Larger dataset - show aggregation
                viz_data['dataOverview'] = {
                    'Total Records': len(df),
                    'Numeric Columns': len(numeric_cols),
                    'Text Columns': len(df.columns) - len(numeric_cols)
                }
    
    return viz_data if viz_data else None


# Keep only the real data extraction function - remove the fallback functions
def extract_viz_data_from_response(response_text, query):
    """Extract ONLY real numerical data from the response - no fallbacks"""
    import re
    
    viz_data = {}
    
    # Extract different types of data based on patterns in your actual files
    patterns = {
        # Project costs with names: "Project Alpha: $150,000"
        'project_costs': r'(?:project|initiative)\s+([^:\n]+):\s*\$?([\d,]+\.?\d*)',
        
        # Budget categories: "Personnel: $45,000"
        'budget_items': r'([A-Za-z\s]+(?:budget|cost|expense|personnel|technology|infrastructure|training)?):\s*\$?([\d,]+\.?\d*)',
        
        # Efficiency scores: "Team A efficiency: 4.2/5"
        'efficiency_scores': r'([^:\n]+)\s+(?:efficiency|score|rating|performance):\s*([\d\.]+)(?:/5)?',
        
        # Percentages: "Availability: 95.5%"
        'percentages': r'([^:\n]+):\s*([\d\.]+)%',
        
        # Time periods: "Q1 2024: $100,000"
        'temporal_data': r'(Q[1-4]\s*20\d{2}|20\d{2}[\-\s]Q[1-4]|[A-Z][a-z]+\s*20\d{2}):\s*\$?([\d,]+\.?\d*)',
    }
    
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            processed_data = {}
            for match in matches:
                try:
                    if len(match) >= 2:
                        name = match[0].strip()
                        value_str = match[1].replace(',', '')
                        if value_str and name and len(name) > 2:  # More strict filtering
                            value = float(value_str)
                            if value > 0:
                                processed_data[name] = value
                except (ValueError, IndexError):
                    continue
            
            # Only include if we have at least 2 meaningful data points
            if processed_data and len(processed_data) >= 2:
                viz_data[pattern_name] = processed_data
                print(f"Found {pattern_name}: {processed_data}")
    
    return viz_data if viz_data else None


def _response_contains_numerical_data(response_text):
    """Check if response contains numerical data worth visualizing"""
    import re
    
    patterns = [
        r'\$[\d,]+',  # Dollar amounts
        r'\d+%',      # Percentages  
        r'\d+\.\d+',  # Decimal numbers
        r':\s*\d+',   # Colon followed by numbers
        r'\d+\s*(hours|days|weeks|months)',  # Time periods
        r'\d+\s*(employees|projects|incidents)',  # Counts
    ]
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        count += len(matches)
        if count >= 2:
            print(f"Found numerical data: {matches[:3]}...")
            return True
    
    print(f"Found {count} numerical patterns (need 2+)")
    return False


def clean_response_text(text):
    """Remove JavaScript code and function definitions from response"""
    import re
    
    # Remove function definitions
    text = re.sub(r'function\s+\w+\s*\([^)]*\)\s*\{[\s\S]*?\}', '', text)
    # Remove variable assignments with functions
    text = re.sub(r'(const|let|var)\s+\w+\s*=\s*function[\s\S]*?\};?', '', text)
    # Remove object method definitions
    text = re.sub(r'\w+\s*:\s*function\s*\([^)]*\)\s*\{[\s\S]*?\}', '', text)
    # Remove DOM manipulation
    text = re.sub(r'document\.createElement[\s\S]*?;', '', text)
    text = re.sub(r'\w+\.innerHTML\s*=[\s\S]*?;', '', text)
    text = re.sub(r'\w+\.appendChild[\s\S]*?;', '', text)
    # Remove chart creation calls
    text = re.sub(r'createChart\([^)]*\);?', '', text)
    # Remove visualization titles that appear as text
    text = re.sub(r'Budget Analysis|Efficiency Scores|Availability Rates', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


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
    http_server_url = "http://0.0.0.0:8000"
    
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
        "ftb": "ftb.jpeg", 
        "dmv": "dmv.jpeg", 
        "sanjose": "sanjose.jpeg",
        "edd": "edd.jpeg", 
        "fiscal": "fiscal.jpeg", 
        "ranchocordova": "ranchocordova.jpeg",
        "calpers": "calpers.jpeg", 
        "cdfa": "cdfa.jpeg", 
        "energy": "energy.jpeg",
    }
    
    display_names = {
        "ftb": "Franchise Tax Board (FTB)", 
        "dmv": "Dept of Motor Vehicles",
        "sanjose": "City of San Jose", 
        "edd": "Employment Development Dept",
        "fiscal": "Fi$cal", 
        "ranchocordova": "Rancho Cordova",
        "calpers": "CalPERS", 
        "cdfa": "CDFA", 
        "energy": "Office of Energy Infrastructure",
    }
    
    modules_list = [
        {"name": "Enterprise QA", "icon": "qa.png", "route": "qa_page"},
        {"name": "Workflow", "icon": "workflow.png", "route": "oops"},
        {"name": "Transaction", "icon": "transaction.png", "route": "oops"},
        {"name": "Insights", "icon": "insights.png", "route": "insights"},
        {"name": "Data Management", "icon": "datamanagement.png", "route": "oops"},
        {"name": "Voice Agent", "icon": "voiceagent.png", "route": "voice_agent"}
    ]
    
    company_icon = icons.get(dept, "default.jpeg")
    company_name = display_names.get(dept, "Department")
    
    return render_template("modules.html", company_icon=company_icon, 
                           company_name=company_name,modules=modules_list)


@app.route("/voice_agent")
@login_required
def voice_agent():
    """Voice Agent Page"""
    return render_template("voice_agent.html")


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
# Add this test route to your app.py to verify integration:

@app.route("/test_viz", methods=["GET"])
@login_required
def test_visualization():
    """Test visualization generation with mock data"""
    
    # Test data that should definitely trigger visualization
    mock_response = """
    Project Alpha Budget Analysis:
    
    Project Alpha: $150,000
    Project Beta: $200,000
    Project Gamma: $175,000
    
    Department Efficiency Scores:
    IT Department: 4.2
    Finance Team: 3.8
    Operations: 4.5
    Marketing: 3.9
    
    Budget Breakdown:
    Personnel: $85,000
    Technology: $45,000
    Infrastructure: $30,000
    Training: $15,000
    """
    
    test_query = "Show me project costs and efficiency data"
    
    try:
        # Test VisualizationAnalyst directly
        if claude_client:
            from visual_analyst import VisualizationAnalyst
            analyst = VisualizationAnalyst(claude_client)
            
            viz_result = analyst.analyze_and_visualize(
                response_text=mock_response,
                user_query=test_query,
                session_context={}
            )
            
            return jsonify({
                "status": "success",
                "mock_response_length": len(mock_response),
                "visualization_generated": viz_result is not None,
                "visualization_data": viz_result,
                "charts_count": len(viz_result.get('visualizations', [])) if viz_result else 0,
                "test_query": test_query
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Claude client not available"
            })
            
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

# Also add this simple test to ensure your imports work:
@app.route("/test_imports", methods=["GET"])
@login_required
def test_imports():
    """Test that all required modules can be imported"""
    
    results = {}
    
    try:
        from claude_processor import ClaudeLikeDocumentProcessor
        results["claude_processor"] = "✓ Imported successfully"
    except Exception as e:
        results["claude_processor"] = f"✗ Import failed: {str(e)}"
    
    try:
        from visual_analyst import VisualizationAnalyst
        results["visual_analyst"] = "✓ Imported successfully"
    except Exception as e:
        results["visual_analyst"] = f"✗ Import failed: {str(e)}"
    
    try:
        import anthropic
        results["anthropic"] = "✓ Imported successfully"
        results["claude_client_status"] = "✓ Available" if claude_client else "✗ Not initialized"
    except Exception as e:
        results["anthropic"] = f"✗ Import failed: {str(e)}"
    
    return jsonify({
        "import_results": results,
        "claude_api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "python_path": sys.path
    })


if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
