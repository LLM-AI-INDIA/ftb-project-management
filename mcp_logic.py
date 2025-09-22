# mcp_logic.py

import os, re, json, ast, asyncio, base64, requests
import pandas as pd
from io import BytesIO
from PIL import Image
from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Load environment variables
load_dotenv()

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
llm_client = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=os.environ.get("OPENAI_MODEL", "gpt-4o")
)

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# ========== CORE FUNCTIONS ==========

def detect_visualization_intent(query: str) -> str:
    visualization_keywords = [
        "visualize", "chart", "graph", "plot", "dashboard", "trends",
        "distribution", "breakdown", "pie chart", "bar graph", "line chart",
        "show me a report", "analytics for"
    ]
    query_lower = query.lower()
    for keyword in visualization_keywords:
        if keyword in query_lower:
            return "Yes"
    return "No"


# ---------- Tool Discovery ----------


async def _discover_tools(server_url="http://0.0.0.0:8080") -> dict:
    """Asynchronously discovers available tools from the MCP server."""
    try:
        transport = StreamableHttpTransport(f"{server_url}/mcp/")
        async with Client(transport) as client:
            tools = await client.list_tools()
            return {tool.name: tool.description for tool in tools}
    except Exception as e:
        print(f"Tool discovery error: {e}")
        return {}

def discover_tools(server_url="http://0.0.0.0:8080") -> dict:
    """Synchronous wrapper for tool discovery."""
    return asyncio.run(_discover_tools(server_url))

def generate_tool_descriptions(tools_dict: dict) -> str:
    if not tools_dict:
        return "No tools available"
    descriptions = ["Available tools:"]
    for i, (tool_name, tool_desc) in enumerate(tools_dict.items(), 1):
        descriptions.append(f"{i}. {tool_name}: {tool_desc}")
    return "\n".join(descriptions)


# ---------- Utilities ----------
def get_image_base64(img_path):
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode()

def _clean_json(raw: str) -> str:
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    return json_match.group(0).strip() if json_match else raw.strip()


# ---------- MCP Tool Call ----------
def call_mcp_tool(tool_name: str, operation: str, args: dict, server_url="http://0.0.0.0:8080") -> dict:
    url = f"{server_url}/call_tool/tools/{tool_name}/invoke"
    payload = {"tool": tool_name, "operation": operation, "args": args}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"sql": None, "result": f"❌ error calling MCP tool: {e}"}

async def _invoke_tool(tool: str, sql: str, server_url="http://0.0.0.0:8080") -> any:
    # Add trailing slash to avoid 307 redirects
    transport = StreamableHttpTransport(f"{server_url}/mcp/")
    async with Client(transport) as client:
        payload = {"sql": sql}
        res_obj = await client.call_tool(tool, payload)
    if res_obj.structured_content is not None:
        return res_obj.structured_content
    text = "".join(b.text for b in res_obj.content).strip()
    if text.startswith("{") and "}{" in text:
        text = "[" + text.replace("}{", "},{") + "]"
    try:
        return json.loads(text)
    except:
        return text

def call_tool_with_sql(tool: str, sql: str, server_url="http://0.0.0.0:8080") -> any:
    return asyncio.run(_invoke_tool(tool, sql, server_url))


# ---------- Query Parsing ----------
def parse_user_query(query: str, available_tools: dict) -> dict:
    if not available_tools:
        return {"error": "No tools available to query."}

    tool_descriptions = "\n".join([f"- **{n}**: {d}" for n, d in available_tools.items()])
    system_prompt = f"""
You are an expert Google BigQuery SQL writer. Your sole function is to act as a deterministic translator.
Your task is to convert a user's natural language request into a single, valid JSON object.
This JSON object MUST contain two keys:
1.  **"tool"**: The exact name of the tool from the list below.
2.  **"sql"**: A valid, complete, and syntactically correct BigQuery SQL query.

**STRICT RULES:**
* **DO NOT** generate any prose, explanations, or text outside the JSON object. Your entire response must be the JSON.
* The `sql` query MUST use the full, exact table names as specified in the tool descriptions (e.g., `genai-poc-424806.MCP_demo.CarData`).
* The `tool` value MUST be one of the exact tool names provided.

**AVAILABLE TOOLS AND THEIR DESCRIPTIONS:**
{tool_descriptions}

**EXAMPLES:**
1.  User Query: "Show me all records from the BigQuery CarData table."
    JSON Output:
    {{
      "tool": "BigQuery_CarData",
      "sql": "SELECT * FROM `genai-poc-424806.MCP_demo.CarData`"
    }}
2.  User Query: "Find all customer feedback records for product 101."
    JSON Output:
    {{
      "tool": "Oracle_CustomerFeedback",
      "sql": "SELECT * FROM `genai-poc-424806.MCP_demo.CustomerFeedback` WHERE product_id = '101'"
    }}
3.  User Query: "How many users are registered?"
    JSON Output:
    {{
      "tool": "tool_Users",
      "sql": "SELECT COUNT(*) FROM `genai-poc-424806.MCP_demo.Users`"
    }}
4.  User Query: "List the top 5 highest-priced cars."
    JSON Output:
    {{
      "tool": "BigQuery_CarData",
      "sql": "SELECT * FROM `genai-poc-424806.MCP_demo.CarData` ORDER BY price DESC LIMIT 5"
    }}
5.  User Query: "Give me the first 20 records from the Youth Health Records."
    JSON Output:
    {{
      "tool": "Bigquery_YouthHealthRecords",
      "sql": "SELECT * FROM `genai-poc-424806.MCP_demo.YouthHealthRecords` LIMIT 20"
    }}

Now, for the user's query, generate ONLY the JSON response. """
    user_prompt = f'User query: "{query}"'

    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = llm_client.invoke(messages)
        raw_json = _clean_json(resp.content)
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            return ast.literal_eval(raw_json)
    except Exception as e:
        return {"tool": None, "sql": None, "error": f"Failed to parse query: {str(e)}"}


# ---------- Response Generation ----------
def generate_llm_response(operation_result: dict, action: str, tool: str, user_query: str, history=None, history_limit: int = 10) -> str:
    messages_for_llm = []
    if history:
        for m in history[-history_limit:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant":
                messages_for_llm.append(HumanMessage(content=f"(assistant) {content}"))
            else:
                messages_for_llm.append(HumanMessage(content=f"(user) {content}"))

    system_prompt = (
        "You are a helpful database assistant. Generate a brief, natural response "
        "explaining what operation was performed and its result. Be conversational "
        "and informative. Focus on the business context and user-friendly explanation."
    )
    user_prompt = f"""
User asked: "{user_query}"
Operation: {action}
Tool used: {tool}
Result: {json.dumps(operation_result, indent=2)}
Please respond naturally and reference prior conversation context where helpful.

"""

    try:
        messages = [SystemMessage(content=system_prompt)] + messages_for_llm + [HumanMessage(content=user_prompt)]
        response = llm_client.invoke(messages)
        return response.content.strip()
    except Exception as e:
        # Fallback response if LLM call fails
        if action == "read":
            return f"Successfully retrieved data from {tool}."
        elif action == "create":
            return f"Successfully created new record in {tool}."
        elif action == "update":
            return f"Successfully updated record in {tool}."
        elif action == "delete":
            return f"Successfully deleted record from {tool}."
        elif action == "describe":
            return f"Successfully retrieved table schema from {tool}."
        else:
            return f"Operation completed successfully using {tool}."


# ---------- Visualization ----------
def generate_visualization(data: any, user_query: str, tool: str) -> tuple:
    if not anthropic_client:
        return None, None # Returns a tuple of Nones if the client is not initialized

    context = {
        "user_query": user_query,
        "tool": tool,
        "data_type": type(data).__name__,
        "data_sample": data[:5] if isinstance(data, list) and len(data) > 0 else data
    }

    system_prompt = """
    You are a JavaScript dashboard designer and visualization expert.

    ⚡ ALWAYS!!! generate a FULL, self-contained HTML document with:
    - <!DOCTYPE html>, <html>, <head>, <body>, and </html> tags included.
    - <style> for modern responsive CSS (gradient backgrounds, glassmorphism cards, shadows, rounded corners).
    - <script> with all JavaScript logic inline (no external JS files except Chart.js).
    - At least two charts (bar, pie, or line) using Chart.js (CDN: https://cdn.jsdelivr.net/npm/chart.js).
    - Summary stat cards (totals, averages, trends).
    - Optional dynamic lists or tables derived from the data.
    - Smooth animations, styled tooltips, and responsive resizing.

    📌 RULES:
    1. Output ONLY raw HTML, CSS, and JS (no markdown, no explanations).
    2. Charts must have fixed max height (350–400px).
    3. The document is INVALID unless it ends with </html>. Do not stop early.
    4. Always close all opened tags and brackets in HTML, CSS, and JS.
    5. The final deliverable must run directly in a browser without edits.

    🎨 Design:
    - Use a clean dashboard layout with cards, charts, and tables.
    - Gradient backgrounds, glassmorphism effects, shadows, rounded corners.
    - Gradient or neon text for headings and KPI values.
    - Responsive layout for both desktop and mobile.

    ❌ Never truncate output.
    ✅ Always finish the document properly with </html>.
    """

    user_prompt = f"""
    Create an interactive visualization for this data:

    User Query: "{user_query}"
    Tool Used: {tool}
    Data Type: {context['data_type']}
    Data Sample: {json.dumps(context['data_sample'], indent=2)}

    📌 Requirements:
    - Return a COMPLETE, browser-ready HTML document.
    - Include <style> and <script> inline.
    - Close all tags properly.
    - End ONLY with </html>.
    Generate a comprehensive visualization that helps understand the data.
    Focus on the most important insights from the query.
    Make sure charts have fixed heights and don't overflow.
    """

    try:
        resp = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=6000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        viz_code = resp.content[0].text.strip()
        print("LLM successfully generated visualization code.")
        return viz_code
    except Exception as e:
        print(f"Error in LLM visualization generation: {e}")
        import traceback
        traceback.print_exc()
        return None

# Updated generate_table_description function
def generate_table_description(df: pd.DataFrame, content: dict, action: str, tool: str, user_query: str) -> str:
    """Generate a simple, direct confirmation message for a successful query."""

    # Get the number of records from the DataFrame
    record_count = len(df)

    # --- REVISED SYSTEM PROMPT ---
    system_prompt = (
        "You are a helpful and efficient database assistant. Your sole purpose is "
        "to confirm a user's request in a single, friendly sentence. "
        "The response must include the number of records retrieved and confirm that the data has been provided. "
        "Do not provide any analysis, insights, or technical details."
    )
    # --- END REVISED SYSTEM PROMPT ---

    user_prompt = f"""
    The user asked: "{user_query}"
    The query successfully retrieved {record_count} records.
    The data is from the "{tool}" tool.

    Please generate a single, friendly, and simple confirmation message.

    Example: "Sure, here is the car data you requested. It contains 321 records."
    Example: "The records for user 'chen.wei' are here. We found 25 matching entries for you."
    """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm_client.invoke(messages)
        return response.content.strip()
    except Exception as e:
        # Fallback to a simple message if the LLM call fails
        return f"Successfully retrieved {record_count} records from the database."

def list_available_tools(available_tools: dict) -> str:
    """
    Formats the dictionary of available tools into a human-readable string.
    
    Args:
        available_tools (dict): A dictionary mapping tool names to descriptions.
        
    Returns:
        str: A formatted string listing the available tools, or an error message if none are found.
    """
    if not available_tools:
        return "I'm sorry, no tools are currently available. Please check the MCP server connection."
    
    formatted_list = "Here are the available tools:\n"
    for name, desc in available_tools.items():
        formatted_list += f"- **{name}**: {desc}\n"
    
    return formatted_list