#!/usr/bin/env python3
"""
HTTP wrapper for the Google Drive MCP Server
This creates HTTP endpoints that your Flask app can call
"""

import asyncio
import json
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import your existing MCP server class
from server import GoogleDriveMCPServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gdrive-http-server")

class ToolRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolResponse(BaseModel):
    success: bool
    data: Any = None
    error: str = None

# Initialize the Google Drive server
gdrive_server = GoogleDriveMCPServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    try:
        gdrive_server.initialize_drive_service()
        logger.info("Google Drive HTTP server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Google Drive service: {e}")
        logger.error("Make sure you have:")
        logger.error("1. credentials.json file in the same directory")
        logger.error("2. Proper Google Cloud Project with Drive API enabled")
        logger.error("3. OAuth consent screen configured")
        # Don't raise here, let the server start but handle errors gracefully
    
    yield
    
    # Shutdown
    logger.info("Shutting down Google Drive HTTP server")

# Create FastAPI app
app = FastAPI(
    title="Google Drive MCP HTTP Server", 
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if Google Drive service is working
        if gdrive_server.service:
            # Try a simple API call
            test_result = gdrive_server.service.files().list(pageSize=1).execute()
            return {
                "status": "healthy", 
                "service": "gdrive-http-server",
                "google_drive": "connected"
            }
        else:
            return {
                "status": "partial", 
                "service": "gdrive-http-server",
                "google_drive": "not_initialized"
            }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "service": "gdrive-http-server",
            "google_drive": "error",
            "error": str(e)
        }

@app.post("/call_tool", response_model=ToolResponse)
async def call_tool(request: ToolRequest):
    """Call a tool with the given arguments"""
    try:
        tool_name = request.name
        arguments = request.arguments
        
        logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")
        
        # Check if Google Drive service is initialized
        if not gdrive_server.service:
            try:
                gdrive_server.initialize_drive_service()
            except Exception as e:
                return ToolResponse(
                    success=False, 
                    error=f"Google Drive service not available: {str(e)}"
                )
        
        if tool_name == "search_gdrive_files":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 10)
            
            files = gdrive_server.list_files(query=query, max_results=max_results)
            
            if not files:
                return ToolResponse(
                    success=True, 
                    data=[], 
                    error="No files found matching the query"
                )
            
            # Return the files list directly for easier processing
            return ToolResponse(success=True, data=files)
        
        elif tool_name == "read_gdrive_file":
            file_id = arguments.get("file_id")
            if not file_id:
                raise HTTPException(status_code=400, detail="file_id is required")
            
            result = await gdrive_server.read_file(file_id)
            
            if "error" in result:
                return ToolResponse(success=False, error=result["error"])
            
            return ToolResponse(success=True, data=result)
        
        elif tool_name == "analyze_file_data":
            file_id = arguments.get("file_id")
            analysis_type = arguments.get("analysis_type")
            
            if not file_id or not analysis_type:
                raise HTTPException(status_code=400, detail="file_id and analysis_type are required")
            
            # Read the file first
            file_data = await gdrive_server.read_file(file_id)
            
            if "error" in file_data:
                return ToolResponse(success=False, error=f"Error reading file: {file_data['error']}")
            
            content = file_data.get("content", {})
            file_type = content.get("type", "unknown")
            
            analysis_result = ""
            
            if analysis_type == "summary":
                analysis_result = f"File Summary for {file_data['file_name']}:\n"
                analysis_result += f"Type: {file_type}\n"
                if file_type in ["csv", "excel"]:
                    analysis_result += f"Shape: {content.get('shape', 'Unknown')}\n"
                    analysis_result += f"Columns: {', '.join(content.get('columns', []))}\n"
                elif file_type == "pdf":
                    analysis_result += f"Pages: {content.get('num_pages', 'Unknown')}\n"
                # Add support for new file types
                elif file_type == "pptx":
                    analysis_result += f"Slides: {content.get('num_slides', 'Unknown')}\n"
                elif file_type in ["docx", "txt"]:
                    analysis_result += f"Content length: {len(content.get('content', ''))}\n"
                elif file_type == "xml":
                    analysis_result += f"Root tag: {content.get('root_tag', 'Unknown')}\n"
                    analysis_result += f"Nodes count: {content.get('nodes_count', 'Unknown')}\n"
            
            elif analysis_type == "statistics":
                if file_type in ["csv", "excel"] and content.get("summary"):
                    analysis_result = f"Statistical Summary:\n{json.dumps(content['summary'], indent=2)}"
                else:
                    analysis_result = "Statistics not available for this file type"
            
            elif analysis_type == "structure":
                analysis_result = f"File Structure:\n{json.dumps(content, indent=2, default=str)}"
            
            elif analysis_type == "content":
                if file_type == "pdf":
                    analysis_result = f"PDF Content Preview:\n{content.get('content', 'No content available')}"
                elif file_type in ["csv", "excel"]:
                    analysis_result = f"Data Preview (first few rows):\n{json.dumps(content.get('data', [])[:5], indent=2)}"
                elif file_type in ["docx", "txt"]:
                    analysis_result = f"Text Content Preview:\n{content.get('content', 'No content available')[:1000]}..."
                elif file_type == "pptx":
                    slides_preview = content.get('slides', [])[:3]  # First 3 slides
                    analysis_result = f"PowerPoint Content Preview:\n{json.dumps(slides_preview, indent=2)}"
                elif file_type == "json":
                    analysis_result = f"JSON Content Preview:\n{content.get('content', 'No content available')[:1000]}..."
                else:
                    analysis_result = "Content preview not available for this file type"
            
            return ToolResponse(success=True, data=analysis_result)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in tool call {tool_name}: {e}")
        return ToolResponse(success=False, error=str(e))

@app.get("/list_files")
async def list_files(query: str = None, max_results: int = 100):
    """List files from Google Drive"""
    try:
        files = gdrive_server.list_files(query=query, max_results=max_results)
        return {"success": True, "files": files}
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return {"success": False, "error": str(e)}

# ------------------------------------------------------------------
#  NEW: small MCP-JSON-RPC router (optional – keeps HTTP working)
# ------------------------------------------------------------------
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import uuid

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str
    params: dict | list | None = None

class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    result: dict | None = None
    error: dict | None = None

# very small state-machine for one client
mcp_state = {"initialised": False}

@app.websocket("/mcp")
async def mcp_ws(websocket: WebSocket):
    """Native MCP WebSocket channel (optional)."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_json()
            req = JSONRPCRequest(**raw)

            if req.method == "initialize":
                mcp_state["initialised"] = True
                await websocket.send_json(
                    JSONRPCResponse(
                        id=req.id,
                        result={
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "serverInfo": {"name": "gdrive-mcp", "version": "0.1.0"},
                        },
                    ).dict(exclude_none=True)
                )
                continue

            if req.method == "tools/list":
                await websocket.send_json(
                    JSONRPCResponse(
                        id=req.id,
                        result={
                            "tools": [
                                {
                                    "name": "search_gdrive_files",
                                    "description": "Search Google Drive files",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"},
                                            "max_results": {"type": "integer"},
                                        },
                                    },
                                },
                                {
                                    "name": "read_gdrive_file",
                                    "description": "Read a Google Drive file",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"file_id": {"type": "string"}},
                                        "required": ["file_id"],
                                    },
                                },
                            ]
                        },
                    ).dict(exclude_none=True)
                )
                continue

            # Unknown method
            await websocket.send_json(
                JSONRPCResponse(
                    id=req.id,
                    error={"code": -32601, "message": "Method not found"},
                ).dict(exclude_none=True)
            )
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------
#  FIXED: non-blocking start-up + correct event-loop policy
# ------------------------------------------------------------------
import asyncio, sys, threading, time

def _warm_google_drive():  # runs in a thread
    try:
        gdrive_server.initialize_drive_service()
        logger.info("Google Drive service warmed-up successfully")
    except Exception as exc:
        logger.error("Background Drive init failed: %s", exc)


def _set_event_loop_policy():
    """Selector event-loop for Python ≥ 3.10 (Windows & Linux)."""
    if sys.version_info >= (3, 10):
        policy = asyncio.WindowsSelectorEventLoopPolicy() if sys.platform == "win32" else asyncio.DefaultEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)


if __name__ == "__main__":
    _set_event_loop_policy()                       # 1. fix hang
    threading.Thread(target=_warm_google_drive, daemon=True).start()  # 2. don’t block
    time.sleep(0.25)                               # tiny head-start
    logger.info("Starting Google Drive HTTP/MCP server …")
    uvicorn.run(
        "http_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
