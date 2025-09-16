import json
import requests
from typing import List, Dict, Any
import anthropic

class ClaudeLikeDocumentProcessor:
    def __init__(self, http_server_url, claude_client):
        self.http_server_url = http_server_url
        self.claude_client = claude_client
        self.conversation_history = []
        self.session_context = {
            "files_mentioned": {},  # Track files that have been discussed
            "topics_discussed": [],  # Track conversation topics
            "user_preferences": {},  # Learn user patterns
            "session_summary": ""   # Running summary of conversation
        }
        self.available_tools = [
            {
                "name": "search_files",
                "description": "Search for files in Google Drive by name or content type. Uses enhanced fallback search strategies.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query - can be partial matches, will try broader searches if needed"},
                        "max_results": {"type": "integer", "description": "Maximum results", "default": 20}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "read_file",
                "description": "Read complete content from a specific file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string", "description": "Google Drive file ID"},
                        "focus_area": {"type": "string", "description": "Optional: specific aspect to focus on (e.g., 'budget', 'timeline', 'personnel')"}
                    },
                    "required": ["file_id"]
                }
            },
            {
                "name": "get_file_metadata",
                "description": "Get metadata about files without reading full content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_pattern": {"type": "string", "description": "Pattern to match files", "default": ""}
                    }
                }
            },
            {
                "name": "recall_session_context",
                "description": "Access previous conversation context and discussed topics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "context_type": {
                            "type": "string", 
                            "enum": ["files_mentioned", "topics_discussed", "full_summary"],
                            "description": "Type of context to retrieve"
                        }
                    },
                    "required": ["context_type"]
                }
            }
        ]
        
    def update_session_context(self, query, files_accessed, topics):
        """Update session context with new information"""
        
        # Update files mentioned
        for file_info in files_accessed:
            file_id = file_info.get("id") or file_info.get("file_id")
            if file_id:
                self.session_context["files_mentioned"][file_id] = {
                    "name": file_info.get("name", file_info.get("file_name", "unknown")),
                    "last_accessed": "current_session",
                    "context": f"Accessed when user asked: {query[:100]}"
                }
        
        # Update topics
        query_topics = self.extract_topics_from_query(query)
        for topic in query_topics:
            if topic not in self.session_context["topics_discussed"]:
                self.session_context["topics_discussed"].append(topic)
        
        # Update session summary
        if len(self.session_context["topics_discussed"]) > 0:
            self.session_context["session_summary"] = f"User has been asking about: {', '.join(self.session_context['topics_discussed'])}. Files accessed: {len(self.session_context['files_mentioned'])} total."
    
    def extract_topics_from_query(self, query):
        """Extract topics from user query"""
        topics = []
        query_lower = query.lower()
        
        # Common project-related topics
        topic_keywords = {
            "projects": ["project", "initiative", "development"],
            "budget": ["budget", "cost", "expense", "financial"],
            "employees": ["employee", "staff", "team", "personnel"],
            "incidents": ["incident", "issue", "problem", "bug"],
            "reports": ["report", "analysis", "summary"],
            "timeline": ["timeline", "schedule", "deadline", "milestone"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return results"""
        
        print(f"TOOL CALL: {tool_name} with args: {arguments}")
        
        try:
            if tool_name == "search_files":
                return self._search_files(arguments)
            elif tool_name == "read_file":
                return self._read_file(arguments)
            elif tool_name == "get_file_metadata":
                return self._get_file_metadata(arguments)
            elif tool_name == "recall_session_context":
                return self._recall_session_context(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def _search_files(self, args):
        """Enhanced search for files with fallback strategies"""
        query = args.get("query", "")
        max_results = args.get("max_results", 20)
        
        # Strategy 1: Try exact query first
        response = requests.post(f"{self.http_server_url}/call_tool", json={
            "name": "search_gdrive_files",
            "arguments": {"query": query, "max_results": max_results}
        })
        
        files_found = []
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                files_found = result.get("data", [])
        
        # Strategy 2: If no results and query looks like a project name, try broader searches
        if not files_found and len(query.split()) > 1:
            broader_terms = []
            words = query.replace('"', '').split()
            
            # Try individual significant words
            for word in words:
                if len(word) > 3 and word.lower() not in ['the', 'and', 'for', 'with', 'project']:
                    broader_terms.append(word)
            
            # Try combinations of 2 words
            if len(words) >= 2:
                broader_terms.extend([f"{words[i]} {words[i+1]}" for i in range(len(words)-1)])
            
            # Search with broader terms
            for term in broader_terms[:3]:  # Try top 3 terms
                response = requests.post(f"{self.http_server_url}/call_tool", json={
                    "name": "search_gdrive_files",
                    "arguments": {"query": term, "max_results": max_results}
                })
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        new_files = result.get("data", [])
                        # Add new files, avoid duplicates
                        existing_ids = {f["id"] for f in files_found}
                        for file in new_files:
                            if file["id"] not in existing_ids:
                                files_found.append(file)
                
                if len(files_found) >= max_results:
                    break
        
        # Strategy 3: If still no results, search for common project file types
        if not files_found:
            common_searches = ["project", "plan", "incident", "report", "data"]
            for search_term in common_searches:
                response = requests.post(f"{self.http_server_url}/call_tool", json={
                    "name": "search_gdrive_files",
                    "arguments": {"query": search_term, "max_results": 10}
                })
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        new_files = result.get("data", [])
                        existing_ids = {f["id"] for f in files_found}
                        for file in new_files:
                            if file["id"] not in existing_ids:
                                files_found.append(file)
                
                if files_found:  # Stop at first successful broader search
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
                for f in files_found
            ],
            "total_found": len(files_found),
            "search_strategy_used": "enhanced_fallback" if len(files_found) > 0 else "no_results"
        }

    def _read_file(self, args):
        """Read file with optional focus area"""
        file_id = args.get("file_id")
        focus_area = args.get("focus_area")
        
        response = requests.post(f"{self.http_server_url}/call_tool", json={
            "name": "read_gdrive_file",
            "arguments": {"file_id": file_id}
        })
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                file_data = result.get("data", {})
                content_info = file_data.get("content", {})
                
                # Extract content intelligently
                extracted = self._extract_focused_content(content_info, focus_area)
                
                return {
                    "success": True,
                    "file_name": file_data.get("file_name", "unknown"),
                    "file_type": content_info.get("type", "unknown"),
                    "content": extracted["content"],
                    "metadata": extracted["metadata"],
                    "focus_applied": focus_area is not None
                }
        
        return {"success": False, "error": "Failed to read file"}
    
    def _extract_focused_content(self, content_info, focus_area):
        """Extract content with optional focus area"""
        file_type = content_info.get("type", "unknown")
        
        if file_type in ["csv", "excel"]:
            data = content_info.get("data", [])
            columns = content_info.get("columns", [])
            
            if focus_area:
                # Filter data based on focus area
                focused_data = []
                focus_lower = focus_area.lower()
                
                for record in data:
                    # Check if any field relates to focus area
                    record_text = " ".join(str(v) for v in record.values() if v).lower()
                    if focus_lower in record_text:
                        focused_data.append(record)
                
                # If focus yields results, use it; otherwise use all data
                relevant_data = focused_data if focused_data else data
            else:
                relevant_data = data
            
            content = f"Dataset: {len(relevant_data)} relevant records\n"
            content += f"Columns: {', '.join(columns)}\n\n"
            
            for i, record in enumerate(relevant_data[:50], 1):  # Show up to 50 records
                record_parts = []
                for k, v in record.items():
                    if v is not None and str(v).strip():
                        record_parts.append(f"{k}: {v}")
                content += f"Record {i}: {', '.join(record_parts)}\n"
            
            if len(relevant_data) > 50:
                content += f"\n[{len(relevant_data) - 50} additional records available]\n"
            
            return {
                "content": content,
                "metadata": {
                    "total_records": len(data),
                    "focused_records": len(relevant_data),
                    "columns": columns,
                    "focus_applied": focus_area is not None
                }
            }
        
        elif file_type == "pdf":
            full_content = content_info.get("content", "")
            
            if focus_area:
                # Extract sections related to focus area
                lines = full_content.split('\n')
                focused_lines = []
                context_lines = 2  # Lines before/after match
                
                for i, line in enumerate(lines):
                    if focus_area.lower() in line.lower():
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        focused_lines.extend(lines[start:end])
                        focused_lines.append("---")
                
                if focused_lines:
                    content = f"Content focused on '{focus_area}':\n\n" + "\n".join(focused_lines)
                else:
                    content = full_content[:5000]  # Fallback to beginning
            else:
                content = full_content
            
            return {
                "content": content,
                "metadata": {
                    "total_length": len(full_content),
                    "pages": content_info.get("num_pages", 0),
                    "focus_applied": focus_area is not None
                }
            }
        
        else:
            # Handle other file types
            full_content = str(content_info.get("content", ""))
            return {
                "content": full_content[:5000],  # Reasonable limit
                "metadata": {
                    "total_length": len(full_content),
                    "focus_applied": False
                }
            }
    
    def _get_file_metadata(self, args):
        """Get file metadata without reading content"""
        file_pattern = args.get("file_pattern", "")
        
        response = requests.post(f"{self.http_server_url}/call_tool", json={
            "name": "search_gdrive_files", 
            "arguments": {"query": file_pattern, "max_results": 100}
        })
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                files = result.get("data", [])
                
                metadata = {
                    "total_files": len(files),
                    "file_types": {},
                    "files_by_type": {},
                    "files": []
                }
                
                for file in files:
                    mime_type = file.get("mimeType", "unknown")
                    file_type = self._mime_to_readable_type(mime_type)
                    
                    metadata["file_types"][file_type] = metadata["file_types"].get(file_type, 0) + 1
                    
                    if file_type not in metadata["files_by_type"]:
                        metadata["files_by_type"][file_type] = []
                    
                    metadata["files_by_type"][file_type].append({
                        "id": file["id"],
                        "name": file["name"],
                        "size": file.get("size", "unknown")
                    })
                    
                    metadata["files"].append({
                        "id": file["id"],
                        "name": file["name"],
                        "type": file_type
                    })
                
                return {"success": True, "metadata": metadata}
        
        return {"success": False, "error": "Failed to get metadata"}
    
    def _mime_to_readable_type(self, mime_type):
        """Convert MIME type to readable format"""
        mapping = {
            "text/csv": "CSV",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel",
            "application/pdf": "PDF",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PowerPoint",
            "text/plain": "Text",
            "application/json": "JSON",
            "text/xml": "XML"
        }
        return mapping.get(mime_type, "Other")
            
    def _recall_session_context(self, args):
        """Recall session context for conversational continuity"""
        context_type = args.get("context_type", "full_summary")
        
        if context_type == "files_mentioned":
            return {
                "success": True,
                "files_mentioned": self.session_context["files_mentioned"],
                "total_files": len(self.session_context["files_mentioned"])
            }
        elif context_type == "topics_discussed":
            return {
                "success": True,
                "topics": self.session_context["topics_discussed"],
                "total_topics": len(self.session_context["topics_discussed"])
            }
        elif context_type == "full_summary":
            return {
                "success": True,
                "session_summary": self.session_context["session_summary"],
                "files_count": len(self.session_context["files_mentioned"]),
                "topics_count": len(self.session_context["topics_discussed"]),
                "files_mentioned": list(self.session_context["files_mentioned"].keys()),
                "topics_discussed": self.session_context["topics_discussed"]
            }
        
        return {"success": False, "error": "Invalid context type"}

    def process_query_iteratively(self, user_query):
        """Process query using iterative tool calling with session memory"""
        
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # Build context-aware system message
        context_info = ""
        if self.session_context["files_mentioned"]:
            context_info += f"\nPREVIOUS FILES ACCESSED THIS SESSION: {list(self.session_context['files_mentioned'].values())}"
        
        if self.session_context["topics_discussed"]:
            context_info += f"\nTOPICS DISCUSSED THIS SESSION: {', '.join(self.session_context['topics_discussed'])}"
        
        if self.session_context["session_summary"]:
            context_info += f"\nSESSION SUMMARY: {self.session_context['session_summary']}"
        
        # Enhanced system message with memory
        system_message = f"""You are an intelligent document analyst with access to Google Drive tools and session memory.

Available tools:
{json.dumps(self.available_tools, indent=2)}

IMPORTANT SEARCH STRATEGY:
- The search_files tool now uses enhanced fallback strategies
- If a specific project name doesn't yield results, it will automatically try broader searches
- It breaks down multi-word queries and tries individual terms
- It searches for related terms like "project", "incident", "report" if specific names fail
- Always start with specific searches, the tool will handle fallbacks automatically

SESSION CONTEXT:{context_info}

When searching for projects or specific topics:
1. Try the specific name first - the search tool will handle broader searches automatically
2. If you suspect the user is referring to something discussed earlier, use recall_session_context
3. Look for patterns in file names and content to identify relevant information
4. Use focus_area parameter when reading large files to target specific information

User query: {user_query}

Approach this strategically, using your session memory and the enhanced search capabilities."""

        max_iterations = 10
        iteration = 0
        files_accessed_this_query = []
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- ITERATION {iteration} ---")
            
            try:
                response = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0.1,
                    system=system_message,
                    tools=self.available_tools,
                    messages=self.conversation_history
                )
                
                # Add Claude's response to conversation (including any tool_use blocks)
                assistant_message = {"role": "assistant", "content": []}
                
                for content_block in response.content:
                    if content_block.type == "text":
                        assistant_message["content"].append({
                            "type": "text",
                            "text": content_block.text
                        })
                    elif content_block.type == "tool_use":
                        assistant_message["content"].append({
                            "type": "tool_use",
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })
                
                self.conversation_history.append(assistant_message)
                
                # Check if Claude wants to use tools
                if response.stop_reason == "tool_use":
                    # Execute tool calls and prepare tool results
                    tool_results = []
                    
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_args = content_block.input
                            tool_use_id = content_block.id
                            
                            # Execute the tool
                            result = self.execute_tool(tool_name, tool_args)
                            
                            # Track files accessed for session context
                            if tool_name == "read_file" and "file_id" in tool_args:
                                files_accessed_this_query.append({
                                    "id": tool_args["file_id"],
                                    "context": f"Read via {tool_name}"
                                })
                            elif tool_name == "search_files" and result.get("success") and result.get("files"):
                                for file_info in result["files"][:3]:  # Track first 3 found files
                                    files_accessed_this_query.append({
                                        "id": file_info["id"],
                                        "name": file_info["name"],
                                        "context": f"Found via search: {tool_args.get('query', '')}"
                                    })
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(result, indent=2)
                            })
                    
                    # Add tool results as a single user message
                    if tool_results:
                        self.conversation_history.append({
                            "role": "user", 
                            "content": tool_results
                        })
                    
                    # Continue the conversation
                    continue
                
                else:
                    # Claude is done, update session context and return the final response
                    query_topics = self.extract_topics_from_query(user_query)
                    self.update_session_context(user_query, files_accessed_this_query, query_topics)
                    
                    final_response = ""
                    for content_block in response.content:
                        if content_block.type == "text":
                            final_response += content_block.text
                    
                    # Add iteration metadata with session info
                    metadata = f"\n\n---\nIterative Analysis Complete: {iteration} iterations, multiple tool calls used."
                    if files_accessed_this_query:
                        metadata += f" Files accessed this query: {len(files_accessed_this_query)}"
                    metadata += f" Session total: {len(self.session_context['files_mentioned'])} files, {len(self.session_context['topics_discussed'])} topics discussed."
                    
                    return final_response + metadata
            
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                return f"Error during iterative analysis: {str(e)}"
        
        return "Analysis completed but reached maximum iterations. Results may be incomplete."
