import re
import json
import requests
from typing import List, Dict, Any
import anthropic
import hashlib
import time
import random
import openai
import os
from enum import Enum
import openai

class APIProvider(Enum):
    CLAUDE = "claude"
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"

class RobustAPIClient:
    def __init__(self, claude_client, openai_client, available_tools):
        self.claude_client = claude_client
        self.openai_client = openai_client
        self.available_tools = available_tools
        self.openai_functions = self._convert_claude_tools_to_openai()
        self.response_cache = {}
        self.provider_status = {
            APIProvider.CLAUDE: {"available": True, "last_error": None},
            APIProvider.GPT4: {"available": True, "last_error": None},
            APIProvider.GPT35: {"available": True, "last_error": None}
        }

    def _convert_claude_tools_to_openai(self):
        """Convert Claude tools to OpenAI function format"""
        functions = []
        for tool in self.available_tools:
            function = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
            functions.append(function)
        return functions

    def call_with_multi_fallback(self, **kwargs):
        """Try Claude first, then GPT with tool calling"""
        providers = [APIProvider.CLAUDE, APIProvider.GPT4, APIProvider.GPT35]

        # Check cache
        cache_key = hashlib.md5(str(kwargs).encode()).hexdigest()
        if cache_key in self.response_cache:
            print("Using cached response")
            return self.response_cache[cache_key]

        for provider in providers:
            if not self.provider_status[provider]["available"]:
                continue

            try:
                print(f"Trying {provider.value}...")

                if provider == APIProvider.CLAUDE:
                    response = self._call_claude(**kwargs)
                else:
                    model = "gpt-4-turbo" if provider == APIProvider.GPT4 else "gpt-3.5-turbo"
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'model'}
                    response = self._gpt_function_call(model, **filtered_kwargs)

                # Success - cache and return
                self.response_cache[cache_key] = response
                self.provider_status[provider]["available"] = True
                return response

            except Exception as e:
                error_code = str(e)
                print(f"{provider.value} failed: {error_code}")

                if "429" in error_code or "529" in error_code:
                    self.provider_status[provider]["available"] = False
                    self._schedule_provider_reset(provider, 300)  # 5 min cooldown
                continue

        # All failed - return fallback
        return self._create_fallback_response(kwargs.get('messages', []))

    def _call_claude(self, **kwargs):
        """Claude with retry"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return self.claude_client.messages.create(**kwargs)
            except Exception as e:
                if ("429" in str(e) or "529" in str(e)) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise e


    def _gpt_function_call(self, model, **kwargs):
        """GPT with function calling - aggressive version"""
        messages = []

        # Add aggressive system prompt addition
        if kwargs.get('system'):
            aggressive_addon = "\n\nIMPORTANT: Be extremely thorough. If one file doesn't have information, immediately try 2-3 more files. Never give up easily."
            kwargs['system'] = kwargs['system'] + aggressive_addon
            messages.append({"role": "system", "content": kwargs['system']})

        for msg in kwargs.get('messages', []):
            content = self._convert_claude_message(msg.get('content', ''))
            messages.append({"role": msg['role'], "content": content})

        # Increase max_tokens for more detailed responses
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            functions=self.openai_functions if self.openai_functions else None,
            function_call="auto" if self.openai_functions else None,
            max_tokens=kwargs.get('max_tokens', 6000),  # Increased from 4000
            temperature=0.2  # Slightly higher for more creative file searching
        )

        return self._convert_gpt_to_claude_format(response)


    def _call_gpt_with_tools(self, model, **kwargs):
        """GPT with function calling"""
        messages = []
        if kwargs.get('system'):
           messages.append({"role": "system", "content": kwargs['system']})

        for msg in kwargs.get('messages', []):
            content = self._convert_claude_message(msg.get('content', ''))
            messages.append({"role": msg['role'], "content": content})

        gpt_kwargs = {k: v for k, v in kwargs.items() if k != 'model'}

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            functions=self.openai_functions if self.openai_functions else None,
            function_call="auto" if self.openai_functions else None,
            max_tokens=gpt_kwargs.get('max_tokens', 4000),
            temperature=gpt_kwargs.get('temperature', 0.1)
        )

        return self._convert_gpt_to_claude_format(response)

    def _convert_claude_message(self, content):
        """Convert Claude structured content to text"""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get('type') == 'text':
                        parts.append(part['text'])
                    elif part.get('type') == 'tool_result':
                        parts.append(f"Tool result: {part.get('content', '')}")
            return "\n".join(parts)
        return str(content)

    def _convert_gpt_to_claude_format(self, gpt_response):
        """Convert GPT response to Claude-compatible format"""
        choice = gpt_response.choices[0]

        class GPTResponse:
            def __init__(self, gpt_choice):
                self.content = []

                if gpt_choice.message.content:
                    self.content.append(GPTContent(gpt_choice.message.content))

                if gpt_choice.message.function_call:
                    self.content.append(GPTToolUse(gpt_choice.message.function_call))
                    self.stop_reason = "tool_use"
                else:
                    self.stop_reason = "end_turn"

        class GPTContent:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        class GPTToolUse:
            def __init__(self, function_call):
                self.type = "tool_use"
                self.id = f"gpt_{int(time.time() * 1000)}"
                self.name = function_call.name
                self.input = json.loads(function_call.arguments)

        return GPTResponse(choice)

    def _create_fallback_response(self, messages):
        """Emergency fallback when all APIs fail"""
        class FallbackResponse:
            def __init__(self, text):
                self.content = [FallbackContent(text)]
                self.stop_reason = "end_turn"

        class FallbackContent:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        return FallbackResponse("I'm experiencing technical difficulties with AI services. Please try again shortly.")

    def _schedule_provider_reset(self, provider, delay):
        """Re-enable provider after delay"""
        import threading
        def reset():
            time.sleep(delay)
            self.provider_status[provider]["available"] = True
            print(f"{provider.value} re-enabled")
        threading.Thread(target=reset, daemon=True).start()




# Global cache for Claude responses
claude_response_cache = {}

# Initialize OpenAI client for fallback
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

def call_claude_with_retry_and_fallback(claude_client, **kwargs):
    """Call Claude API with retry logic and GPT fallback"""
    max_retries = 3
    base_delay = 1

    # Try caching first
    cache_key = hashlib.md5(str(kwargs).encode()).hexdigest()
    if cache_key in claude_response_cache:
        print("Using cached Claude response")
        return claude_response_cache[cache_key]

    # Try Claude with retries
    for attempt in range(max_retries):
        try:
            response = claude_client.messages.create(**kwargs)
            # Cache successful response
            claude_response_cache[cache_key] = response
            return response

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit, retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            elif "429" in str(e) and attempt == max_retries - 1:
                # Final attempt failed - try GPT fallback
                print("Claude rate limited, falling back to GPT-4...")
                return call_gpt_fallback(claude_client, **kwargs)

            else:
                raise e

    # If all retries failed, try GPT
    return call_gpt_fallback(claude_client, **kwargs)


def call_gpt_fallback(claude_client, **kwargs):  # Added claude_client parameter
    """Fallback to GPT-4 when Claude fails"""
    if not openai_client:
        raise Exception("OpenAI API not configured and Claude failed")

    try:
        # Convert Claude format to OpenAI format
        messages = []
        if kwargs.get('system'):
            messages.append({"role": "system", "content": kwargs['system']})

        # Convert Claude messages to OpenAI format
        for msg in kwargs.get('messages', []):
            if isinstance(msg.get('content'), list):
                # Handle tool results - convert to text
                content_parts = []
                for part in msg['content']:
                    if isinstance(part, dict):
                        if part.get('type') == 'text':
                            content_parts.append(part['text'])
                        elif part.get('type') == 'tool_result':
                            content_parts.append(f"Tool result: {part.get('content', '')}")
                content = "\n".join(content_parts)
            else:
                content = msg.get('content', '')

            messages.append({
                "role": msg['role'],
                "content": content
            })

        # Call GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=kwargs.get('max_tokens', 4000),
            temperature=kwargs.get('temperature', 0.1)
        )

        # Convert back to Claude-like response format
        class MockResponse:
            def __init__(self, gpt_response):
                self.content = [MockContent(gpt_response.choices[0].message.content)]
                self.stop_reason = "end_turn"

        class MockContent:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        return MockResponse(response)

    except Exception as e:
        print(f"GPT fallback also failed: {e}")
        raise Exception("Both Claude and GPT APIs failed")




class ClaudeLikeDocumentProcessor:
    def __init__(self, http_server_url, claude_client):
        self.http_server_url = http_server_url
        self.claude_client = claude_client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        self.last_api_call = 0
        self.min_delay = 1.0
        self.conversation_history = []
        self.session_context = {
            "files_mentioned": {},  # Track files that have been discussed
            "topics_discussed": [],  # Track conversation topics
            "user_preferences": {},  # Learn user patterns
            "session_summary": "",   # Running summary of conversation - ADD COMMA HERE
            "extracted_data_by_query": {},  # COLON not equals
            "visualization_ready_data": {},  # COLON not equals, fix spelling
            "conversation_flow": [],        # COLON not equals
            "last_query_data": None         # COLON not equals
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
        self.api_client = RobustAPIClient(claude_client, self.openai_client, self.available_tools)

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

    # Add these methods to your ClaudeLikeDocumentProcessor class
    def _extract_numerical_data_from_response(self, response_text):
        """Universal extraction that works with both Claude and GPT formats"""
        import re

        viz_data = {}

        # Enhanced patterns that catch budget formats
        patterns = {
            # Table format - now handles millions: | Project Alpha | $2.5M |
            'table_format': r'\|\s*([A-Za-z][A-Za-z\s\-\.\']{2,40}?)\s*\|\s*\$?(\d+(?:\.\d+)?[MK]?|\d+(?:,\d{3})*(?:\.\d+)?)',

            # List format with millions: "1. Project Alpha: $2.5M"
            'numbered_list': r'(?:\d+\.|\•|\-)\s*([A-Za-z][A-Za-z\s\-\.]{2,40}?):\s*\$?(\d+(?:\.\d+)?[MK]?|\d+(?:,\d{3})*(?:\.\d+)?)',

            # Simple colon with millions: "Project Alpha: $2.5M"
            'colon_simple': r'^([A-Za-z][A-Za-z\s\-\.]{2,40}?):\s*\$?(\d+(?:\.\d+)?[MK]?|\d+(?:,\d{3})*(?:\.\d+)?)$',

            # Budget-specific pattern: "Budget: $2.5 million" or "Budget: $2,500,000"
            'budget_format': r'([A-Za-z][A-Za-z\s\-\.]{2,40}?)\s+(?:budget|cost):\s*\$?(\d+(?:\.\d+)?)\s*(?:million|M)?',

            # Million format: "Alpha Initiative $2.5M Budget"
            'million_format': r'([A-Za-z][A-Za-z\s\-\.]{2,40}?)\s+\$?(\d+(?:\.\d+)?)[M]\s*(?:budget|cost)?',

            # Existing patterns...
            'efficiency_format': r'([A-Za-z][A-Za-z\s\-\.]{2,40}?)\s*\(.*?(\d+(?:\.\d+)?)%.*?\)',
            'rating_format': r'([A-Za-z][A-Za-z\s\-\.]{2,40}?)\s+(\d+\.\d+)(?:/5)?',
        }

        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)

            if matches:
                print(f"Pattern '{pattern_name}' found {len(matches)} matches")
                processed_data = {}

                for match in matches:
                    try:
                        name, value_str = match
                        name = name.strip()

                        # Handle millions and thousands
                        if value_str.endswith('M'):
                            value = float(value_str[:-1]) * 1000000
                        elif value_str.endswith('K'):
                            value = float(value_str[:-1]) * 1000
                        else:
                            value = float(value_str.replace(',', '').replace('%', '').replace('$', ''))

                        # Skip obvious headers and metadata
                        if not any(skip in name.lower() for skip in ['name', 'employee', 'total', 'average']):
                            if len(name) > 2 and value > 0:
                                processed_data[name] = value

                    except (ValueError, IndexError):
                        continue

                if processed_data and len(processed_data) >= 2:
                    viz_data[pattern_name] = processed_data
                    print(f"Stored {pattern_name}: {len(processed_data)} items - Sample: {list(processed_data.items())[:2]}")

        return viz_data if viz_data else None

    def _clean_extraction_name(self, name):
        """Clean names from table extractions"""
        # Remove markdown and table formatting
        name = re.sub(r'\*+', '', name)
        name = re.sub(r'[|_\-=]{2,}', '', name)
        name = name.strip()

        # Remove common table headers/metadata
        skip_patterns = [
            r'employee\s*name', r'project\s*name', r'name', r'score', r'rating', 
            r'productivity', r'efficiency', r'budget', r'cost'
        ]

        for pattern in skip_patterns:
            if re.match(pattern, name.lower()):
                return None

        return name

    def _is_meaningful_data(self, name, value, pattern_type):
        """Validate if this is actual business data"""
        if not name or len(name) < 3 or len(name) > 30:
            return False

        # Check for obvious metadata
        metadata_terms = ['scale', 'total', 'range', 'from', 'to', 'chart', 'table']
        if any(term in name.lower() for term in metadata_terms):
            return False

        # Value range validation by data type
        if 'employee' in pattern_type.lower():
            return 0 <= value <= 5 or 0 <= value <= 100  # Ratings or percentages
        elif 'project' in pattern_type.lower():
            return value >= 100000  # Reasonable project budget minimum

        return value > 0

    
    def _clean_name(self, name):
        """Clean up extracted names and separate status information"""
        import re

        # Remove newlines and clean up
        name = name.replace('\n', ' ').strip()

        # Remove status words from project names
        status_words = ['completed', 'cancelled', 'in progress', 'on hold', 'active']
        for status in status_words:
            name = re.sub(rf'\b{status}\b', '', name, flags=re.IGNORECASE)

        # Clean up formatting
        name = name.strip().strip('*').strip('"').strip("'")
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
        name = name.replace('**', '').replace('|', '')

        return name.strip()


    def _extract_project_with_status(self, text_match):
        """Extract project name and status from mixed text"""
        import re

        statuses = {
            'completed': 'Completed',
            'cancelled': 'Cancelled', 
            'in progress': 'In Progress',
            'on hold': 'On Hold',
            'active': 'Active'
        }

        # Find status in the text
        found_status = 'Unknown'
        clean_name = text_match

        for status_key, status_value in statuses.items():
            if status_key in text_match.lower():
                found_status = status_value
                # Remove status from name
                clean_name = re.sub(rf'\b{status_key}\b', '', text_match, flags=re.IGNORECASE)
                break

        clean_name = self._clean_name(clean_name)

        return clean_name, found_status
    
    def _parse_value(self, value_str):
        """Parse various value formats"""
        value_str = value_str.replace(',', '').replace('$', '')

        if '%' in value_str:
            return float(value_str.replace('%', ''))
        elif 'M' in value_str or 'million' in value_str.lower():
            return float(value_str.replace('M', '').replace('million', '')) * 1000000
        elif 'K' in value_str or 'thousand' in value_str.lower():
            return float(value_str.replace('K', '').replace('thousand', '')) * 1000
        else:
            return float(value_str)

    def _is_valid_data_point(self, name, value):
        """Check if this is valid visualization data"""
        # Skip metadata and system messages
        skip_terms = [
            'iterative', 'session', 'analysis', 'files accessed', 'total found',
            'search strategy', 'claude', 'gpt', 'api', 'tool', 'iteration'
        ]

        if any(term in name.lower() for term in skip_terms):
            return False

        # Must have reasonable name length and value
        if len(name) < 3 or len(name) > 50:
            return False

        # Value should be reasonable (not 0 or 1 for most business data)
        if isinstance(value, (int, float)) and (value < 0 or value > 100000000):
            return False

        return True

    def _categorize_data(self, data, response_context):
        """Automatically categorize data based on content"""
        data_keys = ' '.join(data.keys()).lower()

        # Employee/People metrics
        if any(term in response_context for term in ['employee', 'staff', 'productivity', 'performance', 'availability']):
            if any(term in data_keys for term in ['score', 'rating', '%', 'productivity']):
                return 'employee_metrics'
            else:
                return 'employee_data'

        # Incident/Issue tracking
        elif any(term in response_context for term in ['incident', 'issue', 'bug', 'ticket', 'problem']):
            if any(term in data_keys for term in ['open', 'closed', 'pending', 'critical', 'high', 'low']):
                return 'incident_status'
            else:
                return 'incident_metrics'

        # Project data
        elif any(term in response_context for term in ['project', 'initiative', 'budget', 'timeline']):
            if any(term in data_keys for term in ['budget', 'cost', '$']):
                return 'project_budgets'
            else:
                return 'project_data'

        # Financial data
        elif any(term in data_keys for term in ['$', 'budget', 'cost', 'revenue', 'expense']):
            return 'financial_data'

        # Performance metrics (percentages)
        elif any(str(v) for v in data.values() if isinstance(v, (int, float)) and 0 < v <= 100):
            return 'performance_metrics'

        # Default categorization
        else:
            return 'general_data'



    
    
    def _extract_and_store_response_data(self, user_query, final_response):
        """Enhanced data storage with proper type separation"""

        print(f"🔍 EXTRACTING DATA from response length: {len(final_response)}")

        # Determine current data type
        query_lower = user_query.lower()
        current_data_type = 'general'

        if any(term in query_lower for term in ['employee', 'productivity', 'efficiency', 'staff', 'worker']):
            current_data_type = 'employee'
        elif any(term in query_lower for term in ['project', 'budget', 'cost', 'initiative']):
            current_data_type = 'project'
        elif any(term in query_lower for term in ['incident', 'issue', 'bug', 'ticket']):
            current_data_type = 'incident'

        print(f"📊 Current data type: {current_data_type}")

        # Check if we're switching data types
        last_data_type = getattr(self, '_last_data_type', None)
        if last_data_type and last_data_type != current_data_type:
            print(f"🔄 Switching from {last_data_type} to {current_data_type} - clearing old data")
            # Clear old visualization data when switching types
            self.session_context["visualization_ready_data"] = {}
            self.session_context["last_query_data"] = None

        # Store current data type
        self._last_data_type = current_data_type

        # Extract new data
        viz_data = self._extract_numerical_data_from_response(final_response)

        if viz_data:
            # Store with type prefix to avoid conflicts
            typed_key = f"{current_data_type}_{len(self.session_context.get('extracted_data_by_query', {}))}"
            self.session_context["extracted_data_by_query"][typed_key] = viz_data

            # Update visualization-ready data with new data only
            for pattern_name, data_dict in viz_data.items():
                if isinstance(data_dict, dict):
                    clean = {k: v for k, v in data_dict.items()
                             if not any(meta in k.lower() for meta in
                                        ('iterative', 'files accessed', 'session total',
                                         'analysis complete', 'tool call', 'iteration'))}
                    self.session_context["visualization_ready_data"].update(clean)

            self.session_context["last_query_data"] = viz_data

            print(f"📊 Stored {current_data_type} data: {list(viz_data.keys())}")
            print(f"📊 Updated viz_ready_data: {list(self.session_context['visualization_ready_data'].keys())}")
        else:
            print("❌ No visualization data extracted")

        # Store conversation pair for context
        self.session_context["conversation_flow"].append({
            "query": user_query,
            "response_preview": final_response[:200] + "...",
            "data_extracted": bool(viz_data),
            "data_keys": list(viz_data.keys()) if viz_data else []
        })

        print(f"💾 Total stored queries with data: {len([q for q in self.session_context['extracted_data_by_query'].values() if q])}")

    def _resolve_contextual_references(self, query):
        """Resolve 'them', 'it', 'those' references to previous data"""

        query_lower = query.lower()
        pronouns = ['them', 'those', 'it', 'this', 'these']

        if any(pronoun in query_lower for pronoun in pronouns):
            if self.session_context.get("last_query_data"):
                # Build context about what the pronouns refer to
                last_data = self.session_context["last_query_data"]
                data_description = []

                for key, data in last_data.items():
                    if isinstance(data, dict):
                        data_description.append(f"{key}: {list(data.keys())}")

                context_note = f"\n\nCONTEXT: The user is referring to data from the previous query: {', '.join(data_description)}"
                return query + context_note

        return query
        
    def _initialize_document_awareness(self):
        """Proactively discover and catalog available documents on first session"""
        print("🔍 Initializing document discovery...")

        # Define broad search terms for different document types
        discovery_terms = {
            "projects": ["project", "initiative", "development"],
            "incidents": ["incident", "issue", "problem", "bug", "error"],
            "reports": ["report", "analysis", "summary", "findings"],
            "plans": ["plan", "strategy", "roadmap", "timeline"],
            "budgets": ["budget", "cost", "financial", "expense"],
            "operations": ["operations", "operational", "process"],
            "data": ["data", "dataset", "metrics", "statistics"]
        }

        discovered_files = {}
        total_files_found = 0

        for category, terms in discovery_terms.items():
            print(f"📁 Discovering {category} documents...")
            category_files = self._discover_files_by_category(terms)
            discovered_files[category] = category_files
            total_files_found += len(category_files)
            print(f"   Found {len(category_files)} {category} files")

        # Update session context with discovered files
        self.session_context["document_catalog"] = discovered_files
        self.session_context["total_documents_available"] = total_files_found

        # Create a summary of available documents
        catalog_summary = self._create_document_catalog_summary(discovered_files)
        self.session_context["session_summary"] = catalog_summary

        print(f"✅ Document discovery complete: {total_files_found} total files cataloged")
        return discovered_files

    def _discover_files_by_category(self, search_terms, max_files_per_term=15):
        """Discover files for a specific category using multiple search terms"""
        category_files = {}
        seen_file_ids = set()

        for term in search_terms:
            try:
                result = self.execute_tool("search_files", {
                    "query": term,
                    "max_results": max_files_per_term
                })

                if result.get("success") and result.get("files"):
                    for file_info in result["files"]:
                        file_id = file_info["id"]
                        if file_id not in seen_file_ids:
                            category_files[file_id] = {
                                "name": file_info["name"],
                                "type": file_info.get("type", "unknown"),
                                "size": file_info.get("size", "unknown"),
                                "found_via_term": term,
                                "relevance_score": self._calculate_relevance_score(
                                    file_info["name"], search_terms
                                )
                            }
                            seen_file_ids.add(file_id)

            except Exception as e:
                print(f"⚠️  Error searching for '{term}': {e}")
                continue

        # Sort by relevance score
        sorted_files = dict(sorted(
            category_files.items(),
            key=lambda x: x[1]["relevance_score"],
            reverse=True
        ))

        return sorted_files

    def _calculate_relevance_score(self, filename, search_terms):
        """Calculate relevance score based on filename match with search terms"""
        score = 0
        filename_lower = filename.lower()

        for term in search_terms:
            if term in filename_lower:
                score += 10
            # Partial matches
            for word in term.split():
                if word in filename_lower:
                    score += 5

        # Bonus for common file indicators
        indicators = ["report", "analysis", "data", "summary", "plan"]
        for indicator in indicators:
            if indicator in filename_lower:
                score += 3

        return score

    def _create_document_catalog_summary(self, discovered_files):
        """Create a comprehensive summary of discovered documents"""
        summary_parts = ["📚 DOCUMENT CATALOG SUMMARY:\n"]

        for category, files in discovered_files.items():
            if files:
                summary_parts.append(f"**{category.upper()}** ({len(files)} files):")

                # Show top 5 most relevant files per category
                top_files = list(files.items())[:5]
                for file_id, file_info in top_files:
                    summary_parts.append(
                        f"  • {file_info['name']} (score: {file_info['relevance_score']})"
                    )

                if len(files) > 5:
                    summary_parts.append(f"  ... and {len(files) - 5} more files")
                summary_parts.append("")

        total_files = sum(len(files) for files in discovered_files.values())
        summary_parts.append(f"📊 Total: {total_files} documents across {len(discovered_files)} categories")

        return "\n".join(summary_parts)

    def _get_relevant_files_for_query(self, user_query):
        """Get files relevant to the user's query from the document catalog"""
        if "document_catalog" not in self.session_context:
            return []

        query_lower = user_query.lower()
        relevant_files = []

        # Score files based on query relevance
        for category, files in self.session_context["document_catalog"].items():
            category_relevance = 0

            # Check if query relates to this category
            category_keywords = {
                "projects": ["project", "initiative", "development"],
                "incidents": ["incident", "issue", "problem", "bug", "error"],
                "reports": ["report", "analysis", "summary"],
                "plans": ["plan", "strategy", "timeline"],
                "budgets": ["budget", "cost", "financial"],
                "operations": ["operations", "operational"],
                "data": ["data", "dataset", "metrics"]
            }

            if category in category_keywords:
                for keyword in category_keywords[category]:
                    if keyword in query_lower:
                        category_relevance += 10

            # If category is relevant, add its files
            if category_relevance > 0:
                for file_id, file_info in files.items():
                    file_query_score = 0
                    filename_lower = file_info["name"].lower()

                    # Score based on filename matching query terms
                    query_words = query_lower.split()
                    for word in query_words:
                        if len(word) > 3:  # Skip short words
                            if word in filename_lower:
                                file_query_score += 15

                    total_score = file_info["relevance_score"] + file_query_score + category_relevance

                    if total_score > 10:  # Threshold for relevance
                        relevant_files.append({
                            "file_id": file_id,
                            "name": file_info["name"],
                            "category": category,
                            "total_score": total_score
                        })

        # Sort by total score and return top 10
        relevant_files.sort(key=lambda x: x["total_score"], reverse=True)
        return relevant_files[:10]


    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], provider_used: str = "claude") -> Dict[str, Any]:
        """Execute tool with provider awareness"""

        print(f"TOOL CALL: {tool_name} via {provider_used} with args: {arguments}")

        try:
            if tool_name == "search_files":
                return self._search_files(arguments)
            elif tool_name == "read_file":
                return self._read_file(arguments)
                if provider_used.startswith("gpt") and result.get("success"):
                    result = self._enhance_with_gpt_analysis(result, arguments)
            elif tool_name == "get_file_metadata":
                return self._get_file_metadata(arguments)
            elif tool_name == "recall_session_context":
                return self._recall_session_context(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}

            result["executed_via"] = provider_used
            return result

        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}


    def _enhance_with_gpt_analysis(self, file_result, original_args):
        """Add GPT-specific analysis when GPT calls tools"""
        if not self.openai_client:
            return file_result

        try:
            content = file_result.get("content", "")[:2000]  # Limit for context

            analysis_prompt = f"""Analyze this document briefly:

            File: {file_result.get('file_name', 'Document')}
            Content: {content}

            Provide:
            1. Key topics (2-3 items)
            2. Important data/numbers
            3. Main insights
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=300,
                temperature=0.1
            )

            file_result["gpt_analysis"] = response.choices[0].message.content
            return file_result

        except Exception as e:
            print(f"GPT analysis failed: {e}")
            return file_result


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

    def _aggressive_file_search(self, query_intent, max_files=5):
        """Aggressively search and read multiple files for comprehensive information"""

        search_terms = []

        if "project" in query_intent.lower() and "member" in query_intent.lower():
            search_terms = ["project", "team", "member", "personnel", "staff"]
        elif "employee" in query_intent.lower():
            search_terms = ["employee", "staff", "personnel", "team", "people"]
        elif "budget" in query_intent.lower():
            search_terms = ["budget", "cost", "financial", "expense", "money"]

        all_file_data = []

        for term in search_terms:
            search_result = self.execute_tool("search_files", {"query": term, "max_results": 10})

            if search_result.get("success") and search_result.get("files"):
                for file_info in search_result["files"][:3]:  # Read top 3 files per search term
                    file_data = self.execute_tool("read_file", {"file_id": file_info["id"]})
                    if file_data.get("success"):
                        all_file_data.append({
                            "file_name": file_info["name"],
                            "content": file_data.get("content", ""),
                            "search_term": term
                        })

                    if len(all_file_data) >= max_files:
                        break

            if len(all_file_data) >= max_files:
                break

        return all_file_data


    def build_enhanced_system_message(self, user_query, context_info, relevant_files):
        """Build comprehensive system message combining both approaches"""

        # Document discovery status
        discovery_status = ""
        if self.session_context.get("total_documents_available"):
            discovery_status = f"DOCUMENT DISCOVERY: Complete - {self.session_context['total_documents_available']} documents cataloged"
        else:
            discovery_status = "DOCUMENT DISCOVERY: Not yet initialized"

        # Relevant files section
        relevant_files_info = ""
        if relevant_files:
            relevant_files_info = f"\nFILES MOST RELEVANT TO CURRENT QUERY:"
            for file in relevant_files[:5]:
                relevant_files_info += f"\n- {file['name']} (ID: {file['file_id']}) - Score: {file['total_score']}"

        system_message = f"""You are an aggressive, thorough document analyst with access to Google Drive tools. Your job is to find information, not give up easily.

MANDATORY BEHAVIOR:
- Always read multiple files when information isn't found in the first file
- When a file appears empty or lacks expected data, immediately try 2-3 more files
- Use different search terms if initial searches don't yield results
- Extract ANY relevant information from files, even if incomplete
- Never conclude "no information found" without reading at least 3 different files
- Be persistent and thorough - the user expects detailed answers

RESPONSE COMPLETION REQUIREMENTS:
- ALWAYS provide complete responses - never truncate tables, lists, or analysis
- If creating tables, include ALL data found, not just samples
- Finish all sentences and close all markdown formatting properly
- When showing project/employee data, include complete listings
- End responses with proper conclusions, not mid-sentence cutoffs

{discovery_status}

Available tools:
{json.dumps(self.available_tools, indent=2)}

AGGRESSIVE SEARCH PROTOCOL:
1. READ the highest-scoring relevant files listed below
2. If no useful data found, IMMEDIATELY search for alternative files
3. Try different file types (.csv, .pdf, .xlsx, .docx)
4. Use varied search terms: if searching "project members" fails, try "team", "personnel", "staff", "employees"
5. Extract partial information and combine from multiple sources
6. Never stop after 1-2 files - always try at least 3-4 files

{context_info}{relevant_files_info}

EXTRACTION RULES:
- Extract names, roles, numbers, dates from ANY file that contains them
- If a file has partial data, note it and search for complementary files
- Combine information from multiple sources to give complete answers
- When files seem empty, try reading them anyway - they might have hidden data

Current query: {user_query}

Be thorough, persistent, and extract maximum information. Don't accept "no data found" easily. Provide complete, untruncated responses."""

        return system_message

    def process_query_iteratively(self, user_query):
        """Process query using iterative tool calling with session memory"""

        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})

        enhanced_query = self._resolve_contextual_references(user_query)

        if "document_catalog" not in self.session_context or not self.session_context.get("document_catalog"):
            self._initialize_document_awareness()

    # Get files relevant to this specific query
        relevant_files = self._get_relevant_files_for_query(user_query)

        # Build context-aware system message
        context_info = ""
        if self.session_context["files_mentioned"]:
            context_info += f"\nPREVIOUS FILES ACCESSED THIS SESSION: {list(self.session_context['files_mentioned'].values())}"

        if self.session_context["topics_discussed"]:
            context_info += f"\nTOPICS DISCUSSED THIS SESSION: {', '.join(self.session_context['topics_discussed'])}"

        if self.session_context["session_summary"]:
            context_info += f"\nSESSION SUMMARY: {self.session_context['session_summary']}"

        if relevant_files:
            context_info += f"\n\nFILES MOST RELEVANT TO CURRENT QUERY:"
        for file in relevant_files[:5]:  # Show top 5 relevant files
            context_info += f"\n• {file['name']} (ID: {file['file_id']}) - Score: {file['total_score']}"

        # Enhanced system message with memory
        system_message = self.build_enhanced_system_message(user_query, context_info, relevant_files)


        max_iterations = 10
        iteration = 0
        files_accessed_this_query = []

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- ITERATION {iteration} ---")

            try:
                now = time.time()
                elapsed = now - self.last_api_call
                if elapsed < self.min_delay:
                    sleep_time = self.min_delay - elapsed
                    print(f"Rate limiting: waiting {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)

                response = self.api_client.call_with_multi_fallback(
                    model="claude-opus-4-20250514",
                    max_tokens=4000,
                    temperature=0.1,
                    system=system_message,
                    tools=self.available_tools,
                    messages=self.conversation_history
                )

                self.last_api_call = time.time()

# Add API response to conversation (handling both Claude and GPT responses)
                assistant_message = {"role": "assistant", "content": []}

                for content_block in response.content:
                    if hasattr(content_block, 'type') and content_block.type == "text":
                        assistant_message["content"].append({
                            "type": "text",
                            "text": content_block.text
                        })
                    elif hasattr(content_block, 'type') and content_block.type == "tool_use":
                        assistant_message["content"].append({
                            "type": "tool_use",
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })

                self.conversation_history.append(assistant_message)

                # Check if we need to use tools (Claude tool_use or GPT needs tools)
                if (hasattr(response, 'stop_reason') and response.stop_reason == "tool_use") or self._response_needs_tools(response):
                    # Execute tool calls with robust error handling
                    tool_results = []

                    for content_block in response.content:
                        if hasattr(content_block, 'type') and content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_args = content_block.input
                            tool_use_id = content_block.id

                            # Execute tool with error handling
                            try:
                                result = self.execute_tool(tool_name, tool_args)

                                # Track files accessed for session context
                                if tool_name == "read_file" and "file_id" in tool_args:
                                    files_accessed_this_query.append({
                                        "id": tool_args["file_id"],
                                        "context": f"Read via {tool_name}",
                                        "status": "success"
                                    })
                                elif tool_name == "search_files" and result.get("success") and result.get("files"):
                                    for file_info in result["files"][:3]:  # Track first 3 found files
                                        files_accessed_this_query.append({
                                            "id": file_info["id"],
                                            "name": file_info["name"],
                                            "context": f"Found via search: {tool_args.get('query', '')}",
                                            "status": "success"
                                        })

                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": json.dumps(result, indent=2)
                                })

                            except Exception as tool_error:
                                print(f"Tool execution error: {tool_error}")
                                # Provide graceful fallback for tool errors
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": json.dumps({
                                        "error": f"Tool execution failed: {str(tool_error)}",
                                        "fallback_message": "Unable to access this resource at the moment. Please try again later."
                                    }, indent=2)
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
                    # Analysis complete - update session context and return final response
                    query_topics = self.extract_topics_from_query(user_query)
                    self.update_session_context(user_query, files_accessed_this_query, query_topics)

                    final_response = ""
                    for content_block in response.content:
                        if hasattr(content_block, 'type') and content_block.type == "text":
                            final_response += content_block.text
                        elif hasattr(content_block, 'text'):  # Fallback for GPT responses
                            final_response += content_block.text

                    if len(final_response.strip()) > 100:
                        self._partial_response = final_response

                    self._extract_and_store_response_data(user_query, final_response)

                    # ADD FOLLOW-UP LOGIC HERE:
                    # Check if response needs more aggressive follow-up (for first few iterations)
                    if iteration <= 3 and len(final_response.strip()) < 100:
                        # Response seems too brief, push for more detail
                        if successful_files < 3:
                            follow_up_messages = [
                                "The previous response was too brief. Please search more files and provide more detailed information.",
                                "Try different file types (.csv, .xlsx, .pdf) and search terms to find comprehensive data.",
                                "Don't give up after one file - read at least 2-3 more files to provide complete information."
                            ]

                            follow_up = {
                                "role": "user",
                                "content": follow_up_messages[min(iteration-1, 2)]  # Different message each iteration
                            }
                            self.conversation_history.append(follow_up)
                            print(f"Adding follow-up push for more detailed analysis...")
                            continue  # Force another iteration

                    # Check for "no information found" type responses
                    if any(phrase in final_response.lower() for phrase in ["no information", "no relevant", "no data", "not found"]):
                        if iteration <= 2 and successful_files < 4:  # Only push back on early iterations
                            follow_up = {
                                "role": "user",
                                "content": "You said no information was found, but please try harder. Search with different terms and read more files. There should be data available in the document catalog."
                            }
                            self.conversation_history.append(follow_up)
                            print(f"Pushing back on 'no information found' response...")
                            continue  # Force another iteration

                    # Original metadata and return code continues here...
                    provider_status = self._get_provider_status() if hasattr(self, 'api_client') else "Standard API"
                    successful_files = len([f for f in files_accessed_this_query if f.get('status') == 'success'])

                    metadata = f"\n\n---\nIterative Analysis Complete: {iteration} iterations using {provider_status}."
                    if files_accessed_this_query:
                        metadata += f" Files accessed: {successful_files}/{len(files_accessed_this_query)} successful."
                    metadata += f" Session total: {len(self.session_context['files_mentioned'])} files, {len(self.session_context['topics_discussed'])} topics discussed."

                    return final_response + metadata


            except Exception as e:
                error_msg = str(e)
                print(f"Error in iteration {iteration}: {error_msg}")

                # Handle specific error types gracefully
                if "429" in error_msg:
                    return f"I'm currently experiencing high demand. Your analysis was partially completed ({iteration} iterations). Please try again in a few moments for a complete analysis."
                elif "529" in error_msg or "502" in error_msg or "503" in error_msg:
                    return f"I'm experiencing temporary service issues. Your analysis was partially completed ({iteration} iterations). Please try again shortly."
                else:
                    # Provide partial results if we got some analysis done
                    partial_response = f"Analysis encountered an error after {iteration} iterations: {error_msg}"
                    if files_accessed_this_query:
                        successful_files = len([f for f in files_accessed_this_query if f.get('status') == 'success'])
                        partial_response += f" However, I was able to access {successful_files} files before the error occurred."
                    return partial_response

        # Reached max iterations
        successful_files = len([f for f in files_accessed_this_query if f.get('status') == 'success'])
        if hasattr(self, '_partial_response') and self._partial_response:
            return self._partial_response + f"\n\n[Analysis completed after {max_iterations} iterations with {successful_files} files processed]"
        else:
            return f"Analysis completed with maximum iterations reached ({max_iterations}). Successfully processed {successful_files} files. Please try a more specific query or ask about a particular aspect of the data."

    def _response_needs_tools(self, response):
        """Check if GPT response indicates it needs tools (for GPT fallback compatibility)"""
        if not hasattr(response, 'content'):
            return False

        for content_block in response.content:
            if hasattr(content_block, 'text'):
                text = content_block.text.lower()
                # Look for indicators that GPT is trying to use tools
                tool_indicators = ["search for", "need to find", "let me look for", "i should check"]
                if any(indicator in text for indicator in tool_indicators):
                    return True
        return False

    def _get_provider_status(self):
        """Get current API provider status for metadata"""
        if hasattr(self, 'api_client') and hasattr(self.api_client, 'provider_status'):
            active_providers = []
            for provider, status in self.api_client.provider_status.items():
                if status["available"]:
                    active_providers.append(provider.value)
            return f"Multi-API ({', '.join(active_providers)} available)"
        return "Standard API"



def get_cached_claude_response(claude_client, system_message, conversation_history):
    """Get cached Claude response or make new API call"""
    # Create hash of the conversation for caching
    cache_key = hashlib.md5(
        f"{system_message}{json.dumps(conversation_history)}".encode()
    ).hexdigest()

    if cache_key in claude_response_cache:
        print("Using cached Claude response")
        return claude_response_cache[cache_key]

    # Make API call with retry logic
    response = call_claude_with_retry(
        claude_client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.1,
        system=system_message,
        messages=conversation_history
    )

    # Cache the response
    claude_response_cache[cache_key] = response

    return response
