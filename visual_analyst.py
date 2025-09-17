"""
Visualization Analyst Agent
Processes Claude responses and generates sophisticated visualizations with insights
"""

import re
import json
from typing import Dict, List, Any, Optional
import anthropic

class VisualizationAnalyst:
    def __init__(self, claude_client):
        self.claude_client = claude_client
        self.last_analysis_time = None
        self.analysis_count = 0
        
    def analyze_and_visualize(self, response_text: str, user_query: str, session_context: Dict = None) -> Dict[str, Any]:
        """
        Main method: Analyzes Claude's response and generates visualization specifications
        """
        import datetime
        self.last_analysis_time = datetime.datetime.now().isoformat()
        self.analysis_count += 1
        
        print(f"🔍 VisualizationAnalyst.analyze_and_visualize() called (#{self.analysis_count})")
        print(f"📊 Response length: {len(response_text)} chars")
        print(f"❓ User query: {user_query[:100]}...")
        
        # Step 1: Extract raw data from response
        extracted_data = self._extract_numerical_data(response_text)
        
        print(f"📈 Extracted data patterns: {list(extracted_data.keys())}")
        for pattern_name, data in extracted_data.items():
            if isinstance(data, dict) and data:
                print(f"   {pattern_name}: {len(data)} items - {list(data.keys())[:3]}...")
        
        # Step 2: Use Claude to intelligently analyze and structure the data
        if extracted_data:
            print("🤖 Generating intelligent visualizations with Claude...")
            viz_specs = self._generate_intelligent_visualizations(
                response_text, extracted_data, user_query, session_context
            )
            
            if viz_specs:
                print(f"✅ Generated {len(viz_specs.get('visualizations', []))} visualizations")
                return viz_specs
            else:
                print("⚠️ Claude analysis returned no visualizations, trying fallback...")
                return self._create_fallback_visualization(extracted_data, user_query)
        else:
            print("❌ No numerical data extracted from response")
        
        return None

    
    def _extract_numerical_data(self, response_text: str) -> Dict[str, Any]:
        """Extract ALL numerical data patterns from Claude's response"""
        data_patterns = {}
        
        # Enhanced pattern matching for different data types
        patterns = {
            # Project costs with names: "Project Alpha: $150,000"
            'project_costs': r'(?:project|initiative)\s+([^:\n]+):\s*\$?([\d,]+\.?\d*)',
            
            # Budget categories: "Personnel: $45,000" or "Technology budget: 30%"
            'budget_items': r'([A-Za-z\s]+(?:budget|cost|expense|personnel|technology|infrastructure|training)?):\s*\$?([\d,]+\.?\d*)(%?)',
            
            # Cost breakdowns: "Hardware costs $25,000, Software $15,000"
            'cost_breakdown': r'([A-Za-z\s]+)\s+(?:costs?\s*)?\$?([\d,]+\.?\d*)',
            
            # Quarterly/temporal data: "Q1 2024: $100,000"
            'temporal_data': r'(Q[1-4]\s*20\d{2}|20\d{2}[\-\s]Q[1-4]|[A-Z][a-z]+\s*20\d{2}):\s*\$?([\d,]+\.?\d*)',
            
            # Efficiency scores: "Team A efficiency: 4.2/5"
            'efficiency_scores': r'([^:\n]+)\s+(?:efficiency|score|rating):\s*([\d\.]+)(?:/5)?',
            
            # Percentages: "Availability: 95.5%"
            'percentages': r'([^:\n]+):\s*([\d\.]+)%',
            
            # Department metrics: "IT Department: 25 employees"
            'department_data': r'([A-Za-z\s]+(?:department|team|division)):\s*([\d,]+\.?\d*)',
            
            # General number patterns with context
            'contextual_numbers': r'([A-Za-z\s]{3,25}):\s*\$?([\d,]+\.?\d*)'
        }
        
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                processed_data = self._process_matches(matches, pattern_name)
                if processed_data:  # Only add if we got valid data
                    data_patterns[pattern_name] = processed_data
                    print(f"   ✓ {pattern_name}: {len(processed_data)} items")
        
        return data_patterns


    
    def _process_matches(self, matches: List, pattern_type: str) -> Dict[str, float]:
        """Convert regex matches to clean numerical data"""
        processed = {}
        
        for match in matches:
            try:
                if pattern_type in ['budget_items', 'cost_breakdown']:
                    name, value, percent_sign = match if len(match) == 3 else (match[0], match[1], '')
                    name = name.strip()
                    num_value = float(value.replace(',', ''))
                    
                    # Skip obviously invalid data
                    if num_value > 0 and len(name) > 1:
                        processed[name] = num_value
                else:
                    name, value = match
                    name = name.strip()
                    num_value = float(value.replace(',', ''))
                    
                    # Skip obviously invalid data
                    if num_value > 0 and len(name) > 1:
                        processed[name] = num_value
                        
            except (ValueError, IndexError):
                continue
        
        return processed

    

    def _generate_intelligent_visualizations(self, response_text: str, extracted_data: Dict, 
                                           user_query: str, session_context: Dict) -> Dict[str, Any]:
        """Use Claude to intelligently analyze data and create visualization specifications"""
        
        print("🤖 Calling Claude for intelligent visualization analysis...")
        
        # Prepare context for Claude analysis
        analysis_prompt = f"""You are a data visualization expert. Analyze the following data extracted from a business document response and create intelligent visualization specifications.

USER QUERY: {user_query}

EXTRACTED DATA:
{json.dumps(extracted_data, indent=2)}

ORIGINAL RESPONSE CONTEXT (first 1500 chars):
{response_text[:1500]}...

Please analyze this data and return a JSON specification for visualizations. Create visualizations that make sense for the data patterns found.

Return ONLY valid JSON in this exact structure:

{{
  "visualizations": [
    {{
      "id": "unique_chart_id",
      "title": "Human-readable chart title",
      "type": "line|bar|doughnut|area",
      "data": {{"label1": value1, "label2": value2}},
      "insights": ["Key insight 1", "Key insight 2"],
      "chart_config": {{
        "color_scheme": "financial|performance|categorical",
        "show_totals": true,
        "format_as_currency": true
      }}
    }}
  ],
  "executive_summary": {{
    "total_value": 123456,
    "key_metrics": ["metric1: value1", "metric2: value2"],
    "trends": ["trend observation 1", "trend observation 2"]
  }}
}}

RULES:
1. Choose chart types intelligently:
   - Time series → line charts
   - Budget/cost breakdowns → doughnut for ≤6 items, bar for >6 items  
   - Comparisons → bar charts
   - Performance metrics → bar charts

2. Only include data that has 2+ meaningful data points
3. Create meaningful titles and insights
4. Generate actionable executive summary
5. Return ONLY valid JSON, no markdown or additional text

ANALYZE THE DATA AND RETURN THE VISUALIZATION SPECIFICATION:"""

        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.1,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            response_text = response.content[0].text.strip()
            print(f"🤖 Claude response length: {len(response_text)} chars")
            
            # Clean the response to ensure it's valid JSON
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Parse Claude's response as JSON
            viz_spec = json.loads(response_text)
            print("✅ Successfully parsed Claude's JSON response")
            
            # Enhance with additional processing
            viz_spec = self._enhance_visualization_spec(viz_spec, extracted_data)
            
            return viz_spec
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {response_text[:500]}...")
            return self._create_fallback_visualization(extracted_data, user_query)
            
        except Exception as e:
            print(f"❌ Error in intelligent visualization generation: {e}")
            return self._create_fallback_visualization(extracted_data, user_query)
    
    
    def _enhance_visualization_spec(self, viz_spec: Dict, extracted_data: Dict) -> Dict[str, Any]:
        """
        Enhance the visualization spec with additional intelligence
        """
        # Add data quality indicators
        for viz in viz_spec.get("visualizations", []):
            data_points = len(viz.get("data", {}))
            viz["data_quality"] = {
                "data_points": data_points,
                "confidence": "high" if data_points >= 3 else "medium" if data_points >= 2 else "low"
            }
        
        # Enhance executive summary with calculated metrics
        all_values = []
        for pattern_data in extracted_data.values():
            if isinstance(pattern_data, dict):
                all_values.extend([v for v in pattern_data.values() if isinstance(v, (int, float))])
        
        if all_values and "executive_summary" in viz_spec:
            viz_spec["executive_summary"]["calculated_metrics"] = {
                "total": sum(all_values),
                "average": sum(all_values) / len(all_values),
                "range": {"min": min(all_values), "max": max(all_values)},
                "data_points": len(all_values)
            }
        
        return viz_spec
    
    def _create_fallback_visualization(self, extracted_data: Dict, user_query: str) -> Dict[str, Any]:
        """
        Create a basic visualization when intelligent analysis fails
        """
        visualizations = []
        
        # Create charts for each data pattern found
        for pattern_name, data in extracted_data.items():
            if isinstance(data, dict) and len(data) > 0:
                
                # Determine chart type based on pattern
                if 'temporal' in pattern_name or 'quarterly' in pattern_name:
                    chart_type = 'line'
                elif 'percentage' in pattern_name or len(data) <= 5:
                    chart_type = 'doughnut'
                else:
                    chart_type = 'bar'
                
                # Create title
                title = pattern_name.replace('_', ' ').title()
                if 'cost' in pattern_name.lower():
                    title += ' Analysis'
                elif 'efficiency' in pattern_name.lower():
                    title += ' Performance'
                
                visualizations.append({
                    "id": f"{pattern_name}_chart",
                    "title": title,
                    "type": chart_type,
                    "data": data,
                    "insights": [f"Found {len(data)} data points", f"Chart type: {chart_type}"],
                    "chart_config": {
                        "color_scheme": "financial" if 'cost' in pattern_name else "performance",
                        "format_as_currency": 'cost' in pattern_name or 'budget' in pattern_name
                    }
                })
        
        return {
            "visualizations": visualizations,
            "executive_summary": {
                "total_visualizations": len(visualizations),
                "key_metrics": [f"Generated {len(visualizations)} charts from data analysis"],
                "data_source": "Automated extraction from document analysis"
            }
        }


def integrate_with_existing_app(app, claude_client):
    """
    Integration function to add visualization analyst to existing Flask app
    """
    analyst = VisualizationAnalyst(claude_client)
    
    # Modify the existing qa_endpoint
    @app.route("/qa_api", methods=["POST"])
    def enhanced_qa_endpoint():
        from claude_processor import ClaudeLikeDocumentProcessor
        
        global document_processor
        
        try:
            query = request.json.get("query")
            
            if not query or not query.strip():
                return jsonify({"answer": "Please provide a valid question."})
            
            print(f"\n=== ENHANCED PROCESSING WITH VISUALIZATION ANALYST ===")
            print(f"User query: '{query}'")
            
            http_server_url = "http://54.172.238.47:8000"
            
            if not claude_client:
                return jsonify({"answer": "Claude API service not available."})
            
            # Initialize processor if not exists
            if document_processor is None:
                document_processor = ClaudeLikeDocumentProcessor(http_server_url, claude_client)
                print("New session started")
            else:
                print(f"Continuing session - {len(document_processor.session_context['files_mentioned'])} files in context")
            
            # Process query iteratively
            response = document_processor.process_query_iteratively(query)
            
            # Clean the response
            cleaned_response = clean_response_text(response)
            
            # Check if visualization is requested
            is_viz_request = any(keyword in query.lower() for keyword in 
                ['visual', 'chart', 'graph', 'show', 'display', 'forecast', 'visualize']) or \
                any(keyword in query.lower() for keyword in 
                ['cost', 'budget', 'data', 'number', 'amount', 'project'])
            
            viz_data = None
            if is_viz_request:
                # Use the intelligent visualization analyst
                viz_data = analyst.analyze_and_visualize(
                    response, query, document_processor.session_context
                )
                print(f"Intelligent visualization generated: {viz_data is not None}")
            
            return jsonify({
                "answer": cleaned_response,
                "visualization": viz_data,
                "has_visualization": viz_data is not None
            })
        
        except Exception as e:
            print(f"Error in enhanced processing: {e}")
            return jsonify({"answer": f"System error: {str(e)}"})
    
    return enhanced_qa_endpoint

# Helper function for response cleaning (from your existing code)
def clean_response_text(text):
    """Remove JavaScript code and function definitions from response"""
    import re
    
    text = re.sub(r'function\s+\w+\s*\([^)]*\)\s*\{[\s\S]*?\}', '', text)
    text = re.sub(r'(const|let|var)\s+\w+\s*=\s*function[\s\S]*?\};?', '', text)
    text = re.sub(r'\w+\s*:\s*function\s*\([^)]*\)\s*\{[\s\S]*?\}', '', text)
    text = re.sub(r'document\.createElement[\s\S]*?;', '', text)
    text = re.sub(r'\w+\.innerHTML\s*=[\s\S]*?;', '', text)
    text = re.sub(r'\w+\.appendChild[\s\S]*?;', '', text)
    text = re.sub(r'createChart\([^)]*\);?', '', text)
    text = re.sub(r'Budget Analysis|Efficiency Scores|Availability Rates', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
