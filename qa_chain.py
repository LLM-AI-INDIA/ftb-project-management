import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

def load_dataframes(csv_files):
    """Load all CSV files into DataFrames with smart sampling"""
    dataframes = {}
    for file in csv_files:
        try:
            df_name = os.path.splitext(file)[0]
            dataframes[df_name] = pd.read_csv(file)
            print(f"✅ Loaded {file} with {len(dataframes[df_name])} rows")
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    return dataframes

def dataframe_to_context(df, df_name, max_sample_rows=5, max_chars=2000):
    """Convert dataframe to context string with smart sampling"""
    context = f"📊 DATAFRAME: {df_name}\n"
    context += f"📈 Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
    context += f"🏷️ Columns: {', '.join(df.columns.tolist())}\n"
    
    # Add sample data - smarter sampling
    if len(df) > 0:
        sample_size = min(max_sample_rows, len(df))
        
        # Try to get diverse sample (not just first rows)
        if len(df) > sample_size * 2:
            sample = pd.concat([
                df.head(sample_size // 2), 
                df.tail(sample_size // 2)
            ])
        else:
            sample = df.head(sample_size)
        
        context += f"🔍 Sample data ({sample_size} rows):\n"
        context += sample.to_string(index=False)
    
    # Truncate if too long
    if len(context) > max_chars:
        context = context[:max_chars] + "...\n[TRUNCATED]"
    
    return context + "\n" + "─" * 50 + "\n"

def build_qa(csv_files):
    dataframes = load_dataframes(csv_files)
    
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

    prompt_template = """You are a smart, helpful assistant for the Franchise Tax Board.
Your job is to answer the user's questions as thoroughly and clearly as possible. Use the following dataframes to answer the question. 
Analyze the available data and provide insights.

- If the answer can be found in the AVAILABLE DATAFRAMES below, use the data to support your response with specific details or analysis.
- If the user's question cannot be answered from the data, or if it's a general knowledge question, respond with the best information you know from your world knowledge.
- Make it clear when your answer is based on general knowledge vs. the enterprise data.
- Always respond accurately, transparently, and with appropriate context.

AVAILABLE DATAFRAMES:
{dataframes_context}

USER QUESTION: {question}

ANALYSIS GUIDELINES:
1. If possible, identify relevant dataframe(s) and look for matching topics or columns that match the question topics.
2. Use specific data when you can, and reference specific data points in your answer.
3. If the data doesn't answer the question, or if the question is general (not covered in the data), answer using your own general knowledge.
4. Be polite and informative if data is not available, but try to always answer the user's question and politely explain what data is available.
5. Format your response clearly, using sections and bullet points where helpful.

ANSWER:
"""
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["dataframes_context", "question"]
    )

    def query_chain(query):
        try:
            # Create context from dataframes
            dataframe_contexts = []
            for df_name, df in dataframes.items():
                context_str = dataframe_to_context(df, df_name)
                dataframe_contexts.append(context_str)
            
            context = "\n".join(dataframe_contexts)
            
            # Format prompt
            formatted_prompt = prompt.format(
                dataframes_context=context, 
                question=query
            )
            
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            
            return response.content
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    return query_chain
