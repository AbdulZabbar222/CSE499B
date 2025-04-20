import pandas as pd
import time
import requests
import psutil

# Function to query the local LLM using Ollama
def query_ollama(prompt: str, context: str = "", model: str = "llama3.2:latest"):
    url = "http://localhost:11434/api/generate"
    
    system_prompt = """You are a helpful bank assistant. Your task is to:
    1. Answer questions based ONLY on the provided context
    2. Keep responses clear and professional
    3. Use bullet points for multiple items
    4. Include relevant numbers/data when available
    5. Say \"I don't have enough information\" if context is insufficient"""

    full_prompt = f"""[SYSTEM]
    {system_prompt}

    [CONTEXT]
    {context}

    [QUESTION]
    {prompt}

    [INSTRUCTIONS]
    - Answer directly based on the above context
    - If numbers or specific data are mentioned, include them
    - Format lists as bullet points
    - Keep explanations concise but complete
    - Cite relevant sections if possible

    [RESPONSE FORMAT]
    Begin your response here..."""

    data = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }
    
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    response_time = end_time - start_time
    
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
    
    if response.status_code == 200:
        return response.json().get('response', "No response"), response_time, memory_usage
    else:
        return f"Error: {response.status_code}", response_time, memory_usage

# Load questions from an Excel file
input_file = "user_queries.xlsx"
output_file = "llama_responses.xlsx"

df = pd.read_excel(input_file)

# Ensure there's a column named 'Question'
if 'Question' not in df.columns:
    raise ValueError("The input Excel file must contain a 'Question' column.")

responses = []
response_times = []
memory_usages = []

# Iterate through each question and print status
print("\nProcessing questions...\n")
for index, row in df.iterrows():
    question = row['Question']
    print(f"Processing {index+1}/{len(df)}: {question[:50]}...")  # Show first 50 chars
    
    response, response_time, memory_usage = query_ollama(question)
    
    responses.append(response)
    response_times.append(response_time)
    memory_usages.append(memory_usage)

# Add responses to DataFrame
df['Response'] = responses
df['Response Time (s)'] = response_times
df['Memory Usage (MB)'] = memory_usages

# Save to a new Excel file
df.to_excel(output_file, index=False)
print(f"\nAll responses saved to {output_file}\n")
