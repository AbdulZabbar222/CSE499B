import pandas as pd

def calculate_human_eval_score(file_path, sheet_name=0, eval_column='Human Evaluation'):
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Count occurrences of each category
    counts = df[eval_column].value_counts()
    
    # Extract values for each category (default to 0 if not present)
    relevant = counts.get('Relevant', 0)
    partially_relevant = counts.get('Partially Relevant', 0)
    misleading = counts.get('Misleading', 0)
    irrelevant = counts.get('Irrelevant', 0)
    
    # Total responses
    total = relevant + partially_relevant + misleading + irrelevant
    if total == 0:
        return "No valid evaluations found."
    
    # Calculate human eval score (formula: weighted sum)
    score = (relevant * 1.0 + partially_relevant * 0.5 - misleading * 0.5 - irrelevant * 1.0) / total
    
    return {
        'Relevant': relevant,
        'Partially Relevant': partially_relevant,
        'Misleading': misleading,
        'Irrelevant': irrelevant,
        'Total': total,
        'Human Eval Score': score
    }

# Example usage
file_path = 'llama_responses.xlsx'
result = calculate_human_eval_score(file_path)
print(result)
