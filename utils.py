import os
import json

def load_essays(directory, limit=None):
    """
    Load essays from the specified directory.
    
    Args:
        directory (str): Path to the directory containing essay files
        limit (int, optional): Maximum number of essays to load
    
    Returns:
        list: List of tuples (essay_id, text)
    """
    essays = []
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found")
    
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Sort files to ensure consistent order
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
    
    # Apply limit if specified
    if limit:
        files = files[:limit]
    
    # Load each essay
    for filename in files:
        if filename.endswith('.txt'):  # Assuming essays are in .txt files
            essay_id = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                essays.append((essay_id, text))
            except Exception as e:
                print(f"Error loading essay {essay_id}: {str(e)}")
    
    return essays