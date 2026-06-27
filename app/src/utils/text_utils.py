import re

def extract_sentences(text: str) -> tuple[list[str], str]:
    """
    Splits text into sentences. Returns a list of complete sentences
    and the remaining unfinished text.
    
    Args:
        text (str): Incoming stream of text chunks.
        
    Returns:
        tuple[list[str], str]: List of full sentences and the remaining partial sentence.
    """
    # Matches sentences ending in ., ?, !, or a newline
    pattern = re.compile(r'([^.!?\n]+[.!?\n]+)')
    matches = pattern.findall(text)
    
    if not matches:
        return [], text
        
    matched_len = sum(len(m) for m in matches)
    remainder = text[matched_len:]
    
    return [m.strip() for m in matches if m.strip()], remainder
