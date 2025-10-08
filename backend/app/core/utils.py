import hashlib
from langdetect import detect

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return "fr" if lang == "fr" else "en"
    except Exception:
        return "en"

def md5(s: str) -> str:
    """
    Generate an MD5 hash for a given string.
    Args:
        s (str): The input string.
    Returns:
        str: A hexadecimal string representing the MD5 hash.
    """
    # 1. Encode the string into bytes using UTF-8 encoding
    # (hash functions always work on bytes, not Python str directly)
    # 2. Compute the MD5 hash of the byte string
    # 3. Convert the hash object into a hexadecimal string (readable format)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def chunk(lst, size):
    """
    Split a list into smaller chunks of a given size.

    Args:
        lst (list): The input list to split.
        size (int): Maximum size of each chunk.

    Yields:
        list: Sub-lists of the original list, each up to 'size' elements.
    
    Example:
        list(_chunk([1,2,3,4,5], 2)) 
        -> [[1,2], [3,4], [5]]
    """
    # Loop over the list with a step of 'size'
    for i in range(0, len(lst), size):
        # Yield a slice from index i to i+size (handles last chunk automatically)
        yield lst[i:i+size]