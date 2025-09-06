from langdetect import detect

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return "fr" if lang == "fr" else "en"
    except Exception:
        return "en"
