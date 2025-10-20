import re
import string

def clean_resume(text):
    """
    Clean resume text by removing URLs, punctuation, numbers, and extra spaces.
    """
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()
