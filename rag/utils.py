import re
import urllib
from collections import Counter

# HELPER FUNCTIONS
def get_unique_urls(collection):  
    urls = []  
    for item in collection:  
        # Extract the URL from the item in the collection  
        url = urllib.parse.urlparse(item['url']).netloc  
        urls.append(url)  
      
    unique_urls = set(urls)  
    url_counts = Counter(urls)  
      
    return unique_urls, url_counts  
def clean_text(text):  
    # Remove non-alphanumeric characters (excluding spaces, underscores, hyphens, periods, and commas)  
    clean = re.sub(r'[^\w\s\.\,\-_]', '', text)  
    return clean 