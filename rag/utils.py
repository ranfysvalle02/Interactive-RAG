import re
import urllib
from datetime import datetime
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

def encode_google_search(query):
        # Remove whitespace and replace with '+'
        query = query.strip().replace(" ", "+")
        # Encode the query using urllib.parse
        encoded_query = urllib.parse.quote(query)
        # Construct the Google search string
        search_string = f"https://www.google.com/search?q={encoded_query}&num=15"
        return search_string    

def format_and_print_user_input(user_input):
    # Get the current timestamp
    current_time = datetime.now()

    # Format the timestamp as a string
    timestamp_str = current_time.strftime("[%I:%M %p]")

    # Combine timestamp and user input
    formatted_message = f"\n{timestamp_str} User \n           {user_input}\n"

    print(formatted_message)

def format_and_print_genai_response(response):
    # Get the current timestamp
    current_time = datetime.now()

    # Format the timestamp as a string
    timestamp_str = current_time.strftime("[%I:%M %p]")

    # Combine timestamp and user input
    formatted_message = f"\n{timestamp_str} GenAI \n           {response}\n"

    print(formatted_message)

def print_log(log):
    # Get the current timestamp
    current_time = datetime.now()

    # Format the timestamp as a string
    timestamp_str = current_time.strftime("[%I:%M %p]")

    # Combine timestamp and user input
    formatted_message = f"\n{timestamp_str} Application Log \n           {log}\n"

    print(formatted_message)    