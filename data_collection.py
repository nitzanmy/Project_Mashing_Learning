import requests
from pathlib import Path

#  Constants 
API_KEY = "924e9425-113e-4278-ba12-b656edcc21fe"
BASE_URL = "https://content.guardianapis.com/search"
MAX_PAGES = 100   # Maximum number of pages to pull per section
PAGE_SIZE = 50    # Number of articles per page

# Configuration List: Maps the API section name to the local folder name
SECTIONS_TO_FETCH = [
    {"api_section": "news", "folder": "news"},
    {"api_section": "sport", "folder": "sport"},
    {"api_section": "culture", "folder": "culture"},
    {"api_section": "commentisfree", "folder": "opinion"} 
]

def save_article(article, folder_name):
    """
    Saves a single article to a text file in the specified folder.
    """
    fields = article.get("fields", {})
    body = (fields.get("bodyText") or "").strip()
    
    # Skip empty articles
    if not body:
        return False

    # Construct the file header
    # use the actual section name from the API for the header text
    section_name_in_header = article.get("sectionName", folder_name)
    date = (article.get("webPublicationDate") or "")[:10]
    byline = fields.get("byline", "Unknown Author")
    url = article.get("webUrl", "")
    title = fields.get("headline", article.get("webTitle", "Untitled"))

    header = (
        f"The Guardian | {section_name_in_header} | {date}\n"
        f"By {byline}\n"
        f"{url}\n\n"
        f"{title}\n"
        f"{'-' * 60}\n"
    )

    # Define output directory based on the 'folder' parameter
    out_dir = Path("data") / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename from the article ID
    article_id = article["id"].replace("/", "_")
    out_path = out_dir / f"{article_id}.txt"
    
    # check if file exists
    if not out_path.exists():
        out_path.write_text(f"{header}{body}\n", encoding="utf-8")
        return True
    
    return False


def fetch_section_data(config):
    """
    Orchestrates the download process for a specific section based on config.
    """
    api_section = config["api_section"]
    folder_name = config["folder"]
    
    print(f"\n Starting collection for: {folder_name.upper()} (API Section: {api_section})...")
    
    total_articles_saved = 0
    current_max_pages = MAX_PAGES
    
    for page_num in range(1, current_max_pages + 1):
        params = {
            "api-key": API_KEY,
            "page-size": PAGE_SIZE,
            "section": api_section, 
            "order-by": "newest",
            "show-fields": "headline,byline,bodyText",
            "page": page_num
        }
        
        try:
            resp = requests.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()["response"]
            
            # On the first page, adjust the total pages based on actual availability
            if page_num == 1:
                actual_pages = data.get("pages", 1)
                current_max_pages = min(MAX_PAGES, actual_pages)
                print(f"Found {data.get('total', 0)} items available. Pulling {current_max_pages} pages.")

            results = data.get("results", [])
            if not results:
                print(" No more results returned from API.")
                break
            
            # Process and save articles
            saved_in_this_page = 0
            for article in results:
                if save_article(article, folder_name):
                    total_articles_saved += 1
                    saved_in_this_page += 1

            print(f"Page {page_num}/{current_max_pages}: Saved {saved_in_this_page} new articles.")
            
            # Stop if we reached the limit
            if page_num >= current_max_pages:
                break

        except requests.exceptions.RequestException as e:
            print(f"API request error on page {page_num}: {e}")
            break

    print(f"Finished {folder_name.upper()}. Total saved: {total_articles_saved}.")


def main():
    print("Starting Bulk Data Collection")
    
    # Iterate over each section configuration and run the fetch process
    for section_config in SECTIONS_TO_FETCH:
        fetch_section_data(section_config)
        
    print("\n All collections completed successfully ")

if __name__ == "__main__":
    main()
