import requests
from pathlib import Path

# הגדרות קבועות
API_KEY = "924e9425-113e-4278-ba12-b656edcc21fe"  
BASE_URL = "https://content.guardianapis.com/search"
MAX_PAGES = 100   # מספר העמודים המקסימלי שנרצה למשוך (עד 5000 כתבות)
PAGE_SIZE = 50   # מספר הכתבות בעמוד (מקסימום מותר על ידי API)
SECTION_NAME = "sport" 


def save_article(article):
    """שומר כתבה אחת לקובץ טקסט בתיקיית sport."""
    fields = article.get("fields", {})
    body = (fields.get("bodyText") or "").strip()
    if not body:
        return False

    # בניית הכותרת לקובץ
    section = article.get("sectionName", SECTION_NAME)
    date = (article.get("webPublicationDate") or "")[:10]
    byline = fields.get("byline", "Unknown Author")
    url = article.get("webUrl", "")
    title = fields.get("headline", article.get("webTitle", "Untitled"))

    header = (
        f"The Guardian | {section} | {date}\n"
        f"By {byline}\n"
        f"{url}\n\n"
        f"{title}\n"
        f"{'-' * 60}\n"
    )

    # שמירה ל-data/sport/<article_id>.txt
    out_dir = Path("data") / SECTION_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    article_id = article["id"].replace("/", "_")
    out_path = out_dir / f"{article_id}.txt"
    
    # ודא שלא שומרים כתבות כפולות
    if not out_path.exists():
        out_path.write_text(f"{header}{body}\n", encoding="utf-8")
        return True
    return False


def main():
    print(f"🚀 מתחיל לאסוף כתבות עבור הנושא: {SECTION_NAME.upper()}...")
    total_articles_saved = 0
    total_pages = MAX_PAGES
    
    for page_num in range(1, total_pages + 1):
        params = {
            "api-key": API_KEY,
            "page-size": PAGE_SIZE,
            "section": SECTION_NAME, 
            "order-by": "newest",
            "show-fields": "headline,byline,bodyText",
            "page": page_num,
        }
        
        try:
            resp = requests.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()["response"]
            
            if page_num == 1:
                total_pages = min(MAX_PAGES, data.get("pages", 1)) 
                print(f"נמצאו סה״כ {data.get('total', 0)} כתבות. מושך עד {total_pages} עמודים.")

            results = data.get("results", [])
            if not results:
                break
            
            articles_saved_in_page = 0
            for article in results:
                if save_article(article):
                    total_articles_saved += 1
                    articles_saved_in_page += 1

            print(f"  ✅ עמוד {page_num}/{total_pages} נשמר: {articles_saved_in_page} כתבות חדשות. סה\"כ נשמרו: {total_articles_saved}")
            
            if page_num >= total_pages:
                break

        except requests.exceptions.RequestException as e:
            print(f"  ❌ שגיאה בבקשת API: {e}")
            break

    print(f"✨ סיום איסוף עבור {SECTION_NAME.upper()}. סה״כ נשמרו: {total_articles_saved} כתבות.")


if __name__ == "__main__":
    main()