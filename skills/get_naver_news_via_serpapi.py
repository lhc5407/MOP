import requests

def get_naver_news_via_serpapi(query, api_key):
    """Uses SerpApi to fetch Naver News results for a given query."""
    url = "https://serpapi.com/search"
    params = {
        "engine": "naver_news",
        "q": query,
        "api_key": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        news_items = []
        for item in data.get("organic_results", []):
            news_items.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", "")
            })
        
        return news_items
    except Exception as e:
        print(f"Error fetching Naver News: {e}")
        return []
