import requests
from bs4 import BeautifulSoup
import json

def get_naver_news(query, count=5):
    """네이버 뉴스 검색 결과를 크롤링하여 구조화된 데이터를 반환합니다."""
    url = f'https://search.naver.com/search.naver?where=news&q={query}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        news_list = []
        items = soup.select('div.news_list li')
        
        for item in items[:count]:
            title = item.select_one('a.news_title')
            link = item.select_one('a.news_link')
            date = item.select_one('span.news_date')
            
            if title and link:
                news_list.append({
                    'title': title.text.strip(),
                    'link': link.get('href', '').strip(),
                    'date': date.text.strip() if date else 'N/A'
                })
        
        return news_list
    except Exception as e:
        return {'error': str(e)}




if __name__ == "__main__":
    test_query = "인공지능"
    result = get_naver_news(test_query)
    print(json.dumps(result, ensure_ascii=False, indent=2))

