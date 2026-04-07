# 1 단계: 모듈 임포트
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# 네이버 뉴스 RSS 피드 URL
NAVER_NEWS_RSS_URL = "https://news.naver.com/main/rss/popular/news?category=popular&order=hot"

def get_naver_news():
    """
    네이버 뉴스 RSS 피드를 크롤링하여 최신 뉴스 목록을 반환합니다.
    """
    try:
        # 2 단계: 요청 보내기
        response = requests.get(NAVER_NEWS_RSS_URL, timeout=10)
        response.raise_for_status()
        
        # 3 단계: XML 파싱
        root = ET.fromstring(response.content)
        
        news_list = []
        # 4 단계: 뉴스 항목 추출
        for item in root.findall('.//item'):
            title_elem = item.find('title')
            link_elem = item.find('link')
            pubdate_elem = item.find('pubDate')
            
            if title_elem is not None and link_elem is not None:
                news_item = {
                    'title': title_elem.text.strip(),
                    'link': link_elem.text.strip(),
                    'pubDate': pubdate_elem.text.strip() if pubdate_elem is not None else None
                }
                news_list.append(news_item)
        
        return news_list
    except Exception as e:
        print(f"Error fetching Naver News: {e}")
        return []

if __name__ == "__main__":
    news = get_naver_news()
    for i, news_item in enumerate(news, 1):
        print(f"{i}. {news_item['title']}")
        print(f"   Link: {news_item['link']}")
        if news_item['pubDate']:
            print(f"   Date: {news_item['pubDate']}")
        print("-" * 30)


