import os
from typing import cast, Any  # 👈 [추가] Pylance 타입 에러를 무마하기 위한 모듈
import chromadb
from chromadb.utils import embedding_functions

class VectorMemoryManager:
    def __init__(self, db_path="./res/vdb_storage"):
        """로컬 벡터 데이터베이스(ChromaDB)를 초기화합니다."""
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        
        # 로컬 환경에 저장되는 영구적인 DB 클라이언트 생성
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 한국어를 잘 지원하는 가벼운 다국어 임베딩 모델 사용
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 👇 [수정] cast(Any, ...)를 사용해 Pylance의 타입 검사를 우회합니다.
        self.collection = self.client.get_or_create_collection(
            name="mop_long_term_memory",
            embedding_function=cast(Any, self.emb_fn) 
        )
        print("🧠 [Vector DB] MOP 장기 기억소(Hippocampus) 로드 완료.")

    # 👇 [수정] metadata: dict | None = None 으로 올바른 타입 힌트 적용
    def add_memory(self, text: str, metadata: dict | None = None):
        """새로운 기억을 벡터로 변환하여 저장합니다."""
        import uuid
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {"source": "agent_learning"}],
            ids=[memory_id]
        )
        return memory_id

    def search_memory(self, query: str, n_results: int = 3) -> list:
        """질문의 의미(Semantic)를 분석하여 가장 유사한 과거 기억을 찾아옵니다."""
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        # 찾은 문서들의 리스트만 깔끔하게 반환
        if results and results['documents']:
            return results['documents'][0]
        return []

# 단독 실행 테스트용
if __name__ == "__main__":
    vdb = VectorMemoryManager()
    vdb.add_memory("어제 파이썬 pandas 라이브러리를 쓸 때 utf-8 인코딩 에러가 발생해서 cp949로 해결했다.", {"type": "error_fix"})
    vdb.add_memory("내 주인의 이름은 차니이고, 치킨을 좋아한다.", {"type": "user_preference"})
    
    print("\n🔍 검색 테스트: '파이썬 에러 어떻게 고치더라?'")
    answers = vdb.search_memory("파이썬 에러 어떻게 고치더라?")
    for a in answers:
        print("-> 기억 회상:", a)