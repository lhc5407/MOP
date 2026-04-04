"""
skills/memory_tools.py
에이전트의 장기 기억(Long-term Memory) 관리를 위한 스킬 모듈 (Pylance Type-Safe)
"""

import os
import json
import argparse
from typing import Dict, Optional

# 기억을 저장할 로컬 파일 경로 (skills 폴더 내에 숨김 파일로 저장)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, ".agent_memory.json")

def _load_memory() -> Dict[str, str]:
    """로컬 파일에서 기억 데이터를 불러옵니다."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_memory(data: Dict[str, str]) -> None:
    """메모리 딕셔너리를 로컬 파일에 안전하게 저장합니다."""
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def write_memory(key: str, value: str) -> str:
    """새로운 기억을 저장하거나 기존 기억을 덮어씁니다."""
    if not key or not value:
        return "오류: 기억을 저장하려면 key와 value가 모두 필요합니다."
    
    data = _load_memory()
    data[key] = value
    _save_memory(data)
    return f"[기억 저장 완료] 핵심 키워드 '{key}'에 정보가 영구적으로 기록되었습니다."

def read_memory(key: str) -> str:
    """특정 키워드에 대한 기억을 불러옵니다."""
    if not key:
        return "오류: 조회할 key가 필요합니다."
        
    data = _load_memory()
    if key in data:
        return f"[기억 조회: {key}]\n{data[key]}"
    return f"'{key}'에 대한 기억을 찾을 수 없습니다."

def list_memories() -> str:
    """저장된 모든 기억의 키워드(목록)를 반환합니다."""
    data = _load_memory()
    if not data:
        return "현재 저장된 장기 기억이 없습니다."
    
    keys_list = "\n".join([f"- {k}" for k in data.keys()])
    return f"[저장된 기억 키워드 목록]\n{keys_list}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["write", "read", "list"])
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--value", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.action == "write":
        # Pylance 안전성: 인자가 None일 경우 빈 문자열로 처리하여 에러 방지
        print(write_memory(args.key or "", args.value or ""))
    elif args.action == "read":
        print(read_memory(args.key or ""))
    elif args.action == "list":
        print(list_memories())