"""
skills/file_tools.py
파일 및 코드 조작을 위한 독립 스킬 모듈 (View, Edit, Find, Search)
"""

import os
import glob
import argparse

def view_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"오류: '{file_path}' 파일을 찾을 수 없습니다."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        numbered_lines = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        return f"[파일 내용: {file_path}]\n" + "".join(numbered_lines)
    except Exception as e:
        return f"파일 읽기 오류: {e}"

def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    if not os.path.exists(file_path):
        return f"오류: '{file_path}' 파일을 찾을 수 없습니다."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_text not in content:
            return "오류: 변경할 원본 텍스트(old_text)를 파일에서 찾을 수 없습니다. 들여쓰기와 줄바꿈이 정확히 일치해야 합니다."
            
        new_content = content.replace(old_text, new_text)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return f"성공: '{file_path}' 교체 완료."
    except Exception as e:
        return f"파일 수정 오류: {e}"

def find_files(pattern: str, dir_path: str = ".") -> str:
    try:
        # 확장자만 들어올 경우를 대비해 *. 처리
        if not pattern.startswith("*"):
            pattern = "*" + pattern
            
        search_path = os.path.join(dir_path, "**", pattern)
        files = glob.glob(search_path, recursive=True)
        if not files:
            return f"'{pattern}' 패턴과 일치하는 파일을 찾을 수 없습니다."
        return "[검색된 파일 목록]\n" + "\n".join(files)
    except Exception as e:
        return f"파일 검색 오류: {e}"

def search_text(query: str, dir_path: str = ".") -> str:
    results = []
    try:
        for root, _, files in os.walk(dir_path):
            if '.git' in root or '__pycache__' in root or 'venv' in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if query in line:
                                results.append(f"{file_path}:{i+1}: {line.strip()}")
                except (UnicodeDecodeError, PermissionError):
                    continue 
        
        if not results:
            return f"'{query}' 텍스트를 찾을 수 없습니다."
        return f"[텍스트 검색 결과]\n" + "\n".join(results[:100])
    except Exception as e:
        return f"텍스트 검색 오류: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["view", "edit", "find", "search"])
    parser.add_argument("--path", "--file_path", type=str, default=".") # 별칭 추가
    parser.add_argument("--old_text", "--search_string", type=str, default="") # 별칭 추가
    parser.add_argument("--new_text", "--replace_string", type=str, default="") # 별칭 추가
    parser.add_argument("--pattern", "--extension", "--ext", type=str, default="") # 별칭 통합
    parser.add_argument("--query", "--text", "--search_text", type=str, default="") # 별칭 통합
    
    args = parser.parse_args()
    
    if args.action == "view":
        print(view_file(args.path))
    elif args.action == "edit":
        print(edit_file(args.path, args.old_text, args.new_text))
    elif args.action == "find":
        print(find_files(args.pattern, args.path))
    elif args.action == "search":
        print(search_text(args.query, args.path))