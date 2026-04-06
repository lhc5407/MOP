"""
skills/file_tools.py
파일 및 코드 조작을 위한 독립 스킬 모듈 (View, Edit, Find, Search)
"""

import os
import glob
import argparse

def view_file(file_path: str) -> str:
    """파일의 내용을 줄 번호와 함께 읽어옵니다."""
    if not os.path.exists(file_path):
        return f"오류: '{file_path}' 파일을 찾을 수 없습니다."
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # AI가 수정할 위치를 쉽게 파악하도록 줄 번호 추가
        numbered_lines = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        return f"[파일 내용: {file_path}]\n" + "".join(numbered_lines)
    except Exception as e:
        return f"파일 읽기 오류: {e}"

def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    """파일 내의 특정 텍스트를 새로운 텍스트로 교체(수정)합니다."""
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
        return f"[수정 완료] '{file_path}' 파일이 성공적으로 업데이트되었습니다."
    except Exception as e:
        return f"파일 수정 오류: {e}"

def find_files(pattern: str, dir_path: str = ".") -> str:
    """특정 패턴(*.py 등)과 일치하는 파일 목록을 재귀적으로 찾습니다."""
    try:
        search_path = os.path.join(dir_path, "**", pattern)
        files = glob.glob(search_path, recursive=True)
        if not files:
            return f"'{pattern}' 패턴과 일치하는 파일을 찾을 수 없습니다."
        return "[검색된 파일 목록]\n" + "\n".join(files)
    except Exception as e:
        return f"파일 검색 오류: {e}"

def search_text(query: str, dir_path: str = ".") -> str:
    """디렉토리 내의 모든 파일에서 특정 텍스트(변수명 등)가 포함된 위치를 찾습니다."""
    results = []
    try:
        for root, _, files in os.walk(dir_path):
            # 깃허브 폴더나 캐시 폴더는 검색에서 제외
            if '.git' in root or '__pycache__' in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if query in line:
                                results.append(f"{file_path}:{i+1}: {line.strip()}")
                except (UnicodeDecodeError, PermissionError):
                    continue # 바이너리 파일 등 읽을 수 없는 파일은 무시
        
        if not results:
            return f"'{query}' 텍스트를 찾을 수 없습니다."
        return f"[텍스트 검색 결과]\n" + "\n".join(results[:100]) # 출력 길이 제한
    except Exception as e:
        return f"텍스트 검색 오류: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["view", "edit", "find", "search"])
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--old_text", type=str, default="")
    parser.add_argument("--new_text", type=str, default="")
    parser.add_argument("--pattern", type=str, default="")
    parser.add_argument("--query", type=str, default="")
    
    args = parser.parse_args()
    
    if args.action == "view":
        print(view_file(args.path))
    elif args.action == "edit":
        print(edit_file(args.path, args.old_text, args.new_text))
    elif args.action == "find":
        print(find_files(args.pattern, args.path))
    elif args.action == "search":
        print(search_text(args.query, args.path))