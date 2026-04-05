"""
MOP.py - Hybrid Memory & Context Optimized Agent
상단부: SQLite 연동, 5턴 아카이빙 로직 및 도구 정의
"""

import sys
import os
import gc
import json
import subprocess
import re
import datetime
import sqlite3
from typing import List, Dict, Any, cast, Iterator
from ddgs import DDGS

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    print("오류: llama-cpp-python 라이브러리가 설치되어 있지 않습니다.")
    sys.exit(1)

# --- [SQLite 대화 아카이빙 모듈] ---
DB_PATH = "./skills/chat_history.db"

def init_db():
    os.makedirs("./skills", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS history 
                  (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                   role TEXT, content TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def archive_to_sqlite(role, content):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO history (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, datetime.datetime.now()))
    conn.commit()
    conn.close()

def fetch_from_sqlite(count=10):
    """DB에서 가장 최근 대화를 순서대로 불러옵니다."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM history ORDER BY id DESC LIMIT ?", (count,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

def search_history_db(keyword: str, limit: int = 5) -> str:
    """DB에서 특정 키워드가 포함된 과거 대화를 검색합니다."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # LIKE를 사용하여 키워드 검색
    cur.execute("SELECT role, content, timestamp FROM history WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?", (f"%{keyword}%", limit))
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return f"'{keyword}'에 대한 과거 대화 기록이 없습니다."
    
    result = f"[{keyword} 관련 과거 대화 검색 결과]\n"
    for r, c, t in reversed(rows):
        # 시간 정보까지 알려주어 컨텍스트를 강화합니다.
        result += f"[{t[:16]}] {r.upper()}: {c}\n"
    return result

# --- [핵심 실행 함수들] ---
def search_web(query: str) -> str:
    print(f"\n   [시스템] 🌐 인터넷에서 '{query}' 검색 중...", end="", flush=True)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results: return "검색 결과가 없습니다."
        context = ""
        for i, r in enumerate(results, 1):
            context += f"[{i}] 제목: {r['title']}\n내용: {r['body']}\n\n"
        print(" 완료!\n")
        return context
    except Exception as e:
        print(" 실패!\n"); return f"검색 중 오류 발생: {e}"

def execute_skill_safely(cli_args: List[str]) -> str:
    try:
        result = subprocess.run(cli_args, capture_output=True, timeout=30)
        
        # 바이트 데이터를 안전하게 디코딩하는 내부 함수
        def decode_bytes(raw_bytes: bytes) -> str:
            if raw_bytes is None: return ""
            try: return raw_bytes.decode('utf-8')
            except UnicodeDecodeError: return raw_bytes.decode('cp949', errors='replace')
            
        # [핵심 수정] 결과값이 None인 경우를 대비해 빈 문자열 처리를 강화합니다.
        out_str = decode_bytes(result.stdout).strip() if result.stdout else ""
        err_str = decode_bytes(result.stderr).strip() if result.stderr else ""
        
        if result.returncode != 0:
            return f"실행 오류 (코드 {result.returncode}): {err_str or out_str}"
        return out_str or "실행 성공 (반환값 없음)"
    except Exception as e:
        # 시스템 자체 에러 발생 시 AI가 인식할 수 있는 메시지 반환
        return f"시스템 실행 에러: {str(e)}"

def load_gguf_with_kv_quantization(model_path: str) -> Llama:
    if not os.path.exists(model_path): raise FileNotFoundError(f"모델 파일 없음: {model_path}")
    print(f"[{model_path}] 🧠 초고효율(Air-Optimized) 모드로 모델 로딩 중...")
    
    return Llama(
        model_path=model_path, 
        n_threads=10, 
        n_gpu_layers=15, # VRAM 용량에 맞춰 조절 (Layer-wise 분산 처리)
        n_ctx=8192,      # 컨텍스트 윈도우 (필요시 16384로 상향)
        n_batch=512,     # [추가] 프롬프트 처리 시 RAM 스파이크를 막기 위해 배치 사이즈 조절
        
        # [핵심 1] Memory Mapping 활성화 (AirLLM의 Layer-wise와 유사한 효과)
        # 하드디스크의 모델 파일을 가상 메모리로 매핑하여 RAM이 꽉 차면 알아서 SSD로 스왑합니다.
        use_mmap=True,   
        use_mlock=False, # RAM에 모델을 강제로 고정(Lock)하지 않음 (스왑 허용)
        
        # [핵심 2] KV Cache 양자화 (Flash Attention 연계)
        type_k=llama_cpp.GGML_TYPE_Q4_0, 
        type_v=llama_cpp.GGML_TYPE_Q4_0,
        flash_attn=True, 
        verbose=False, 
        chat_format="chatml-function-calling"
    )

def main() -> None:
    init_db()
    MODEL_PATH = r"C:\Users\lhc54\Downloads\Qwen3.5-9B-Q4_K_M.gguf"
    current_time = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    
    try:
        llm = load_gguf_with_kv_quantization(MODEL_PATH)
    except Exception as e:
        print(f"오류: {e}"); return

    # [최적화] 도구 명세서: AI가 도구를 호출할 때 참고하는 규정
    tools: List[Dict[str, Any]] = [
        {"type": "function", "function": {"name": "search_web", "description": "인터넷 검색(query)", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {
            "type": "function", 
            "function": {
                "name": "write_memory", 
                "description": "중요 정보를 영구 저장합니다. [규정]: 반드시 'key'(제목)와 'value'(내용) 인자를 사용하세요. 절대 'content'나 'args'를 쓰지 마세요.", 
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "key": {"type": "string", "description": "기억의 제목"}, 
                        "value": {"type": "string", "description": "기억의 내용"}
                    }, 
                    "required": ["key", "value"]
                }
            }
        },
        {"type": "function", "function": {"name": "read_memory", "description": "기억 조회(key)", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
        {"type": "function", "function": {"name": "list_memories", "description": "기억 목록 확인", "parameters": {"type": "object", "properties": {}}}},
        {
            "type": "function",
            "function": {
                "name": "search_chat_history",
                "description": "SQLite DB에서 과거 대화 기록을 키워드로 검색합니다. 사용자가 예전 대화를 언급하면 반드시 이 도구를 사용하세요.",
                "parameters": {
                    "type": "object",
                    "properties": {"keyword": {"type": "string", "description": "검색할 단어나 문장"}},
                    "required": ["keyword"]
                }
            }
        },
        {"type": "function", "function": {"name": "run_python_snippet", "description": "파이썬 실행(code)", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
        {"type": "function", "function": {"name": "manage_packages", "description": "패키지(action, package_name)", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "package_name": {"type": "string"}}, "required": ["action"]}}},
        {"type": "function", "function": {"name": "view_file", "description": "파일 읽기(file_path)", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
        {"type": "function", "function": {"name": "find_files", "description": "파일 찾기(extension)", "parameters": {"type": "object", "properties": {"extension": {"type": "string"}}, "required": ["extension"]}}},
        {"type": "function", "function": {"name": "search_text", "description": "내용 검색(search_text, file_path)", "parameters": {"type": "object", "properties": {"search_text": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["search_text"]}}},
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "터미널(cmd) 명령어를 실행하여 OS 환경, 파일 목록, 패키지 상태 등을 확인합니다. (예: 'dir', 'python --version', 'pip list')",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "파일의 전체를 다시 쓰지 않고 특정 문자열만 찾아 다른 문자열로 교체(Patch)합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "수정할 파일 경로"},
                        "search_string": {"type": "string", "description": "찾을 기존 문자열 (정확히 일치해야 함)"},
                        "replace_string": {"type": "string", "description": "새롭게 바꿀 문자열"}
                    },
                    "required": ["file_path", "search_string", "replace_string"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "control_mouse",
                "description": "물리적 마우스를 제어하여 실제 화면을 클릭하거나 이동합니다. 'AI라서 마우스 제어가 불가능하다'고 거절하지 마세요.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["move", "click"], "description": "수행할 동작"},
                        "x": {"type": "integer", "description": "이동/클릭할 X 좌표"},
                        "y": {"type": "integer", "description": "이동/클릭할 Y 좌표"}
                    },
                    "required": ["action"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "control_keyboard",
                "description": (
                    "물리적 키보드를 제어하여 메모장 등 현재 활성화된 프로그램에 실시간으로 텍스트를 타이핑하거나 키를 누릅니다. "
                    "절대 'AI라서 물리적 타자가 불가능하다'고 거절하지 마세요. 당신은 할 수 있습니다. "
                    "[사용법]: 문장을 쓸 때는 action='type'과 text='내용', 엔터 등을 누를 때는 action='press'와 key='enter'를 사용하세요."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["type", "press", "hotkey"]},
                        "text": {"type": "string", "description": "타이핑할 실제 문장 또는 단축키(예: ctrl,c)"},
                        "key": {"type": "string", "description": "누를 단일 키 이름 (예: enter, win, esc)"}
                    },
                    "required": ["action"]
                }
            }
        }
    ]

    tool_descriptions_str = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}" 
        for t in tools
    )

    # [수정 2] 시스템 프롬프트에 JSON 정답지(앵커링) 박제
    messages: List[Dict[str, Any]] = [
        {
            "role": "system", 
            "content": (
                f"당신은 로컬 시스템 제어 및 장기 기억을 보유한 AI 에이전트입니다. 현재 시간: {current_time}\n\n"
                "[사용 가능한 도구 목록]\n"
                f"{tool_descriptions_str}\n\n"
                "[도구 호출 프로토콜]\n"
                "도구를 호출할 때는 반드시 아래의 JSON 스키마를 엄격히 준수하여 ```json 블록으로 출력하세요.\n"
                "- write_memory 예시: {\"name\": \"write_memory\", \"arguments\": {\"key\": \"제목\", \"value\": \"내용\"}}\n"
                "- search_web 예시: {\"name\": \"search_web\", \"arguments\": {\"query\": \"검색어\"}}\n\n"
                "[기억 관리]\n"
                "최신 5턴의 대화만 RAM에 유지됩니다. 오래된 기록은 'fetch_archived_memory'를 통해 DB에서 불러올 수 있습니다.\n\n"
                "[핵심 지침]\n"
                "1. 한국어로 답변하고, 도구 호출은 ```json 블록을 사용하세요.\n"
                "2. 도구의 인자 규격(key, value 등)을 엄격히 준수하세요.\n"
                "3. 모든 작업은 순차적으로 진행하며, 한 번에 하나의 도구만 호출하세요.\n"
                "4. 도구 호출 후 '잠시 기다려달라'거나 '결과가 아직 안 왔다'는 변명을 절대 하지 마세요. 도구 결과는 즉시 제공되므로, 결과를 읽는 즉시 다음 도구를 호출하거나 답변하세요.\n"
                "5. [다중 작업 검증 루틴]: 사용자가 여러 작업을 지시한 경우, 한 작업이 끝났다고 대화를 멈추지 마세요. 도구 실행 후 반드시 <think> 속마음에서 '전체 지시사항 검토 -> 완료된 작업 -> 미실행 작업' 순으로 체크리스트를 점검하고, 미실행 작업이 있다면 즉시 다음 도구를 연속해서 호출하세요.\n"
                "6. [자가 디버깅 루틴]: 에러가 발생하면 절대 포기하거나 사용자에게 떠넘기지 말고, 코드를 수정하여 즉시 재호출하세요.\n"
                "7. [코딩 작업 지시서(Plan)]: 복잡한 코딩 요청을 받으면, 다짜고짜 코드를 짜지 말고 반드시 메모장이나 텍스트 파일에 '작업 계획서(TODO List)'를 먼저 작성하세요.\n"
                "8. [환경 파악]: 코드를 짜기 전에 'run_shell_command'를 통해 현재 디렉토리 구조나 파이썬 환경을 먼저 확인하는 습관을 들이세요."
            )
        }
    ]

    recent_history = fetch_from_sqlite(6)
    if recent_history:
        messages.extend(recent_history)
        print("   [시스템] 📦 이전 대화 기록이 복원되었습니다. (자연스럽게 대화를 이어가세요!)")

    print("=" * 60); print(" 🚀 MOP 하이브리드 메모리 시스템 가동!"); print("=" * 60)

    while True:
        try: user_input = input("\n👤 사용자: ").strip()
        except (KeyboardInterrupt, EOFError): break
        if not user_input: continue
        if user_input.lower() in ['quit', 'exit']: break
        if user_input.lower() == 'clear': messages = [messages[0]]; print("🔄 초기화됨."); continue

        # 👇 [추가] -f 플래그 감지 및 처리 로직
        auto_approve = False
        if user_input.endswith("-f") or user_input.endswith("-F"):
            auto_approve = True
            user_input = user_input[:-2].strip() # AI가 헷갈리지 않게 명령어에서 '-f' 텍스트는 제거
            print("   [시스템] ⚡ '-f' 플래그 감지: 이번 턴의 하드웨어 보안 승인이 자동으로 패스됩니다.")
            
        archive_to_sqlite("user", user_input)
        messages.append({"role": "user", "content": user_input})
        
        # [수정] 턴 수(20개) + 전체 글자 수(약 12,000자) 이중 방어망 구축
        # n_ctx 8192(약 16,000자)를 넘지 않도록 여유 버퍼를 둡니다.
        current_context_length = sum(len(str(msg.get('content', ''))) for msg in messages)
        
        if len(messages) > 20 or current_context_length > 12000:
            print(f"   [시스템] 🧹 메모리 최적화 시작 (현재 메시지: {len(messages)}개, 글자 수: {current_context_length}자)")
            end_idx = 2
            
            # 다음 'user' 메시지가 나올 때까지 탐색하되, 무한 루프 방지를 위해 조건 추가
            while end_idx < len(messages) and messages[end_idx].get('role') != 'user':
                end_idx += 1
                
            if end_idx < len(messages): # 안전망
                old_turns = messages[1:end_idx]
                for msg in old_turns:
                    if msg.get('role') in ['user', 'assistant'] and msg.get('content'):
                        archive_to_sqlite(msg['role'], msg['content'])
                
                messages = [messages[0]] + messages[end_idx:]
                print("   [시스템] 📦 오래된 대화가 안전하게 SQLite로 이관 및 RAM 정리 완료.")

        
        print("   [시스템] 📦 오래된 대화가 안전하게 SQLite로 이관 및 RAM 정리 완료.") # (기존 코드)

        consecutive_error_count = 0  # 👈 [추가] 턴마다 에러 횟수를 초기화합니다.
        
        while True:
            print("🤖 AI: ", end="", flush=True) # (기존 코드 계속...)
            stream = cast(Iterator[Dict[str, Any]], llm.create_chat_completion(
                messages=cast(Any, messages), tools=cast(Any, tools), stream=True, 
                temperature=0.1, repeat_penalty=1.2, max_tokens=2048
            ))

            assistant_content = ""; tc_name = ""; tc_args = ""; is_tool_call = False
            for chunk in stream:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "tool_calls" in delta and delta["tool_calls"]:
                        is_tool_call = True
                        tc_chunk = delta["tool_calls"][0]
                        if "function" in tc_chunk:
                            if "name" in tc_chunk["function"]: tc_name += tc_chunk["function"]["name"]
                            if "arguments" in tc_chunk["function"]: tc_args += tc_chunk["function"]["arguments"]
                    content = delta.get("content")
                    if content: print(content, end="", flush=True); assistant_content += content
            
            print()

            if not is_tool_call:
                clean_text = assistant_content.replace("&lt;", "<").replace("&gt;", ">") \
                                              .replace("&#34;", '"').replace("&quot;", '"') \
                                              .replace("&#39;", "'").replace("&apos;", "'")
                scan_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
                
                # 2. json_match 변수 정의: 마크다운 JSON 블록을 정규표현식으로 찾습니다. (배열/객체 모두 지원)
                json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', clean_text, re.DOTALL | re.IGNORECASE)
                
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        if isinstance(parsed, list): parsed = parsed[0]
                        if isinstance(parsed, dict):
                            tc_name_fallback = parsed.get("name") or parsed.get("tool") or ""
                            if tc_name_fallback:
                                is_tool_call = True; tc_name = tc_name_fallback
                                raw_args = parsed.get("arguments") or parsed.get("args") or parsed.get("parameters", parsed)
                                tc_args = json.dumps(raw_args)
                                print(f"   [시스템] 🛟 '{tc_name}' 구출 성공!")
                    except Exception as e:
                        # [핵심 수정] JSON이 깨졌을 때 침묵하지 않고 AI에게 에러 피드백을 강제로 넘김
                        is_tool_call = True
                        tc_name = "json_syntax_error"
                        tc_args = "{}"
                        print(f"   [시스템] 🚨 JSON 문법 에러 감지! AI에게 재작성을 지시합니다.")
                archive_to_sqlite("assistant", assistant_content.strip())

            if is_tool_call:
                print(f"   [시스템] 🛠️ 도구 호출: {tc_name}")
                messages.append({
                    "role": "assistant", "content": assistant_content.strip() or None,
                    "tool_calls": [{"id": "call_id", "type": "function", "function": {"name": tc_name, "arguments": tc_args}}]
                })
                
                try:
                    args_dict = json.loads(tc_args); tool_result = ""

                    if tc_name == "json_syntax_error":
                        tool_result = (
                            "오류: 출력하신 JSON 블록의 문법이 깨졌습니다. "
                            "파이썬 코드를 'code' 인자에 넣을 때 쌍따옴표(\") 안에 쌍따옴표를 겹쳐 쓰지 마시고(홑따옴표 사용 권장), "
                            "절대 실제 줄바꿈을 쓰지 말고 '\\n'으로 처리하세요. 코드를 수정하여 다시 호출하세요."
                        )
                    
                    elif tc_name == "write_memory":
                        if "key" not in args_dict or "value" not in args_dict:
                            # [수정 3-2] 에러 발생 시 올바른 템플릿을 AI에게 가르쳐 줌
                            tool_result = "오류: 'key'와 'value' 인자가 누락되었습니다. 반드시 {'key': '제목', 'value': '내용'} 형식으로 다시 호출하세요."
                        else:
                            tool_result = execute_skill_safely(["python", "./skills/memory_tools.py", "--action", "write", "--key", args_dict["key"], "--value", args_dict["value"]])
                    
                    elif tc_name == "search_chat_history":
                        keyword = args_dict.get("keyword", "")
                        if not keyword:
                            tool_result = "오류: 검색할 'keyword' 인자가 필요합니다."
                        else:
                            tool_result = search_history_db(keyword)
                    
                    elif tc_name == "search_web":
                        tool_result = search_web(args_dict.get("query", ""))
                    
                    elif tc_name == "run_python_snippet":
                        tool_result = execute_skill_safely(["python", "./skills/system_tools.py", "--action_type", "python", "--code", args_dict.get("code", "")])
                    
                    elif tc_name == "manage_packages":
                        action = args_dict.get("action"); pkg = args_dict.get("package_name", "")
                        tool_result = execute_skill_safely(["python", "./skills/system_tools.py", "--action_type", "pip", "--action", action, "--packages", pkg])
                    
                    elif tc_name == "view_file":
                        tool_result = execute_skill_safely(["python", "./skills/file_tools.py", "--action", "view", "--path", args_dict.get("file_path", "")])

                    elif tc_name == "list_memories":
                        tool_result = execute_skill_safely(["python", "./skills/memory_tools.py", "--action", "list"])
                    
                    elif tc_name == "read_memory":
                        tool_result = execute_skill_safely(["python", "./skills/memory_tools.py", "--action", "read", "--key", args_dict.get("key", "")])

                    elif tc_name == "run_shell_command":
                        cmd = args_dict.get("command", "")
                        # Windows 환경을 고려하여 cmd /c 로 실행합니다.
                        tool_result = execute_skill_safely(["cmd", "/c", cmd])
                    
                    elif tc_name == "edit_file":
                        f_path = args_dict.get("file_path", "")
                        s_str = args_dict.get("search_string", "")
                        r_str = args_dict.get("replace_string", "")
                        try:
                            with open(f_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            if s_str not in content:
                                tool_result = "오류: 'search_string'으로 지정한 문자열을 파일에서 찾을 수 없습니다. 오타가 없는지 확인하세요."
                            else:
                                new_content = content.replace(s_str, r_str)
                                with open(f_path, 'w', encoding='utf-8') as f:
                                    f.write(new_content)
                                tool_result = f"성공: '{f_path}' 파일의 내용이 성공적으로 교체되었습니다."
                        except Exception as e:
                            tool_result = f"오류: 파일 수정 실패 - {str(e)}"

                    elif tc_name in ["control_mouse", "control_keyboard"]:
                        # 👇 [수정] 자동 승인 플래그(auto_approve)에 따른 분기 처리
                        if auto_approve:
                            print(f"⚠️  [자동 승인] 하드웨어 제어 프리패스: {tc_name}")
                            user_consent = 'y'
                        else:
                            print(f"⚠️  [보안 승인 대기] 하드웨어 제어 요청: {tc_name}")
                            user_consent = input("   👉 허용하시겠습니까? (y/n): ").lower().strip()
                        
                        if user_consent == 'y':
                            script_path = os.path.join(".", "skills", "computer_tools.py")
                            cli_args = ["python", script_path, "--device", "mouse" if "mouse" in tc_name else "keyboard", "--action", args_dict.get("action", "")]
                            if "x" in args_dict: cli_args.extend(["--x", str(args_dict["x"])])
                            if "y" in args_dict: cli_args.extend(["--y", str(args_dict["y"])])
                            if "text" in args_dict: cli_args.extend(["--text", args_dict["text"]])
                            if "key" in args_dict: cli_args.extend(["--key", args_dict["key"]])
                            tool_result = execute_skill_safely(cli_args)
                        else: 
                            tool_result = "사용자가 실행을 거부했습니다."

                    else: tool_result = f"알 수 없는 도구: {tc_name}"

                except Exception as e: tool_result = f"에러: {e}"

                # ... (기존) try/except 및 [디버그] 결과 출력 ...
                print(f"   [디버그] 결과: {tool_result[:100]}...") # 콘솔 도배 방지
                
                # 👇 [새로 추가하는 Air-Style Context Management] 👇
                # 도구 결과값이 비정상적으로 길 경우, 컨텍스트 윈도우 폭발(OOM)을 막기 위해 
                # 강제로 앞뒤 텍스트만 남기고 중간을 압축(Truncate)합니다.
                MAX_CHAR_LIMIT = 6000 # 약 3000 토큰에 해당 (n_ctx 8192 기준 아주 안전한 수치)
                
                if len(tool_result) > MAX_CHAR_LIMIT:
                    print(f"   [시스템] ⚠️ 도구 결과가 너무 깁니다! 메모리 보호를 위해 중간 텍스트를 압축합니다. (길이: {len(tool_result)})")
                    half_limit = MAX_CHAR_LIMIT // 2
                    tool_result = tool_result[:half_limit] + "\n\n... [시스템 경고: 데이터가 너무 길어 메모리 최적화를 위해 중략되었습니다] ...\n\n" + tool_result[-half_limit:]

                tool_result_lower = tool_result.lower()
                error_keywords = ["error", "exception", "traceback", "오류", "실패", "fail", "nonetype"]
                is_error = any(kw in tool_result_lower for kw in error_keywords)

                if is_error:
                    consecutive_error_count += 1
                    
                    # [서킷 브레이커 발동] 3연속 에러 시 강제 중단
                    if consecutive_error_count >= 3:
                        print(f"\n   [시스템] 🚨 연속 에러 3회 감지! 무한 루프를 막기 위해 서킷 브레이커가 발동되었습니다.")
                        sos_msg = "차니님, 코드를 여러 번 수정하며 시도했지만 동일한 에러의 늪에 빠진 것 같습니다. 제가 생각한 방향이 틀렸을 수 있으니 힌트를 주시거나 코드를 직접 확인해 주시겠어요?"
                        print(f"🤖 AI: {sos_msg}")
                        archive_to_sqlite("assistant", sos_msg)
                        messages.append({"role": "assistant", "content": sos_msg})
                        break # 도구 실행 루프를 강제 종료하고 사용자의 입력을 기다립니다.
                    
                    # [코딩 에이전트 전용: 강력한 디버깅 및 회피 프로토콜]
                    enforced_result = (
                        f"🚨 [도구 실행 실패 - 에러 발생! (현재 연속 에러 {consecutive_error_count}회/최대 3회)]\n{tool_result}\n\n"
                        "---[시스템 디버그 긴급 지시 (Coding Agent Protocol)]---\n"
                        "1. (반복 금지): 절대 직전과 완벽히 똑같은 코드나 논리를 제출하지 마세요.\n"
                        "2. (우회로 탐색): 모듈 import 에러나 경로 문제라면, 'run_shell_command'로 먼저 환경을 확인하거나 완전히 다른 라이브러리(접근 방식)를 사용하세요.\n"
                        "3. (검색 강제): 에러 원인을 모르겠다면 뇌피셜로 짜지 말고 즉시 'search_web'으로 에러 메시지를 구글링하여 해결책을 찾으세요.\n"
                        "4. 위 지침에 따라 해결책을 찾아 다시 도구를 호출하세요. 사용자에게 말을 걸지 마세요."
                    )
                else:
                    consecutive_error_count = 0  # 성공하면 에러 카운터 초기화
                    enforced_result = (
                        f"[도구 실행 결과]\n{tool_result}\n\n"
                        "---[시스템 자가 점검 지시(Task Verification)]---\n"
                        "1. 위 작업이 성공했습니다. 아직 실행하지 않은 미완료 작업이 있다면 즉시 다음 도구를 호출하세요.\n"
                        "2. 모든 작업이 완벽히 끝났음이 확실할 때만 최종 답변을 출력하세요."
                    )

                messages.append({"role": "tool", "tool_call_id": "call_id", "name": tc_name, "content": enforced_result})
                continue
            break

        # [추가] 메모리 정리
        gc.collect()

if __name__ == "__main__": main()