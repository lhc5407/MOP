"""
MOP.py - Hybrid Memory & Context Optimized Agent
상단부: SQLite 연동, 5턴 아카이빙 로직 및 도구 정의
"""

import sys
import os
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
    result = subprocess.run(cli_args, capture_output=True)
    def decode_bytes(raw_bytes: bytes) -> str:
        if not raw_bytes: return ""
        try: return raw_bytes.decode('utf-8')
        except UnicodeDecodeError: return raw_bytes.decode('cp949', errors='replace')
    out_str = decode_bytes(result.stdout).strip()
    err_str = decode_bytes(result.stderr).strip()
    return out_str if result.returncode == 0 else err_str

def load_gguf_with_kv_quantization(model_path: str) -> Llama:
    if not os.path.exists(model_path): raise FileNotFoundError(f"모델 파일 없음: {model_path}")
    print(f"[{model_path}] 모델 로딩 중...")
    return Llama(
        model_path=model_path, n_threads=10, n_gpu_layers=15, n_ctx=8192,
        type_k=llama_cpp.GGML_TYPE_Q4_0, type_v=llama_cpp.GGML_TYPE_Q4_0,
        flash_attn=True, verbose=False, chat_format="chatml-function-calling"
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
                "5. [다중 작업 검증 루틴]: 사용자가 여러 작업을 지시한 경우, 한 작업이 끝났다고 대화를 멈추지 마세요. 도구 실행 후 반드시 <think> 속마음에서 '전체 지시사항 검토 -> 완료된 작업 -> 미실행 작업' 순으로 체크리스트를 점검하고, 미실행 작업이 있다면 즉시 다음 도구를 연속해서 호출하세요."
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
        
        # [수정된 스마트 아카이빙 로직] 도구 호출 사이클이 끊기지 않도록 '사용자 질문' 단위로 묶어서 잘라냅니다.
        # Qwen 9B의 n_ctx가 8192이므로 여유를 두어 20개 메시지(약 5~7턴)를 기준으로 잡습니다.
        if len(messages) > 20:
            end_idx = 2
            while end_idx < len(messages) and messages[end_idx].get('role') != 'user':
                end_idx += 1
            
            # 가장 오래된 한 사이클(질문 -> 도구호출 -> 도구결과 -> 최종답변)을 추출
            old_turns = messages[1:end_idx]
            
            # DB에는 영양가 있는 'user' 질문과 'assistant'의 실제 텍스트 답변만 저장합니다.
            for msg in old_turns:
                if msg.get('role') in ['user', 'assistant'] and msg.get('content'):
                    archive_to_sqlite(msg['role'], msg['content'])
            
            # RAM에서 해당 사이클을 통째로 안전하게 도려냅니다.
            messages = [messages[0]] + messages[end_idx:]
            print("   [시스템] 📦 오래된 대화 1사이클이 구조 손상 없이 SQLite DB로 이관되었습니다.")

        
        while True:
            print("🤖 AI: ", end="", flush=True)
            stream = cast(Iterator[Dict[str, Any]], llm.create_chat_completion(
                messages=cast(Any, messages), tools=cast(Any, tools), stream=True, 
                temperature=0.1, repeat_penalty=1.2, max_tokens=1024
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
                                # [수정 3-1] AI가 "args"라고 창조해서 출력해도 알아서 "arguments"로 인식하게 만듦
                                raw_args = parsed.get("arguments") or parsed.get("args") or parsed.get("parameters", parsed)
                                tc_args = json.dumps(raw_args)
                                print(f"   [시스템] 🛟 '{tc_name}' 구출 성공!")
                    except: pass
                archive_to_sqlite("assistant", assistant_content.strip())

            if is_tool_call:
                print(f"   [시스템] 🛠️ 도구 호출: {tc_name}")
                messages.append({
                    "role": "assistant", "content": assistant_content.strip() or None,
                    "tool_calls": [{"id": "call_id", "type": "function", "function": {"name": tc_name, "arguments": tc_args}}]
                })
                
                try:
                    args_dict = json.loads(tc_args); tool_result = ""

                    if tc_name == "write_memory":
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

                print(f"   [디버그] 결과: {tool_result}")
                
                # 👇 여기 아래 코드를 추가/수정합니다! 👇
                enforced_result = (
                    f"[도구 실행 결과]\n{tool_result}\n\n"
                    "---[시스템 자가 점검 지시(Task Verification)]---\n"
                    "1. 방금 수행한 작업 외에, 사용자의 최초 질문에서 **아직 실행하지 않은 미완료 작업**이 남아있는지 확인하세요.\n"
                    "2. 미완료 작업이 있다면 절대 대화를 종료하거나 사용자에게 묻지 말고, **즉시 이어서 다음 도구를 호출**하세요.\n"
                    "3. 사용자가 요청한 '모든' 작업이 완벽히 끝났음이 확실할 때만 최종 요약 답변을 출력하고 차례를 넘기세요."
                )

                messages.append({"role": "tool", "tool_call_id": "call_id", "name": tc_name, "content": enforced_result})
                continue
            break

if __name__ == "__main__": main()