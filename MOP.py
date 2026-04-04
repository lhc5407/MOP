"""
MOP.py - My Ollama Python (Function Calling + Pylance Safe)
Qwen3.5 GGUF 모델 전용 + 네이티브 KV 캐시 양자화 + 자율 웹 검색 + 스킬 연동
"""

import sys
import os
import json
import subprocess  # [수정됨] 누락되었던 subprocess 임포트 추가
from typing import List, Dict, Any, cast, Iterator
from duckduckgo_search import DDGS

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    print("오류: llama-cpp-python 라이브러리가 설치되어 있지 않습니다.")
    sys.exit(1)


def search_web(query: str) -> str:
    """DuckDuckGo를 이용해 웹 검색을 수행하고 결과를 반환하는 실제 함수"""
    print(f"\n   [시스템] 🌐 인터넷에서 '{query}' 검색 중...", end="", flush=True)
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "검색 결과가 없습니다."
            
        context = ""
        for i, r in enumerate(results, 1):
            context += f"[{i}] 제목: {r['title']}\n내용: {r['body']}\n\n"
        print(" 완료!\n")
        return context
    except Exception as e:
        print(" 실패!\n")
        return f"검색 중 오류가 발생했습니다: {e}"


def load_gguf_with_kv_quantization(model_path: str, num_threads: int = 10, kv_quant_type: int = llama_cpp.GGML_TYPE_Q4_0) -> Llama:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    print(f"[{model_path}] 모델 적재 중...")
    
    model = Llama(
        model_path=model_path,
        n_threads=num_threads,
        n_gpu_layers=20,     # 환경에 맞춰 조절
        n_ctx=2048,
        type_k=kv_quant_type,
        type_v=kv_quant_type,
        flash_attn=True,     # V 캐시 양자화를 위한 필수 옵션
        verbose=False
    )
    print("✅ 모델 로딩 완료!\n")
    return model


def main() -> None:
    # 1. 모델 경로 설정 (본인 환경에 맞게 수정)
    MODEL_PATH = r"C:\Users\lhc54\Downloads\Qwen3.5-9B-Q4_K_M.gguf"
    
    try:
        llm = load_gguf_with_kv_quantization(MODEL_PATH)
    except Exception as e:
        print(f"오류 발생: {e}")
        return

    # 2. AI에게 알려줄 도구(Tool) 명세서 작성
    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "최신 정보, 뉴스, 날씨, 또는 학습 데이터에 없는 사실에 답해야 할 때 웹 검색을 수행합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색 엔진에 입력할 핵심 키워드"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_python_snippet",
                "description": "검증이나 수학적 계산이 필요할 때 파이썬 코드를 실행하고 터미널 출력(print) 결과를 확인합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "실행할 완전한 파이썬 소스 코드"}},
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "manage_packages",
                "description": "파이썬 pip 패키지를 설치, 삭제 또는 목록을 확인합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["install", "uninstall", "list"]},
                        "packages": {"type": "string", "description": "패키지 이름 (공백으로 구분, 예: numpy pandas)"}
                    },
                    "required": ["action"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_terminal_command",
                "description": "시스템 터미널 명령어(dir, ls, git 등)를 실행하고 결과를 반환합니다. 파괴적인 명령어는 사용하지 마세요.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "실행할 터미널 명령어"}},
                    "required": ["command"]
                }
            }
        }
    ]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "당신은 한국어로 답변하는 유용한 AI입니다. 모르는 최신 정보는 반드시 search_web 함수를 호출하여 검색하세요."}
    ]

    print("=" * 60)
    print(" 🚀 MOP (My Ollama Python) 터미널 - Function Calling 활성화!")
    print("    - 일반적인 질문을 던지면, AI가 필요에 따라 스스로 인터넷을 검색합니다.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 사용자: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit']:
            break
        if user_input.lower() == 'clear':
            messages = [{"role": "system", "content": "당신은 한국어로 답변하는 유용한 AI입니다. 모르는 최신 정보는 반드시 search_web 함수를 호출하여 검색하세요."}]
            print("🔄 대화 기록 초기화됨.")
            continue

        messages.append({"role": "user", "content": user_input})
        
        while True:
            print("🤖 AI: ", end="", flush=True)

            stream = cast(Iterator[Dict[str, Any]], llm.create_chat_completion(
                messages=cast(Any, messages),
                tools=cast(Any, tools),      # [수정됨] Pylance 타입 에러 우회를 위해 cast 적용
                stream=True,
                temperature=0.7,
                max_tokens=1024
            ))

            assistant_content = ""
            tc_name = ""
            tc_args = ""
            tc_id = "call_auto_id"
            is_tool_call = False

            # 스트리밍 청크(조각) 처리기
            for chunk in stream:
                choices: List[Dict[str, Any]] = chunk.get("choices", [])
                if choices and len(choices) > 0:
                    delta: Dict[str, Any] = choices[0].get("delta", {})
                    
                    if "tool_calls" in delta and delta["tool_calls"]:
                        is_tool_call = True
                        tc_chunk = delta["tool_calls"][0]
                        if "id" in tc_chunk and tc_chunk["id"]: 
                            tc_id = tc_chunk["id"]
                        if "function" in tc_chunk:
                            if "name" in tc_chunk["function"] and tc_chunk["function"]["name"]:
                                tc_name += tc_chunk["function"]["name"]
                            if "arguments" in tc_chunk["function"] and tc_chunk["function"]["arguments"]:
                                tc_args += tc_chunk["function"]["arguments"]
                    
                    content = delta.get("content")
                    if content is not None:
                        text = str(content)
                        print(text, end="", flush=True)
                        assistant_content += text
            
            print() 

            # --- 스트리밍 종료 후 판단 ---
            if is_tool_call:
                print(f"\n   [시스템] 🛠️ AI가 도구를 호출했습니다: {tc_name}(...생략...)")
                messages.append({
                    "role": "assistant", "content": None, 
                    "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": tc_name, "arguments": tc_args}}]
                })
                
                try:
                    args_dict = json.loads(tc_args)
                    system_tools = ["execute_terminal_command", "run_python_snippet", "manage_packages"]
                    
                    # 1. 시스템 도구인 경우: 승인 절차 + system_tools.py 외부 호출
                    if tc_name in system_tools:
                        print(f"⚠️  [보안 승인 대기] AI가 시스템 권한을 요청했습니다.")
                        confirm = input("   👉 실행을 허용하시겠습니까? (y/n): ").lower().strip()
                        
                        if confirm != 'y':
                            tool_result = "사용자가 보안상의 이유로 해당 명령어 실행을 거부했습니다."
                        else:
                            script_path = os.path.join(".", "skills", "system_tools.py")
                            cli_args = ["python", script_path]
                            
                            if tc_name == "execute_terminal_command":
                                cli_args.extend(["--action_type", "terminal", "--command", args_dict.get("command", "")])
                            elif tc_name == "run_python_snippet":
                                cli_args.extend(["--action_type", "python", "--code", args_dict.get("code", "")])
                            elif tc_name == "manage_packages":
                                cli_args.extend(["--action_type", "pip", "--action", args_dict.get("action", "")])
                                if "packages" in args_dict:
                                    cli_args.extend(["--packages", args_dict.get("packages", "")])
                            
                            print("   ⚙️ 백그라운드 실행 중...", end="", flush=True)
                            result = subprocess.run(cli_args, capture_output=True, text=True, check=False)
                            tool_result = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
                            print(" 완료!")

                    # 2. 기타 기존 도구 (search_web 등)
                    elif tc_name == "search_web":
                        tool_result = search_web(args_dict.get("query", ""))
                    else:
                        tool_result = "알 수 없는 스킬입니다."
                        
                except Exception as e:
                    tool_result = f"오류 발생: {e}"
                    
                messages.append({"role": "tool", "tool_call_id": tc_id, "name": tc_name, "content": tool_result})
                continue 
            
            else:
                messages.append({"role": "assistant", "content": assistant_content})
                break

if __name__ == "__main__":
    main()