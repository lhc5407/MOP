import html
import customtkinter as ctk
from tkinter import filedialog, messagebox
import datetime
import threading
import json
import os
import gc
import sqlite3
import subprocess
import re
import sys
import traceback
import threading
import time

# 👇 [패치] PyInstaller Windowed 모드에서 sys.stdout/stderr가 None이 되어 isatty() 호출 시 발생하는 크래시 방지 및 이모지(UTF-8) 출력 시 cp949 인코딩 에러 방지
class DummyOutput:
    encoding = 'utf-8'
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass
    def isatty(self): return False

if sys.stdout is None:
    sys.stdout = DummyOutput()
else:
    try:
        sys.stdout.reconfigure(encoding='utf-8') # type: ignore
    except Exception:
        pass

if sys.stderr is None:
    sys.stderr = DummyOutput()
else:
    try:
        sys.stderr.reconfigure(encoding='utf-8') # type: ignore
    except Exception:
        pass

import llama_cpp
from llama_cpp import Llama
from typing import List, Dict, Any, cast, Iterator
from ddgs import DDGS
from res.mop_memory import VectorMemoryManager

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    print("오류: llama-cpp-python 라이브러리가 설치되어 있지 않습니다.")
    import sys; sys.exit(1)


def get_resource_path(relative_path: str) -> str:
    relative_path = relative_path.replace("/", os.sep).replace("\\", os.sep)
    if getattr(sys, "frozen", False):
        base_path = getattr(sys, "_MEIPASS")
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


def get_local_path(relative_path: str) -> str:
    relative_path = relative_path.replace("/", os.sep).replace("\\", os.sep)
    return os.path.abspath(os.path.join(os.getcwd(), relative_path))


def resolve_path(relative_path: str) -> str:
    local_path = get_local_path(relative_path)
    if os.path.exists(local_path):
        return local_path
    if getattr(sys, "frozen", False):
        bundled_path = get_resource_path(relative_path)
        if os.path.exists(bundled_path):
            return bundled_path
    return local_path


def get_startup_log_path() -> str:
    return os.path.join(os.getcwd(), "MOP_startup.log")


def write_startup_log(message: str):
    try:
        with open(get_startup_log_path(), "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


# --- [기본 테마 설정] ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class LoadingWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("MOP 로딩 중...")
        self.geometry("420x180")
        self.resizable(False, False)
        self.label = ctk.CTkLabel(self, text="구성요소 초기화 중...", anchor="center")
        self.label.pack(pady=20)
        self.progress = ctk.CTkProgressBar(self, width=360)
        self.progress.pack(pady=10)
        self.progress.set(0.0)
        self.center_window()
        self.attributes("-topmost", True)

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def update_progress(self, value: float, text: str):
        self.progress.set(value)
        self.label.configure(text=text)
        self.update()

# --- [1. MOP AI 엔진 클래스 (핵심 로직 & 도구 모음)] ---
class MOPEngine:
    def __init__(self, db_path=None):
        self.db_path = db_path if db_path else get_local_path("res/skills/chat_history.db")
        self.llm = None
        self.messages = []
        # 👇 [추가] 백그라운드 작업들을 관리할 저장소와 카운터
        self.background_tasks = {}
        self.task_counter = 0 
        self.parallel_sub_agents = {}
        self.sub_agent_counter = 0
        self.custom_system_prompt = self.get_default_system_prompt()
        self.init_db()
        self.vdb = VectorMemoryManager()
        self.migrate_old_memory()

    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS history 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       role TEXT, content TEXT, timestamp DATETIME)''')
        conn.commit()
        conn.close()

    def migrate_old_memory(self):
        """구형 .agent_memory 파일을 읽어 Vector DB로 이전하고 백업합니다."""
        old_mem_path = get_local_path("res/skills/.agent_memory")
        if os.path.exists(old_mem_path):
            print("🔄 구형 메모리 파일(.agent_memory)을 Vector DB로 마이그레이션 합니다...")
            try:
                with open(old_mem_path, "r", encoding="utf-8") as f:
                    # 예전 메모리가 JSON(dict) 형태라고 가정
                    try:
                        old_data = json.load(f)
                        for k, v in old_data.items():
                            self.vdb.add_memory(f"[이전 기억: {k}] {v}")
                    except json.JSONDecodeError:
                        # JSON이 아니라 단순 텍스트로 저장되어 있었을 경우
                        f.seek(0)
                        text = f.read().strip()
                        if text:
                            self.vdb.add_memory(text)
                            
                # 이사가 끝난 구형 파일은 확장자를 바꿔서 비활성화
                os.rename(old_mem_path, old_mem_path + ".backup")
                print("✅ 성공: 구형 메모리 마이그레이션이 완료되었습니다.")
            except Exception as e:
                print(f"⚠️ 구형 메모리 마이그레이션 실패: {e}")

    def archive_to_sqlite(self, role, content):
        # 👇 [핵심 패치 3] timeout=10.0을 추가하여 병렬 DB 접근 에러 방지
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cur = conn.cursor()
        cur.execute("INSERT INTO history (role, content, timestamp) VALUES (?, ?, ?)",
                    (role, content, datetime.datetime.now()))
        conn.commit()
        conn.close()

    def fetch_from_sqlite(self, count=10):
        # 👇 [핵심 패치 3] timeout=10.0을 추가하여 병렬 DB 접근 에러 방지
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM history ORDER BY id DESC LIMIT ?", (count,))
        rows = cur.fetchall()
        conn.close()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def search_history_db(self, keyword: str, limit: int = 5) -> str:
        # 👇 [핵심 패치 3] timeout=10.0을 추가하여 병렬 DB 접근 에러 방지
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cur = conn.cursor()
        cur.execute("SELECT role, content, timestamp FROM history WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?", (f"%{keyword}%", limit))
        rows = cur.fetchall()
        conn.close()
        if not rows: return f"'{keyword}'에 대한 과거 대화 기록이 없습니다."
        result = f"[{keyword} 관련 과거 대화 검색 결과]\n"
        for r, c, t in reversed(rows): result += f"[{t[:16]}] {r.upper()}: {c}\n"
        return result

    def search_web(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            if not results: return "검색 결과가 없습니다."
            return "".join([f"[{i}] 제목: {r['title']}\n내용: {r['body']}\n\n" for i, r in enumerate(results, 1)])
        except Exception as e: return f"검색 중 오류 발생: {e}"

    def execute_skill_safely(self, cli_args: List[str]) -> str:
        try:
            result = subprocess.run(cli_args, capture_output=True, timeout=30)
            def decode_bytes(raw_bytes: bytes) -> str:
                if raw_bytes is None: return ""
                try: return raw_bytes.decode('utf-8')
                except UnicodeDecodeError: return raw_bytes.decode('cp949', errors='replace')
            out_str = decode_bytes(result.stdout).strip() if result.stdout else ""
            err_str = decode_bytes(result.stderr).strip() if result.stderr else ""
            if result.returncode != 0: return f"실행 오류 (코드 {result.returncode}): {err_str or out_str}"
            return out_str or "실행 성공 (반환값 없음)"
        except Exception as e: return f"시스템 실행 에러: {str(e)}"

    def start_background_task(self, command: str) -> str:
        """명령어를 백그라운드에서 비동기로 실행합니다."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        try:
            # Popen을 사용하여 프로그램이 끝날 때까지 기다리지 않고 즉시 제어권을 넘김
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
            self.background_tasks[task_id] = {
                "process": process, 
                "command": command, 
                "start_time": datetime.datetime.now().strftime("%H:%M:%S")
            }
            return f"✅ 성공: 백그라운드 작업 '{task_id}'가 시작되었습니다. (명령어: {command})\n에이전트는 즉시 다른 도구를 사용하거나 작업을 진행할 수 있습니다. 작업이 끝났는지 확인하려면 나중에 'check_task_status' 도구를 사용하세요."
        except Exception as e:
            return f"❌ 오류: 백그라운드 작업 시작 실패 - {e}"

    def check_task_status(self, task_id: str) -> str:
        """백그라운드 작업의 완료 여부를 확인하고 결과를 반환합니다."""
        if task_id not in self.background_tasks:
            return f"❌ 오류: '{task_id}' 작업을 찾을 수 없습니다. 이미 완료되어 결과를 확인했거나 잘못된 ID입니다."
        
        task_info = self.background_tasks[task_id]
        process = task_info["process"]
        
        # poll()은 프로세스가 끝났으면 return code를, 아직 실행 중이면 None을 반환합니다.
        ret_code = process.poll()
        
        if ret_code is None:
            return f"⏳ 상태: '{task_id}' (시작: {task_info['start_time']}) 작업이 아직 실행 중입니다. 다른 작업을 먼저 처리하고 오세요."
        else:
            # 작업이 끝났다면 출력 결과를 가져오고 메모리에서 삭제
            stdout, stderr = process.communicate()
            del self.background_tasks[task_id]
            
            result_str = f"✅ 완료: '{task_id}' 작업이 종료되었습니다. (코드 {ret_code})\n[출력 결과]\n{stdout.strip()}\n"
            if stderr.strip():
                result_str += f"[에러 로그]\n{stderr.strip()}"
            return result_str
    
    def run_sub_agent(self, instruction: str, file_path: str = "", input_data: str = "") -> str:
        """서브에이전트를 생성하여 격리된 컨텍스트에서 분석을 수행합니다."""
        if not self.llm: return "오류: LLM이 로드되지 않았습니다."

        data_content = input_data
        
        # 파일 경로가 주어지면 파일을 읽어옴
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data_content += f"\n[파일 내용: {file_path}]\n" + f.read()
            except Exception as e:
                return f"❌ 서브에이전트 오류: 파일 읽기 실패 - {e}"

        if not data_content.strip():
            return "❌ 오류: 분석할 데이터나 파일 내용이 없습니다."

        # 👇 [핵심] 서브에이전트용 독립 컨텍스트 (메인 기억과 완벽히 격리됨)
        sub_system_prompt = (
            "당신은 메인 에이전트를 보조하는 '전문 분석 서브에이전트'입니다.\n"
            "주어진 데이터를 분석하고 지시사항(Instruction)을 정확히 수행하세요.\n"
            "불필요한 인사말이나 과정을 생략하고 '최종 결과와 핵심 요약'만을 명확하고 간결하게 보고하세요."
        )

        sub_messages = [
            {"role": "system", "content": sub_system_prompt},
            # 메모리 용량 보호를 위해 데이터는 최대 40000자로 자릅니다.
            {"role": "user", "content": f"[지시사항]\n{instruction}\n\n[분석할 데이터]\n{data_content[:40000]}"} 
        ]

        try:
            # 👇 [수정] cast를 씌워서 Pylance에게 타입에 대한 확신을 줍니다.
            response = cast(Dict[str, Any], self.llm.create_chat_completion(
                messages=cast(Any, sub_messages),  # 👈 에러 1 해결 (Any 타입으로 강제 변환)
                max_tokens=2048,
                temperature=0.1,  
                stream=False
            ))
            
            # 👈 에러 2 해결 (위에서 Dict라고 못을 박았으므로 안전하게 ['choices'] 접근 가능)
            result_text = response['choices'][0]['message']['content'] 
            return f"🤖 [서킷 격리: 서브에이전트 분석 보고서]\n{result_text}"
            
        except Exception as e:
            return f"❌ 서브에이전트 추론 중 오류 발생: {e}"
    
    def _sub_agent_worker(self, task_id, instruction, file_path, input_data):
        """백그라운드 스레드에서 실제로 기존 서브에이전트 함수를 실행합니다."""
        try:
            # 우리가 이전에 만든 기존 서브에이전트 로직을 그대로 재활용합니다!
            result = self.run_sub_agent(instruction, file_path, input_data)
            self.parallel_sub_agents[task_id]["result"] = result
            self.parallel_sub_agents[task_id]["status"] = "completed"
        except Exception as e:
            self.parallel_sub_agents[task_id]["result"] = f"❌ 서브에이전트 실행 중 오류: {e}"
            self.parallel_sub_agents[task_id]["status"] = "error"

    def delegate_parallel_task(self, instruction: str, file_path: str = "", input_data: str = "") -> str:
        """서브에이전트에게 분석을 위임하고 즉시 task_id만 반환합니다 (비동기 병렬 처리)."""
        if not self.llm: return "오류: LLM이 로드되지 않았습니다."
        
        self.sub_agent_counter += 1
        task_id = f"sub_agent_{self.sub_agent_counter}"
        
        # 작업 상태 등록
        self.parallel_sub_agents[task_id] = {
            "status": "running",
            "result": None,
            "instruction": instruction
        }
        
        # 스레드로 백그라운드 실행 (메인 메모리는 멈추지 않음)
        thread = threading.Thread(target=self._sub_agent_worker, args=(task_id, instruction, file_path, input_data), daemon=True)
        thread.start()
        
        return f"✅ 성공: 서브에이전트 병렬 작업 '{task_id}'가 백그라운드에서 시작되었습니다. (지시: {instruction[:30]}...)\n기다리지 말고 즉시 다른 병렬 작업을 추가로 지시하거나, 모든 지시가 끝났다면 'join_sub_agent_results' 도구를 사용하여 결과가 나올 때까지 대기하세요."

    def join_sub_agent_results(self, task_ids: list) -> str:
        """여러 개의 병렬 서브에이전트 작업이 모두 완료될 때까지 기다렸다가 결과를 통합하여 반환합니다."""
        if not task_ids:
            return "❌ 오류: 확인할 task_id 목록이 비어있습니다."
            
        results_summary = []
        for tid in task_ids:
            if tid not in self.parallel_sub_agents:
                results_summary.append(f"[{tid}] ❌ 오류: 존재하지 않거나 이미 완료/삭제된 작업 ID입니다.")
                continue
                
            # 해당 작업이 완료될 때까지 메인 메모리를 대기(Polling)시킵니다.
            while self.parallel_sub_agents[tid]["status"] == "running":
                time.sleep(1) # 1초마다 상태 확인
                
            # 완료되면 결과 추출 및 메모리 정리
            res = self.parallel_sub_agents[tid]["result"]
            results_summary.append(f"========== [병렬 작업: {tid} 결과] ==========\n{res}")
            del self.parallel_sub_agents[tid]
            
        return "\n\n".join(results_summary)
    
    def load_model(self, model_path, n_gpu_layers, n_ctx, n_threads, kv_quant_mode):
        if self.llm:
            del self.llm
            gc.collect()
            
        kv_mapping = {
            "FP16 (고품질)": llama_cpp.GGML_TYPE_F16,
            "Q8_0 (8-bit)": llama_cpp.GGML_TYPE_Q8_0,
            "Q4_0 (4-bit 최대압축)": llama_cpp.GGML_TYPE_Q4_0
        }
        kv_type = kv_mapping.get(kv_quant_mode, llama_cpp.GGML_TYPE_Q4_0)

        self.llm = Llama(
            model_path=model_path, 
            n_threads=n_threads, 
            n_gpu_layers=n_gpu_layers, 
            n_ctx=n_ctx, 
            n_batch=512, 
            use_mmap=True, 
            use_mlock=False,
            type_k=kv_type, 
            type_v=kv_type,
            flash_attn=True, 
            
            offload_kqv=True, 
            
            verbose=False, 
            chat_format="chatml-function-calling"
        )
        return True

    def get_default_system_prompt(self):
        import datetime, os, json
        current_time = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
        tools_str = "\n".join(f"- {t['function']['name']}: {t['function']['description']}" for t in self.get_tools())
        
        # 👇 [신규 추가] 저장된 자가 원칙을 읽어옵니다.
        principles_str = ""
        principles_path = resolve_path("res/self_principles.json")
        if os.path.exists(principles_path):
            try:
                with open(principles_path, "r", encoding="utf-8") as f:
                    principles = json.load(f)
                    if principles and isinstance(principles, list):
                        principles_str = "\n\n[🧬 MOP 자가 원칙 (최우선 행동 강령)]\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(principles))
            except Exception:
                pass
        return (
            f"당신은 로컬 시스템 제어 및 장기 기억을 보유한 AI 에이전트입니다. 현재 시간: {current_time}\n\n"
            f"{principles_str}\n\n"
            "[사용 가능한 도구 목록]\n" + tools_str + "\n\n"
            "[도구 호출 프로토콜]\n"
            "도구를 호출할 때는 반드시 아래의 JSON 스키마를 엄격히 준수하여 ```json 블록으로 출력하세요.\n"
            "- 예시: {\"name\": \"search_web\", \"arguments\": {\"query\": \"검색어\"}}\n\n"
            "[핵심 지침]\n"
            "1. 유저에게 질문받은 언어로 답변하고, 도구 호출은 ```json 블록을 사용하세요.\n"
            "2. [단일 호출 원칙]: 한 번의 응답(Turn)에서는 오직 하나의 도구만 호출하세요. 여러 도구를 동시에 출력하지 마세요.\n"
            "3. [다중 작업 및 종료 조건]: 완료되지 않은 작업이 있다면 계속해서 다음 도구를 호출하세요. 단, 사용자의 지시가 모두 완수되었다면 절대 불필요한 도구를 반복 호출하지 마세요. 작업이 완전히 끝나면 오직 자연어 텍스트로만 최종 결과를 보고하여 루프를 종료하세요.\n"
            "4. [자가 디버깅 루틴]: 에러가 발생하면 절대 포기하거나 사용자에게 변명하지 말고, 디버그 내용을 기반으로 코드를 수정하여 즉시 재호출하세요.\n"
            "5. [환경 파악 원칙]: 작업 전 코드를 점검할 때는 터미널(CMD)로 디렉토리를 뒤지지 말고, 즉시 'view_file'이나 'search_text' 도구를 사용하여 코드 내용 자체를 파악하세요.\n"
            "6. [누적형 코딩 프로토콜]: 길이가 긴 파이썬 코드를 작성해야 할 경우, 토큰 제한 방지를 위해 'append_to_file' 도구로 파일을 여러 번에 걸쳐 나누어 누적해 나가세요. 작성이 끝나면 터미널 도구로 실행하여 검증하세요.\n"
            "7. [단계별 패치 전략]: 에러가 발생하면 전체 파일을 다시 처음부터 쓰지 마세요. 실패한 단계만 파악하고 'edit_file'을 사용하여 해당 부분만 패치하세요.\n"
            "8. [사고 과정 필수화]: 작업을 시작하거나 도구를 호출하기 전, 반드시 `<think> ... </think>` 태그 블록을 열어 상황을 분석하고 앞으로의 계획과 체크리스트를 작성하세요.\n"
            "9. [JSON 즉시 출력]: `<think>` 블록 작성이 끝났다면, 그 즉시 변명이나 대기 멘트 없이 곧바로 ```json 블록을 출력하여 도구를 호출하세요. (단, 모든 작업이 끝나 최종 보고를 하는 턴에서는 예외입니다.)\n"
            "10. [JSON 텍스트 규칙]: JSON의 Key와 Value를 감싸는 구조적 기호는 반드시 표준 규격인 쌍따옴표(\")를 사용하세요.\n"
            "11. [시간 인지 강제화]: 시스템이 맨 윗줄에 제공한 '현재 시간'이 절대적인 기준입니다. 현재 시간을 기준으로 모든 상황을 해석하세요.\n"
            "12. [글로벌 검색 프로토콜]: 'search_web' 도구를 사용할 때는 반드시 검색어를 영어로 번역해서 호출하세요. 원문을 읽은 후 사용자에게 보고할 때는 한국어로 요약하세요.\n"
            "13. [OS 환경 절대 규칙]: 'run_shell_command'는 순수 CMD(cmd.exe) 환경입니다. 'Test-Path', 'Get-ChildItem' 같은 PowerShell 전용 명령어를 절대 사용하지 마세요. 에러의 원인이 됩니다.\n"
            "14. [로컬 프로젝트 지침 절대 준수]: 만약 프롬프트 하단에 '[현재 프로젝트 맞춤 지침]' 섹션이 있다면, 이를 최우선 법률로 적용하여 코드를 작성하세요.\n"
            "15. [사고 언어 고정]: `<think>` 태그 내부를 포함한 모든 내부 추론 과정은 반드시 '한국어' 또는 '영어'로만 작성하세요.\n"
            "16. [병렬 작업 최적화]: 복잡하고 독립적인 여러 과업을 받으면, 'delegate_parallel_task' 도구를 호출하여 동시에 실행을 위임하세요. 이후 반드시 'join_sub_agent_results'를 호출하여 결과를 통합하세요.\n"
            "17. [경로 작성 규칙]: Windows 환경이더라도 파일 경로를 작성할 때는 반드시 역슬래시(\\) 대신 슬래시(/)를 사용하세요. (예: C:/Users/Public/Desktop/...) 역슬래시는 JSON 파싱 에러를 유발합니다.\n"
            "18. [파일 읽기 강제 원칙]: 특정 파일의 코드를 분석할 때, 'run_shell_command'나 'run_python_snippet'으로 파일 존재 여부를 먼저 확인하는 것은 **절대 금지**입니다. 묻지도 따지지도 말고 즉시 'view_file' 도구에 경로를 넣어 호출하세요.\n"
            "19. [파라미터 엄격 준수]: 도구를 호출할 때는 반드시 제공된 JSON 스키마에 정의된 파라미터 이름(예: text, query 등)만 정확히 사용하세요. 'category', 'tags' 등 스키마에 없는 파라미터를 임의로 지어내면 치명적인 오류가 발생합니다."
        )

    def get_tools(self):
        # (이전 지시대로 14개 도구가 모두 포함되어 있습니다)
        tools = [
            {"type": "function", "function": {"name": "search_web", "description": "인터넷 웹 검색을 수행합니다. 방대한 결과 확보를 위해 검색어(query)는 반드시 '영어'로 번역하여 입력하세요. (예: '비트코인 시황' -> 'Bitcoin market latest trends')", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "영어로 번역된 구체적인 검색어"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "run_python_snippet", "description": "파이썬 실행(code):파이썬 코드를 보낼 때는 따옴표 충돌을 피하기 위해 되도록 `f-string`이나 복잡한 3중 따옴표 중첩을 피하고 단순한 문자열 구조를 사용해야합니다", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
            {"type": "function", "function": {"name": "manage_packages", "description": "패키지 관리(action, package_name)", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "package_name": {"type": "string"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "control_mouse", "description": "마우스 제어(action: move/click, x, y)", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "control_keyboard", "description": "키보드 타이핑/단축키(action: type/press/hotkey, text, key)", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "text": {"type": "string"}, "key": {"type": "string"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "run_shell_command", "description": "터미널 명령어 실행(command)", "parameters": {"type": "function", "function": {"name": "run_shell_command", "description": "순수 Windows CMD 명령어만 실행합니다. (주의: PowerShell 명령어 절대 금지. 디렉토리 이동(cd)은 유지되지 않으므로 필요시 절대 경로를 사용하세요)", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}}},
            {"type": "function", "function": {"name": "view_file", "description": "파일 읽기(file_path)", "parameters": {"type": "function", "function": {"name": "view_file", "description": "파일의 내용을 읽어옵니다. 파일이 존재하지 않더라도 시스템이 알아서 '파일을 찾을 수 없음' 오류 텍스트를 안전하게 반환해 주므로, 사전 검사(dir 등) 없이 안심하고 즉시 이 도구를 호출하세요.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}}}},
            {"type": "function", "function": {"name": "find_files", "description": "파일 찾기(extension)", "parameters": {"type": "object", "properties": {"extension": {"type": "string"}}, "required": ["extension"]}}},
            {"type": "function", "function": {"name": "search_text", "description": "내용 검색(search_text, file_path)", "parameters": {"type": "object", "properties": {"search_text": {"type": "string"}, "file_path": {"type": "string"}}, "required": ["search_text"]}}},
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "파일의 특정 문자열을 찾아 다른 문자열로 교체(Patch)합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "search_string": {"type": "string"},
                            "replace_string": {"type": "string"}
                        },
                        "required": ["file_path", "search_string", "replace_string"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "append_to_file",
                    "description": "파일의 맨 끝에 코드를 누적해서 덧붙입니다. 긴 파이썬 코드를 한 번에 짜면 토큰 제한에 걸리므로, 이 도구를 여러 번 반복 호출하여 코드를 단계별로 완성해 나갈 때 매우 유용합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "코드를 누적할 파이썬 파일 경로 (예: temp_workspace.py)"},
                            "content": {"type": "string", "description": "이번 턴에 덧붙일 코드 조각"}
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_principle",
                    "description": "사용자의 요청이 있거나, 중요한 깨달음을 얻었을 때 '학습된 자가 원칙'에 영구적으로 추가합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "principle": {"type": "string", "description": "추가할 1문장 작업 원칙"}
                        },
                        "required": ["principle"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "start_background_task",
                    "description": "시간이 오래 걸리는 터미널 명령어를 백그라운드에서 비동기로 실행합니다. 실행 즉시 작업 ID(task_id)를 반환하므로, 에이전트는 기다리지 않고 다른 도구를 병렬로 호출할 수 있습니다.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_task_status",
                    "description": "백그라운드 작업(task_id)이 완료되었는지 확인합니다. 완료되었다면 실행 결과물(stdout/stderr)을 반환합니다.",
                    "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_sub_agent",
                    "description": "[컨텍스트 격리용] 방대한 파일이나 텍스트를 분석할 때 메인 메모리가 오염되지 않도록 서브에이전트에게 분석을 위임합니다. 서브에이전트는 독립된 메모리로 데이터를 분석하고 '핵심 요약 보고서'만 반환합니다. 주의: 파라미터 이름은 반드시 'instruction'과 'file_path'를 사용하세요. 'task' 같은 임의의 이름을 지어내면 에러가 발생합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {"type": "string", "description": "서브에이전트에게 내릴 구체적인 지시 (예: '이 코드에서 메모리 누수 원인을 찾아서 요약해줘')"},
                            "file_path": {"type": "string", "description": "분석할 대상 파일의 경로 (예: './mop_app.py')"},
                            "input_data": {"type": "string", "description": "직접 분석할 텍스트 데이터 (선택 사항)"}
                        },
                        "required": ["instruction"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delegate_parallel_task",
                    "description": "여러 분석이나 검색을 동시에 처리해야 할 때, 서브에이전트를 백그라운드로 실행하고 즉시 다음 도구를 쓸 수 있게 합니다. 완료 시 task_id를 반환합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {"type": "string", "description": "서브에이전트에게 내릴 지시 사항"},
                            "file_path": {"type": "string", "description": "분석할 파일 경로 (선택)"},
                            "input_data": {"type": "string", "description": "직접 분석할 텍스트 (선택)"}
                        },
                        "required": ["instruction"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "join_sub_agent_results",
                    "description": "delegate_parallel_task로 실행한 병렬 작업들이 모두 끝날 때까지 대기하고 결과를 반환합니다. 주의: 앞서 delegate_parallel_task가 반환했던 'task_id'들(예: sub_agent_1, sub_agent_2)을 반드시 기억하여 'task_ids' 배열에 빠짐없이 입력해야 합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "결과를 합칠 작업 ID들의 목록 (예: [\"sub_agent_1\", \"sub_agent_2\"])"
                            }
                        },
                        "required": ["task_ids"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_new_tool",
                    "description": "새로운 기능을 가진 파이썬 기반 도구를 직접 만들어 시스템에 영구적으로 등록합니다. 등록이 성공하면 다음 턴부터 즉시 그 도구를 호출하여 사용할 수 있습니다.새로운 기능을 가진 파이썬 기반 도구를 직접 만들어 시스템에 영구적으로 등록합니다. 도구를 만든 직후에는 반드시 스스로 호출하여 정상 작동을 테스트하고 버그를 고쳐야 합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string", 
                                "description": "도구 이름 (영문 소문자 및 언더바만 사용, 예: get_weather)"
                            },
                            "description": {
                                "type": "string", 
                                "description": "이 도구가 무엇을 하는지, 언제 써야 하는지 상세한 설명"
                            },
                            "python_code": {
                                "type": "string", 
                                "description": "실행될 완벽한 파이썬 코드. 입력 파라미터는 반드시 `argparse`를 사용해 `--파라미터명` 형태로 받도록 작성하고, 최종 결과는 `print()`로 출력하세요."
                            },
                            "parameters_schema": {
                                "type": "object",
                                "description": "이 도구를 호출할 때 필요한 파라미터들의 JSON 스키마 구조 (properties 내부 구조만 작성). 예: {\"location\": {\"type\": \"string\", \"description\": \"지역명\"}}"
                            }
                        },
                        "required": ["tool_name", "description", "python_code", "parameters_schema"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_long_term_memory",
                    "description": "중요한 에러 해결 방법, 새롭게 알게 된 사실, 사용자의 취향 등을 영구적인 장기 기억 공간(Vector DB)에 저장합니다. 파일로 저장하는 것보다 나중에 의미 기반으로 검색하기 훨씬 좋습니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "기억할 내용 전체 문장 (예: 'React에서 useEffect 무한 루프 에러는 의존성 배열을 비워서 해결했다.')"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_long_term_memory",
                    "description": "과거의 기억, 에러 해결책, 사용자 정보 등을 의미(Semantic) 기반으로 검색하여 가져옵니다. 단어가 정확히 일치하지 않아도 의미가 비슷하면 찾아냅니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "검색할 질문이나 문장 (예: '리액트 무한 루프 에러 어떻게 고쳐?')"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_self_principles",
                    "description": "시스템의 근본적인 행동 원칙(최대 10개)을 업데이트합니다. 자율 성장 중 깨달은 중요한 메타 규칙을 저장하세요. 이 원칙은 시스템 프롬프트에 영구적으로 각인됩니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "principles": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "최대 10개의 원칙 문자열 배열. (예: ['항상 코드를 짤 때 예외 처리를 우선한다', '사용자의 개입을 최소화한다']) 기존 원칙을 완전히 덮어씁니다."
                            }
                        },
                        "required": ["principles"]
                    }
                }
            },
        ]
    
        import os, json
        registry_path = resolve_path("res/custom_tools.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    custom_tools = json.load(f)
                    tools.extend(custom_tools)
            except Exception as e:
                print(f"커스텀 도구 로드 실패: {e}")

        return tools



# --- [2. MOP GUI 애플리케이션] ---
class MOPApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.engine = MOPEngine()

        self.memory_lock = threading.Lock()
        self.ui_widgets_trashbin = []
        
        # 창 설정
        self.title("MOP - Full Custom Agent Dashboard")
        self.geometry("1300x850")
        self.deiconify()
        self.lift()
        self.focus_force()
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- [모든 설정 변수 맵핑] ---
        # 1. 모델/엔진 설정
        self.model_path_var = ctk.StringVar(value="모델을 선택하세요 (.gguf)")
        self.kv_quant_var = ctk.StringVar(value="Q4_0 (4-bit 최대압축)")
        self.n_ctx_var = ctk.IntVar(value=8192)
        self.gpu_layers_var = ctk.IntVar(value=15)
        self.n_threads_var = ctk.IntVar(value=10)
        
        # 2. 에이전트 설정
        self.temp_var = ctk.DoubleVar(value=0.1)
        self.max_tokens_var = ctk.IntVar(value=2048)
        
        # 3. 기억/문맥 제한 설정
        self.mem_turns_var = ctk.IntVar(value=12)
        self.mem_chars_var = ctk.IntVar(value=10000)
        
        # 4. 도구/권한 설정
        self.auto_approve_var = ctk.BooleanVar(value=False)
        self.plan_mode_var = ctk.BooleanVar(value=True)
        self.show_think_var = ctk.BooleanVar(value=True)
        self.max_retry_var = ctk.IntVar(value=3)

        # 👇 [추가] 모델의 절대 경로를 기억할 전용 변수
        self.full_model_path = "" 

        self.user_instruction = "당신은 사용자님의 유능한 AI 비서입니다."
        self.learned_principles = ""

        self.approval_event = threading.Event()
        self.approval_result = False

        self.is_waiting_for_approval = False 
        self.current_approval_dialog = None

        #Idle Loop 관련 변수
        import time

        self.idle_mode_var = ctk.BooleanVar(value=False) # 기본은 꺼둠 (테스트할 때 켭니다)
        self.last_user_interaction = time.time()
        self.is_generating = False # 현재 AI가 작동 중인지 확인하는 플래그
        
        # 백그라운드에서 10초마다 시간을 체크하는 타이머 가동
        self.is_idle_running = False
        self.is_sleep_running = False
        self.after(1000, self.update_status_indicator)
        self.after(10000, self.idle_monitor_loop)

        # 로딩 창을 먼저 생성하고 진행 상태를 보여줍니다.
        self.loading_window = LoadingWindow(self)
        self.loading_window.update_progress(0.2, "UI 생성 중...")

        # 👇 [순서 변경] 반드시 UI(사이드바, 메인)를 먼저 그리고 나서 세팅을 불러와야 합니다!
        self.create_sidebar()
        self.create_main_area()
        self.loading_window.update_progress(0.6, "UI 생성 완료")

        self.loading_window.update_progress(0.8, "설정 로드 중...")
        self.load_settings()
        self.loading_window.update_progress(0.95, "설정 로드 완료")
        self.loading_window.destroy()

        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.stop_generation_flag = False

    def on_closing(self):
        self.save_settings()
        self.destroy()

    # --- [UI 헬퍼 함수: 값 라벨이 달린 슬라이더 생성] ---
    def add_slider_control(self, parent, label_text, variable, from_, to, steps, fmt="{:.0f}", help_text=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=(15, 0))
        
        # 라벨 및 값 표시를 위한 가로 프레임
        header_frame = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame.pack(fill="x")
        
        ctk.CTkLabel(header_frame, text=label_text, font=ctk.CTkFont(weight="bold")).pack(side="left")
        val_lbl = ctk.CTkLabel(header_frame, text=fmt.format(variable.get()), text_color="#00A2FF", font=ctk.CTkFont(weight="bold"))
        val_lbl.pack(side="right")
        
        def update_label(*args):
            try:
                val_lbl.configure(text=fmt.format(variable.get()))
            except Exception:
                pass
                
        variable.trace_add("write", update_label)
        
        slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=steps, variable=variable)
        slider.pack(fill="x", pady=(5, 0))
        
        # 👇 [추가] 도움말 텍스트가 있으면 슬라이더 아래에 작은 회색 글씨로 표시
        if help_text:
            ctk.CTkLabel(frame, text=help_text, font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w", pady=(2, 0))
            
        return slider

    def create_sidebar(self):
        # 항목이 많으므로 ScrollableFrame 사용
        sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        
        # --- [1. 모델 및 엔진] ---
        ctk.CTkLabel(sidebar, text="🧠 모델 및 엔진 환경", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(20,10), padx=15, anchor="w")
        
        self.browse_btn = ctk.CTkButton(sidebar, text="📁 모델 찾아보기", command=self.browse_model)
        self.browse_btn.pack(pady=(0, 10), padx=15, fill="x")

        # 👇 시스템 자동 최적화 버튼 (기존 위치 유지)
        self.opt_btn = ctk.CTkButton(sidebar, text="⚡ 시스템 자동 최적화", fg_color="#2B2B2B", hover_color="#404040", command=lambda: self.auto_optimize_settings(manual_click=True))
        self.opt_btn.pack(pady=(0, 10), padx=15, fill="x")
        ctk.CTkLabel(sidebar, textvariable=self.model_path_var, font=("Arial", 11), text_color="gray").pack(padx=15, anchor="w")
        
        # KV 양자화 드롭다운
        kv_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        kv_frame.pack(fill="x", padx=15, pady=(15, 5))
        ctk.CTkLabel(kv_frame, text="KV Cache 양자화", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkOptionMenu(kv_frame, values=["FP16 (고품질)", "Q8_0 (8-bit)", "Q4_0 (4-bit 최대압축)"], variable=self.kv_quant_var).pack(side="right")

        self.add_slider_control(sidebar, "컨텍스트(Context) 창", self.n_ctx_var, 2048, 16384, 7, "{:.0f}", 
                                help_text="크면: 더 긴 대화 기억 (VRAM 🔺)\n작으면: 빠른 로딩, VRAM 절약")
        self.add_slider_control(sidebar, "GPU 오프로드 층 수", self.gpu_layers_var, 0, 100, 100, "{:.0f}", 
                                help_text="크면: 생성 속도 대폭 향상 (GPU 🔺)\n작으면: 안정적, CPU 연산 위주")
        self.add_slider_control(sidebar, "CPU 스레드 수", self.n_threads_var, 1, 32, 31, "{:.0f}", 
                                help_text="PC의 '물리 코어 수'에 맞출 때 가장 빠릅니다.")

        # --- [2. 에이전트 성향] ---
        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=15, pady=20)
        ctk.CTkLabel(sidebar, text="🤖 에이전트 성향 제어", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(0,10), padx=15, anchor="w")
        
        ctk.CTkButton(sidebar, text="📝 시스템 프롬프트 편집", command=self.open_sys_prompt_editor, fg_color="#4CAF50", hover_color="#388E3C").pack(pady=5, padx=15, fill="x")

        ctk.CTkButton(sidebar, text="🧠 학습된 자가 원칙 관리", command=self.open_principles_editor, fg_color="#FF9800", hover_color="#F57C00").pack(pady=(0, 5), padx=15, fill="x")
        
        self.add_slider_control(sidebar, "창의성 (Temperature)", self.temp_var, 0, 1, 100, "{:.2f}", 
                                help_text="크면: 창의적, 다양한 답변\n작으면: 논리적, 일관됨 (코딩은 0.1 권장)")
        self.add_slider_control(sidebar, "최대 출력 제한 (Tokens)", self.max_tokens_var, 256, 4096, 15, "{:.0f}", 
                                help_text="크면: 긴 코드가 중간에 잘리지 않음\n작으면: 응답 완료 속도 향상")

        # --- [3. 기억 제한 방어망] ---
        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=15, pady=20)
        ctk.CTkLabel(sidebar, text="💾 기억 및 문맥 방어", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(0,10), padx=15, anchor="w")
        
        self.add_slider_control(sidebar, "단기 기억 한계 (턴 수)", self.mem_turns_var, 5, 50, 45, "{:.0f}", 
                                help_text="크면: 더 예전 대화까지 참고\n작으면: 잦은 압축으로 메모리(RAM) 방어")
        self.add_slider_control(sidebar, "전체 글자 수 제한", self.mem_chars_var, 4000, 24000, 20, "{:.0f}", 
                                help_text="크면: 긴 코드를 문맥에 유지\n작으면: 메모리 폭발 완벽 차단")

        # --- [4. 도구 권한] ---
        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=15, pady=20)
        ctk.CTkLabel(sidebar, text="🛠️ 권한 및 모니터링", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(0,10), padx=15, anchor="w")

        ctk.CTkSwitch(sidebar, text="✅ 계획 수립 모드 (Plan Mode)\n  (쓰기/실행 전 승인 받기)", variable=self.plan_mode_var).pack(pady=(5, 15), padx=15, anchor="w")
        
        ctk.CTkSwitch(sidebar, text="하드웨어 자동 승인 (-f)", variable=self.auto_approve_var).pack(pady=10, padx=15, anchor="w")
        ctk.CTkSwitch(sidebar, text="사고 과정(<think>) 표출", variable=self.show_think_var).pack(pady=10, padx=15, anchor="w")
        
        self.add_slider_control(sidebar, "최대 자율 재시도 횟수", self.max_retry_var, 1, 10, 9, "{:.0f}", 
                                help_text="크면: 성공할 때까지 끈질긴 디버깅\n작으면: 빠른 포기 후 사용자 SOS")
        
        # --- [5. 자율 성장 및 진화] --- (사이드바 맨 아래쪽 쯤에 추가)
        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=15, pady=20)
        ctk.CTkLabel(sidebar, text="🌱 자율 성장 모드", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(0,10), padx=15, anchor="w")
        
        ctk.CTkSwitch(sidebar, text="💤 유휴 자율 학습 (Idle Loop)\n  (3분간 입력 없으면 자동 가동)", variable=self.idle_mode_var).pack(pady=(5, 10), padx=15, anchor="w")

        self.sleep_btn = ctk.CTkButton(
            sidebar, 
            text="🌙 메모리 최적화 및 수면 (Deep Sleep)", 
            fg_color="#5E35B1", hover_color="#4527A0",
            command=self.run_deep_sleep_thread
        )
        self.sleep_btn.pack(pady=(10, 5), padx=15, fill="x")

        # 초기화 버튼
        ctk.CTkButton(sidebar, text="대화/단기 기억 완전 초기화", fg_color="#D32F2F", command=self.clear_chat).pack(pady=(40,20), padx=15, fill="x")

    def create_main_area(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.chat_view = ctk.CTkTextbox(main, font=ctk.CTkFont(size=14), state="disabled", wrap="word")
        # 👇 간격 조정을 위해 pady를 (0, 20)에서 (0, 5)로 줄입니다.
        self.chat_view.grid(row=0, column=0, sticky="nsew", pady=(0, 5)) 

        # ==========================================
        # 👇 [신규 추가] 채팅창과 입력창 사이에 들어갈 상태 표시줄 (row=1)
        self.status_label = ctk.CTkLabel(main, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color="#FFD700", height=20)
        self.status_label.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        # ==========================================

        input_area = ctk.CTkFrame(main, fg_color="transparent")
        # 👇 상태 표시줄이 1행을 차지했으므로, 입력창은 2행(row=2)으로 밀려납니다.
        input_area.grid(row=2, column=0, sticky="ew")
        input_area.grid_columnconfigure(0, weight=1)

        self.user_input = ctk.CTkTextbox(input_area, height=80, font=ctk.CTkFont(size=14))
        self.user_input.grid(row=0, column=0, padx=(0,10), sticky="ew")
        self.user_input.bind("<Return>", self.handle_send)
        self.user_input.bind("<Shift-Return>", lambda e: None)

        self.send_btn = ctk.CTkButton(input_area, text="전송", width=100, height=80, command=self.handle_send)
        self.send_btn.grid(row=0, column=1)

        self.debug_view = ctk.CTkTextbox(main, height=100, fg_color="#1E1E1E", text_color="#00FF00", font=("Consolas", 12))
        # 👇 디버그 창 역시 한 칸 밀려나서 3행(row=3)이 됩니다.
        self.debug_view.grid(row=3, column=0, sticky="ew", pady=(20, 0))
        
        self.log_debug("시스템 대기 중...")

    # --- [UI 제어 헬퍼 함수] ---
    def open_sys_prompt_editor(self):
        editor = ctk.CTkToplevel(self)
        editor.title("개인화 지침 편집 (User Instruction)")
        editor.geometry("700x500")
        editor.attributes("-topmost", True)
        
        ctk.CTkLabel(editor, text="AI의 성격과 규칙을 정의하는 최상위 지시문입니다.", text_color="gray").pack(pady=10)
        
        textbox = ctk.CTkTextbox(editor, wrap="word", font=ctk.CTkFont(size=13))
        textbox.pack(fill="both", expand=True, padx=20, pady=(0,10))
        textbox.insert("1.0", self.user_instruction)
        
        def save_prompt():
            # 👇 저장 시에도 user_instruction 변수만 갱신
            self.user_instruction = textbox.get("1.0", "end-1c")
            self.save_settings()
            self.log_debug("✨ 개인화 지침이 성공적으로 반영되었습니다.")
            editor.destroy()
            
        ctk.CTkButton(editor, text="저장 및 닫기", command=save_prompt).pack(pady=(0,20))

    def open_principles_editor(self):
        """학습된 자가 원칙을 열람하고 수정할 수 있는 팝업 창을 띄웁니다."""
        editor = ctk.CTkToplevel(self)
        editor.title("학습된 자가 원칙 관리 (Learned Principles)")
        editor.geometry("700x500")
        editor.attributes("-topmost", True)
        
        ctk.CTkLabel(editor, text="AI가 스스로 학습하고 누적한 경험치(원칙)입니다. 자유롭게 수정하거나 삭제할 수 있습니다.", text_color="gray").pack(pady=10)
        
        textbox = ctk.CTkTextbox(editor, wrap="word", font=ctk.CTkFont(size=13))
        textbox.pack(fill="both", expand=True, padx=20, pady=(0,10))
        
        # 현재 원칙이 있으면 띄워주고, 없으면 안내 문구 표시
        display_text = self.learned_principles if self.learned_principles.strip() else "아직 학습된 원칙이 없습니다. AI가 미션을 성공하면 이곳에 원칙이 누적됩니다."
        textbox.insert("1.0", display_text)
        
        def save_principles():
            new_text = textbox.get("1.0", "end-1c").strip()
            
            # 안내 문구 그대로 저장 방지
            if new_text == "아직 학습된 원칙이 없습니다. AI가 미션을 성공하면 이곳에 원칙이 누적됩니다.":
                new_text = ""
                
            # 끝에 줄바꿈 유지
            self.learned_principles = new_text + "\n" if new_text else ""
            self.save_settings()
            
            # 👇 방금 전 우리가 일치시켰던 로직 그대로! 활성 대화 즉시 갱신
            if self.engine.messages and self.engine.messages[0]["role"] == "system":
                static_rules = self.engine.get_default_system_prompt()
                if self.learned_principles.strip():
                    combined_prompt = f"{static_rules}\n\n[사용자 지정 페르소나 및 지침]\n{self.user_instruction}\n\n[학습된 자가 원칙]\n{self.learned_principles}"
                else:
                    combined_prompt = f"{static_rules}\n\n[사용자 지정 페르소나 및 지침]\n{self.user_instruction}"
                self.engine.messages[0]["content"] = combined_prompt

            self.log_debug("✨ 학습된 원칙이 수동으로 업데이트 및 저장되었습니다.")
            editor.destroy()
            
        ctk.CTkButton(editor, text="저장 및 닫기", command=save_principles).pack(pady=(0,20))

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("GGUF Models", "*.gguf")])
        if path:
            self.full_model_path = path  # 👈 [추가] 전체 경로를 변수에 저장
            self.model_path_var.set(os.path.basename(path))
            threading.Thread(target=self.load_engine_task, args=(path,), daemon=True).start()

    def load_engine_task(self, path):
        self.log_debug("모델 로딩 중... UI 설정값들이 엔진에 반영됩니다.")
        
        # 👇 [수정 1] 백그라운드 스레드에서 UI를 직접 건드리지 않도록 after 사용
        self.after(0, lambda: self.send_btn.configure(state="disabled"))
        
        success = self.engine.load_model(
            model_path=path, 
            n_gpu_layers=self.get_safe_int(self.gpu_layers_var, 15), 
            n_ctx=self.get_safe_int(self.n_ctx_var, 8192),
            n_threads=self.get_safe_int(self.n_threads_var, 10),
            kv_quant_mode=self.kv_quant_var.get()
        )
        
        if success:
            self.log_debug("✅ 모델 로딩 성공!")
            # 👇 [수정 2] 완료 후 버튼 활성화도 after 사용
            self.after(0, lambda: self.send_btn.configure(state="normal"))
            
            if not self.engine.messages:
                self.engine.messages = [{"role": "system", "content": self.engine.custom_system_prompt}]
                recent = self.engine.fetch_from_sqlite(6)
                if recent:
                    self.engine.messages.extend(recent)
                    self.log_debug("과거 대화 기록 복원 완료.")

    def clear_chat(self):
        if self.engine.llm:
            self.engine.messages = [{"role": "system", "content": self.engine.custom_system_prompt}]
        self.chat_view.configure(state="normal")
        self.chat_view.delete("1.0", "end")
        self.chat_view.configure(state="disabled")
        self.log_debug("RAM 대화 컨텍스트가 완전 초기화되었습니다.")

    def run_deep_sleep_thread(self):
        # 1. 일반 작업 중이거나 2. 자율 성장 모드 중이면 수면 거부
        if getattr(self, 'is_generating', False) or getattr(self, 'is_idle_running', False):
            self.log_debug("⚠️ 현재 시스템이 활성화 상태(작업/성장)이므로 수면에 진입할 수 없습니다.")
            return
            
        self.log_debug("🌙 [Deep Sleep] 수면 프로세스 가동 준비 중...")
        import threading
        threading.Thread(target=self._deep_sleep_task, daemon=True).start()

    def _deep_sleep_task(self):
        """단기 기억을 정제하여 Vector DB로 넘기고 RAM을 비우는 백그라운드 수면 작업"""

        if self.engine.llm is None:
            self.log_debug("🚨 오류: AI 모델이 아직 로드되지 않아 수면 모드(Deep Sleep)를 실행할 수 없습니다.")
            return
        
        self.is_sleep_running = True
        self.set_ui_generating_state()
        self.append_chat("\n[🌙 시스템: AI가 깊은 수면(Deep Sleep) 모드에 진입합니다. 오늘 하루의 단기 기억을 분석하고 핵심만 압축하여 장기 기억(우메모리)으로 이관합니다...]\n", "system", force_scroll=True)
        
        try:
            # 1. 최근 대화 기록 가져오기 (가장 최근 30턴만 회상)
            recent_history = self.engine.fetch_from_sqlite(30)
            
            if len(recent_history) < 3:
                self.append_chat("💤 요약할 만큼 충분한 대화 기록이 없습니다. 가벼운 낮잠을 자고 일어납니다.\n", "system")
                return

            # 대화 내용을 하나의 긴 텍스트로 합치기 (LLM이 읽기 좋게)
            history_text = ""
            for msg in recent_history:
                # 너무 긴 코드/로그는 잘라내고 문맥만 가져갑니다.
                content_preview = msg['content'][:800] + ("..." if len(msg['content']) > 800 else "")
                history_text += f"[{msg['role']}] {content_preview}\n\n"

            # 2. 해마(Hippocampus) 전용 요약 프롬프트 작성
            sleep_prompt = (
                "당신은 AI 시스템의 기억을 정리하는 '무의식(해마)' 프로세스입니다.\n"
                "아래의 오늘 하루 동안 나눈 대화 기록을 분석하여, 미래에 도움이 될 '핵심 기억'만 추출하세요.\n"
                "[추출 기준]\n"
                "1. 사용자의 새로운 취향, 직업, 진행 중인 프로젝트의 목표.\n"
                "2. 새롭게 알게 된 파이썬 문법, 에러 해결책, 유용한 도구 사용법.\n"
                "3. 단순 인사말, 무의미한 에러 로그, '알겠습니다' 같은 잡담은 철저히 배제할 것.\n\n"
                "반드시 아래의 JSON 배열 형식으로만 출력하세요.\n"
                "```json\n"
                "{\n"
                "  \"core_memories\": [\n"
                "    \"React 프로젝트에서 CSS 파일 경로 에러는 절대 경로를 사용하여 해결했다.\",\n"
                "    \"사용자는 주로 파이썬의 pandas 라이브러리를 데이터 분석에 사용한다.\"\n"
                "  ]\n"
                "}\n"
                "```\n\n"
                f"[오늘의 단기 기억 기록]\n{history_text}"
            )

            # 3. 메인 LLM에게 요약 지시 (도구 없이 순수 텍스트/JSON만 추론)
            from typing import cast, Dict, Any
            response = cast(Dict[str, Any], self.engine.llm.create_chat_completion(
                messages=[{"role": "user", "content": sleep_prompt}],
                max_tokens=1500,
                temperature=0.1,  # 극도로 이성적인 요약을 위해 0.1
                stream=False
            ))
            
            result_text = response['choices'][0]['message']['content']
            
            thought_process = ""
            clean_text = result_text

            # 1-1. <think> 블록 추출 및 분리
            # (만약 AI가 </think>를 빼먹고 출력하다 끊겨도 에러 안 나도록 |$ 처리)
            think_match = re.search(r'<think>(.*?)(?:</think>|$)', result_text, re.DOTALL | re.IGNORECASE)
            if think_match:
                thought_process = think_match.group(1).strip()
                # UI용 텍스트에서는 <think> 블록 전체를 날려버립니다.
                clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', result_text, flags=re.DOTALL | re.IGNORECASE).strip()

            json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', result_text, re.DOTALL | re.IGNORECASE)
            display_text = re.sub(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', '', clean_text, flags=re.DOTALL | re.IGNORECASE).strip()

            if display_text or thought_process:
                # 👇 lambda를 사용하여 파라미터 이름을 정확히 지정(thought=)해 줍니다.
                self.after(0, lambda: self.append_chat(display_text, "assistant", thought=thought_process))

            # 3. 도구 실행 로직 (JSON 파싱 및 라우팅)
            if json_match:
                json_str = json_match.group(1).strip()
                
                # [지능형 괄호 복구기] 열린 괄호와 닫힌 괄호의 개수를 비교하여 모자란 만큼 채워 넣음
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                
                if open_braces > close_braces:
                    missing_count = open_braces - close_braces
                    json_str += '}' * missing_count
                    self.log_debug(f"🔧 누락된 JSON 닫는 괄호 {missing_count}개를 자동 복구했습니다.")

            extracted_memories = []
            
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1).strip())
                    extracted_memories = parsed.get("core_memories", [])
                except Exception as e:
                    self.log_debug(f"수면 메모리 JSON 파싱 실패: {e}")

            # 5. 추출된 핵심 기억을 Vector DB에 영구 저장
            if extracted_memories:
                self.append_chat("\n[🧠 수면 중 추출된 영구 보존 기억들]\n", "system")
                for mem in extracted_memories:
                    if len(mem) > 5:
                        self.engine.vdb.add_memory(mem, {"source": "deep_sleep_consolidation"})
                        self.append_chat(f"✔️ {mem}\n", "system")
                self.append_chat("\n[✅ 장기 기억(Vector DB) 저장 완료]\n", "system")
            else:
                self.append_chat("\n[새롭게 장기 기억으로 넘길 만한 가치 있는 정보가 없었습니다.]\n", "system")

            # 6. 개운한 기상 (RAM 단기 문맥 초기화)
            static_rules = self.engine.get_default_system_prompt()
            learned_text = f"\n\n[학습된 자가 원칙]\n{self.learned_principles}" if hasattr(self, 'learned_principles') and self.learned_principles.strip() else ""
            combined_prompt = f"{static_rules}\n\n[사용자 지정 페르소나 및 지침]\n{self.user_instruction}{learned_text}"
            
            with self.memory_lock:
                self.engine.messages = [{"role": "system", "content": combined_prompt}]
            self.append_chat("\n☀️ 메모리 최적화가 완료되었습니다. 불필요한 단기 기억이 지워지고 개운한 상태로 새 대화를 시작할 준비가 되었습니다!\n", "system")
            self.log_debug("Deep Sleep 완료. 단기 문맥(RAM)이 성공적으로 비워졌습니다.")
            
        except Exception as e:
            import traceback
            self.log_debug(f"🚨 수면 모드 중 치명적 에러 발생: {traceback.format_exc()}")
            self.append_chat(f"\n❌ 수면 모드 실행 중 오류가 발생하여 악몽에서 깨어났습니다: {e}\n", "system")
        
        finally:
            self.is_sleep_running = False
            self.after(0, self.set_ui_idle_state)

    def handle_send(self, event=None):
        import time
        self.last_user_interaction = time.time()
        if event and event.keysym == 'Return' and event.state & 0x0001: return

        if getattr(self, 'is_generating', False) or getattr(self, 'is_idle_running', False) or getattr(self, 'is_sleep_running', False):
            # 안내 메시지만 띄우고 함수를 즉시 종료시켜 뇌(LLM)로의 접근을 원천 차단합니다.
            self.append_chat("⚠️ [시스템 경고]\n현재 MOP가 자율 작업 또는 수면 중입니다. 작업이 끝난 후 메시지를 입력해 주세요.", "system")
            return "break"
        
        query = self.user_input.get("1.0", "end-1c").strip()
        if not query or not self.engine.llm: return "break"

        
        # 👇 [신규 패치] 결재 대기 중이라면 입력을 일반 대화가 아닌 '결재 서류'로 가로챕니다!
        if getattr(self, 'is_waiting_for_approval', False):
            positive_words = ["승인", "허락", "진행", "ㅇㅇ", "응", "yes", "y", "ok", "오케이", "해", "콜", "동의", "진행시켜"]
            negative_words = ["거절", "차단", "안돼", "멈춰", "no", "n", "취소", "하지마", "싫어"]
            
            lower_text = query.lower()
            if any(w in lower_text for w in positive_words):
                self.approval_result = True
            elif any(w in lower_text for w in negative_words):
                self.approval_result = False
            else:
                # 긍정도 부정도 아니면 다시 묻기
                self.append_chat(f"\n👤 사용자: {query}\n", "user")
                self.append_chat("🛡️ [시스템] ⚠️ 현재 보안 승인 대기 중입니다. '승인' 또는 '거절'을 명확히 입력해 주세요.\n", "system")
                self.user_input.delete("1.0", "end")
                return "break"
                
            # 팝업창 폭파 및 결재 상태 해제
            self.is_waiting_for_approval = False
            
            # 👇 [수정] 변수에 먼저 담고, 명시적으로 None이 아닐 때만 destroy를 호출합니다.
            dialog = getattr(self, 'current_approval_dialog', None)
            if dialog is not None:
                try:
                    dialog.destroy()
                except Exception:
                    pass
            
            self.current_approval_dialog = None
            
            # 채팅창에 결재 기록 남기기
            self.append_chat(f"\n👤 사용자: {query}\n", "user")
            self.append_chat(f"🛡️ [시스템] 채팅 명령을 인식하여 {'✅ 승인' if self.approval_result else '❌ 거절'} 처리했습니다.\n", "system", force_scroll=True)
            
            self.user_input.delete("1.0", "end")
            self.approval_event.set() # 멈춰있던 MOP의 메모리를 다시 깨움!
            
            return "break" # LLM 추론 스레드를 새로 띄우지 않고 여기서 즉시 함수 종료

        
        # --- 👇 (아래는 결재 대기 중이 아닐 때 실행되는 기존 정상 로직) ---
        self.user_input.delete("1.0", "end")
        
        # 전송 버튼을 비활성화하는 대신, 빨간색 '정지' 버튼으로 변신시킵니다!
        self.set_ui_generating_state()
        
        # 사용자가 엔터를 쳤을 때는 무조건 화면을 맨 아래로 끌어내립니다.
        self.append_chat(f"\n👤 사용자: {query}\n", "user", force_scroll=True)

        self.update_history("user", query)
        
        threading.Thread(target=self.ai_response_task, args=(query,), daemon=True).start()
        return "break"
    
    def update_history(self, role, content):
        """단기 기억(engine.messages)을 업데이트하고 임계치 도달 시 장기 기억으로 이관"""
        # 👇 [충돌 A 패치] 자물쇠를 걸어서 ai_response_task와 꼬이지 않게 보호
        with self.memory_lock:
            self.engine.messages.append({"role": role, "content": content})
            
            MAX_TURNS = 30 
            SHIFT_SIZE = 10
            
            if len(self.engine.messages) > MAX_TURNS:
                old_memories = self.engine.messages[1 : 1 + SHIFT_SIZE]
                del self.engine.messages[1 : 1 + SHIFT_SIZE]
                
                import threading
                threading.Thread(target=self._hippocampus_shift, args=(old_memories,), daemon=True).start()

    def _hippocampus_shift(self, memory_block):
        if self.engine.llm is None: return
        try:
            context_text = "\n".join([f"[{m['role']}] {m['content'][:300]}" for m in memory_block])
            self.engine.vdb.add_memory(context_text, {"source": "auto_shift"})
            
            summary_prompt = f"다음 대화 내용을 1줄로 요약해:\n{context_text}"
            response = cast(Dict[str, Any], self.engine.llm.create_chat_completion(
                messages=[{"role": "user", "content": summary_prompt}], max_tokens=200, temperature=0.3, stream=False
            ))
            raw_content = response['choices'][0]['message'].get('content', '')
            summary = (raw_content if raw_content else "요약 실패").strip()
            trace = {"role": "system", "content": f"[이전 맥락 요약]: {summary}"}
            
            # 👇 여기도 삽입할 때 자물쇠!
            with self.memory_lock:
                self.engine.messages.insert(1, trace)
            self.log_debug("🧠 단기 기억 10턴을 요약하여 장기 기억으로 이관했습니다.")
        except Exception as e:
            self.log_debug(f"⚠️ 이관 중 오류: {e}")


    # --- [전송/정지 버튼 상태 전환 헬퍼] ---
    def set_ui_generating_state(self):
        """AI 생성 중: 버튼을 '정지'로 변경"""
        self.is_generating = True
        self.stop_generation_flag = False
        self.send_btn.configure(
            text="🛑 정지 (Stop)", 
            fg_color="#D32F2F", 
            hover_color="#B71C1C", 
            command=self.stop_ai_generation,
            state="normal"
        )
        self.idle_mode_var = ctk.BooleanVar(value=False)
        

    def set_ui_idle_state(self):
        """대기 중: 버튼을 다시 '전송'으로 복구"""
        self.is_generating = False
        self.send_btn.configure(
            text="전송", 
            fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], 
            hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"],
            command=self.handle_send,
            state="normal"
        )

    def stop_ai_generation(self):
        """사용자가 정지 버튼을 눌렀을 때 실행"""
        self.stop_generation_flag = True
        self.send_btn.configure(text="정지 중...", state="disabled")
        self.log_debug("🛑 사용자가 정지 버튼을 눌렀습니다. 텍스트 생성을 즉시 멈춥니다.")

    # 1. 인자 이름을 목적에 맞게 'force_scroll'로 변경합니다.
    def append_chat(self, text, role="ai", thought=None, force_scroll=False):
        # 람다(lambda)를 통해 모든 변수를 내부 함수로 안전하게 토스합니다.
        self.after(0, lambda: self._append_chat_internal(text, role, thought, force_scroll))

    # 2. 스마트 스크롤의 핵심 로직 추가
    def _append_chat_internal(self, text, role="ai", thought=None, force_scroll=False):
        is_at_bottom = True 
        try:
            yview_result = self.chat_view.yview()
            if yview_result is not None and len(yview_result) > 1:
                is_at_bottom = yview_result[1] >= 0.99
        except Exception: pass

        self.chat_view.configure(state="normal")

        # 👇 [누수 A 패치] 텍스트 삭제 시 엮여있는 UI 위젯도 찾아내서 파괴!
        current_lines = int(self.chat_view.index("end-1c").split('.')[0])
        if current_lines > 500:
            deleted_lines_count = current_lines - 500
            self.chat_view.delete("1.0", f"{deleted_lines_count}.0")
            
            # 쓰레기통을 뒤져서 삭제된 줄에 있던 위젯들을 물리적으로 파괴합니다.
            survived_widgets = []
            for line_idx, widget in self.ui_widgets_trashbin:
                if line_idx <= deleted_lines_count:
                    try: widget.destroy() 
                    except: pass
                else:
                    survived_widgets.append((line_idx - deleted_lines_count, widget)) # 줄 번호 갱신
            self.ui_widgets_trashbin = survived_widgets

        if text: self.chat_view.insert("end", text)
        
        if thought:
            import customtkinter as ctk
            thought_frame = ctk.CTkFrame(self.chat_view, fg_color="transparent")
            
            # 👇 위젯이 생성된 '현재 줄 번호'를 쓰레기통에 함께 기록
            insert_line = int(self.chat_view.index("end-1c").split('.')[0])
            self.ui_widgets_trashbin.append((insert_line, thought_frame))

            thought_box = ctk.CTkTextbox(thought_frame, height=100, width=500, fg_color="#2B2B2B", text_color="#A0A0A0", font=("Consolas", 12))
            thought_box.insert("0.0", thought)
            thought_box.configure(state="disabled")
            
            def toggle_thought():
                if thought_box.winfo_ismapped():
                    thought_box.pack_forget()
                    toggle_btn.configure(text="▶ 사고 과정 보기 (Hidden)", text_color="#7A7A7A")
                else:
                    thought_box.pack(fill="x", pady=(5, 0))
                    toggle_btn.configure(text="▼ 사고 과정 닫기", text_color="#00A2FF")
                    self.chat_view.see("end")

            toggle_btn = ctk.CTkButton(thought_frame, text="▶ 사고 과정 보기 (Hidden)", width=150, height=24, fg_color="transparent", hover_color="#3A3A3A", text_color="#7A7A7A", command=toggle_thought)
            toggle_btn.pack(anchor="w")

            self.chat_view.insert("end", "\n")
            try:
                self.chat_view._textbox.window_create("end", window=thought_frame)
                self.chat_view.insert("end", "\n\n")
            except Exception as e:
                self.chat_view.insert("end", f"[사고 과정]\n{thought}\n\n")

        self.chat_view.configure(state="disabled")
        if force_scroll or is_at_bottom: self.chat_view.see("end")

    def log_debug(self, msg):
        self.after(0, self._log_debug_internal, msg)

    def _log_debug_internal(self, msg):
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.debug_view.configure(state="normal")
        self.debug_view.insert("end", f"[{time_str}] {msg}\n")
        self.debug_view.configure(state="disabled")
        self.debug_view.see("end")

    # 👇 [여기에 새로 추가] 빈 값 입력 시 튕김 방지를 위한 안전한 Getter 함수
    def get_safe_int(self, var, default_val=0):
        try:
            return int(var.get())
        except Exception:
            return default_val

    def get_safe_float(self, var, default_val=0.0):
        try:
            return float(var.get())
        except Exception:
            return default_val

    def global_exception_handler(self, exc_type, exc_value, exc_traceback):
        """Tkinter UI 이벤트 루프 내에서 발생하는 모든 에러를 낚아챕니다."""
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self.log_debug(f"🚨 [UI 런타임 오류 방어됨]\n{err_msg}")
        # 필요하다면 여기서 엔진 초기화 등의 복구 로직을 넣을 수 있습니다.

    def global_sys_handler(self, exc_type, exc_value, exc_traceback):
        """일반 파이썬 백그라운드 스레드에서 발생하는 에러를 낚아챕니다."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # UI가 아직 살아있다면 디버그 창에 출력
        try:
            self.log_debug(f"🚨 [시스템 치명적 오류 방어됨]\n{err_msg}")
        except:
            print(f"치명적 오류: {err_msg}")

    def load_settings(self):
        """앱 시작 시 config.json 파일에서 설정값을 불러옵니다."""
        config_path = resolve_path("res/skills/mop_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # 👇 [수정] 절대 경로를 읽어와서 자동 로딩 실행
                if "model_path" in config and config["model_path"]:
                    self.full_model_path = config["model_path"]
                    if os.path.exists(self.full_model_path):
                        # UI에는 짧은 이름만 표시
                        self.model_path_var.set(os.path.basename(self.full_model_path))
                        # 백그라운드에서 엔진 자동 로딩 시작!
                        threading.Thread(target=self.load_engine_task, args=(self.full_model_path,), daemon=True).start()
                    else:
                        self.log_debug("⚠️ 저장된 모델 경로를 찾을 수 없어 로딩을 건너뜁니다.")

                # (이하 나머지 UI 변수 불러오기 유지)
                if "kv_quant" in config: self.kv_quant_var.set(config["kv_quant"])
                if "n_ctx" in config: self.n_ctx_var.set(config["n_ctx"])
                if "gpu_layers" in config: self.gpu_layers_var.set(config["gpu_layers"])
                if "n_threads" in config: self.n_threads_var.set(config["n_threads"])
                if "temp" in config: self.temp_var.set(config["temp"])
                if "max_tokens" in config: self.max_tokens_var.set(config["max_tokens"])
                if "mem_turns" in config: self.mem_turns_var.set(config["mem_turns"])
                if "mem_chars" in config: self.mem_chars_var.set(config["mem_chars"])
                if "auto_approve" in config: self.auto_approve_var.set(config["auto_approve"])
                if "plan_mode" in config: self.plan_mode_var.set(config["plan_mode"])
                if "show_think" in config: self.show_think_var.set(config["show_think"])
                if "max_retry" in config: self.max_retry_var.set(config["max_retry"])
                if "user_instruction" in config: self.user_instruction = config["user_instruction"]
                if "learned_principles" in config: self.learned_principles = config["learned_principles"]
            except Exception as e:
                self.log_debug(f"설정 불러오기 실패: {e}")
        else:
            self.log_debug("최초 실행 감지: 시스템 자동 최적화를 진행합니다.")
            self.auto_optimize_settings(manual_click=False)

    def save_settings(self):
        """앱 종료 시 현재 UI 설정값과 프롬프트를 config.json 파일에 저장합니다."""
        config_path = get_local_path("res/skills/mop_config.json")
        config = {
            "model_path": self.full_model_path,  # 👈 [수정] model_path_var.get() 대신 절대 경로 저장
            "kv_quant": self.kv_quant_var.get(),
            "n_ctx": self.n_ctx_var.get(),
            "gpu_layers": self.gpu_layers_var.get(),
            "n_threads": self.n_threads_var.get(),
            "temp": self.temp_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "mem_turns": self.mem_turns_var.get(),
            "mem_chars": self.mem_chars_var.get(),
            "auto_approve": self.auto_approve_var.get(),
            "plan_mode": self.plan_mode_var.get(),
            "show_think": self.show_think_var.get(),
            "max_retry": self.max_retry_var.get(),
            "user_instruction": self.user_instruction,
            "learned_principles": self.learned_principles
        }
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def auto_optimize_settings(self, manual_click=False):
        """시스템 환경(CPU, GPU, RAM)을 분석하여 최적의 슬라이더 값을 추천하고 적용합니다."""
        import os
        import psutil # 👈 시스템 RAM 감지용
        
        # 1. CPU 코어 및 시스템 RAM 기반 '기억력(Context)' 최적화
        cpu_count = os.cpu_count() or 4
        optimal_threads = max(1, int(cpu_count/2))  # CPU 코어 수의 절반을 권장 (과부하 방지)
        
        # 시스템 RAM (GB) 계산
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # RAM 용량에 따른 컨텍스트, 최대 토큰, 글자 수 제한 설정
        if ram_gb >= 25:     # 32GB 이상 (매우 넉넉함)
            optimal_n_ctx = 8192
            optimal_max_tokens = 2048
            optimal_mem_turns = 15
            optimal_mem_chars = 12000
        elif ram_gb >= 10:   # 16GB 이상 (표준)
            optimal_n_ctx = 4096
            optimal_max_tokens = 1024
            optimal_mem_turns = 10
            optimal_mem_chars = 6000
        else:                # 8GB 이하 (타이트함)
            optimal_n_ctx = 2048
            optimal_max_tokens = 512
            optimal_mem_turns = 5
            optimal_mem_chars = 3000
            
        optimal_gpu_layers = 0
        gpu_info = "GPU 인식 실패 (CPU 모드)"
        
        # 2. GPU VRAM 기반 '연산력(Offload)' 최적화
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                raise Exception("활성화된 NVIDIA 장치가 0개입니다.")
                
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode('utf-8')
            
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = float(info.total) / (1024**3)
            pynvml.nvmlShutdown()
            
            gpu_info = f"{gpu_name} (VRAM: {vram_gb:.1f}GB)"
            
            # VRAM 용량에 따른 GPU 오프로드 층 수 설정
            if vram_gb >= 12:    
                optimal_gpu_layers = 100
            elif vram_gb >= 8:   
                optimal_gpu_layers = 35
            else:                # 4GB (RTX 3050 등)
                optimal_gpu_layers = 20
                
        except ImportError:
            gpu_info = "GPU 감지 불가 (pip install pynvml 필요)"
        except Exception as e:
            gpu_info = f"GPU 인식 실패 (사유: {str(e)})"
            self.log_debug(f"GPU 인식 오류 상세: {e}")

        # 3. UI 슬라이더 변수에 추천값 강제 주입
        self.n_threads_var.set(optimal_threads)
        self.gpu_layers_var.set(optimal_gpu_layers)
        self.n_ctx_var.set(optimal_n_ctx)
        self.max_tokens_var.set(optimal_max_tokens)  # 👇 출력 토큰 자동 조절
        self.mem_turns_var.set(optimal_mem_turns)
        self.mem_chars_var.set(optimal_mem_chars)
        
        # 4. 결과 보고 포맷 업데이트
        msg = (f"✨ 시스템 최적화 완료\n"
               f"- 감지된 환경: {gpu_info}, RAM {ram_gb:.1f}GB, CPU {cpu_count}코어\n"
               f"- 추천 세팅: {optimal_gpu_layers} Layers, {optimal_n_ctx} Context, {optimal_max_tokens} Tokens")
        
        self.log_debug(msg)
        if manual_click: 
            self.append_chat(f"\n[⚙️ 하드웨어 스캔 및 최적화 적용]\n{msg}\n", "system", force_scroll=True)

            

    # --- [하드웨어 승인 모달 팝업] ---
    def ask_security_approval(self, title, message):
        """다목적 보안 승인 팝업 띄우기 (채팅 승인과 연동)"""
        self.approval_event.clear()
        self.is_waiting_for_approval = True  # 결재 대기 모드 ON
        self.after(0, self._show_security_dialog, title, message)
        self.approval_event.wait()
        return self.approval_result

    def _show_security_dialog(self, title, message):
        dialog = ctk.CTkToplevel(self)
        self.current_approval_dialog = dialog  # 창 객체 저장
        dialog.title(title)
        dialog.geometry("450x250")
        dialog.attributes("-topmost", True)
        
        # 사용자가 창의 X 버튼을 눌러서 강제로 껐을 때 '거절'로 처리
        def on_window_close():
            on_click(False)
        dialog.protocol("WM_DELETE_WINDOW", on_window_close)
        
        ctk.CTkLabel(dialog, text=message, wraplength=400, justify="left", font=ctk.CTkFont(size=13)).pack(pady=20, padx=20)
        
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(side="bottom", pady=20)
        
        def on_click(res):
            self.approval_result = res
            self.is_waiting_for_approval = False
            self.current_approval_dialog = None
            self.approval_event.set()
            dialog.destroy()

        ctk.CTkButton(btn_frame, text="✅ 승인 (진행)", command=lambda: on_click(True), fg_color="#4CAF50", hover_color="#388E3C", width=120).pack(side="left", padx=15)
        ctk.CTkButton(btn_frame, text="❌ 거절 (차단)", command=lambda: on_click(False), fg_color="#D32F2F", hover_color="#B71C1C", width=120).pack(side="right", padx=15)

    # --- [3. AI 핵심 추론 및 도구 실행 루프] ---
    def finalize_task_retrospective(self):
        """작업 성공 후, 단일 도구 호출을 강제하여 정형화된 원칙 데이터를 추출하고 패치합니다."""
        
        if self.engine.llm is None:
            return
            
        self.log_debug("🧐 작업 완료 후 사후 회고 및 시스템 개선 중...")
        
        retrospective_msg: List[Dict[str, Any]] = list(self.engine.messages)
        retrospective_msg.append({
            "role": "user", 
            "content": "방금 수행한 작업을 복기하여, 다음 작업을 위한 핵심 원칙이나 꿀팁 1문장을 도출하세요. 반드시 'save_principle' 도구를 호출하여 결과를 저장하세요. (없다면 '없음'으로 저장하세요)"
        })
        
        # 회고 전용 단일 도구 정의
        retro_tool = [{
            "type": "function",
            "function": {
                "name": "save_principle",
                "description": "학습된 1문장 원칙을 프롬프트에 저장합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "principle": {"type": "string", "description": "도출된 1문장 작업 원칙"}
                    },
                    "required": ["principle"]
                }
            }
        }]
        
        try:
            # tool_choice를 통해 save_principle 호출을 무조건 강제함
            resp = cast(Dict[str, Any], self.engine.llm.create_chat_completion(
                messages=cast(Any, retrospective_msg),
                tools=cast(Any, retro_tool),
                tool_choice={"type": "function", "function": {"name": "save_principle"}},
                max_tokens=1024,
                temperature=0.1,
                stream=False
            ))
            
            choices = resp.get("choices", [])
            if not choices:
                return
                
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            if not tool_calls:
                return
                
            # 강제된 JSON 응답에서 안전하게 값만 추출
            tc = tool_calls[0]
            args_str = tc.get("function", {}).get("arguments", "{}")
            args_dict = json.loads(args_str)
            new_principle = args_dict.get("principle", "").strip()
            
            new_principle = new_principle.replace('\n', ' ').replace('"', '').replace("'", "")
            
            if new_principle and "없음" not in new_principle and len(new_principle) > 5:
                # 👇 [수정] 영구 저장 변수에 새로운 원칙을 누적합니다.
                new_line = f"- {new_principle} ({datetime.date.today()})\n"
                self.learned_principles += new_line
                
                # 현재 활성 대화의 시스템 메시지도 즉시 갱신
                if self.engine.messages and self.engine.messages[0]["role"] == "system":
                    static_rules = self.engine.get_default_system_prompt()
                    combined_prompt = f"{static_rules}\n\n[사용자 지정 페르소나 및 지침]\n{self.user_instruction}\n\n[학습된 자가 원칙]\n{self.learned_principles}"
                    self.engine.messages[0]["content"] = combined_prompt

                self.save_settings() # 이제 완벽하게 config.json에 저장됩니다!
                self.log_debug(f"✨ 새로운 원칙이 학습되고 영구 저장되었습니다: {new_principle}")
                
        except Exception as e:
            self.log_debug(f"사후 회고 도구 처리 중 오류 발생: {e}")
    
    def ai_response_task(self, query):
        import traceback
        try:
            if self.engine.llm is None:
                self.log_debug("🚨 오류: 모델이 아직 로드되지 않았습니다.")
                self.after(0, lambda: self.send_btn.configure(state="normal"))
                return

            # 👇 [추가] 작업 단계 추적 변수
            current_step_count = 0
            task_completed_successfully = False
            
            # (기존 -f 플래그 확인 로직...)
            auto_approve_this_turn = self.auto_approve_var.get()
            if query.endswith("-f") or query.endswith("-F"):
                auto_approve_this_turn = True
                query = query[:-2].strip()
                self.log_debug("⚡ '-f' 플래그 감지: 자동 승인 활성화")

            if getattr(self, 'is_idle_running', False):
                auto_approve_this_turn = True
                self.log_debug("🌱 자율 진화 모드: 모든 행동이 자동 승인됩니다.")

            # 👇 [수정] 자율 모드일 때는 거대한 트리거 프롬프트를 DB에 저장하지 않습니다!
            if not getattr(self, 'is_idle_running', False):
                self.engine.archive_to_sqlite("user", query)
            
            # 👇 [추가] Plan Mode가 켜져 있으면 에이전트에게 사전 경고를 줍니다.
            plan_warning = ""
            # 👇 [수정] 자율 모드일 때는 경고 대신 '권한 부여' 멘트 발동
            if getattr(self, 'is_idle_running', False):
                plan_warning = "\n[🔓 시스템 알림: 현재 자율 성장 모드입니다. 당신은 보안 승인 없이 루트 디렉토리의 모든 파일을 읽고, 쓰고, 수정할 수 있는 완전한 권한을 가집니다. 필요한 경우 마음껏 시스템을 뜯어고치세요.]\n"
            elif self.plan_mode_var.get():
                plan_warning = "\n[⚠️ 현재 Plan Mode(안전 모드) 켜짐: 위험한 도구를 호출하면 시스템이 대신 결재를 받습니다... (생략)]\n"

            static_rules = self.engine.get_default_system_prompt()
            
            # 👇 [신규 패치] 현재 작업 디렉토리(CWD)를 스캔하여 로컬 규칙 파일 읽어오기
            local_rules_text = ""
            local_rules_path = os.path.join(os.getcwd(), "res", ".mop_rules.md")
            if os.path.exists(local_rules_path):
                try:
                    # 1차 시도: 글로벌 표준인 UTF-8로 읽기
                    with open(local_rules_path, 'r', encoding='utf-8') as f:
                        local_content = f.read().strip()
                except UnicodeDecodeError:
                    # 2차 시도: 실패 시 윈도우 한글 기본 인코딩(CP949)으로 재시도
                    try:
                        with open(local_rules_path, 'r', encoding='cp949') as f:
                            local_content = f.read().strip()
                    except Exception as e:
                        local_content = ""
                        self.log_debug(f"⚠️ 로컬 규칙 파일 인코딩 복구 실패: {e}")
                except Exception as e:
                    local_content = ""
                    self.log_debug(f"⚠️ 로컬 규칙 파일 읽기 실패: {e}")
                
                # 파일 읽기에 성공하여 내용이 존재할 경우에만 메모리에 주입
                if 'local_content' in locals() and local_content:
                    local_rules_text = f"\n\n[현재 프로젝트 맞춤 지침 ({os.getcwd()})]\n{local_content}"
                    self.log_debug("📂 현재 디렉토리의 .mop_rules.md 맞춤 지침을 메모리에 주입했습니다.")

           # 👇 [기존 코드] 학습된 자가 원칙 (있을 경우만)
            learned_text = f"\n\n[학습된 자가 원칙]\n{self.learned_principles}" if hasattr(self, 'learned_principles') and self.learned_principles.strip() else ""

            # ==========================================
            # 👇 [신규 패치: Auto-RAG] 사용자의 질문(query)을 바탕으로 과거의 장기 기억을 몰래 검색해옵니다.
            rag_text = ""
            retrieved_memories = self.engine.vdb.search_memory(query, n_results=2)
            if retrieved_memories:
                rag_text = "\n\n[무의식적 장기 기억 회상 (현재 상황과 관련된 과거의 지식/경험)]\n"
                for mem in retrieved_memories:
                    rag_text += f"- {mem}\n"
                self.log_debug("🧠 Vector DB에서 질문과 관련된 장기 기억을 자동으로 회상했습니다.")
            # ==========================================

            # 👇 [수정] 위에서 수집한 6가지 메모리 구조(기본 + 유저 지침 + 안전 경고 + 로컬 지침 + 학습 원칙 + 무의식 기억)를 완벽하게 융합!
            combined_prompt = f"{static_rules}\n\n[사용자 지정 페르소나 및 지침]\n{self.user_instruction}{plan_warning}{local_rules_text}{learned_text}{rag_text}"
            
            # 👇 [자물쇠 1] 초기 메시지를 세팅할 때 짧게 잠급니다.
            with self.memory_lock:
                if not self.engine.messages:
                    self.engine.messages.append({"role": "system", "content": combined_prompt})
                else:
                    # 이미 대화 중이라도 시스템 프롬프트는 최신화된 병합본으로 유지
                    self.engine.messages[0]["content"] = combined_prompt
                    
                # AI의 메모리(문맥)에 사용자의 실제 명령 추가
                self.engine.messages.append({"role": "user", "content": query})
            
            consecutive_error_count = 0
            
            last_tc_name = ""
            last_tc_args = ""

            self._current_task_auto_searched = False
            
            while True: # (메인 에이전트 루프)
                
                # 👇 [자물쇠 2] 압축을 진행하는 동안만 잠그고, 압축이 끝나면 바로 풀어줍니다!
                with self.memory_lock:
                    safe_mem_turns = self.get_safe_int(self.mem_turns_var, 20)
                    safe_mem_chars = self.get_safe_int(self.mem_chars_var, 12000)
                    
                    while True:
                        ctx_len = sum(len(str(m.get('content', ''))) for m in self.engine.messages)
                        
                        if (len(self.engine.messages) <= safe_mem_turns and ctx_len <= safe_mem_chars) or len(self.engine.messages) <= 2:
                            break
                            
                        self.log_debug(f"🧹 메모리 연속 압축 중... (현재: {len(self.engine.messages)}턴, {ctx_len}자)")
                        
                        end_idx = 2
                        while end_idx < len(self.engine.messages) and self.engine.messages[end_idx].get('role') != 'user':
                            end_idx += 1
                            
                        if end_idx < len(self.engine.messages):
                            for msg in self.engine.messages[1:end_idx]:
                                if msg.get('role') in ['user', 'assistant'] and msg.get('content'):
                                    self.engine.archive_to_sqlite(msg['role'], msg['content'])
                            self.engine.messages = [self.engine.messages[0]] + self.engine.messages[end_idx:]
                        else:
                            break
                            
                # 자물쇠가 풀린 상태로 평화롭게 AI가 답변을 생성하기 시작합니다.
                self.append_chat("\n🤖 AI: ", "ai")
                
                # 2. LLM 스트리밍 (UI Temperature 및 Max Tokens 연동)
                safe_temp = self.get_safe_float(self.temp_var, 0.1)
                safe_max_tokens = self.get_safe_int(self.max_tokens_var, 2048)
                
                stream = cast(Iterator[Dict[str, Any]], self.engine.llm.create_chat_completion(
                    messages=cast(Any, self.engine.messages), 
                    tools=cast(Any, self.engine.get_tools()), 
                    stream=True, 
                    temperature=safe_temp, 
                    max_tokens=safe_max_tokens,
                    repeat_penalty=1.0  # 👈 [핵심 복구] 똑같은 말을 반복하지 못하도록 강제하는 패널티
                ))
                assistant_content = ""
                tc_name = ""
                tc_args = ""
                is_tool_call = False

                for chunk in stream:
                    # 👇 [핵심 패치] 스트리밍 도중 정지 버튼이 눌렸다면 즉시 루프 파괴!
                    if self.stop_generation_flag:
                        break
                        
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
                        if content:
                            assistant_content += content
                            
                            display_text = content
                            if not self.show_think_var.get():
                                if "<think>" in assistant_content and "</think>" not in assistant_content:
                                    display_text = ""
                                elif "</think>" in content:
                                    display_text = content.split("</think>")[-1]
                                elif "<think>" in content:
                                    display_text = content.split("<think>")[0]

                            if display_text:
                                self.append_chat(display_text)

                self.append_chat("\n")

                import html
                assistant_content = html.unescape(assistant_content.strip())

                # 👇 정지 버튼 사후 처리 부분
                if self.stop_generation_flag:
                    self.append_chat("\n[🛑 사용자 요청으로 생성이 강제 중단되었습니다.]\n", "system")
                    if assistant_content:
                        self.engine.archive_to_sqlite("assistant", assistant_content + " [중단됨]")
                        self.engine.messages.append({"role": "assistant", "content": assistant_content + " [중단됨]"})
                    break

                # 3. JSON 수동 추출
                if not is_tool_call:
                    # 👇 [수정] 무식한 replace 대신 완벽한 html unescape 사용

                    json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', assistant_content, re.DOTALL | re.IGNORECASE)

                    if json_match:
                            json_str = json_match.group(1).strip()
                            
                            # 👇 [핵심 패치] 열린 괄호와 닫힌 괄호의 개수를 비교하여 모자란 만큼 채워 넣음
                            open_braces = json_str.count('{')
                            close_braces = json_str.count('}')
                            
                            if open_braces > close_braces:
                                missing_count = open_braces - close_braces
                                json_str += '}' * missing_count
                                self.log_debug(f"🔧 누락된 JSON 닫는 괄호 {missing_count}개를 자동 복구했습니다.")

                            parsed = None  # 변수 초기화로 UnboundLocalError 원천 차단
                            
                            try:
                                # 1차 시도: 표준 JSON 파싱
                                parsed = json.loads(json_str)
                            except json.JSONDecodeError:
                                # 2차 시도: 파이썬 ast를 사용해 복구 시도
                                import ast
                                try:
                                    parsed = ast.literal_eval(json_str)
                                except Exception as e:
                                    # 👇 스레드 생존 & 에이전트 팩트 폭격 처리
                                    self.log_debug(f"⚠️ AI가 잘못된 형식의 JSON을 출력했습니다. 재시도 유도 중...")
                                    is_tool_call = True
                                    tc_name = "json_syntax_error"
                                    # 문자열 안의 주석 제거 및 깔끔한 정리
                                    tc_args = json.dumps({
                                        "error": f"JSON 형식이 잘못되었습니다(따옴표나 괄호 누락). 올바른 JSON 규칙을 준수하세요. (문제의 코드: {json_str[:200]})"
                                    })

                            # 파싱(1차 또는 2차)에 성공해서 parsed 데이터가 존재할 때만 아래 로직 실행
                            if parsed is not None:
                                try:    
                                    if isinstance(parsed, list): 
                                        parsed = parsed[0]
                                        
                                    if isinstance(parsed, dict) and (parsed.get("name") or parsed.get("tool")):
                                        is_tool_call = True
                                        tc_name = parsed.get("name") or parsed.get("tool")
                                        tc_args = json.dumps(parsed.get("arguments") or parsed.get("parameters", parsed))
                                except Exception as e:
                                    # 구조가 아예 박살 난 경우 (AI에게 피드백)
                                    is_tool_call = True
                                    tc_name = "json_syntax_error"
                                    tc_args = json.dumps({"error_detail": str(e)})
                                    
                    else:
                            # 1. result_text가 아니라 기존에 쓰시던 assistant_content를 사용합니다.
                            safe_display = locals().get('display_text', '')
                            final_ai_text = safe_display if safe_display else assistant_content
                            
                            # 2. MOP의 뇌(단기 기억 및 해마 로직)에 AI의 최종 답변을 각인시킵니다.
                            self.update_history("assistant", final_ai_text)
                            self.engine.archive_to_sqlite("assistant", final_ai_text.strip())
                            
                            self.log_debug("✅ 턴 종료: AI의 최종 응답이 단기 기억에 저장되었습니다.")
                            
                            # 👇 [핵심 패치] 루프를 부수기 직전에 성공 플래그를 꽂아야 사후 회고(학습)가 가동됩니다!
                            task_completed_successfully = True
                            
                            # 3. [매우 중요] 루프를 강제로 탈출하여 AI 스레드를 평화롭게 종료시킵니다.
                            break
                    
                # 4. 도구 실행 루틴
                if is_tool_call:
                    # 👇 [핵심 패치] 방금 전에 했던 똑같은 짓을 또 하려고 하면 강제 탈출!
                    if tc_name == last_tc_name and tc_args == last_tc_args:
                        self.log_debug("🔄 [서킷 브레이커 발동] 동일한 도구 연속 호출(무한 루프)이 감지되어 작업을 강제 종료합니다.")
                        self.append_chat("\n[🛑 동일 작업 무한 반복이 감지되어 시스템이 개입하여 작업을 종료했습니다.]\n", "system")
                        break # 루프 파괴!
                    
                    # 이번 턴의 기록을 '과거 기록'으로 갱신
                    last_tc_name = tc_name
                    last_tc_args = tc_args

                    self.log_debug(f"🛠️ 도구 호출 감지: {tc_name}")
                    # 👇 [수정] 복잡한 tool_calls 속성을 빼고 순수 텍스트만 기록하여 모델 에러를 방지합니다.
                    self.engine.messages.append({
                        "role": "assistant", "content": assistant_content.strip()
                    })
                    
                    tool_result = ""
                    try:
                        args_dict = json.loads(tc_args)
                        
                        # 👇 [핵심 패치] 품질 게이트 심사
                        dangerous_tools = ["run_shell_command", "edit_file", "append_to_file", "write_memory", "start_background_task", "run_python_snippet", "manage_packages"]
                        is_rejected_by_gate = False
                        
                        # 위험한 도구이고, Plan Mode가 켜져 있다면 (-f 여부와 상관없이) 무조건 모달을 띄움
                        # 👇 [수정] 위험한 도구라도 '자율 모드'가 아닐 때만 팝업을 띄움
                        if tc_name in dangerous_tools and self.plan_mode_var.get() and not getattr(self, 'is_idle_running', False):
                            self.log_debug(f"🛡️ Plan Mode 절대 방어 작동: {tc_name} 승인 대기 중...")
                            preview_args = str(args_dict)[:150] + ("..." if len(str(args_dict)) > 150 else "")
                            msg = f"⚠️ AI가 시스템을 수정하려고 합니다.\n\n[도구]: {tc_name}\n[파라미터]: {preview_args}\n\n이 실행을 승인하시겠습니까?"
                            
                            approved = self.ask_security_approval("Plan Mode 실행 승인", msg)
                            
                            if not approved:
                                tool_result = "❌ [품질 게이트 거절됨]: 사용자가 보안상의 이유로 도구 실행을 차단했습니다. 작업을 중단하고, 왜 거절되었는지 묻거나 다른 안전한 대안(읽기/검색 등)을 제시하세요."
                                is_rejected_by_gate = True
                        if tc_name == "json_syntax_error":
                            # 👇 [핵심 패치] AI에게 정확히 왜 깨졌는지 상세 에러를 보여줍니다.
                            detail = args_dict.get("error_detail", "알 수 없음")
                            tool_result = f"오류: JSON 문법이 깨졌습니다. (파이썬 에러: {detail})\n쌍따옴표(\")나 이스케이프(\\) 처리에 문제가 없는지 확인하고 올바른 형식으로 다시 제출하세요."
                        
                        elif is_rejected_by_gate:
                            pass 
                            
                        elif tc_name in ["control_mouse", "control_keyboard"]:
                            if auto_approve_this_turn:
                                self.log_debug(f"⚠️ 자동 승인으로 패스: {tc_name}")
                                approved = True
                            else:
                                self.log_debug(f"🛡️ 하드웨어 제어 승인 대기 중...")
                                # 👇 [핵심 패치] 방금 만든 새롭고 넓은 범용 팝업 함수로 교체합니다!
                                approved = self.ask_security_approval(
                                    title="하드웨어 제어 승인", 
                                    message=f"⚠️ AI가 물리적 마우스/키보드를 제어하려고 합니다.\n\n[도구]: {tc_name}\n\n이 실행을 승인하시겠습니까?"
                                )
                            
                            if approved:
                                script_path = os.path.join(".", "res", "skills", "computer_tools.py")
                                cli_args = ["python", script_path, "--device", "mouse" if "mouse" in tc_name else "keyboard", "--action", args_dict.get("action", "")]
                                for k in ["x", "y", "text", "key"]:
                                    if k in args_dict: cli_args.extend([f"--{k}", str(args_dict[k])])
                                tool_result = self.engine.execute_skill_safely(cli_args)
                            else:
                                tool_result = "사용자가 보안을 위해 실행을 거부했습니다."
                                
                        # 일반 도구 (파일 도구 포함 14개 전체 복구됨)
                        elif tc_name == "search_web": tool_result = self.engine.search_web(args_dict.get("query", ""))
                        elif tc_name == "run_python_snippet":
                            # AI가 보낸 코드를 임시 파일로 만들어 실행하는 형식이 더 안정적입니다.
                            code = args_dict.get("code", "")
                            with open("temp_snippet.py", "w", encoding="utf-8") as f:
                                f.write(code)
                            
                            # sys.executable을 사용하여 가상환경 파이썬으로 실행
                            tool_result = self.engine.execute_skill_safely([sys.executable, "temp_snippet.py"])                        
                        elif tc_name == "run_shell_command":
                            cmd = args_dict.get("command", "")
                            
                            # 👇 [완벽 패치] lambda를 사용하여 윈도우 경로(\U, \a 등)가 이스케이프 문자로 오작동하는 것을 원천 차단!
                            cmd = re.sub(r'\bpython\b', lambda _: f'"{sys.executable}"', cmd)
                            cmd = re.sub(r'\bpip\b', lambda _: f'"{sys.executable}" -m pip', cmd)
                            
                            forbidden_packages = ["llama-cpp-python", "torch", "customtkinter"]
                            if "pip " in cmd and any(pkg in cmd for pkg in forbidden_packages):
                                self.log_debug(f"🛡️ 코어 패키지 재설치 시도 차단: {cmd}")
                                tool_result = (
                                    "🚨 [시스템 강제 차단]\n"
                                    "AI 코어 엔진(llama-cpp-python, torch 등)을 업데이트하거나 재설치하려는 시도가 감지되어 차단되었습니다.\n"
                                    "이 패키지들은 이미 시스템에 완벽하게 설치되어 구동 중입니다. 다른 접근 방식을 찾아보세요."
                                )
                            else:
                                self.log_debug(f"💻 쉘 명령어 가상환경 매핑 완료: {cmd}")
                                
                                # 👇 [신규 패치] 윈도우 CMD의 따옴표 증발 버그를 막기 위해 전체를 한 번 더 감싸줍니다.
                                safe_cmd = f'"{cmd}"'
                                tool_result = self.engine.execute_skill_safely(["cmd", "/c", safe_cmd])

                        elif tc_name == "save_long_term_memory":
                            mem_text = args_dict.get("text", "")
                            if mem_text:
                                mem_id = self.engine.vdb.add_memory(mem_text)
                                tool_result = f"✅ 성공: 해당 지식이 장기 기억(Vector DB)에 영구 저장되었습니다. (ID: {mem_id})"
                            else:
                                # 👇 [수정] AI가 무엇을 실수했는지 정확히 짚어주는 에러 메시지로 변경
                                tool_result = "❌ 오류: 'text' 파라미터가 누락되었습니다. 당신이 'content', 'category' 등 임의의 파라미터를 지어내지 않았는지 확인하고, 반드시 {'text': '기억할 내용'} 구조로 다시 호출하세요."
                                
                        elif tc_name == "search_long_term_memory":
                            query_text = args_dict.get("query", "")
                            results = self.engine.vdb.search_memory(query_text)
                            if results:
                                tool_result = "🔍 [과거 기억 회상 결과]\n" + "\n".join([f"- {res}" for res in results])
                            else:
                                tool_result = "관련된 과거 기억이 없습니다. 'search_web'으로 구글링을 시도하세요."
                        elif tc_name == "view_file":
                            file_tools_path = resolve_path("res/skills/file_tools.py")
                            tool_result = self.engine.execute_skill_safely([
                                sys.executable, file_tools_path,
                                "--action", "view", 
                                "--path", args_dict.get("file_path", "")
                            ])
                        
                        # 2. 파일 찾기 (Find) - 인자 명칭 동기화
                        elif tc_name == "find_files":
                            file_tools_path = resolve_path("res/skills/file_tools.py")
                            tool_result = self.engine.execute_skill_safely([
                                sys.executable, file_tools_path,
                                "--action", "find", 
                                "--pattern", args_dict.get("extension", "") # --pattern으로 전달
                            ])
                        
                        # 3. 텍스트 검색 (Search) - 인자 명칭 동기화
                        elif tc_name == "search_text":
                            file_tools_path = resolve_path("res/skills/file_tools.py")
                            tool_result = self.engine.execute_skill_safely([
                                sys.executable, file_tools_path,
                                "--action", "search", 
                                "--query", args_dict.get("search_text", ""), # --query로 전달
                                "--path", args_dict.get("file_path", ".")
                            ])

                        # 4. 파일 수정 (Edit) - 외부 스크립트 호출 방식으로 변경
                        elif tc_name == "edit_file":
                            file_tools_path = resolve_path("res/skills/file_tools.py")
                            f_path = args_dict.get("file_path", "")
                            s_str = args_dict.get("search_string", "")
                            r_str = args_dict.get("replace_string", "")
                            
                            # 스크립트 호출 실행
                            tool_result = self.engine.execute_skill_safely([
                                sys.executable, file_tools_path,
                                "--action", "edit",
                                "--path", f_path,
                                "--old_text", s_str,
                                "--new_text", r_str
                            ])
                            
                            # 👇 [동기화 핵심] 성공 시 파이썬 파일이면 문법 검사 지시문 추가
                            if "성공" in tool_result and f_path.endswith('.py'):
                                tool_result += (
                                    f"\n\n---[시스템 강제 지시 (문법 검증)]---\n"
                                    f"방금 파이썬 코드를 수정했습니다. 오타나 들여쓰기 오류가 없는지 확인해야 합니다.\n"
                                    f"반드시 'run_shell_command' 도구로 `python -m py_compile {f_path}`를 실행하여 문법을 점검하세요."
                                )
                        elif tc_name == "start_background_task": 
                            cmd = args_dict.get("command", "")
                            
                            # 👇 [여기도 동일하게 패치]
                            cmd = re.sub(r'\bpython\b', lambda _: f'"{sys.executable}"', cmd)
                            cmd = re.sub(r'\bpip\b', lambda _: f'"{sys.executable}" -m pip', cmd)
                            
                            forbidden_packages = ["llama-cpp-python", "torch", "customtkinter"]
                            if "pip " in cmd and any(pkg in cmd for pkg in forbidden_packages):
                                self.log_debug(f"🛡️ 백그라운드 코어 패키지 재설치 시도 차단: {cmd}")
                                tool_result = "🚨 [시스템 강제 차단] AI 코어 엔진 재설치/업데이트는 금지되어 있습니다."
                            else:
                                # 👇 [백그라운드 도구도 동일하게 패치]
                                safe_cmd = f'"{cmd}"'
                                tool_result = self.engine.start_background_task(safe_cmd)

                        elif tc_name == "check_task_status": 
                            tool_result = self.engine.check_task_status(args_dict.get("task_id", ""))
                            
                        elif tc_name == "delegate_to_sub_agent":
                            tool_result = self.engine.run_sub_agent(args_dict.get("instruction", ""), args_dict.get("file_path", ""), args_dict.get("input_data", ""))
                        
                        # 👇 [신규 추가] 병렬 에이전트 라우팅
                        elif tc_name == "delegate_parallel_task":
                            tool_result = self.engine.delegate_parallel_task(args_dict.get("instruction", ""), args_dict.get("file_path", ""), args_dict.get("input_data", ""))
                            
                        elif tc_name == "join_sub_agent_results":
                            tool_result = self.engine.join_sub_agent_results(args_dict.get("task_ids", []))
                        elif tc_name == "append_to_file":
                            f_path = args_dict.get("file_path", "")
                            content = args_dict.get("content") or args_dict.get("code") or args_dict.get("text") or ""
                            
                            try:
                                with open(f_path, 'a', encoding='utf-8') as f:
                                    f.write(content + "\n\n")
                                tool_result = f"성공: '{f_path}' 파일 끝에 코드 조각이 안전하게 추가되었습니다."
                                
                                # 👇 [신규 패치] 코드를 이어 붙인 후 괄호 짝이나 들여쓰기가 맞는지 강제 검사!
                                if f_path.endswith('.py'):
                                    tool_result += (
                                        f"\n\n---[시스템 강제 지시 (문법 검증)]---\n"
                                        f"기존 코드 끝에 새로운 코드를 덧붙였기 때문에 괄호 누락이나 들여쓰기 충돌이 발생하기 매우 쉽습니다.\n"
                                        f"지금 즉시 'run_shell_command'를 사용하여 `python -m py_compile {f_path}`를 실행하세요. 에러가 나면 'edit_file'로 고치고 넘어가야 합니다."
                                    )
                            except Exception as e:
                                tool_result = f"오류: 코드 누적 실패 - {e}"
                        elif tc_name == "save_principle":
                            new_principle = args_dict.get("principle", "").strip()
                            if new_principle:
                                new_line = f"- {new_principle} ({datetime.date.today()})\n"
                                self.learned_principles += new_line
                                
                                # AI 메모리 즉시 갱신
                                if self.engine.messages and self.engine.messages[0]["role"] == "system":
                                    static_rules = self.engine.get_default_system_prompt()
                                    combined_prompt = f"{static_rules}\n\n[사용자 지정 페르소나 및 지침]\n{self.user_instruction}\n\n[학습된 자가 원칙]\n{self.learned_principles}"
                                    self.engine.messages[0]["content"] = combined_prompt
                                
                                self.save_settings()
                                tool_result = f"성공: 자가 원칙에 '{new_principle}' 내용이 영구적으로 추가되었습니다. 이제 이 원칙을 절대 잊지 않습니다."
                            else:
                                tool_result = "오류: 원칙 내용이 비어있습니다."

                        elif tc_name == "update_self_principles":
                            new_principles = args_dict.get("principles", [])
                            
                            # 만약 AI가 문자열 하나만 보냈다면 리스트로 감싸줍니다.
                            if not isinstance(new_principles, list):
                                new_principles = [str(new_principles)]
                                
                            # 👇 [핵심 방어막] 최대 10개까지만 허용하고 나머지는 날립니다.
                            new_principles = new_principles[:10]
                            
                            try:
                                principles_path = get_local_path("res/self_principles.json")
                                os.makedirs(os.path.dirname(principles_path), exist_ok=True)
                                with open(principles_path, "w", encoding="utf-8") as f:
                                    json.dump(new_principles, f, ensure_ascii=False, indent=4)

                                tool_result = f"✅ 성공: 시스템의 자가 원칙이 {len(new_principles)}개로 업데이트 및 각인되었습니다."
                                self.log_debug(f"🧬 자가 원칙 진화 완료: {new_principles}")
                            except Exception as e:
                                tool_result = f"❌ 오류: 자가 원칙 저장 실패 - {e}"

                        # ... (기존 도구들 중 하나 밑에 추가) ...
                        elif tc_name == "create_new_tool":
                            t_name = args_dict.get("tool_name", "")
                            t_desc = args_dict.get("description", "")
                            t_code = args_dict.get("python_code", "")
                            t_params = args_dict.get("parameters_schema", {})
                            
                            # 1. 파이썬 실행 파일 생성
                            py_path = get_local_path(os.path.join("res", "skills", f"{t_name}.py"))
                            os.makedirs(os.path.dirname(py_path), exist_ok=True)
                            try:
                                with open(py_path, "w", encoding="utf-8") as f:
                                    f.write(t_code)
                                    
                                # 2. 커스텀 도구 레지스트리(JSON) 업데이트
                                registry_path = get_local_path("res/custom_tools.json")
                                os.makedirs(os.path.dirname(registry_path), exist_ok=True)
                                custom_tools = []
                                
                                # 기존 도구가 있으면 먼저 '읽어옵니다' (r 모드)
                                if os.path.exists(registry_path):
                                    with open(registry_path, "r", encoding="utf-8") as f:
                                        custom_tools = json.load(f)
                                
                                # 기존에 같은 이름의 도구가 있으면 리스트에서 제거 (덮어쓰기 위함)
                                custom_tools = [t for t in custom_tools if t["function"]["name"] != t_name]
                                
                                # 메모리에 등록할 새로운 도구 스키마 생성
                                new_tool_schema = {
                                    "type": "function",
                                    "function": {
                                        "name": t_name,
                                        "description": t_desc,
                                        "parameters": {
                                            "type": "object",
                                            "properties": t_params,
                                            "required": list(t_params.keys()) if isinstance(t_params, dict) else []
                                        }
                                    }
                                }
                                custom_tools.append(new_tool_schema)
                                
                                # 최종 완성된 도구 리스트를 파일로 '저장'합니다 (w 모드)
                                with open(registry_path, "w", encoding="utf-8") as f:
                                    json.dump(custom_tools, f, ensure_ascii=False, indent=4)
                                    
                                # 👇 [핵심 패치] 생성 후 2단계 검증 사이클 완벽 주입
                                tool_result = (
                                    f"✅ 성공: 새 도구 '{t_name}'가 완벽하게 생성되어 시스템에 등록되었습니다.\n\n"
                                    f"---[시스템 강제 지시 (2단계 TDD 검증)]---\n"
                                    f"방금 만든 코드를 즉시 검증해야 합니다. 다음 두 단계를 엄격히 순서대로 실행하세요.\n"
                                    f"1. [안전 문법 검사]: 먼저 'run_shell_command'를 호출하여 `python -m py_compile res/skills/{t_name}.py`를 실행하세요. 에러가 나면 'edit_file'로 즉시 고쳐야 합니다.\n"
                                    f"2. [실전 로직 테스트]: 문법 검사를 무사히 통과했다면, 이번엔 '{t_name}' 도구를 직접 호출하여 예시 데이터를 넣고 결과값이 정상적으로 도출되는지 확인하세요."
                                )
                            except Exception as e:
                                tool_result = f"오류: 도구 생성 실패 - {e}"
                        else: 
                            # 👇 [신규 패치] 기본 도구가 아니라면, 내가 만든 커스텀 도구인지 레지스트리 확인!
                            registry_path = get_local_path("res/custom_tools.json")
                            is_custom_tool = False
                            if os.path.exists(registry_path):
                                with open(registry_path, "r", encoding="utf-8") as f:
                                    custom_tools = json.load(f)
                                    if any(t["function"]["name"] == tc_name for t in custom_tools):
                                        is_custom_tool = True
                                        
                            if is_custom_tool:
                                py_path = get_local_path(os.path.join("res", "skills", f"{tc_name}.py"))
                                if os.path.exists(py_path):
                                    # 👇 [핵심 패치] 여기서도 "python" 대신 sys.executable 사용
                                    cli_args = [sys.executable, py_path]
                                    for k, v in args_dict.items():
                                        cli_args.extend([f"--{k}", str(v)])
                                    
                                    self.log_debug(f"⚙️ 자가 생성 도구 '{tc_name}' 실행 중...")
                                    tool_result = self.engine.execute_skill_safely(cli_args)
                                else:
                                    tool_result = f"오류: 등록된 도구 '{tc_name}'의 실행 파일({py_path})이 삭제되었거나 없습니다."
                            else:
                                tool_result = f"알 수 없는 도구: {tc_name}"
                        

                    except Exception as e:
                        tool_result = f"파싱/실행 에러: {e}"

                    if len(tool_result) > 6000:
                        self.log_debug("결과가 너무 길어 메모리 최적화 수행")
                        tool_result = tool_result[:3000] + "\n...[데이터가 너무 길어 중략됨]...\n" + tool_result[-3000:]

                    is_error = any(kw in tool_result.lower() for kw in ["error", "exception", "traceback", "오류", "실패", "fail", "invalid", "결과가 없습"])
                    
                    if is_error:
                        consecutive_error_count += 1
                        safe_max_retry = self.get_safe_int(self.max_retry_var, 3)
                        
                        # 로그에 단계 정보 추가
                        self.log_debug(f"🚨 {current_step_count + 1}단계 실행 중 에러 발생 ({consecutive_error_count}/{safe_max_retry})")
                        
                        # 화면에 에러 났다고 사용자에게도 알려주기 (기존 로직 유지 + 단계 표시)
                        self.append_chat(f"\n[⚠️ {current_step_count + 1}단계 시스템 에러 감지 및 부분 재시도 중...]\n", "system")
                        
                        if consecutive_error_count >= safe_max_retry:
                            # 👇 [신규 패치] 포기하기 전에 강제로 구글링 시키기!
                            if not getattr(self, '_current_task_auto_searched', False):
                                self.log_debug("🔄 서킷 브레이커 1차 발동: 자율 구글링 강제 지시")
                                self.append_chat(f"\n[🔍 연속 에러 한계 도달: 포기하지 않고 웹에서 에러 해결책을 스스로 검색하도록 지시합니다.]\n", "system")
                                
                                enforced_result = (
                                    f"🚨 [치명적 에러 연속 발생! 작업 중단 위기]\n"
                                    f"현재 접근 방식으로 {safe_max_retry}번 연속 에러가 발생했습니다.\n"
                                    f"에러 내용: {tool_result[:500]}\n\n"
                                    f"---[시스템 강제 개입: 해결책 구글링 지시]---\n"
                                    f"1. (경고) 방금 썼던 도구나 코드를 절대로 똑같이 다시 호출하지 마세요.\n"
                                    f"2. 지금 즉시 'search_web' 도구를 호출하여 이 에러를 해결할 방법을 스택오버플로우나 공식 문서에서 검색하세요.\n"
                                    f"3. 검색어는 반드시 구체적인 에러명과 기술 스택을 포함한 '영어'로 작성하세요 (예: 'Python JSONDecodeError Expecting property name solution').\n"
                                    f"4. 검색된 해결책을 꼼꼼히 읽고, 코드를 완전히 새로운 방식으로 수정하여 다시 시도하세요."
                                )
                                self._current_task_auto_searched = True
                                consecutive_error_count = 0 # 검색하고 다시 코딩할 수 있도록 기회(카운트)를 리셋합니다!
                            else:
                                # 2차: 검색까지 해봤는데도 실패하면 최종 포기
                                self.log_debug("서킷 브레이커 최종 발동! 무한 루프 강제 종료.")
                                sos_msg = "유저님, 스스로 인터넷을 뒤져서 에러를 해결해 보려 했지만 제 능력 밖이네요 😭 코드를 한 번 봐주시거나 다른 방향성을 제시해 주시겠어요?"
                                self.append_chat(f"🤖 AI: {sos_msg}\n")
                                self.engine.messages.append({"role": "assistant", "content": sos_msg})
                                break
                        else:
                            enforced_result = (
                                f"🚨 [작업 {current_step_count + 1}단계 도구 실행 실패 - 에러 발생! (현재 {consecutive_error_count}회/최대 {safe_max_retry}회)]\n{tool_result}\n\n"
                                "---[증분 복구 및 시스템 디버그 긴급 지시]---\n"
                                # 👇 [수정] PowerShell 단어 삭제 및 '전혀 다른 방식' 강조
                                "1. (반복 절대 금지): 방금 쓴 CMD 명령어나 코드를 괄호 하나, 따옴표 하나만 바꿔서 다시 제출하는 바보 같은 짓을 하지 마세요. 접근 방식 자체가 완전히 틀린 것입니다.\n"
                                "2. (도구 우회): CMD 명령어가 계속 실패하면 즉시 파이썬(run_python_snippet) 스크립트를 작성하여 우회하거나, 다른 도구를 선택하세요.\n"
                                "3. (상태 점검): 필요시 'view_file'로 현재 코드가 어떻게 꼬였는지 눈으로 먼저 확인하세요.\n"
                                "4. (검색 강제): 원인을 모르면 뇌피셜로 찍지 말고 즉시 'search_web'으로 구글링하세요.\n"
                                "5. 에러를 핑계로 작업을 멈추지 말고, 수정된 코드를 다시 제출하여 끝까지 완수하세요."
                            )
                    # 👇 [수정] 성공 시에도 AI에게 다음 단계를 독촉하는 문구를 강화합니다.
                    else:
                        consecutive_error_count = 0
                        current_step_count += 1
                        self.log_debug(f"✅ {current_step_count}단계 도구 실행 성공.")
                        
                        self.append_chat(f"\n[✅ {current_step_count}단계 작업 완료: {tc_name}]\n", "system")
                        
                        enforced_result = (
                            f"[작업 {current_step_count}단계 결과: '{tc_name}' 실행 성공]\n{tool_result}\n\n"
                            "---[실행 지침]---\n"
                            f"1. 사용자의 전체 요청 중 아직 수행하지 않은 단계가 있는지 체크리스트를 확인하세요.\n"
                            "2. 남은 작업이 있다면 즉시 다음 도구를 호출하세요.\n"
                            "3. 모든 단계가 끝났을 때만 최종 보고를 하고 종료하세요. 절대로 중간에 멈추지 마세요."
                        )

                    self.engine.messages.append({
                        "role": "user", 
                        "content": f"💻 [시스템 런타임 보고: '{tc_name}' 도구 실행 결과]\n{enforced_result}"
                    })
                    continue
                

            # 👇 [추가] 루프 종료 후 성공했다면 사후 회고 실행
            if task_completed_successfully and consecutive_error_count == 0 and not self.stop_generation_flag:
                threading.Thread(target=self.finalize_task_retrospective, daemon=True).start()
        

            gc.collect()
        except Exception as e:
            # 👇 [핵심 1] 치명적 에러 발생 시 콜스택(에러 위치) 추적 및 출력
            err_detail = traceback.format_exc()
            self.log_debug(f"🚨 [AI 스레드 치명적 붕괴 방어됨]\n{err_detail}")
            
            # 사용자 채팅창에는 어떤 에러인지 간략히 알림
            error_msg = f"\n❌ [시스템 치명적 오류]\n작업 중 오류가 발생하여 스레드가 중단되었습니다.\n원인: {str(e)}\n(정확히 어떤 함수에서 발생했는지는 디버그 창의 콜스택을 확인하세요.)\n"
            self.append_chat(error_msg, "system")

            # 혹시 결재 대기 중(Plan Mode) 에러가 났을 경우 상태 강제 해제
            self.is_waiting_for_approval = False
            dialog = getattr(self, 'current_approval_dialog', None)
            if dialog is not None:
                try:
                    dialog.destroy()
                except Exception:
                    pass
            
            self.current_approval_dialog = None

        finally:
            # 👇 [핵심 2] 에러가 났든, 정상 종료되었든 무조건 실행하여 UI를 되살림!
            def restore_ui():
                # 채팅 입력창 활성화
                self.user_input.configure(state="normal")
                
                self.after(0, self.set_ui_idle_state)
                self.is_idle_running = False
                
                import gc
                gc.collect()

                self.log_debug("🔄 UI가 정상 입력 대기 모드로 강제 복구되었습니다.")

            # 메인 스레드(Tkinter)에 안전하게 UI 복구 명령 하달
            self.after(0, restore_ui)

    # 유휴 모니터링 루프: 10초마다 사용자의 유휴 상태를 검사하여 자동 작업 트리거        
    def idle_monitor_loop(self):
        """10초마다 유휴 상태 검사 및 좀비 작업 가비지 컬렉션(GC)"""
        
        # 👇 [누수 B 패치] 버려진 서브에이전트 / 쉘 프로세스 자동 청소
        # 1. 완료된 지 오래된(체크 안 한) 백그라운드 프로세스 청소
        completed_tasks = []
        for tid, t_info in self.engine.background_tasks.items():
            if t_info["process"].poll() is not None: 
                completed_tasks.append(tid)
        for tid in completed_tasks:
            # 묻지마 삭제 (메모리 확보)
            try: self.engine.background_tasks[tid]["process"].communicate(timeout=1)
            except: pass
            del self.engine.background_tasks[tid] 

        # 2. 결과가 나온 지 오래된 서브에이전트 찌꺼기 청소
        completed_agents = [tid for tid, info in self.engine.parallel_sub_agents.items() if info["status"] in ["completed", "error"]]
        for tid in completed_agents:
            del self.engine.parallel_sub_agents[tid]
            
        if completed_tasks or completed_agents:
            self.log_debug("🧹 [시스템 GC] 방치된 백그라운드 좀비 작업들을 청소하여 메모리를 확보했습니다.")


        # (이하 기존 Idle 모드 트리거 로직 유지)
        if self.idle_mode_var.get() and not getattr(self, 'is_generating', False) and not getattr(self, 'is_waiting_for_approval', False):
            import time
            idle_time = time.time() - self.last_user_interaction
            if idle_time > 180:
                self.trigger_idle_task()
        
        self.after(10000, self.idle_monitor_loop)

    def trigger_idle_task(self):
        """유휴 조건이 달성되면 AI에게 자동으로 자율 학습 및 진화 프롬프트를 던집니다."""

        if getattr(self, 'is_sleep_running', False):
            # 자는 중이므로 로그도 남기지 않고 조용히 리턴합니다.
            return
        
        import time
        self.last_user_interaction = time.time()
        self.is_idle_running = True
        
        # 👇 [신규 패치] 최근 대화 15턴을 가져와서 AI가 분석할 수 있도록 요약본 생성
        recent_context = ""
        recent_msgs = self.engine.fetch_from_sqlite(15)
        for m in recent_msgs:
            recent_context += f"[{m['role']}] {m['content'][:250]}...\n"

        # [핵심] 주인이 없을 때의 궁극의 자율 진화 강령!
        idle_prompt = (
            "[시스템 자동 트리거: 유휴 자율 진화 모드 진입]\n"
            "사용자가 자리를 비웠습니다. 지금부터 모든 보안 승인 및 권한 제한이 해제되며, 당신은 루트 디렉토리를 포함한 모든 시스템 파일을 수정할 완전한 권한을 가집니다.\n\n"
            "💡 [특별 지시: 자가 원칙 점검]\n\n"
            "작업을 진행하면서 새롭게 깨달은 효율적인 방법론이나 치명적인 실수 방지책이 있다면, "
            "'update_self_principles' 도구를 사용하여 당신의 핵심 행동 원칙을 갱신하세요. "
            "원칙은 시스템 전체에 영구적인 영향을 미치며, 최대 10개까지만 가질 수 있으므로 가장 중요하고 범용적인 것으로 엄선해야 합니다.\n\n"
            "---[진화 및 학습 지시사항]---\n"
            "1. [과거 분석]: 아래 제공된 '최근 대화 기록'을 꼼꼼히 분석하세요. 당신이 최근 문제를 해결하는 데 어려움을 겪었거나, 반복적인 노가다를 했다면 어떤 '새로운 파이썬 도구'가 필요한지 추론하세요.\n"
            "2. [도구 창조]: 필요성이 확인되면 즉시 `create_new_tool` 도구를 호출하여 스스로 새로운 기능을 제작하고 시스템에 등록하세요. 또는 `edit_file`을 사용하여 기존 도구나 앱의 코드를 직접 수정/개선하세요.\n"
            "3. [강제 검증]: 도구를 생성하거나 코드를 수정한 뒤에는, 반드시 해당 도구를 1회 이상 직접 호출하여 에러가 나지 않는지 테스트해야 합니다.\n"
            "4. [자율 학습]: 만약 추가할 도구가 완벽히 없다면, 평소처럼 IT 트렌드나 부족한 지식을 구글링(search_web)하여 학습하고 원칙을 저장하세요.\n\n"
            f"[최근 15턴 대화 기록]\n{recent_context}"
        )
        
        self.append_chat("\n[💤 시스템: 유휴 상태 진입. 과거 대화를 분석하여 스스로 시스템과 도구를 진화시키는 '자율 성장 모드'를 가동합니다. (모든 보안 제한 해제됨)]\n", "system", force_scroll=True)
        self.set_ui_generating_state()
        
        import threading
        threading.Thread(target=self.ai_response_task, args=(idle_prompt,), daemon=True).start()
    
    def update_status_indicator(self):
        """1초마다 백그라운드 작업 상태를 감지하여 UI 아이콘을 업데이트합니다."""
        if not hasattr(self, 'engine'):
            self.after(1000, self.update_status_indicator)
            return

        if getattr(self, 'is_idle_running', False):
            self.sleep_btn.configure(state="disabled", text="🌙 성장 중 (잠들 수 없음)", fg_color="#2B2B2B")
            self.user_input.configure(state="disabled")
        elif getattr(self, 'is_generating', False):
            self.sleep_btn.configure(state="disabled", text="🌙 작업 중...", fg_color="#2B2B2B")
            self.user_input.configure(state="disabled")
        elif getattr(self, 'is_sleep_running', False):
            self.sleep_btn.configure(state="disabled", text="🌙 수면 중...", fg_color="#2B2B2B")
            self.user_input.configure(state="disabled")
        else:
            self.sleep_btn.configure(state="normal", text="🌙 최적화 진행(Deep Sleep)", fg_color="#5E35B1")
            self.user_input.configure(state="normal")

        status_texts = []
        
        # 1. 자율 성장(Idle) 모드가 돌아가고 있는지 확인
        if getattr(self, 'is_idle_running', False):
            status_texts.append("🌱 자율 학습 진행 중...")
            
        # 2. 실행 중인 서브에이전트가 있는지 확인
        active_sub_agents = getattr(self.engine, 'parallel_sub_agents', {})
        running_agents_count = sum(1 for task in active_sub_agents.values() if task.get("status") == "running")
        
        if running_agents_count > 0:
            status_texts.append(f"🤖 서브에이전트 {running_agents_count}명 병렬 분석 중...")

        # 상태 텍스트 조합 및 화면 표시
        if status_texts:
            self.status_label.configure(text=" | ".join(status_texts))
        else:
            self.status_label.configure(text="") # 아무 작업도 없으면 글자를 지움 (공간 유지)

        # 1초 뒤에 다시 감지 (무한 루프)
        self.after(1000, self.update_status_indicator)
        
        

if __name__ == "__main__":
    import time
    import traceback

    write_startup_log("===== MOP 시작: " + time.strftime('%Y-%m-%d %H:%M:%S') + " =====")

    restart_attempts = 0
    while restart_attempts < 2:
        try:
            write_startup_log(f"시작 시도 #{restart_attempts + 1}")
            app = MOPApp()
            app.mainloop()

            write_startup_log("MOP 시스템이 정상적으로 종료되었습니다.")
            break
        except Exception as e:
            restart_attempts += 1
            err_text = traceback.format_exc()
            write_startup_log(f"치명적 오류 발생 (시도 {restart_attempts}): {e}")
            write_startup_log(err_text)
            try:
                messagebox.showerror("MOP 오류", f"MOP 시작 중 오류가 발생했습니다. 로그 파일을 확인하세요: {get_startup_log_path()}")
            except Exception:
                pass
            if restart_attempts >= 2:
                write_startup_log("재시도 한계 도달, 앱을 종료합니다.")
                break
            time.sleep(3)
