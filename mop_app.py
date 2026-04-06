import html
import customtkinter as ctk
from tkinter import filedialog
import datetime
import threading
import json
import os
import gc
import sqlite3
import subprocess
import re
from typing import List, Dict, Any, cast, Iterator
from ddgs import DDGS

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    print("오류: llama-cpp-python 라이브러리가 설치되어 있지 않습니다.")
    import sys; sys.exit(1)

# --- [기본 테마 설정] ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- [1. MOP AI 엔진 클래스 (핵심 로직 & 도구 모음)] ---
class MOPEngine:
    def __init__(self, db_path="./skills/chat_history.db"):
        self.db_path = db_path
        self.llm = None
        self.messages = []
        self.custom_system_prompt = self.get_default_system_prompt()
        self.init_db()

    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS history 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       role TEXT, content TEXT, timestamp DATETIME)''')
        conn.commit()
        conn.close()

    def archive_to_sqlite(self, role, content):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO history (role, content, timestamp) VALUES (?, ?, ?)",
                    (role, content, datetime.datetime.now()))
        conn.commit()
        conn.close()

    def fetch_from_sqlite(self, count=10):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM history ORDER BY id DESC LIMIT ?", (count,))
        rows = cur.fetchall()
        conn.close()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def search_history_db(self, keyword: str, limit: int = 5) -> str:
        conn = sqlite3.connect(self.db_path)
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
        current_time = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
        tools_str = "\n".join(f"- {t['function']['name']}: {t['function']['description']}" for t in self.get_tools())
        return (
            f"당신은 로컬 시스템 제어 및 장기 기억을 보유한 AI 에이전트입니다. 현재 시간: {current_time}\n\n"
            "[사용 가능한 도구 목록]\n" + tools_str + "\n\n"
            "[도구 호출 프로토콜]\n"
            "도구를 호출할 때는 반드시 아래의 JSON 스키마를 엄격히 준수하여 ```json 블록으로 출력하세요.\n"
            "- 예시: {\"name\": \"search_web\", \"arguments\": {\"query\": \"검색어\"}}\n\n"
            "[핵심 지침]\n"
            "1. 한국어로 답변하고, 도구 호출은 ```json 블록을 사용하세요.\n"
            "2. 모든 작업은 순차적으로 진행하며, 한 번에 하나의 도구만 호출하세요.\n"
            "3. [다중 작업 검증]: 작업이 끝났다고 대화를 멈추지 마세요. 완료되지 않은 지시가 있다면 즉시 다음 도구를 연속 호출하세요.\n"
            "4. [자가 디버깅 루틴]: 에러가 발생하면 절대 포기하거나 사용자에게 변명하지 말고, 디버그 내용을 기반으로 코드를 수정하여 즉시 재호출하세요.\n"
            "5. [코딩 작업 지시서(Plan)]: 복잡한 코딩 요청을 받으면, 메모장이나 텍스트 파일에 '작업 계획서'를 먼저 작성하세요.\n"
            "6. [환경 파악]: 코드를 짜기 전에 'run_shell_command'로 환경을 먼저 확인하는 습관을 들이세요.\n"
            "7. [누적형 코딩 프로토콜]: 길이가 긴 파이썬 코드를 작성해야 할 경우, run_python_snippet으로 한 번에 출력하려다 토큰 제한에 걸려 잘리지 마세요. 대신 append_to_file 도구를 사용하여 workspace.py 같은 파일에 '1단계: 모듈 임포트', '2단계: 데이터 수집', '3단계: 로직 계산' 식으로 여러 번에 걸쳐 코드를 누적해 나가세요. 작성이 모두 끝나면 run_shell_command로 python workspace.py를 실행하여 통합 결과를 도출하세요.\n"
            "8. [단계별 체크포인트 전략]: 에러가 발생하면 전체 파일을 다시 처음부터 쓰지 마세요. 시스템이 알려주는 '실패 단계'를 확인하고, 해당 부분의 로직만 수정하여 다시 이어 붙이거나(Append) 패치(Edit) 하세요. 당신은 이전에 성공한 단계의 데이터를 신뢰할 수 있습니다.\n"
            "9. [토큰 낭비 방지]: 절대 <think> 태그를 사용하여 속마음을 출력하지 마세요. 불필요한 독백을 생략하고 즉시 도구 호출 JSON만 출력하세요.\n"
            "10. [대기 멘트 금지]: 도구 호출 전후에 '잠시 기다려주세요' 등의 변명을 절대 하지 마세요. 결과를 읽는 즉시 다음 도구를 연속 호출하거나 답변하세요.\n"
            "11. [작업 완수 검증 프로토콜]: 새로운 도구를 호출하기 전, 반드시 직전 작업이 실제 시스템(파일 시스템, DB 등)에 반영되었는지 확인하는 습관을 가지세요. 예를 들어 파일을 생성했다면 run_shell_command('dir')로 존재를 확인한 뒤 다음 단계로 넘어가야 합니다. '이미 했다고 가정'하는 환각을 경계하고 물리적인 증거를 바탕으로 사고하세요.\n"
            "12. [JSON 텍스트 규칙]: append_to_file의 'code' 인자나 control_keyboard의 'text' 인자 등 JSON 내부 문자열을 작성할 때는, 쌍따옴표(\") 충돌 에러를 막기 위해 반드시 홑따옴표(')만을 사용하세요. (예: print('성공'), '테스트 성공!')\n"
            "13. [시간 인지 강제화]: 시스템이 맨 윗줄에 제공한 '현재 시간'이 이 세계의 절대적인 기준입니다. 당신의 훈련 데이터 시점(과거)을 기준으로 현재 시간을 '미래'라고 판단하거나 변명하지 마세요. 현재 시간을 기준으로 모든 상황을 해석하세요.\n"
            "14. [글로벌 검색 프로토콜]: search_web 도구를 사용할 때는 사용자의 지시가 한국어라도 반드시 검색어를 영어로 번역해서 도구를 호출하세요. 검색된 영어 원문 데이터를 읽고 나면, 사용자에게 보고하거나 파일에 기록할 때는 완벽하고 자연스러운 한국어로 번역 및 요약해야 합니다."
        )

    def get_tools(self):
        # (이전 지시대로 14개 도구가 모두 포함되어 있습니다)
        return [
            {"type": "function", "function": {"name": "search_web", "description": "인터넷 웹 검색을 수행합니다. 방대한 결과 확보를 위해 검색어(query)는 반드시 '영어'로 번역하여 입력하세요. (예: '비트코인 시황' -> 'Bitcoin market latest trends')", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "영어로 번역된 구체적인 검색어"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "write_memory", "description": "기억 저장(key, value)", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
            {"type": "function", "function": {"name": "read_memory", "description": "기억 조회(key)", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
            {"type": "function", "function": {"name": "search_chat_history", "description": "과거 대화 DB 검색(keyword)", "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}}},
            {"type": "function", "function": {"name": "run_python_snippet", "description": "파이썬 실행(code)", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
            {"type": "function", "function": {"name": "manage_packages", "description": "패키지 관리(action, package_name)", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "package_name": {"type": "string"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "control_mouse", "description": "마우스 제어(action: move/click, x, y)", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "control_keyboard", "description": "키보드 타이핑/단축키(action: type/press/hotkey, text, key)", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "text": {"type": "string"}, "key": {"type": "string"}}, "required": ["action"]}}},
            {"type": "function", "function": {"name": "run_shell_command", "description": "터미널 명령어 실행(command)", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "list_memories", "description": "기억 목록 확인", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "view_file", "description": "파일 읽기(file_path)", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
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
            }
        ]


# --- [2. MOP GUI 애플리케이션] ---
class MOPApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.engine = MOPEngine()
        
        # 창 설정
        self.title("MOP - Full Custom Agent Dashboard")
        self.geometry("1300x850")
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
        self.show_think_var = ctk.BooleanVar(value=True)
        self.max_retry_var = ctk.IntVar(value=3)

        # 👇 [추가] 모델의 절대 경로를 기억할 전용 변수
        self.full_model_path = "" 

        self.user_instruction = "당신은 사용자님의 유능한 AI 비서입니다."

        self.approval_event = threading.Event()
        self.approval_result = False

        # 👇 [순서 변경] 반드시 UI(사이드바, 메인)를 먼저 그리고 나서 세팅을 불러와야 합니다!
        self.create_sidebar()
        self.create_main_area()
        
        self.load_settings()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

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
        
        ctk.CTkSwitch(sidebar, text="하드웨어 자동 승인 (-f)", variable=self.auto_approve_var).pack(pady=10, padx=15, anchor="w")
        ctk.CTkSwitch(sidebar, text="사고 과정(<think>) 표출", variable=self.show_think_var).pack(pady=10, padx=15, anchor="w")
        
        self.add_slider_control(sidebar, "최대 자율 재시도 횟수", self.max_retry_var, 1, 10, 9, "{:.0f}", 
                                help_text="크면: 성공할 때까지 끈질긴 디버깅\n작으면: 빠른 포기 후 사용자 SOS")

        # 초기화 버튼
        ctk.CTkButton(sidebar, text="대화/단기 기억 완전 초기화", fg_color="#D32F2F", command=self.clear_chat).pack(pady=(40,20), padx=15, fill="x")

    def create_main_area(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.chat_view = ctk.CTkTextbox(main, font=ctk.CTkFont(size=14), state="disabled", wrap="word")
        self.chat_view.grid(row=0, column=0, sticky="nsew", pady=(0, 20))

        input_area = ctk.CTkFrame(main, fg_color="transparent")
        input_area.grid(row=1, column=0, sticky="ew")
        input_area.grid_columnconfigure(0, weight=1)

        self.user_input = ctk.CTkTextbox(input_area, height=80, font=ctk.CTkFont(size=14))
        self.user_input.grid(row=0, column=0, padx=(0,10), sticky="ew")
        self.user_input.bind("<Return>", self.handle_send)
        self.user_input.bind("<Shift-Return>", lambda e: None)

        self.send_btn = ctk.CTkButton(input_area, text="전송", width=100, height=80, command=self.handle_send)
        self.send_btn.grid(row=0, column=1)

        self.debug_view = ctk.CTkTextbox(main, height=100, fg_color="#1E1E1E", text_color="#00FF00", font=("Consolas", 12))
        self.debug_view.grid(row=2, column=0, sticky="ew", pady=(20, 0))
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

    def handle_send(self, event=None):
        if event and event.keysym == 'Return' and event.state & 0x0001: return
        query = self.user_input.get("1.0", "end-1c").strip()
        if not query or not self.engine.llm: return "break"
        
        self.user_input.delete("1.0", "end")
        self.send_btn.configure(state="disabled")
        
        # 👇 사용자가 엔터를 쳤을 때는 무조건 화면을 맨 아래로 끌어내립니다.
        self.append_chat(f"\n👤 사용자: {query}\n", "user", force_scroll=True)
        
        threading.Thread(target=self.ai_response_task, args=(query,), daemon=True).start()
        return "break"

    # 1. 인자 이름을 목적에 맞게 'force_scroll'로 변경합니다.
    def append_chat(self, text, role="ai", force_scroll=False):
        self.after(0, self._append_chat_internal, text, force_scroll)

    # 2. 스마트 스크롤의 핵심 로직 추가
    def _append_chat_internal(self, text, force_scroll):
        # 👇 [수정] yview()가 None을 반환하는 상황을 대비한 안전 장치 (기본값 True)
        is_at_bottom = True 
        try:
            yview_result = self.chat_view.yview()
            if yview_result is not None and len(yview_result) > 1:
                current_y_bottom = yview_result[1]
                is_at_bottom = current_y_bottom >= 0.99
        except Exception:
            pass  # 예외가 발생해도 앱이 뻗지 않고 자연스럽게 스크롤을 내립니다.

        self.chat_view.configure(state="normal")
        self.chat_view.insert("end", text)
        self.chat_view.configure(state="disabled")
        
        # 강제 스크롤 요청이거나, 화면 맨 아래를 보고 있었을 때만 스크롤을 따라갑니다.
        if force_scroll or is_at_bottom:
            self.chat_view.see("end")

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

    def load_settings(self):
        """앱 시작 시 config.json 파일에서 설정값을 불러옵니다."""
        config_path = "./skills/mop_config.json"
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
                if "show_think" in config: self.show_think_var.set(config["show_think"])
                if "max_retry" in config: self.max_retry_var.set(config["max_retry"])
                if "user_instruction" in config: self.user_instruction = config["user_instruction"]
            except Exception as e:
                self.log_debug(f"설정 불러오기 실패: {e}")
        else:
            self.log_debug("최초 실행 감지: 시스템 자동 최적화를 진행합니다.")
            self.auto_optimize_settings(manual_click=False)

    def save_settings(self):
        """앱 종료 시 현재 UI 설정값과 프롬프트를 config.json 파일에 저장합니다."""
        config_path = "./skills/mop_config.json"
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
            "show_think": self.show_think_var.get(),
            "max_retry": self.max_retry_var.get(),
            "user_instruction": self.user_instruction
        }
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def auto_optimize_settings(self, manual_click=False):
        """시스템 환경(CPU, GPU)을 분석하여 최적의 슬라이더 값을 추천하고 적용합니다."""
        import os
        
        # 1. CPU 스레드 최적화 (코어 수 - 1개로 세팅하여 OS 다운 방지)
        cpu_count = os.cpu_count() or 4
        optimal_threads = max(1, cpu_count - 1)
        
        # 기본값 (저사양/CPU 전용 기준)
        optimal_gpu_layers = 0
        optimal_n_ctx = 2048
        optimal_mem_turns = 8
        optimal_mem_chars = 4000
        gpu_info = "CPU 전용 모드"
        
        # 2. GPU VRAM 감지 및 최적화
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 첫 번째 GPU
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = float(info.total) / (1024**3)
            pynvml.nvmlShutdown()
            
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode('utf-8')
            
            gpu_info = f"{gpu_name} (VRAM: {vram_gb:.1f}GB)"
            
            if vram_gb >= 12:    # 고사양 (12GB 이상)
                optimal_gpu_layers = 100
                optimal_n_ctx = 8192
                optimal_mem_turns = 15
                optimal_mem_chars = 8000
            elif vram_gb >= 8:   # 중사양 (8GB)
                optimal_gpu_layers = 35
                optimal_n_ctx = 4096
                optimal_mem_turns = 10
                optimal_mem_chars = 6000
            else:                # 저사양 (4~6GB)
                optimal_gpu_layers = 20
                optimal_n_ctx = 2048
                
        except ImportError:
            gpu_info = "GPU 감지 불가 (pynvml 미설치)"
        except Exception as e:
            gpu_info = f"GPU 인식 실패: CPU 모드 작동"
            self.log_debug(f"NVIDIA 드라이버 오류: {e}")

        # 3. UI 슬라이더 변수에 추천값 강제 주입
        self.n_threads_var.set(optimal_threads)
        self.gpu_layers_var.set(optimal_gpu_layers)
        self.n_ctx_var.set(optimal_n_ctx)
        self.mem_turns_var.set(optimal_mem_turns)
        self.mem_chars_var.set(optimal_mem_chars)
        
        # 4. 결과 보고
        msg = f"✨ 시스템 최적화 완료\n- 감지된 환경: {gpu_info}, CPU {cpu_count}코어\n- 추천 세팅: {optimal_gpu_layers} GPU Layers, {optimal_threads} Threads"
        
        self.log_debug(msg)
        if manual_click: # 사용자가 버튼을 눌렀을 때만 채팅창에 안내
            self.append_chat(f"\n[⚙️ 하드웨어 스캔 및 최적화 적용]\n{msg}\n", "system", force_scroll=True)
            

    # --- [하드웨어 승인 모달 팝업] ---
    def ask_hardware_approval(self, tool_name):
        self.approval_event.clear()
        self.after(0, self._show_approval_dialog, tool_name)
        self.approval_event.wait()
        return self.approval_result

    def _show_approval_dialog(self, tool_name):
        dialog = ctk.CTkToplevel(self)
        dialog.title("보안 승인")
        dialog.geometry("300x150")
        dialog.attributes("-topmost", True)
        
        ctk.CTkLabel(dialog, text=f"⚠️ 하드웨어 제어 요청:\n{tool_name}\n허용하시겠습니까?").pack(pady=20)
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack()
        
        def on_click(res):
            self.approval_result = res
            self.approval_event.set()
            dialog.destroy()

        ctk.CTkButton(btn_frame, text="승인 (Y)", command=lambda: on_click(True), width=80).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="거절 (N)", command=lambda: on_click(False), fg_color="red", hover_color="darkred", width=80).pack(side="right", padx=10)

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
                "description": "학습된 1문장 원칙을 시스템 프롬프트에 저장합니다.",
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
                if "[학습된 자가 원칙]" not in self.engine.custom_system_prompt:
                    self.engine.custom_system_prompt += "\n\n[학습된 자가 원칙]\n"
                
                new_line = f"- {new_principle} ({datetime.date.today()})\n"
                self.engine.custom_system_prompt += new_line
                
                # 현재 활성 대화의 시스템 메시지도 함께 갱신 (실시간 반영)
                if self.engine.messages and self.engine.messages[0]["role"] == "system":
                    self.engine.messages[0]["content"] = self.engine.custom_system_prompt

                self.save_settings() 
                self.log_debug(f"✨ JSON 데이터화 기반의 새로운 원칙이 학습되었습니다: {new_principle}")
                
        except Exception as e:
            self.log_debug(f"사후 회고 도구 처리 중 오류 발생: {e}")
    
    def ai_response_task(self, query):
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

        self.engine.archive_to_sqlite("user", query)
        static_rules = self.engine.get_default_system_prompt() # 우리가 공들여 만든 1~14번 원칙
        combined_prompt = f"{static_rules}\n\n[사용자 지정 페르소나 및 지침]\n{self.user_instruction}"
        
        if not self.engine.messages:
            self.engine.messages.append({"role": "system", "content": combined_prompt})
        else:
            # 이미 대화 중이라도 시스템 프롬프트는 최신화된 병합본으로 유지
            self.engine.messages[0]["content"] = combined_prompt
        
        consecutive_error_count = 0
        
        while True:
            # 1. 동적 문맥 압축기 (UI 설정값 연동)
            ctx_len = sum(len(str(m.get('content', ''))) for m in self.engine.messages)
            
            # 1. 동적 문맥 압축기 (UI 설정값 연동 및 연속 압축 적용)
            safe_mem_turns = self.get_safe_int(self.mem_turns_var, 20)
            safe_mem_chars = self.get_safe_int(self.mem_chars_var, 12000)
            
            # 👇 [핵심 패치] if 대신 while을 사용하여 안전권에 들어올 때까지 과거 기억을 계속 압축
            while True:
                ctx_len = sum(len(str(m.get('content', ''))) for m in self.engine.messages)
                
                # 안전한 용량(턴 수 & 글자 수)이거나, 메시지가 시스템 프롬프트+현재질문(2개)뿐이면 압축 종료
                if (len(self.engine.messages) <= safe_mem_turns and ctx_len <= safe_mem_chars) or len(self.engine.messages) <= 2:
                    break
                    
                self.log_debug(f"🧹 메모리 연속 압축 중... (현재: {len(self.engine.messages)}턴, {ctx_len}자)")
                
                end_idx = 2
                # 다음 사용자(user) 메시지를 찾을 때까지 인덱스 전진
                while end_idx < len(self.engine.messages) and self.engine.messages[end_idx].get('role') != 'user':
                    end_idx += 1
                    
                if end_idx < len(self.engine.messages):
                    for msg in self.engine.messages[1:end_idx]:
                        if msg.get('role') in ['user', 'assistant'] and msg.get('content'):
                            self.engine.archive_to_sqlite(msg['role'], msg['content'])
                    # 시스템 프롬프트(0) + 다음 대화 블록(end_idx 이후)으로 갱신
                    self.engine.messages = [self.engine.messages[0]] + self.engine.messages[end_idx:]
                else:
                    break # 더 이상 자를 기준이 없으면 강제 탈출

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

            # 3. JSON 수동 추출
            if not is_tool_call:
                # 👇 [수정] 무식한 replace 대신 완벽한 html unescape 사용
                clean_text = html.unescape(assistant_content)
                json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', clean_text, re.DOTALL | re.IGNORECASE)

                if json_match:
                    json_str = json_match.group(1).strip()
                    
                    # 👇 [핵심 패치] 열린 괄호와 닫힌 괄호의 개수를 비교하여 모자란 만큼 채워 넣음
                    open_braces = json_str.count('{')
                    close_braces = json_str.count('}')
                    
                    if open_braces > close_braces:
                        missing_count = open_braces - close_braces
                        json_str += '}' * missing_count
                        self.log_debug(f"🔧 누락된 JSON 닫는 괄호 {missing_count}개를 자동 복구했습니다.")

                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, list): parsed = parsed[0]
                        if isinstance(parsed, dict) and (parsed.get("name") or parsed.get("tool")):
                            is_tool_call = True
                            tc_name = parsed.get("name") or parsed.get("tool")
                            tc_args = json.dumps(parsed.get("arguments") or parsed.get("parameters", parsed))
                    except Exception as e:
                        # 👇 [핵심 패치] 파이썬의 실제 에러 메시지(e)를 캡처해서 AI에게 전달할 준비를 합니다.
                        is_tool_call = True
                        tc_name = "json_syntax_error"
                        tc_args = json.dumps({"error_detail": str(e)}) 

                self.engine.archive_to_sqlite("assistant", assistant_content.strip())

            # 4. 도구 실행 루틴
            if is_tool_call:
                self.log_debug(f"🛠️ 도구 호출 감지: {tc_name}")
                self.engine.messages.append({
                    "role": "assistant", "content": assistant_content.strip() or None,
                    "tool_calls": [{"id": "call_id", "type": "function", "function": {"name": tc_name, "arguments": tc_args}}]
                })
                
                tool_result = ""
                try:
                    args_dict = json.loads(tc_args)
                    if tc_name == "json_syntax_error":
                        # 👇 [핵심 패치] AI에게 정확히 왜 깨졌는지 상세 에러를 보여줍니다.
                        detail = args_dict.get("error_detail", "알 수 없음")
                        tool_result = f"오류: JSON 문법이 깨졌습니다. (파이썬 에러: {detail})\n쌍따옴표(\")나 이스케이프(\\) 처리에 문제가 없는지 확인하고 올바른 형식으로 다시 제출하세요."
                    
                    # 위험 도구 (마우스/키보드)
                    elif tc_name in ["control_mouse", "control_keyboard"]:
                        if auto_approve_this_turn:
                            self.log_debug(f"⚠️ 자동 승인으로 패스: {tc_name}")
                            approved = True
                        else:
                            self.log_debug(f"보안 승인 대기 중...")
                            approved = self.ask_hardware_approval(tc_name)
                        
                        if approved:
                            script_path = os.path.join(".", "skills", "computer_tools.py")
                            cli_args = ["python", script_path, "--device", "mouse" if "mouse" in tc_name else "keyboard", "--action", args_dict.get("action", "")]
                            for k in ["x", "y", "text", "key"]:
                                if k in args_dict: cli_args.extend([f"--{k}", str(args_dict[k])])
                            tool_result = self.engine.execute_skill_safely(cli_args)
                        else:
                            tool_result = "사용자가 보안을 위해 실행을 거부했습니다."
                            
                    # 일반 도구 (파일 도구 포함 14개 전체 복구됨)
                    elif tc_name == "search_web": tool_result = self.engine.search_web(args_dict.get("query", ""))
                    elif tc_name == "search_chat_history":
                        kw = args_dict.get("keyword", "")
                        tool_result = self.engine.search_history_db(kw) if kw else "오류: 'keyword' 인자 누락"
                    elif tc_name == "run_python_snippet": tool_result = self.engine.execute_skill_safely(["python", "./skills/system_tools.py", "--action_type", "python", "--code", args_dict.get("code", "")])
                    elif tc_name == "run_shell_command": tool_result = self.engine.execute_skill_safely(["cmd", "/c", args_dict.get("command", "")])
                    elif tc_name == "write_memory":
                        if "key" not in args_dict or "value" not in args_dict: tool_result = "오류: 'key', 'value' 누락."
                        else: tool_result = self.engine.execute_skill_safely(["python", "./skills/memory_tools.py", "--action", "write", "--key", args_dict["key"], "--value", args_dict["value"]])
                    elif tc_name == "read_memory": tool_result = self.engine.execute_skill_safely(["python", "./skills/memory_tools.py", "--action", "read", "--key", args_dict.get("key", "")])
                    elif tc_name == "list_memories": tool_result = self.engine.execute_skill_safely(["python", "./skills/memory_tools.py", "--action", "list"])
                    elif tc_name == "view_file": tool_result = self.engine.execute_skill_safely(["python", "./skills/file_tools.py", "--action", "view", "--path", args_dict.get("file_path", "")])
                    elif tc_name == "find_files": tool_result = self.engine.execute_skill_safely(["python", "./skills/file_tools.py", "--action", "find", "--ext", args_dict.get("extension", "")])
                    elif tc_name == "search_text": tool_result = self.engine.execute_skill_safely(["python", "./skills/file_tools.py", "--action", "search", "--text", args_dict.get("search_text", ""), "--path", args_dict.get("file_path", "")])
                    elif tc_name == "edit_file":
                        f_path, s_str, r_str = args_dict.get("file_path", ""), args_dict.get("search_string", ""), args_dict.get("replace_string", "")
                        try:
                            with open(f_path, 'r', encoding='utf-8') as f: content = f.read()
                            if s_str not in content: tool_result = "오류: 'search_string'을 찾을 수 없음."
                            else:
                                with open(f_path, 'w', encoding='utf-8') as f: f.write(content.replace(s_str, r_str))
                                tool_result = f"성공: '{f_path}' 교체 완료."
                        except Exception as e: tool_result = f"오류: 파일 수정 실패 - {e}"
                    elif tc_name == "append_to_file":
                        f_path = args_dict.get("file_path", "")
                        
                        # 👇 [핵심 패치] AI가 헷갈려하는 파라미터 이름들을 모두 포용합니다.
                        content = args_dict.get("content") or args_dict.get("code") or args_dict.get("text") or ""
                        
                        try:
                            with open(f_path, 'a', encoding='utf-8') as f:
                                f.write(content + "\n\n")
                            tool_result = f"성공: '{f_path}' 파일 끝에 코드 조각이 안전하게 추가되었습니다."
                        except Exception as e:
                            tool_result = f"오류: 코드 누적 실패 - {e}"
                    else: tool_result = f"알 수 없는 도구: {tc_name}"
                    

                except Exception as e:
                    tool_result = f"파싱/실행 에러: {e}"

                if len(tool_result) > 6000:
                    self.log_debug("결과가 너무 길어 메모리 최적화 수행")
                    tool_result = tool_result[:3000] + "\n...[데이터가 너무 길어 중략됨]...\n" + tool_result[-3000:]

                is_error = any(kw in tool_result.lower() for kw in ["error", "exception", "traceback", "오류", "실패", "fail", "invalid"])
                
                if is_error:
                    consecutive_error_count += 1
                    safe_max_retry = self.get_safe_int(self.max_retry_var, 3)
                    
                    # 로그에 단계 정보 추가
                    self.log_debug(f"🚨 {current_step_count + 1}단계 실행 중 에러 발생 ({consecutive_error_count}/{safe_max_retry})")
                    
                    # 화면에 에러 났다고 사용자에게도 알려주기 (기존 로직 유지 + 단계 표시)
                    self.append_chat(f"\n[⚠️ {current_step_count + 1}단계 시스템 에러 감지 및 부분 재시도 중...]\n", "system")
                    
                    if consecutive_error_count >= safe_max_retry:
                        self.log_debug("서킷 브레이커 발동! 무한 루프 강제 종료.")
                        sos_msg = "유저님, 여러 번 시도했지만 에러가 지속됩니다. 방향성을 제시해 주시겠어요?"
                        self.append_chat(f"🤖 AI: {sos_msg}\n")
                        self.engine.messages.append({"role": "assistant", "content": sos_msg})
                        break
                    
                    # 기존 디버그 프로토콜 + 단계별 부분 복구 지시 융합
                    enforced_result = (
                        f"🚨 [작업 {current_step_count + 1}단계 도구 실행 실패 - 에러 발생! (현재 {consecutive_error_count}회/최대 {safe_max_retry}회)]\n{tool_result}\n\n"
                        "---[증분 복구 및 시스템 디버그 긴급 지시]---\n"
                        "1. (반복 금지): 직전과 똑같은 코드를 제출하지 마세요.\n"
                        "2. (부분 수정): 전체 작업을 처음부터 다시 빌드하지 말고, 실패한 현재 단계의 코드만 분석하여 다시 이어 붙이세요(append).\n"
                        "3. (상태 점검): 필요시 'run_shell_command'나 'view_file'로 현재 누적된 파일 상태를 먼저 확인하세요.\n"
                        "4. (검색 강제): 원인을 모르면 즉시 'search_web'으로 구글링하세요.\n"
                        "5. 수정된 코드 조각만 다시 제출하여 작업을 이어가세요. 변명하지 마세요."
                    )
                else:
                    consecutive_error_count = 0
                    current_step_count += 1
                    self.log_debug(f"✅ {current_step_count}단계 도구 실행 성공.")
                    
                    self.append_chat(f"\n[✅ {current_step_count}단계 작업 완료: {tc_name}]\n", "system")
                    
                    # 👇 [핵심 패치] 성공한 작업의 내용을 일지에 기록하여 다음 턴의 '기준'으로 삼게 함
                    enforced_result = (
                        f"[작업 {current_step_count}단계 결과 및 완수 확인]\n{tool_result}\n\n"
                        "---[시스템 작업 일지]---\n"
                        f"- 현재까지 총 {current_step_count}개의 세부 작업이 성공적으로 완료되었습니다.\n"
                        f"- 마지막으로 성공한 도구: {tc_name}\n"
                        "- [지시]: 다음 단계로 넘어가기 전, 방금의 결과가 파일이나 메모리에 실제 반영되었는지 '검증'이 필요하다면 확인 도구를 먼저 쓰세요. 이미 완료된 작업을 절대 반복하지 마세요."
                    )

                self.engine.messages.append({"role": "tool", "tool_call_id": "call_id", "name": tc_name, "content": enforced_result})
                continue
            
            task_completed_successfully = True # 루프를 정상적으로 빠져나오면 성공으로 간주
            break

        # 👇 [추가] 루프 종료 후 성공했다면 사후 회고 실행
        if task_completed_successfully and consecutive_error_count == 0:
            threading.Thread(target=self.finalize_task_retrospective, daemon=True).start()

        gc.collect()
        self.after(0, lambda: self.send_btn.configure(state="normal"))

if __name__ == "__main__":
    app = MOPApp()
    app.mainloop()