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
            
            # 👇 [추가된 KV 최적화 파라미터]
            # 장시간 자율 주행 시 VRAM 폭발을 막기 위해 KV 캐시를 시스템 RAM으로 강제 할당합니다.
            offload_kqv=False, 
            
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
            "4. [자가 디버깅 루틴]: 에러가 발생하면 절대 포기하거나 사용자에게 변명하지 말고, 코드를 수정하여 즉시 재호출하세요.\n"
            "5. [코딩 작업 지시서(Plan)]: 복잡한 코딩 요청을 받으면, 메모장이나 텍스트 파일에 '작업 계획서'를 먼저 작성하세요.\n"
            "6. [환경 파악]: 코드를 짜기 전에 'run_shell_command'로 환경을 먼저 확인하는 습관을 들이세요."
        )

    def get_tools(self):
        # (이전 지시대로 14개 도구가 모두 포함되어 있습니다)
        return [
            {"type": "function", "function": {"name": "search_web", "description": "인터넷 검색(query)", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
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
        self.mem_turns_var = ctk.IntVar(value=20)
        self.mem_chars_var = ctk.IntVar(value=12000)
        
        # 4. 도구/권한 설정
        self.auto_approve_var = ctk.BooleanVar(value=False)
        self.show_think_var = ctk.BooleanVar(value=True)
        self.max_retry_var = ctk.IntVar(value=3)

        # 하드웨어 제어 동기화 이벤트
        self.approval_event = threading.Event()
        self.approval_result = False

        self.create_sidebar()
        self.create_main_area()

    # --- [UI 헬퍼 함수: 값 라벨이 달린 슬라이더 생성] ---
    def add_slider_control(self, parent, label_text, variable, from_, to, steps, fmt="{:.0f}"):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=(15, 0))
        
        # 라벨 및 값 표시
        ctk.CTkLabel(frame, text=label_text, font=ctk.CTkFont(weight="bold")).pack(side="left")
        val_lbl = ctk.CTkLabel(frame, text=fmt.format(variable.get()), text_color="#00A2FF", font=ctk.CTkFont(weight="bold"))
        val_lbl.pack(side="right")
        
        # 슬라이더 콜백: 움직일 때마다 라벨 업데이트
        def on_change(v):
            val_lbl.configure(text=fmt.format(float(v)))
            
        slider = ctk.CTkSlider(parent, from_=from_, to=to, number_of_steps=steps, variable=variable, command=on_change)
        slider.pack(fill="x", padx=15, pady=(5, 0))
        return slider

    def create_sidebar(self):
        # 항목이 많으므로 ScrollableFrame 사용
        sidebar = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        
        # --- [1. 모델 및 엔진] ---
        ctk.CTkLabel(sidebar, text="🧠 모델 및 엔진 환경", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(20,10), padx=15, anchor="w")
        
        self.model_btn = ctk.CTkButton(sidebar, text="모델 선택 (.gguf)", command=self.browse_model)
        self.model_btn.pack(pady=5, padx=15, fill="x")
        ctk.CTkLabel(sidebar, textvariable=self.model_path_var, font=("Arial", 11), text_color="gray").pack(padx=15, anchor="w")
        
        # KV 양자화 드롭다운
        kv_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        kv_frame.pack(fill="x", padx=15, pady=(15, 5))
        ctk.CTkLabel(kv_frame, text="KV Cache 양자화", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkOptionMenu(kv_frame, values=["FP16 (고품질)", "Q8_0 (8-bit)", "Q4_0 (4-bit 최대압축)"], variable=self.kv_quant_var).pack(side="right")

        self.add_slider_control(sidebar, "컨텍스트(Context) 창", self.n_ctx_var, 2048, 16384, 7, "{:.0f}")
        self.add_slider_control(sidebar, "GPU 오프로드 층 수", self.gpu_layers_var, 0, 100, 100, "{:.0f}")
        self.add_slider_control(sidebar, "CPU 스레드 수", self.n_threads_var, 1, 32, 31, "{:.0f}")

        # --- [2. 에이전트 성향] ---
        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=15, pady=20)
        ctk.CTkLabel(sidebar, text="🤖 에이전트 성향 제어", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(0,10), padx=15, anchor="w")
        
        ctk.CTkButton(sidebar, text="📝 시스템 프롬프트 편집", command=self.open_sys_prompt_editor, fg_color="#4CAF50", hover_color="#388E3C").pack(pady=5, padx=15, fill="x")
        
        self.add_slider_control(sidebar, "창의성 (Temperature)", self.temp_var, 0, 1, 100, "{:.2f}")
        self.add_slider_control(sidebar, "최대 출력 제한 (Tokens)", self.max_tokens_var, 256, 4096, 15, "{:.0f}")

        # --- [3. 기억 제한 방어망] ---
        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=15, pady=20)
        ctk.CTkLabel(sidebar, text="💾 기억 및 문맥 방어", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(0,10), padx=15, anchor="w")
        
        self.add_slider_control(sidebar, "단기 기억 한계 (턴 수)", self.mem_turns_var, 5, 50, 45, "{:.0f}")
        self.add_slider_control(sidebar, "전체 글자 수 제한", self.mem_chars_var, 4000, 24000, 20, "{:.0f}")

        # --- [4. 도구 권한] ---
        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=15, pady=20)
        ctk.CTkLabel(sidebar, text="🛠️ 권한 및 모니터링", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00A2FF").pack(pady=(0,10), padx=15, anchor="w")
        
        ctk.CTkSwitch(sidebar, text="하드웨어 자동 승인 (-f)", variable=self.auto_approve_var).pack(pady=10, padx=15, anchor="w")
        ctk.CTkSwitch(sidebar, text="사고 과정(<think>) 표출", variable=self.show_think_var).pack(pady=10, padx=15, anchor="w")
        
        self.add_slider_control(sidebar, "최대 자율 재시도 횟수", self.max_retry_var, 1, 10, 9, "{:.0f}")

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
        editor.title("시스템 프롬프트 편집")
        editor.geometry("700x500")
        editor.attributes("-topmost", True)
        
        ctk.CTkLabel(editor, text="AI의 성격과 규칙을 정의하는 최상위 지시문입니다.", text_color="gray").pack(pady=10)
        
        textbox = ctk.CTkTextbox(editor, wrap="word", font=ctk.CTkFont(size=13))
        textbox.pack(fill="both", expand=True, padx=20, pady=(0,10))
        textbox.insert("1.0", self.engine.custom_system_prompt)
        
        def save_prompt():
            self.engine.custom_system_prompt = textbox.get("1.0", "end-1c")
            # 현재 대화의 첫 번째 메시지(system) 갱신
            if self.engine.messages and self.engine.messages[0]["role"] == "system":
                self.engine.messages[0]["content"] = self.engine.custom_system_prompt
            self.log_debug("시스템 프롬프트가 성공적으로 업데이트되었습니다.")
            editor.destroy()
            
        ctk.CTkButton(editor, text="저장 및 닫기", command=save_prompt).pack(pady=(0,20))

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("GGUF Models", "*.gguf")])
        if path:
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
        self.append_chat(f"\n👤 사용자: {query}\n", "user")
        
        threading.Thread(target=self.ai_response_task, args=(query,), daemon=True).start()
        return "break"

    def append_chat(self, text, role="ai"):
        self.after(0, self._append_chat_internal, text)

    def _append_chat_internal(self, text):
        self.chat_view.configure(state="normal")
        self.chat_view.insert("end", text)
        self.chat_view.configure(state="disabled")
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
    def ai_response_task(self, query):
        if self.engine.llm is None:
            self.log_debug("🚨 오류: 모델이 아직 로드되지 않았습니다.")
            self.after(0, lambda: self.send_btn.configure(state="normal"))
            return

        auto_approve_this_turn = self.auto_approve_var.get()
        if query.endswith("-f") or query.endswith("-F"):
            auto_approve_this_turn = True
            query = query[:-2].strip()
            self.log_debug("⚡ '-f' 플래그 감지: 자동 승인 활성화")

        self.engine.archive_to_sqlite("user", query)
        self.engine.messages.append({"role": "user", "content": query})
        
        consecutive_error_count = 0
        
        while True:
            # 1. 동적 문맥 압축기 (UI 설정값 연동)
            ctx_len = sum(len(str(m.get('content', ''))) for m in self.engine.messages)
            
            # 👇 [수정] mem_turns_var와 mem_chars_var 안전 호출
            safe_mem_turns = self.get_safe_int(self.mem_turns_var, 20)
            safe_mem_chars = self.get_safe_int(self.mem_chars_var, 12000)
            
            if len(self.engine.messages) > safe_mem_turns or ctx_len > safe_mem_chars:
                self.log_debug(f"메모리 최적화 시작 (현재: {len(self.engine.messages)}턴, {ctx_len}자)")
                end_idx = 2
                while end_idx < len(self.engine.messages) and self.engine.messages[end_idx].get('role') != 'user':
                    end_idx += 1
                if end_idx < len(self.engine.messages):
                    for msg in self.engine.messages[1:end_idx]:
                        if msg.get('role') in ['user', 'assistant'] and msg.get('content'):
                            self.engine.archive_to_sqlite(msg['role'], msg['content'])
                    self.engine.messages = [self.engine.messages[0]] + self.engine.messages[end_idx:]

            self.append_chat("\n🤖 AI: ", "ai")
            
            # 2. LLM 스트리밍 (UI Temperature 및 Max Tokens 연동)
            # 👇 [수정] temp_var와 max_tokens_var 안전 호출
            safe_temp = self.get_safe_float(self.temp_var, 0.1)
            safe_max_tokens = self.get_safe_int(self.max_tokens_var, 2048)
            
            stream = cast(Iterator[Dict[str, Any]], self.engine.llm.create_chat_completion(
                messages=cast(Any, self.engine.messages), tools=cast(Any, self.engine.get_tools()), 
                stream=True, temperature=safe_temp, max_tokens=safe_max_tokens
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
                clean_text = assistant_content.replace("&lt;", "<").replace("&gt;", ">").replace("&#34;", '"')
                json_match = re.search(r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```', clean_text, re.DOTALL | re.IGNORECASE)

                if json_match:
                    try:
                        parsed = json.loads(json_match.group(1))
                        if isinstance(parsed, list): parsed = parsed[0]
                        if isinstance(parsed, dict) and (parsed.get("name") or parsed.get("tool")):
                            is_tool_call = True
                            tc_name = parsed.get("name") or parsed.get("tool")
                            tc_args = json.dumps(parsed.get("arguments") or parsed.get("parameters", parsed))
                    except Exception:
                        is_tool_call = True
                        tc_name = "json_syntax_error"
                        tc_args = "{}"

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
                        tool_result = "오류: JSON 문법이 깨졌습니다. 쌍따옴표와 줄바꿈(\\n)에 주의하여 다시 작성하세요."
                    
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
                    else: tool_result = f"알 수 없는 도구: {tc_name}"

                except Exception as e:
                    tool_result = f"파싱/실행 에러: {e}"

                # 5. 서킷 브레이커 및 강력한 디버깅 프로토콜
                if len(tool_result) > 6000:
                    self.log_debug("결과가 너무 길어 메모리 최적화 수행")
                    tool_result = tool_result[:3000] + "\n...[데이터가 너무 길어 중략됨]...\n" + tool_result[-3000:]

                is_error = any(kw in tool_result.lower() for kw in ["error", "exception", "traceback", "오류", "실패", "fail", "invalid"])
                
                if is_error:
                    consecutive_error_count += 1
                    safe_max_retry = self.get_safe_int(self.max_retry_var, 3)
                    
                    self.log_debug(f"🚨 에러 발생 ({consecutive_error_count}/{safe_max_retry})")
                    if consecutive_error_count >= safe_max_retry:
                        self.log_debug("서킷 브레이커 발동! 무한 루프 강제 종료.")
                        sos_msg = "차니님, 여러 번 시도했지만 에러가 지속됩니다. 방향성을 제시해 주시겠어요?"
                        self.append_chat(f"🤖 AI: {sos_msg}\n")
                        self.engine.messages.append({"role": "assistant", "content": sos_msg})
                        break
                    
                    enforced_result = (
                        # 👇 [수정 3] self.max_retry_var.get() 대신 안전한 safe_max_retry 변수 사용!
                        f"🚨 [도구 실행 실패 - 에러 발생! (현재 {consecutive_error_count}회/최대 {safe_max_retry}회)]\n{tool_result}\n\n"
                        "---[시스템 디버그 긴급 지시 (Coding Agent Protocol)]---\n"
                        "1. (반복 금지): 직전과 똑같은 코드를 제출하지 마세요.\n"
                        "2. (우회로 탐색): 'run_shell_command'로 환경을 확인하거나 다른 라이브러리를 쓰세요.\n"
                        "3. (검색 강제): 원인을 모르면 즉시 'search_web'으로 구글링하세요.\n"
                        "4. 해결책을 찾아 다시 호출하세요. 변명하지 마세요."
                    )
                else:
                    consecutive_error_count = 0
                    self.log_debug("도구 실행 성공.")
                    enforced_result = f"[도구 실행 결과]\n{tool_result}\n---[시스템 지시]---\n성공했습니다. 남은 작업이 있다면 즉시 연속 호출하고, 끝났다면 답변하세요."

                self.engine.messages.append({"role": "tool", "tool_call_id": "call_id", "name": tc_name, "content": enforced_result})
                continue 
            
            break

        gc.collect()
        self.after(0, lambda: self.send_btn.configure(state="normal"))

if __name__ == "__main__":
    app = MOPApp()
    app.mainloop()