# 🤖 MOP (Multi-agent Orchestration Platform)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](#)
[![GUI](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet)](#)
[![LLM Engine](https://img.shields.io/badge/Engine-llama.cpp-green)](#)
[![License](https://img.shields.io/badge/License-MIT-orange)](#)

**MOP**는 로컬 환경에서 구동되는 고성능 자율 AI 에이전트 오케스트레이션 플랫폼입니다. 단순한 대화형 인터페이스를 넘어, 파일 시스템을 제어하고 다중 에이전트를 생성하며 장기적인 과업을 병렬로 수행할 수 있는 완벽한 '에이전트 하네스(Agent Harness)' 인프라를 제공합니다.

## ✨ 핵심 기능 (Key Features)

### 1. 🚀 비동기 백그라운드 오케스트레이션 (Background Tasks)
에이전트가 시간이 오래 걸리는 터미널 작업(예: 대규모 패키지 설치, 파일 압축, 서버 구동)을 백그라운드(`start_background_task`)로 던져두고 즉시 다른 작업을 병렬로 수행합니다. 진정한 의미의 멀티태스킹을 지원하여 전체 작업 시간을 획기적으로 단축합니다.

### 2. 🧠 서브에이전트를 통한 컨텍스트 격리 (Context Isolation)
방대한 코드나 문서를 분석할 때, 메인 에이전트의 메모리(Context Window)가 오염되는 것을 방지합니다. 일회용 서브에이전트(`delegate_to_sub_agent`)를 소환하여 독립된 환경에서 데이터를 분석하고 핵심 요약 보고서만 메인 뇌로 전달받아 환각(Hallucination) 현상을 원천 차단합니다.

### 3. 🛡️ 품질 게이트와 안전 통제 체계 (Plan Mode & Quality Gate)
강력한 힘에는 통제가 필요합니다. `Plan Mode`가 활성화되면 에이전트가 파일 쓰기, 삭제, 시스템 명령어 등 '위험한 도구'를 사용하기 직전, 직관적인 UI 모달을 띄워 사용자에게 승인(결재)을 요청합니다.
- 승인 시: 즉시 작업 수행
- 거절 시: 에이전트에게 '차단됨' 컨텍스트를 반환하여 안전한 우회 경로를 스스로 탐색하게 유도

### 4. 🦎 디렉토리 맞춤형 동적 환경 적응 (Meta-Harness)
에이전트가 실행되는 작업 폴더(CWD)에 `.mop_rules.md` 파일이 존재할 경우, 이를 자동으로 감지하여 에이전트의 뇌(System Prompt)에 최우선 지침으로 융합합니다. 폴더를 이동할 때마다 '프론트엔드 전문가', '파이썬 데이터 분석가' 등으로 스스로 페르소나와 작업 원칙을 갈아끼웁니다.

### 5. 📈 자가 학습 및 영구적 기억 (Self-Learning Principles)
작업 중 에러를 극복하거나 사용자의 피드백을 받으면, 에이전트가 스스로 깨달음을 요약하여 `학습된 자가 원칙`에 영구적으로 저장(`mop_config.json`)합니다. 앱을 재부팅해도 에이전트는 어제의 실수를 반복하지 않고 진화합니다.

### 6. 🧟 불사조 아키텍처 및 서킷 브레이커 (Resilience)
- **전역 에러 핸들러:** GUI나 백그라운드 스레드에서 발생하는 파이썬 런타임 에러를 낚아채어 앱이 튕기는 것을 방지하고 디버그 로그로 전환합니다.
- **오토 리부트:** C++ 레벨의 VRAM 초과 등 치명적 오류로 앱이 죽더라도, 즉시 재가동 루프를 통해 시스템을 자동 복구합니다.
- **무한 루프 방지:** AI의 환각으로 인한 동일 도구 연속 호출을 감지하면 서킷 브레이커가 작동하여 즉시 작업을 중단시키고 안정화합니다.

---

## 🛠️ 시스템 아키텍처 (Architecture)

MOP는 크게 세 가지 계층으로 이루어져 있습니다.
1. **UI Layer (`MOPApp`):** CustomTkinter 기반의 유려한 다크모드 인터페이스. 에이전트 상태 모니터링, 시스템 프롬프트 편집, Plan Mode 토글 등을 지원합니다.
2. **Engine Layer (`MOPEngine`):** 에이전트의 사고 루프(ReAct), 도구 라우팅, 메모리(SQLite) 관리, 서브에이전트 관리를 담당합니다.
3. **Execution Layer:** 파일 I/O, 터미널 제어, 마우스/키보드 자동화, 웹 검색 등 실제 환경과 상호작용하는 40여 개의 도구 스택.

---

## 🚀 시작하기 (Getting Started)

### 사전 요구 사항 (Prerequisites)
- Python 3.10 이상
- 로컬 LLM 구동을 위한 `llama-cpp-python` 및 호환되는 `.gguf` 모델 파일
- (권장) VRAM 8GB 이상의 GPU 환경

### 설치 및 실행
```bash
# 1. 저장소 클론
git clone [https://github.com/your-username/MOP.git](https://github.com/your-username/MOP.git)
cd MOP

# 2. 의존성 패키지 설치
pip install -r requirements.txt

# 3. 모델 파일 배치
# 호환되는 .gguf 파일을 ./models/ 디렉토리에 위치시킵니다.

# 4. 시스템 실행
python mop_app.py
💡 활용 팁 (Pro Tips)
프로젝트 도메인 지식 주입: 현재 작업 중인 코드 디렉토리 루트에 .mop_rules.md 파일을 만들고 팀의 코딩 컨벤션이나 요구사항을 적어두세요. MOP가 이를 가장 먼저 숙지하고 작업을 시작합니다.

UI 설정 세밀 조정: 좌측 사이드바 메뉴를 통해 에이전트의 창의성(Temperature)을 조절하거나, '학습된 자가 원칙 관리' 버튼을 눌러 AI의 뇌 구조를 직접 튜닝할 수 있습니다.

📄 라이선스 (License)
이 프로젝트는 MIT License에 따라 배포됩니다. 자유롭게 활용하고, 수정하고, 공유하세요!