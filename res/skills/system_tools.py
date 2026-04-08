"""
res/skills/system_tools.py
시스템 제어 및 실행을 위한 독립 스킬 모듈
"""

import sys
import os
import subprocess
import tempfile
import argparse

def execute_terminal_command(command: str) -> str:
    """시스템 터미널 명령어를 실행합니다."""
    try:
        # shell=True는 편리하지만 위험할 수 있으므로 주의가 필요합니다.
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return f"[터미널 출력]\n{output.strip()}"
    except Exception as e:
        return f"실행 중 오류 발생: {e}"

def run_python_snippet(code: str) -> str:
    """임시 파일을 생성하여 파이썬 코드를 실행하고 결과를 반환합니다."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path], capture_output=True, text=True, timeout=30
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return f"[파이썬 실행 결과]\n{output.strip()}"
    except Exception as e:
        return f"실행 오류: {e}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def manage_packages(action: str, packages: str) -> str:
    """pip를 통해 패키지를 관리합니다 (install, uninstall, list)."""
    cmd = [sys.executable, "-m", "pip", action]
    if action in ["install", "uninstall"]:
        if packages:
            cmd.extend(packages.split())
        if action == "uninstall":
            cmd.append("-y")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return f"[PIP {action} 결과]\n{result.stdout.strip()}"
    except Exception as e:
        return f"패키지 관리 오류: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System Control Tools for LLM")
    
    # 어떤 기능을 실행할지 결정하는 필수 인자
    parser.add_argument("--action_type", required=True, choices=["terminal", "python", "pip"], help="실행할 도구 유형")
    
    # 각 도구별 파라미터
    parser.add_argument("--command", type=str, help="터미널 명령어 (terminal 선택 시)")
    parser.add_argument("--code", type=str, help="파이썬 코드 (python 선택 시)")
    parser.add_argument("--action", type=str, help="pip 액션 (pip 선택 시: install, uninstall, list)")
    parser.add_argument("--packages", type=str, default="", help="패키지 이름 (pip 선택 시)")

    args = parser.parse_args()

    # 인자에 따른 라우팅
    if args.action_type == "terminal":
        if not args.command:
            print("오류: terminal 모드에서는 --command 인자가 필수입니다.")
        else:
            print(execute_terminal_command(args.command))
            
    elif args.action_type == "python":
        if not args.code:
            print("오류: python 모드에서는 --code 인자가 필수입니다.")
        else:
            print(run_python_snippet(args.code))
            
    elif args.action_type == "pip":
        if not args.action:
            print("오류: pip 모드에서는 --action 인자가 필수입니다.")
        else:
            print(manage_packages(args.action, args.packages))