def run_shell_command_safe(command: str) -> str:
    """
    터미널 명령어를 안전하게 실행하고 결과를 반환합니다.
    
    Args:
        command (str): 실행할 터미널 명령어.
    
    Returns:
        str: 명령어 실행 결과 (stdout 또는 stderr).
    """
    try:
        # shell=True는 편리하지만 위험할 수 있으므로 주의가 필요합니다.
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return f"[터미널 출력]\n{output.strip()}"
    except Exception as e:
        return f"실행 중 오류 발생: {e}"

if __name__ == "__main__":
    # 테스트용 명령어
    test_command = "echo Hello World"
    print(run_shell_command_safe(test_command))


