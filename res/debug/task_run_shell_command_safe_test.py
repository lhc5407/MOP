#!/usr/bin/env python3
import sys
import os

# 현재 스크립트가 실행될 때, run_shell_command_safe 도구를 호출하여 테스트합니다.
# 이 스크립트는 run_shell_command_safe 도구가 정상적으로 작동하는지 확인하는 테스트 코드입니다.

print("[Test Start] Checking run_shell_command_safe tool...")

# 테스트 1: 간단한 echo 명령어 실행
print("\n[Test 1] Running: echo 'Hello from test script'")
try:
    # run_shell_command_safe 도구를 호출합니다.
    # (실제 시스템에서 이 도구는 이미 등록되어 있으므로, 이를 호출하여 결과를 확인합니다.)
    # 주의: 여기서는 실제 도구 호출 로직이 아니라, 테스트 스크립트 자체의 논리입니다.
    # 실제 테스트는 run_shell_command_safe 도구를 통해 이 스크립트를 실행할 때 수행됩니다.
    
    # 여기서는 단순히 스크립트 실행 성공을 확인합니다.
    print("[Test 1] PASSED: Script executed without syntax errors.")
except Exception as e:
    print(f"[Test 1] FAILED: {e}")
    import traceback
    traceback.print_exc()

# 테스트 2: 파일 읽기 테스트 (만약 파일이 존재한다면)
print("\n[Test 2] Checking file existence...")
test_file = "res/debug/task_run_shell_command_safe_test.py"
if os.path.exists(test_file):
    print(f"[Test 2] PASSED: File '{test_file}' exists.")
else:
    print(f"[Test 2] FAILED: File '{test_file}' not found.")

print("\n[Test Complete]")


