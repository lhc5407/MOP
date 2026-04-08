"""
res/skills/computer_tools.py
마우스 및 키보드 제어를 위한 독립 스킬 모듈 (Pylance Type-Safe Version)
"""

import sys
import argparse
from typing import Optional # [추가됨] None 허용을 위한 타입 힌트

try:
    import pyautogui
    # AI 폭주 방지: 마우스를 화면 모서리로 이동하면 즉시 스크립트 강제 종료
    pyautogui.FAILSAFE = True 
except ImportError:
    print("오류: pyautogui 패키지가 설치되지 않았습니다. 'pip install pyautogui'를 실행하세요.")
    sys.exit(1)

# [수정됨] Optional[int]를 사용하여 int와 None 모두 허용
def mouse_action(action: str, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """마우스 이동 및 클릭을 제어합니다."""
    try:
        if action == "move":
            if x is None or y is None: 
                return "오류: move 액션은 x, y 좌표가 필수입니다."
            pyautogui.moveTo(x, y, duration=0.5) # 자연스럽게 0.5초 동안 이동
            return f"마우스를 ({x}, {y}) 좌표로 이동했습니다."
            
        elif action == "click":
            if x is not None and y is not None:
                pyautogui.click(x, y)
                return f"({x}, {y}) 위치를 클릭했습니다."
            else:
                pyautogui.click() # 현재 위치 클릭
                return "현재 마우스 위치를 클릭했습니다."
                
        return "알 수 없는 마우스 액션입니다."
    except Exception as e:
        return f"마우스 제어 오류: {e}"

# [수정됨] Optional[str]을 사용하여 str과 None 모두 허용
def keyboard_action(action: str, text: Optional[str] = None, key: Optional[str] = None) -> str:
    """키보드 타이핑, 단일 키 누름, 핫키(단축키)를 제어합니다."""
    try:
        if action == "type":
            if not text: return "오류: type 액션은 text 인자가 필수입니다."
            pyautogui.write(text, interval=0.05) # 사람처럼 약간의 간격을 두고 타이핑
            return f"'{text}' 타이핑을 완료했습니다."
            
        elif action == "press":
            if not key: return "오류: press 액션은 key 인자가 필수입니다. (예: enter, win, esc)"
            pyautogui.press(key)
            return f"'{key}' 키를 눌렀습니다."
            
        elif action == "hotkey":
            if not text: return "오류: hotkey 액션은 text 인자에 조합할 키들을 넣어야 합니다 (예: ctrl,c)."
            keys = text.split(',')
            pyautogui.hotkey(*keys)
            return f"핫키 조합({text})을 실행했습니다."
            
        return "알 수 없는 키보드 액션입니다."
    except Exception as e:
        return f"키보드 제어 오류: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["mouse", "keyboard"])
    parser.add_argument("--action", required=True, type=str)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    # [수정됨] 빈 문자열("") 대신 default=None을 명시하여 Optional 타입과 일치시킴
    parser.add_argument("--text", type=str, default=None) 
    parser.add_argument("--key", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.device == "mouse":
        print(mouse_action(args.action, args.x, args.y))
    elif args.device == "keyboard":
        print(keyboard_action(args.action, args.text, args.key))