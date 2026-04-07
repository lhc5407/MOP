import re
import traceback

def auto_debug_and_fix(error_message):
    """
    에러 메시지를 분석하고 해결책을 제안하는 도구.
    """
    print(f"[DEBUG] 분석 시작: {error_message}")
    
    # 1. 에러 메시지 패턴 매칭
    diagnosis = "알 수 없음"
    suggested_fix = "코드 로직을 재검토하세요."
    
    if "TypeError: string indices must be integers" in error_message:
        diagnosis = "문자열을 딕셔너리처럼 접근하는 오류 (예: item['key'] where item is str)"
        suggested_fix = "1. 리스트 요소가 딕셔너리가 아닌 문자열인지 확인하세요.\n2. 데이터 파싱 단계 (예: JSON, HTML) 에서 타입이 제대로 변환되었는지 확인하세요.\n3. 예시 수정: if isinstance(item, dict): item['key']"
    elif "ModuleNotFoundError" in error_message:
        diagnosis = "모듈이 설치되지 않거나 경로가 잘못되었습니다."
        suggested_fix = "1. pip install <모듈명> 을 실행하세요.\n2. sys.path 에 모듈 경로가 포함되어 있는지 확인하세요."
    elif "SyntaxError" in error_message:
        diagnosis = "파이썬 문법 오류 (문자열 미완성, 괄호 불일치 등)"
        suggested_fix = "1. 에러가 발생한 줄의 문법을 다시 확인하세요.\n2. 문자열은 반드시 ' 또는 \" 로 시작하고 끝내야 합니다."
    
    print(f"[DIAGNOSIS] 원인: {diagnosis}")
    print(f"[SUGGESTED FIX]\n{suggested_fix}")
    
    return {
        "diagnosis": diagnosis,
        "suggested_fix": suggested_fix
    }

if __name__ == "__main__":
    # 테스트
    test_error = 'TypeError: string indices must be integers, not str'
    result = auto_debug_and_fix(test_error)
    print(f"\n[테스트 결과]\n{result}")


