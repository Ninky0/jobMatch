import re
import torch


def extract_section(text: str, section_title: str) -> str:
    """
    LLM 출력에서 특정 섹션 제목(예: <공고 제목>)에 해당하는 내용을 추출.
    """
    pattern = rf"<\s*{re.escape(section_title.strip('<>'))}\s*>\s*(.*?)(?=(<[^>]+>|\Z))"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        content = match.group(1)
        if '<주요업무>' in match.group(0) or '<우대 사항>' in match.group(0):
            return content.strip('*').strip("\n")
        else:
            return content.strip('*').replace('\n', '')
    return ""


def clear_gpu_memory():
    """
    GPU 캐시 비우기 (모델 초기화 시 메모리 정리용)
    """
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.memory_stats()
        torch.cuda.ipc_collect()
