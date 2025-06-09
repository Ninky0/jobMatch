import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils.text_utils import clear_gpu_memory


def load_llm_pipeline(model_path: str):
    """
    주어진 경로에서 LLM 텍스트 생성 파이프라인을 로드하고 반환.

    Args:
        model_path (str): 사전 학습된 모델의 경로
    Returns:
        transformers.Pipeline: 텍스트 생성 파이프라인
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    pipe.model.eval()
    clear_gpu_memory()
    return pipe