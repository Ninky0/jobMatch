import asyncio
from typing import Callable

# 요청 큐 (다른 모듈에서 접근 가능)
request_queue = asyncio.Queue()


async def worker(handler: Callable, pipe):
    """
    큐에서 요청을 꺼내 handler 함수로 처리하는 작업자 코루틴.
    Args:
        handler (Callable): 요청 처리 함수 (예: llm_generate_job_posting)
        pipe: LLM 파이프라인 (FastAPI 앱 시작 시 주입됨)
    """
    while True:
        input_data, result_future = await request_queue.get()
        try:
            result = handler(input_data, pipe)
            result_future.set_result(result)
        except Exception as e:
            result_future.set_exception(e)
        finally:
            request_queue.task_done()


async def start_workers(handler: Callable, pipe, num_workers: int = 1):
    """
    서버 시작 시 작업자 코루틴을 등록.

    Args:
        handler (Callable): 요청 처리 함수
        pipe: 모델 파이프라인
        num_workers (int): 동시 처리할 worker 수 (기본 1)
    """
    for _ in range(num_workers):
        asyncio.create_task(worker(handler, pipe))
