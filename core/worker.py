import asyncio
from typing import Callable, Awaitable
from schemas.job import JobInput, JobOutput

# 요청 큐 (다른 모듈에서 접근 가능)
request_queue = asyncio.Queue()


async def worker(handler: Callable[[JobInput], Awaitable[JobOutput]]):
    """
    큐에서 요청을 꺼내 비동기 handler 함수로 처리하는 작업자 코루틴.
    Args:
        handler (Callable): 요청 처리 함수 (async def)
    """
    while True:
        input_data, result_future = await request_queue.get()
        try:
            result = await handler(input_data)  # await 추가
            result_future.set_result(result)
        except Exception as e:
            result_future.set_exception(e)
        finally:
            request_queue.task_done()


async def start_workers(handler: Callable[[JobInput], Awaitable[JobOutput]], num_workers: int = 1):
    """
    서버 시작 시 작업자 코루틴을 등록.

    Args:
        handler (Callable): 요청 처리 async 함수
        num_workers (int): 동시 처리할 worker 수
    """
    for _ in range(num_workers):
        asyncio.create_task(worker(handler))  # pipe 제거
