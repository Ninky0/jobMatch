import asyncio
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from schemas.job import JobInput, JobOutput
from core.worker import request_queue
from core.model_loader import load_llm_pipeline

# 라우터 객체 생성
router = APIRouter()


@router.get("/")
def index():
    return {"message": "구인공고 작성지원 API 입니다."}


@router.get("/docs")
async def custom_swagger_ui_html():
    return RedirectResponse(url="/static/docs.html")  # static에서 커스텀한 Swagger UI 사용 가능


@router.get("/swagger_css")
def swagger_css():
    with open("static/swagger-ui.css", "rt", encoding="utf-8") as f:
        content = f.read()
    return Response(content, media_type="text/css")


@router.get("/swagger_js")
def swagger_js():
    with open("static/swagger-ui-bundle.js", "rt", encoding="utf-8") as f:
        content = f.read()
    return Response(content, media_type="application/javascript")


@router.post("/generate-job-posting", response_model=JobOutput)
async def generate_job_posting(input_data: JobInput, background_tasks: BackgroundTasks, request: Request):
    """
    구인공고 생성을 위한 POST 엔드포인트
    - 모델 파이프라인은 main.py에서 앱 전역 상태(app.state.pipe)로 전달됨
    """
    pipe = request.app.state.pipe
    result_future = asyncio.Future()
    await request_queue.put((input_data, result_future, pipe))
    result = await result_future
    return result
