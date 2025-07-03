import asyncio
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from schemas.job import JobInput, JobOutput
from tasks.job_tasks import generate_job_posting_task


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


# Celery에 작업을 맡기고 즉시 task_id만 응답
@router.post("/generate-job-posting")
async def generate_job(input_data: JobInput):
    task = generate_job_posting_task.delay(input_data.dict())
    return {"task_id": task.id}


@router.get("/job-status/{task_id}")
async def get_task_result(task_id: str):
    from core.celery_app import celery_app

    # 즉시 반환되므로 timeout 없음
    result = celery_app.AsyncResult(task_id)
    
    if result.state == "PENDING":
        return {"status": "PENDING"}
    elif result.state == "SUCCESS":
        return {"status": "SUCCESS", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "FAILURE", "error": str(result.result)}
    else:
        return {"status": result.state}