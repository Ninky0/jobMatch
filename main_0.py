from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes import router as api_router
from api.custom_handler import validation_exception_handler
from core.worker import start_workers
from services.job_generator import llm_generate_job_posting
from core.model_loader import load_llm_pipeline


# FastAPI 앱 생성
app = FastAPI(
    title='Job Posting API',
    version='1.0.0',
    docs_url=None,  # /docs로 별도 정의
    openapi_url='/openapi.json'
)

# 정적 파일 서빙 (/static 경로)
static_dir = Path("./static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS 설정정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (필요 시 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 경로 및 파이프라인 로드
MODEL_PATH = "/workspace/volume/models--rtzr--ko-gemma-2-9b-it/snapshopts/c9aea5c"

# 앱 시작 시 백그라운드 작업자 등록
@app.on_event("startup")
async def startup_event():
    app.state.pipe = load_llm_pipeline(MODEL_PATH)  # 모델을 앱 전역 상태로 등록
    await start_workers(llm_generate_job_posting, app.state.pipe)


# 라우터 등록
app.include_router(api_router)

# 커스텀 예외 핸들러 등록
app.add_exception_handler(RequestValidationError, validation_exception_handler)


# 개발용 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
