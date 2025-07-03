from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes import router as api_router
from api.custom_handler import validation_exception_handler

# FastAPI 앱 생성
app = FastAPI(
    title='Job Posting API (vLLM 기반)',
    version='2.0.0',
    docs_url=None,
    openapi_url='/openapi.json'
)

# 정적 파일 서빙
static_dir = Path("./static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(api_router)

# 커스텀 예외 핸들러 등록
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# 개발용 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)