# main_vllm.py

import re
import httpx
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from celery import Celery
from celery.result import AsyncResult

# --- Celery 설정 ---
celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery_app.conf.update(
    task_track_started=True,
    task_time_limit=300,
)

# --- FastAPI 설정 ---
app = FastAPI(title='Job Posting API (vLLM + Celery)', version='2.0.0', docs_url='/docs')

templates = Jinja2Templates(directory="templates")
static_dir = Path('./static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 모델 정의 ---
class JobInput(BaseModel):
    business_registration_number: str = Field(..., pattern=r"^\d{3}-\d{2}-\d{5}$")
    company_intro: str
    job_description: str

    @field_validator("job_description")
    def validate_job_description_length(cls, value):
        if len(value) < 10:
            raise ValueError("직무 내용은 최소 10자 이상이어야 합니다.")
        return value


class JobOutput(BaseModel):
    job_title: str
    recommended_occupation_main: str
    recommended_occupation_sub: str
    recommended_job: str
    job_intro: str
    main_tasks: str
    preferred_qualifications: str
    search_keywords: str


# --- 유틸 함수 ---
def extract_section(text, section_title):
    pattern = rf"<\s*{re.escape(section_title.strip('<>'))}\s*>\s*(.*?)(?=(<[^>]+>|\Z))"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip('*').replace('\n', '').strip()
        return cleaned_text
    return ""
    
# 프롬프트 구성
def build_prompt(company_intro, job_description, main_job, sub_job):
    return f"""
사용자의 입력을 바탕으로 구인 공고 작성에 필요한 항목들을 구성하세요. 각 항목에 알맞은 내용을 채우기 위해 아래 [설명]과 [예시]를 참고하여 작성하세요.

[설명]
입력:
    <회사 소개>: 해당 회사에 대한 간단한 설명입니다.
    <직무 내용>: 입사자가 맡을 직무 내용을 1~2문장으로 설명합니다.
    <모집 직종>: 직무내용과 가장 관련 있는 직종들입니다.
    <관련 직종>: 직무내용과 관련 있는 직종들입니다.

출력:
    <공고 제목>
    실제 화면에 올라갈 해당 공고의 제목을 작성하세요.
    <모집 직종>
    입력으로 받은 <모집 직종>을 그대로 출력하세요.
    <관련 직종>
    입력으로 받은 <관련 직종>을 그대로 출력하세요.
    <모집 직무>
    직무 내용을 기반으로 어울리는 직무 5개를 콤마로 구분해 주세요.
    <직무 소개>
    회사 소개 + 직무 설명을 합쳐 5문장 이내로 써주세요.
    <주요 업무>
    하이픈으로 구분하여 주요 업무를 작성하세요.
    <우대 사항>
    관련 자격증, 경력, 기술 등을 하이픈으로 구분하여 작성하세요.
    <검색키워드>
    키워드 5개를 콤마로 구분해서 작성하세요.

<회사 소개>: {company_intro}
<직무 내용>: {job_description}
<모집 직종>: {main_job}
<관련 직종>: {sub_job}
"""

# LLM 호출 함수
async def call_vllm_chat(prompt: str):
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "rtzr/ko-gemma-2-9b-it",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 512,
            },
            timeout=60.0
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]


# --- Celery Task 정의 ---
@celery_app.task(name="generate_job_posting_task")
def generate_job_posting_task(input_data: dict):
    import asyncio
    main_job = "회계원, 경리, 경영지원직, 회계사무원, 수금업무"
    sub_job = "총무직, 재무직, 경영관리직, 일반사무직, 일반관리직"
    prompt = build_prompt(input_data["company_intro"], input_data["job_description"], main_job, sub_job)

    # asyncio 루프 돌려서 httpx 호출
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    llm_output = loop.run_until_complete(call_vllm_chat(prompt))

    return {
        "job_title": extract_section(llm_output, "<공고 제목>"),
        "recommended_occupation_main": extract_section(llm_output, "<모집 직종>"),
        "recommended_occupation_sub": extract_section(llm_output, "<관련 직종>"),
        "recommended_job": extract_section(llm_output, "<모집 직무>"),
        "job_intro": extract_section(llm_output, "<직무 소개>"),
        "main_tasks": extract_section(llm_output, "<주요 업무>"),
        "preferred_qualifications": extract_section(llm_output, "<우대 사항>"),
        "search_keywords": extract_section(llm_output, "<검색키워드>"),
    }


# --- 라우터 ---
@app.post("/generate-job-posting")
async def generate_job(input_data: JobInput):
    task = generate_job_posting_task.delay(input_data.dict())
    return {"task_id": task.id}


@app.get("/job-status/{task_id}")
def get_task_result(task_id: str):
    result: AsyncResult = celery_app.AsyncResult(task_id)

    if result.state == "PENDING":
        return {"status": "PENDING"}
    elif result.state == "STARTED":
        return {"status": "STARTED"}
    elif result.state == "SUCCESS":
        return {"status": "SUCCESS", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "FAILURE", "error": str(result.result)}
    else:
        return {"status": result.state}


# --- 실행 엔트리포인트 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)