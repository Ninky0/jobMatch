# main_vllm.py

import httpx
import uvicorn
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

templates = Jinja2Templates(directory="templates")
static_dir = Path('./static')

app = FastAPI(title='Job Posting API (vLLM)', version='2.0.0', docs_url='/docs')
app.mount("/static", StaticFiles(directory=static_dir), name='static')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 입력 데이터 모델
class JobInput(BaseModel):
    business_registration_number: str = Field(..., pattern=r"^\d{3}-\d{2}-\d{5}$")
    company_intro: str
    job_description: str

    @field_validator("job_description")
    def validate_job_description_length(cls, value):
        if len(value) < 10:
            raise ValueError("직무 내용은 최소 10자 이상이어야 합니다.")
        return value

# 출력 데이터 모델
class JobOutput(BaseModel):
    job_title: str
    recommended_occupation_main: str
    recommended_occupation_sub: str
    recommended_job: str
    job_intro: str
    main_tasks: str
    preferred_qualifications: str
    search_keywords: str

def extract_section(text, section_title):
    pattern = rf"<\s*{re.escape(section_title.strip('<>'))}\s*>\s*(.*?)(?=(<[^>]+>|\Z))"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip('*').replace('\n','').strip()
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

# POST API 엔드포인트
@app.post("/generate-job-posting", response_model=JobOutput)
async def generate_job_posting(input_data: JobInput):
    try:
        # 직종 추천 임시 하드코딩 (원래는 embed_matching 결과)
        main_job = "회계원, 경리, 경영지원직, 회계사무원, 수금업무"
        sub_job = "총무직, 재무직, 경영관리직, 일반사무직, 일반관리직"

        prompt = build_prompt(input_data.company_intro, input_data.job_description, main_job, sub_job)
        llm_output = await call_vllm_chat(prompt)

        return JobOutput(
            job_title=extract_section(llm_output, "<공고 제목>"),
            recommended_occupation_main=extract_section(llm_output, "<모집 직종>"),
            recommended_occupation_sub=extract_section(llm_output, "<관련 직종>"),
            recommended_job=extract_section(llm_output, "<모집 직무>"),
            job_intro=extract_section(llm_output, "<직무 소개>"),
            main_tasks=extract_section(llm_output, "<주요 업무>"),
            preferred_qualifications=extract_section(llm_output, "<우대 사항>"),
            search_keywords=extract_section(llm_output, "<검색키워드>")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 오류: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
