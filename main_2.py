# 이게 찐 원본
import transformers
import torch
import uvicorn
import re
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

# from embed_matching import embed_matching

templates = Jinja2Templates(directory="templates")

PATH_APP_PY = Path(__file__)
PATH_PROJECT = PATH_APP_PY.parent

app = FastAPI(title='Job Posting API', version='1.0.0', docs_url=None, openapi_url='/openapi.json')

static_dir = Path('./static')
app.mount("/static", StaticFiles(directory=static_dir), name='static')

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title='Job announcement API',
        version='0.5.0',
        description='This is a Job announcement API using Swagger UI.',
        routes=app.routes,
    )
    openapi_schema['openapi'] = '3.0.3'
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get('/')
def index():
    return {'message':'구인공고 작성지원 api입니다.'}

@app.get('/docs')
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url='/openapi.json',
        title='Custom Swagger UI',
        swagger_js_url='./static/swagger-ui-bundle.js',
        swagger_css_url='./static/swagger-ui.css',
    )

@app.get('/swagger_css')
def swagger_css():
    with open(static_dir, 'swagger-ui.css', 'rt', encoding='utf-8') as f:
        swagger_css = f.read()
    return Response(swagger_css,headers={"Content-type:":"text/css"})


@app.get('/swagger_js')
def swagger_js():
    with open(static_dir, 'swagger-ui-bundle.js', 'rt', encoding='utf-8') as f:
        swagger_js = f.read()
    return Response(swagger_js,headers={"Content-type:":"text/javascript"})


# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# model_path="/workspace/volume/models--rtzr--ko-gemma-2-9b-it/snapshots/c9aea5c899021d60dc0e0b051b00a504e5d9c7ba"
# model_path = '/mnt/hdd_sda/gemma2_model/ko-gemma-2-9b-it'

# model_path = Path("/workspace/gemma2_model/ko-gemma-2-9b-it")

pipe = transformers.pipeline("text-generation",
                             model = "/workspace/gemma2_model/ko-gemma-2-9b-it",
                             model_kwargs = {"torch_dtype": torch.bfloat16},
                             device_map = "auto",
                        )
pipe.model.eval()

def clear_gpu_memory():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.memory_stats()
    torch.cuda.ipc_collect()

clear_gpu_memory()

# 특정 섹션을 추출하는 함수 정의
def extract_section(text, section_title):
    pattern = rf"<\s*{re.escape(section_title.strip('<>'))}\s*>\s*(.*?)(?=(<[^>]+>|\Z))"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # 텍스트 앞뒤에 불필요한 \n, 공백 등을 제거
        if '<주요 업무>' in match.group(0) or '<우대 사항>' in match.group(0):
            cleaned_text = match.group(1).strip('*').strip("\n")
        else:
            cleaned_text = match.group(1).strip('*').replace('\n','')
        return cleaned_text
    return ""

# 입력 데이터 모델
class JobInput(BaseModel):
    business_registration_number: str = Field(
        ...,
        pattern=r"^\d{3}-\d{2}-\d{5}$",
        description="사업자등록번호는 ***-**-***** 형식이어야 합니다."
    )
    company_intro: str
    job_description: str

    # Custom validation for job_description length
    @field_validator("job_description")
    def validate_job_description_length(cls, value):
        if len(value) < 10:
            raise ValueError("직무 내용은 최소 10자 이상이어야 합니다.")
        return value

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    errors = exc.errors()
    custom_errors = []
    for error in errors:
        if "ctx" in error and "pattern" in error["ctx"]:
            custom_errors.append({
                "loc": error["loc"],
                "msg": "사업자등록번호는 ***-**-***** 형식이어야 합니다.",
                "type": error["type"],
                "input": error.get("input", None)
            })
        elif error.get("msg") == "직무 내용은 필수 입력값이며, 10자 이상이어야 합니다.":
            custom_errors.append({
                "loc": error["loc"],
                "msg": "직무 내용은 필수 입력값이며, 10자 이상이어야 합니다.",
                "type": "value_error.min_length",
                "input": error.get("input", None)
            })
        else:
            custom_errors.append(error)

    # JSON 직렬화 문제 해결을 위해 ValueError를 문자열로 변환
    for custom_error in custom_errors:
        if "ctx" in custom_error and "error" in custom_error["ctx"]:
            custom_error["ctx"]["error"] = str(custom_error["ctx"]["error"])


    return JSONResponse(
        status_code=400,
        content={"detail": custom_errors},
    )

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


# 프롬프트 구성 함수
def task_prompt(input, pipeline):
    messages = [
            {"role": "user", "content": f"{input}"}
        ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    return outputs, len(prompt)

# LLM 호출 함수 (이 부분은 사용자가 구현)
def llm_generate_job_posting(input_data: JobInput) -> JobOutput:
    """
    LLM을 호출하여 구인공고 초안을 생성하는 함수.
    """

    # jikjong_preds = embed_matching(input_data.job_description)
    # main_job = ', '.join(jikjong_preds[:5])
    # sub_job = ', '.join(jikjong_preds[5:])
    main_job = '회계원, 경리, 경영지원직, 회계사무원, 수금업무'
    sub_job = '총무직, 재무직, 경영관리직, 일반사무직, 일반관리직'

    prompt_input = """사용자의 입력을 바탕으로 구인 공고 작성에 필요한 항목들을 구성하세요. 각 항목에 알맞은 내용을 채우기 위해 아래 [설명]과 [예시]를 참고하여 작성하세요.

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
        입력으로 받은 직무 내용과 어울리는 직무를 콤마(",")로 구분하여 총 5개 나열하세요.

        <직무 소개>
        입력으로 받은 <회사 소개>와 <직무 내용>을 종합하여 직무를 소개하는 문장들을 5문장 이내로 만들어 주세요. 문장마다 개행해 주세요. 회사 소개는 약간 짧게, 직무 소개는 약간 길게 해 주세요.

        <주요 업무>
        업무의 주요 내용을 구체적으로 나열하세요. 하이픈('-')으로 구분하여 작성하세요.

        <우대 사항>
        관련 자격증, 경력, 필요 기술 스택에 대한 우대 사항을 명시하세요. 하이픈('-')으로 구분하여 작성하세요.

        <검색키워드>
        해당 구인공고를 검색할 때 쓰면 좋을 것 같은 키워드를 콤마(",")로 구분하여 총 5개 나열하세요.

    [예시]
    입력:
        <회사 소개>: 주식회사 브레인코어는 AI 교육과 에듀테크 분야에서 앞서 나가는 혁신적인 기업입니다. 우리는 성인과 청소년을 대상으로 인공지능 교육 프로그램을 개발하고 제공하며, 이를 통해 AI 기술의 대중화를 촉진하고 있습니다.
        <직무 내용>: 프로그래밍 언어 교육 자료 개발 및 강의, Transformer, CNN 등 최신 인공지능 주요 모델 개발 및 강의

    출력:
        <공고 제목>
        브레인코어와 함께 앞으로 나아갈 열정적인 인공지능 교육 개발자를 모집합니다.

        <모집 직종>
        인공지능 연구원, 커리큘럼 개발자, 인공지능 교육자, 교재 검수원, 컴퓨터 선생님

        <관련 직종>
        교육 관리자, 학원 강사, 전문 강사, 인공지능 개발자, 머신러닝 개발자

        <모집 직무>
        교육자료 개발, 강의 기획, 인공지능 모델 개발, 데이터 분석, 프로그래밍 교육

        <직무 소개>
        주식회사 브레인코어는 우리는 성인과 청소년을 대상으로 인공지능 교육 프로그램을 개발하고 제공하며, 이를 통해 AI 기술의 대중화를 촉진하고 있습니다.
        당사에서는 최신 인공지능 기술과 프로그래밍 언어 교육에 열정을 가진 전문가를 찾습니다.
        본 직무는 인공지능과 관련된 최신 모델을 이해하고, 이를 바탕으로 교육 자료를 개발하며, 강의를 통해 배움을 전수할 수 있는 기회를 제공합니다.

        <주요 업무>
        - 인공지능 및 프로그래밍 언어 교육 자료 개발 및 강의 진행
        - Transformer, CNN 등 최신 인공지능 모델 연구 및 응용
        - 데이터 분석을 통한 학습 자료의 품질 관리 및 개선)

        <우대 사항>
        - 정보처리기사, 데이터분석 준전문가(ADsP), 인공지능 데이터 전문가 등 자격증 보유자
        - 인공지능 연구원, 데이터 과학자, 소프트웨어 개발자 등으로서의 실무 경험)

        <검색키워드>
        교육자료, 프로그래밍, 인공지능, 딥러닝, 신경망

    <회사 소개>: {},
    <직무 내용>: {},
    <모집 직종>: {},
    <관련 직종>: {}""".format(input_data.company_intro, input_data.job_description, main_job, sub_job)

    task_outputs, task_prompt_len = task_prompt(prompt_input, pipe)
    output = task_outputs[0]['generated_text'][task_prompt_len:]

    title = extract_section(output, "<공고 제목>")
    occu_main = extract_section(output, "<모집 직종>")
    occu_sub = extract_section(output, "<관련 직종>")
    jobs = extract_section(output, "<모집 직무>")
    job_intro = extract_section(output, "<직무 소개>")
    main_tasks = extract_section(output, "<주요 업무>")
    pref = extract_section(output, "<우대 사항>")
    kwrd = extract_section(output, "<검색키워드>")

    return JobOutput(
        job_title=title,
        recommended_occupation_main=occu_main,
        recommended_occupation_sub=occu_sub,
        recommended_job=jobs,
        job_intro=job_intro,
        main_tasks=main_tasks,
        preferred_qualifications=pref,
        search_keywords=kwrd
    )

# API 엔드포인트
@app.post("/generate-job-posting", response_model=JobOutput)
async def generate_job_posting(input_data: JobInput):
    try:
        job_posting = llm_generate_job_posting(input_data)
        return job_posting
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating job posting: {str(e)}")

@app.post("/generate-job-posting-html", response_class=HTMLResponse)
async def generate_job_posting_html(input_data: JobInput, request: Request):
    try:
        job_posting = llm_generate_job_posting(input_data)
        # print("Job Posting:", job_posting)
        return templates.TemplateResponse("job_posting.html", {
            "request": request,
            "job_title": job_posting.job_title,
            "recommended_occupation_main": job_posting.recommended_occupation_main,
            "recommended_occupation_sub": job_posting.recommended_occupation_sub,
            "recommended_job": job_posting.recommended_job,
            "job_intro": job_posting.job_intro,
            "main_tasks": job_posting.main_tasks,
            "preferred_qualifications": job_posting.preferred_qualifications,
            "search_keywords": job_posting.search_keywords,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating job posting: {str(e)}")


# 실행 코드
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
