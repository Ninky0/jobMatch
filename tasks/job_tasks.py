# tasks/job_tasks.py

from core.celery_app import celery_app
import httpx
from schemas.job import JobInput
from utils.embed_matching import embed_matching
from utils.text_utils import extract_section

@celery_app.task(bind=True, name="tasks.job_tasks.generate_job_posting_task")
def generate_job_posting_task(self, input_data: dict):
    """
    Celery 비동기 작업: 입력 데이터를 바탕으로 직무 예측 및 공고 생성
    """
    try:
        # 1. 임베딩 기반 직무 예측
        predictions = embed_matching(input_data["job_description"])

        main_job = ", ".join(predictions["jobs"][:3])
        sub_job = ", ".join(predictions["jobs"][3:])
        license = ", ".join(predictions["licenses"])
        major = ", ".join(predictions["majors"])

        # 2. vLLM 호출을 위한 입력 구성
        prompt = f"""사용자의 입력을 바탕으로 구인 공고 작성에 필요한 항목들을 구성하세요. 각 항목에 알맞은 내용을 채우기 위해 아래 [설명]과 [예시]를 참고하여 작성하세요.

    [설명]
    입력:
        <회사 소개>: 해당 회사에 대한 간단한 설명입니다.
        <직무 내용>: 입사자가 맡을 직무 내용을 1~2문장으로 설명합니다.
        <모집 직종>: 직무내용과 가장 관련 있는 직종들입니다.
        <관련 직종>: 직무내용과 관련 있는 직종들입니다.
        <관련 자격증>: 직무와 관련된 자격증들입니다.
        <관련 전공>: 직무와 관련된 전공들입니다.

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

        <관련 자격증>
        입력으로 받은 <관련 자격증>을 그대로 출력하세요.

        <관련 전공>
        입력으로 받은 <관련 전공>을 그대로 출력하세요.

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
        - 데이터 분석을 통한 학습 자료의 품질 관리 및 개선

        <우대 사항>
        - 정보처리기사, 데이터분석 준전문가(ADsP), 인공지능 데이터 전문가 등 자격증 보유자
        - 인공지능 연구원, 데이터 과학자, 소프트웨어 개발자 등으로서의 실무 경험

        <검색키워드>
        교육자료, 프로그래밍, 인공지능, 딥러닝, 신경망

        <관련 자격증>
        정보처리기사, 데이터분석 준전문가(ADsP), 인공지능 데이터 전문가

        <관련 전공>
        컴퓨터공학, 인공지능, 데이터사이언스

    <회사 소개>: {input_data.company_intro}
    <직무 내용>: {input_data.job_description}
    <모집 직종>: {main_job}
    <관련 직종>: {sub_job}
    <관련 자격증>: {license}
    <관련 전공>: {major}
    """

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://vllm:8000/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                }
            )
            response.raise_for_status()
            result_text = response.json()["choices"][0]["text"]

        return {
            "job_posting": result_text.strip(),
            "main_job": main_job,
            "sub_job": sub_job,
            "license": license,
            "major": major
        }

    except Exception as e:
        raise self.retry(exc=e, countdown=10, max_retries=3)
