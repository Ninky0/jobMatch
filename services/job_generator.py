from schemas.job import JobInput, JobOutput
from embed_matching import embed_matching
from utils.prompt_builder import task_prompt
from utils.text_utils import extract_section


# 🔹 LLM을 호출하여 구인공고 초안을 생성하는 서비스 함수
def llm_generate_job_posting(input_data: JobInput, pipe) -> JobOutput:
    # 직종 예측 (10개)
    jikjong_preds = embed_matching(input_data.job_description)
    main_job = ', '.join(jikjong_preds[:5])
    sub_job = ', '.join(jikjong_preds[5:])

    # 프롬프트 구성
    prompt_input = """
    (프롬프트)
    """.format(
        input_data.company_intro,
        input_data.job_description,
        main_job,
        sub_job
    )

    # LLM 실행
    task_outputs, task_prompt_len = task_prompt(prompt_input, pipe)
    output = task_outputs[0]['generated_text'][task_prompt_len:]

    # 출력 섹션별 파싱
    return JobOutput(
        job_title=extract_section(output, "<공고 제목>"),
        recommended_occupation_main=extract_section(output, "<모집 직종>"),
        recommended_occupation_sub=extract_section(output, "<관련 직종>"),
        recommended_job=extract_section(output, "<모집 직무>"),
        job_intro=extract_section(output, "<직무 소개>"),
        main_tasks=extract_section(output, "<주요 업무>"),
        preferred_qualifications=extract_section(output, "<우대 사항>"),
        search_keywords=extract_section(output, "<검색키워드>")
    )
