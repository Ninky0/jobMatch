# JobInput, JobOutput 모델 정의
from pydantic import BaseModel, Field, field_validator

# 요청 모델: 구인공고 생성 요청에 사용
class JobInput(BaseModel):
    business_registration_number: str = Field(
        ...,
        pattern=r"^\d{3}-\d{2}-\d{5}$",
        description="사업자등록번호는 ***-**-***** 형식이어야 합니다."
    )
    company_intro: str
    job_description: str

    # 직무 내용 길이 유효성 검사
    @field_validator("job_description")
    @classmethod
    def validate_job_description_length(cls, value):
        if len(value) < 10:
            raise ValueError("직무 내용은 최소 10자 이상이어야 합니다.")
        return value


# 응답 모델: 구인공고 생성 결과
class JobOutput(BaseModel):
    job_title: str
    recommended_occupation_main: str
    recommended_occupation_sub: str
    recommended_job: str
    job_intro: str
    main_tasks: str
    preferred_qualifications: str
    search_keywords: str
