# core/celery_app.py
from celery import Celery

celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",   # Redis 브로커 주소
    backend="redis://localhost:6379/0"   # 결과 저장용 백엔드도 Redis 사용
)

# 필요 시 task 관련 라우팅 등 추가 설정 가능
celery_app.conf.task_routes = {
    "tasks.job_tasks.generate_job_posting_task": {"queue": "job_tasks"},
}

# 타임아웃, 재시도 정책 등 추가 설정도 가능
celery_app.conf.update(
    task_track_started=True,
    task_time_limit=300,  # 초 단위
)