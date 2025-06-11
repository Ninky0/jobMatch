FROM python:3.10-slim

# 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y nginx git curl && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 전체 복사
COPY . /app

# nginx 설정 복사 및 설정 교체
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/nginx.conf

# start.sh 실행 권한 부여
RUN chmod +x start.sh

# 포트 오픈
EXPOSE 80

# 컨테이너 시작 시 실행할 명령어
CMD ["./start.sh"]
