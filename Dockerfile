# 1. 베이스 이미지로 Python 3.11 사용
FROM python:3.11-slim

# 2. 작업 디렉터리 설정
WORKDIR /app

# 3. 시스템 패키지 업데이트 및 필수 툴 설치 (옵션)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 코드 복사
COPY . .

# 6. 환경 변수 설정 (예: Flask 실행 모드)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# 7. 컨테이너 시작 시 실행할 명령
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]