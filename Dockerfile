FROM python:3.10

# 2. 환경 변수 설정
#    - PYTHONUNBUFFERED: Python의 출력이 버퍼링 없이 즉시 터미널에 표시되도록 합니다. (로그 확인에 유용)
#    - PYTHONDONTWRITEBYTECODE: .pyc 파일을 생성하지 않아 컨테이너를 가볍게 유지합니다.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 3. 시스템 의존성 설치
#    - pydub, openai-whisper가 사용하는 FFmpeg와 soundfile이 사용하는 libsndfile1을 설치합니다.
#    - --no-install-recommends: 불필요한 추천 패키지를 설치하지 않아 이미지 크기를 줄입니다.
#    - apt-get clean && rm -rf /var/lib/apt/lists/*: 설치 후 캐시를 정리하여 이미지 크기를 최적화합니다.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. 애플리케이션 디렉토리 설정
WORKDIR /app

# 5. Python 의존성 설치
#    - requirements.txt만 먼저 복사하여 Docker의 레이어 캐시를 활용합니다.
#    - --no-cache-dir: pip 캐시를 사용하지 않아 이미지 크기를 줄입니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. 애플리케이션 코드 복사
COPY . .

# 8. Gunicorn이 사용할 포트 노출
EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.app:app"]