FROM python:3.10

# 2. 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 3. 시스템 의존성 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. 애플리케이션 디렉토리 설정
WORKDIR /app

# 5. Python 의존성 설치 (FastAPI와 Gradio 의존성을 모두 설치)
# requirements.txt에 fastapi, uvicorn, gunicorn, gradio, requests, transformers, torch 등이 모두 포함되어야 합니다.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m unidic download
# 6. AI 모델 미리 다운로드 및 캐시 (기존 로직 유지 - 아주 좋습니다!)
ARG STTMODELS="openai/whisper-large-v3-turbo"
RUN python -c "import os; from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq; [AutoModelForSpeechSeq2Seq.from_pretrained(name.strip()) for name in os.environ.get('STTMODELS', '${STTMODELS}').split(',') if name.strip()]"
RUN python -c "import whisper; whisper.load_model('base', device='cpu')"

# 7. 프로젝트의 모든 파일을 컨테이너의 /app 디렉토리로 복사
COPY . .

# 8. 포트 노출 (FastAPI와 Gradio가 사용할 포트 모두)
EXPOSE 8080
#EXPOSE 7860