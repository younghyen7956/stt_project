# 1. 아키텍처에 유연한 공식 이미지 사용
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 5. Gunicorn이 사용할 포트 하나만 명시 (예: 8000번)
EXPOSE 8000

# 6. 실서비스용 Gunicorn으로 FastAPI 서버 실행
# main.py 파일 안에 app = FastAPI() 객체가 있다고 가정
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]