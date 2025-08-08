import os
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from stt_project.controller.stt_controller import SttRouter
from stt_project.repository.stt_repository_impl_kokoro import SttRepositoryImpl
from contextlib import asynccontextmanager  # [추가] lifespan을 위해 import

load_dotenv()


# [수정] lifespan 함수 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- 애플리케이션 시작 ---")
    SttRepositoryImpl.getInstance()
    print("✅ Startup: model checked and loaded.")

    yield  # yield 이전은 시작, 이후는 종료 시점입니다.

    # --- 애플리케이션 종료 시 실행될 로직 (필요 시 추가) ---
    print("--- 애플리케이션 종료 ---")


# [수정] FastAPI 앱 생성 시 lifespan 연결
app = FastAPI(debug=True, lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함도 그대로 유지
app.include_router(SttRouter)

if __name__ == "__main__":
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8282))
    uvicorn.run(app, host=host, port=port, log_level="debug")