import os
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from stt_project.controller.stt_controller import SttRouter
from stt_project.repository.stt_repository_impl import SttRepositoryImpl

load_dotenv()
app = FastAPI(debug=True)

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(SttRouter)


@app.on_event("startup")
async def on_startup():
    repo = SttRepositoryImpl.getInstance()
    # (원한다면 임베딩 캐시나 인덱스 로드도 이곳에서)
    print("✅ Startup: model checked.")  # 메시지 약간 수정


if __name__ == "__main__":
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run(app, host=host, port=port, log_level="debug")