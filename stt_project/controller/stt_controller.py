import os
import io
import base64
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Response
from enum import Enum
from typing import Literal
from stt_project.service.stt_service_impl import SttServiceImpl
from fastapi.responses import StreamingResponse, JSONResponse
import urllib.parse # URL 인코딩을 위해 추가

# 라우터 초기화
SttRouter = APIRouter()

# 서비스 의존성 주입 함수
def injectService() -> SttServiceImpl:
    return SttServiceImpl.getInstance()

# --- 환경 변수에서 허용된 STT 모델 목록을 읽어 동적으로 Enum 생성 ---
allowed_stt_models_str = os.getenv("STTMODELS", "openai/whisper-large-v3-turbo")
allowed_stt_models = [
    model.strip() for model in allowed_stt_models_str.split(',') if model.strip()
]

stt_model_enum_members = {}
for model_path in allowed_stt_models:
    member_name = model_path.split('/')[-1].replace('-', '_').upper()
    if member_name in stt_model_enum_members:
        member_name = f"{member_name}_{len(stt_model_enum_members)}"
    stt_model_enum_members[member_name] = model_path

SttModel = Enum("SttModel", stt_model_enum_members)

# --- 새로운 응답 타입 Enum 정의 ---
class ResponseType(str, Enum):
    JSON = "json"
    AUDIO = "audio"


@SttRouter.post(
    "/transcribe",
    summary="오디오 파일을 텍스트 변환 및 LLM 답변 생성 후, 응답 타입을 선택하여 반환합니다.",
    description="오디오 파일을 받아 STT, LLM 답변, TTS를 수행하고, `response_type` 파라미터에 따라 JSON 객체 또는 음성 파일 스트림을 반환합니다.",
    response_model=None
)
async def transcribe_audio_endpoint(
        model: SttModel = Query(
            default=list(SttModel.__members__.values())[0] if SttModel.__members__ else None,
            description="사용할 STT 모델을 선택합니다."
        ),
        audio_file: UploadFile = File(..., description="변환할 오디오 파일 (예: .wav, .mp3, .m4a)"),
        response_type: ResponseType = Query(
            default=ResponseType.JSON,
            description="API 응답 형식: 'json' (모든 정보 JSON) 또는 'audio' (음성 파일 스트리밍)"
        ),
        stt_service: SttServiceImpl = Depends(injectService)
) -> StreamingResponse | JSONResponse:
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail=f"잘못된 파일 형식입니다.")

    try:
        print(f"컨트롤러 [/transcribe]: 수신 파일 '{audio_file.filename}', 모델: '{model.value}', 응답 타입: '{response_type.value}'")

        service_response = await stt_service.transcription(audio_file, model_name=model.value)

        if "error" in service_response and service_response["error"]:
            error_message = service_response.get("message", "STT 처리 중 알 수 없는 오류")
            raise HTTPException(status_code=500, detail=error_message)

        if response_type == ResponseType.AUDIO:
            audio_base64 = service_response.get("original_language_answer_audio_base64")

            if audio_base64:
                audio_bytes = base64.b64decode(audio_base64)

                # --- 파일 이름 인코딩 수정 부분 ---
                original_filename_without_ext = os.path.splitext(audio_file.filename)[0]
                # RFC 5987에 따라 UTF-8로 인코딩하고 URL-인코딩합니다.
                encoded_filename = urllib.parse.quote(f"{original_filename_without_ext}_response.wav", encoding='utf-8')

                print(f"컨트롤러 [/transcribe]: 음성 파일 스트리밍 시작 (파일: {audio_file.filename})")
                return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav", headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
                })
            else:
                print(f"컨트롤러 [/transcribe]: 음성 파일 생성 실패 (Base64 데이터 없음). 텍스트 응답: {service_response.get('original_language_answer', 'N/A')}")
                raise HTTPException(status_code=500, detail="요청된 음성 파일을 생성할 수 없었습니다. 지원 언어를 확인하거나 서비스 로그를 참조하십시오.")
        else: # response_type == ResponseType.JSON
            print(f"컨트롤러 [/transcribe]: JSON 응답 반환 (파일: {audio_file.filename})")
            return JSONResponse(content=service_response)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"컨트롤러 [/transcribe]: 예상치 못한 심각한 오류 발생 - {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="서버 내부 오류 발생")