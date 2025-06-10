from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from enum import Enum
from fastapi import Query
from stt_project.service.stt_service_impl import SttServiceImpl

SttRouter = APIRouter()

def injectService() -> SttServiceImpl:
    return SttServiceImpl.getInstance()

class SttModel(str, Enum):
    # 사용자가 정의한 모델명 유지
    medium = "openai/whisper-medium"
    large_v3_turbo = "openai/whisper-large-v3-turbo" # 모델 이름은 .env와 일치시키는 것이 좋습니다.
# --- ▼▼▼ 라우터 코드 수정 시작 ▼▼▼ ---

@SttRouter.post(
    "/transcribe",
    # response_model=... # 필요시 최종 키에 맞는 새 응답 모델 정의
    summary="오디오 파일을 텍스트로 변환하고 LLM 답변을 생성합니다.", # 요약 수정
    description=(
        "오디오 파일을 받아 STT를 수행하고, 감지된 언어 정보, "
        "STT 원문(original_query)과 한국어 번역본(korean_query), "
        "그리고 최종 LLM 답변(원본 언어/한국어)을 반환합니다."
    ) # 설명 수정
)
async def transcribe_audio_endpoint(
        model: SttModel = Query(
            default=SttModel.large_v3_turbo,
            description="사용할 STT 모델을 선택합니다."
        ),
        audio_file: UploadFile = File(..., description="변환할 오디오 파일 (예: .wav, .mp3, .m4a)"),
        stt_service: SttServiceImpl = Depends(injectService)
):
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=f"잘못된 파일 형식입니다. '{audio_file.content_type}'는 오디오 파일이 아닙니다."
        )

    try:
        print(f"컨트롤러 [/transcribe]: 수신 파일 '{audio_file.filename}', 선택된 모델: '{model.value}'")
        service_response = await stt_service.transcription(
            audio_file,
            model_name=model.value,
        )

        # 오류 처리 로직은 기존과 동일하게 유지 (매우 중요)
        if "error" in service_response and service_response["error"]:
            error_message = service_response.get("message", "STT 처리 중 알 수 없는 오류 발생")
            error_type = service_response.get("error", "UnknownError")
            print(f"컨트롤러 [/transcribe]: 서비스로부터 오류 수신 - 타입: {error_type}, 메시지: {error_message}")
            status_code = 500
            if error_type == "ModelNotFoundError":
                status_code = 404
            elif error_type == "AudioProcessingError":
                status_code = 422
            raise HTTPException(status_code=status_code, detail=error_message)

        # --- 응답 처리 로직을 새 구조에 맞게 대폭 단순화 ---
        original_query_text = service_response.get("original_query")
        if original_query_text is None:
            print(f"컨트롤러 [/transcribe]: 서비스 응답에 필수 STT 정보(original_query) 누락 (파일: {audio_file.filename})")
            raise HTTPException(status_code=500, detail="STT 결과를 처리하는 중 서버 내부 오류가 발생했습니다 (필수 정보 누락).")

        # 로그 출력을 새 키에 맞게 수정합니다.
        print(f"컨트롤러 [/transcribe]: 변환 완료 (파일: {audio_file.filename})")
        print(f"  STT 결과 (original_query): {str(original_query_text)[:70]}...")
        print(f"  한국어 쿼리 (korean_query): {str(service_response.get('korean_query'))[:70]}...")
        print(f"  최종 한국어 답변 (korean_answer): {str(service_response.get('korean_answer'))[:70]}...")

        # --- 최종 API 응답: 서비스가 만들어준 결과물을 그대로 반환 ---
        return service_response

    except HTTPException as http_exc:
        # 이미 처리된 HTTP 예외는 그대로 다시 발생시킴
        raise http_exc
    except Exception as e:
        import traceback
        print(f"컨트롤러 [/transcribe]: 예상치 못한 심각한 오류 발생 - {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"음성 변환 및 답변 생성 중 서버 내부에서 심각한 오류가 발생했습니다."
        )
