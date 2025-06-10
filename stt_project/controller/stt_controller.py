from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from enum import Enum
from fastapi import Query
from stt_project.service.stt_service_impl import SttServiceImpl
import os

SttRouter = APIRouter()

def injectService() -> SttServiceImpl:
    return SttServiceImpl.getInstance()

allowed_stt_models_str = os.getenv("STTMODELS", "")
# 쉼표로 분리된 문자열을 리스트로 변환하고, 공백을 제거합니다.
allowed_stt_models = [
    model.strip() for model in allowed_stt_models_str.split(',') if model.strip()
]

# SttModel Enum을 동적으로 생성합니다.
# Enum 멤버의 이름은 모델 경로의 마지막 부분을 대문자로 변환하여 사용합니다.
# 예: "openai/whisper-medium" -> MEDIUM
# 예: "openai/whisper-large-v3-turbo" -> LARGE_V3_TURBO
# 예: "openai/whisper-large-v3" -> LARGE_V3
stt_model_enum_members = {}
if not allowed_stt_models:
    # 환경 변수가 비어 있거나 설정되지 않은 경우, 기본 모델을 포함합니다.
    # 이 부분은 프로젝트의 정책에 따라 조정할 수 있습니다.
    print("경고: STTMODELS 환경 변수가 설정되지 않았습니다. 기본 STT 모델(large-v3-turbo)을 사용합니다.")
    stt_model_enum_members["large_v3_turbo"] = "openai/whisper-large-v3-turbo"
else:
    for model_path in allowed_stt_models:
        # 모델 경로에서 마지막 부분을 추출하여 Enum 멤버 이름으로 사용
        # 예: "openai/whisper-medium" -> "medium"
        # "openai/whisper-large-v3-turbo" -> "large-v3-turbo"
        # Enum 멤버 이름은 유효한 Python 식별자여야 하므로, 하이픈을 언더스코어로 변환합니다.
        member_name = model_path.split('/')[-1].replace('-', '_').upper()
        # 이미 존재하는 이름인지 확인 (중복 방지)
        if member_name in stt_model_enum_members:
            # 중복된 이름이 있다면 모델 경로를 포함하여 이름을 더 구체적으로 만듭니다.
            member_name = f"{member_name}_{len(stt_model_enum_members)}"
        stt_model_enum_members[member_name] = model_path

# type()을 사용하여 SttModel Enum을 동적으로 생성합니다.
SttModel = Enum("SttModel", stt_model_enum_members)

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
            default=list(SttModel.__members__.values())[0] if SttModel.__members__ else None,
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
