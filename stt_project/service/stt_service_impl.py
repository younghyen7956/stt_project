from fastapi import UploadFile

from stt_project.repository.stt_repository_impl_kokoro import SttRepositoryImpl
from stt_project.service.stt_service import SttService


# RAGService 인터페이스가 있다면 그것을 상속, 없다면 object 상속
class SttServiceImpl(SttService):
    __instance = None

    def __new__(cls, *args, **kwargs): # 싱글턴 인수 처리
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            # __init__에서 repo 초기화
        return cls.__instance

    def __init__(self):
        if not hasattr(self, '_initialized_service'): # 서비스 초기화 플래그
            # ⭐️ 클래스 이름에 맞게 프린트 로그 수정
            print("--- SttServiceImpl: __init__ 최초 초기화 ---")
            self.stt_repository = SttRepositoryImpl.getInstance() # 레포지토리 인스턴스 생성 및 저장
            self._initialized_service = True
        else:
            # ⭐️ 클래스 이름에 맞게 프린트 로그 수정
            print("--- SttServiceImpl: __init__ (이미 초기화됨) ---")


    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls() # __new__ 와 __init__ 호출
        return cls.__instance

    # ⭐️ enable_translation 파라미터를 추가하고 repository로 전달합니다.
    async def transcription(self, Uploadfile: UploadFile, model_name: str, enable_translation: bool):
        audio_transcription = await self.stt_repository.transcription(
            audio_file=Uploadfile,
            model_name=model_name,
            enable_translation=enable_translation
        )
        return audio_transcription

    def get_model_list(self):
        return self.stt_repository.get_model_list()

