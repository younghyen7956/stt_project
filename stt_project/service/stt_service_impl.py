from fastapi import UploadFile

from stt_project.repository.stt_repository_impl import SttRepositoryImpl
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
            print("--- RAGServiceImpl: __init__ 최초 초기화 ---")
            self.stt_repository = SttRepositoryImpl.getInstance() # 레포지토리 인스턴스 생성 및 저장
            self._initialized_service = True
        else:
            print("--- RAGServiceImpl: __init__ (이미 초기화됨) ---")


    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls() # __new__ 와 __init__ 호출
        return cls.__instance

    async def transcription(self,Uploadfile: UploadFile,model_name):
        audio_transcription = await self.stt_repository.transcription(Uploadfile,model_name)
        return audio_transcription

    async def translation(self, Uploadfile: UploadFile) -> dict:
        audio_translation = await self.stt_repository.translation(Uploadfile)
        return audio_translation



