from abc import ABC, abstractmethod
from fastapi import UploadFile


class SttService(ABC):
    @abstractmethod
    async def transcription(self, Uploadfile: UploadFile, model_name: str, enable_translation: bool):
        pass

    @abstractmethod
    def get_model_list(self):
        pass