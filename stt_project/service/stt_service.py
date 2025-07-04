from abc import ABC, abstractmethod
from fastapi import UploadFile


class SttService(ABC):
    @abstractmethod
    def transcription(self,Uploadfile: UploadFile,model_name):
        pass