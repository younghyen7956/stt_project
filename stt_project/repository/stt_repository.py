from abc import ABC, abstractmethod
from fastapi import UploadFile


class SttRepository(ABC):

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def transcription(self, audio_file: UploadFile,model_name):
        pass