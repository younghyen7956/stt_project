import base64

import pandas as pd
from pydub import AudioSegment
from dotenv import load_dotenv
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor,AutoModel
from transformers import pipeline
import soundfile as sf
import numpy as np
import io,ssl,whisper,torch,openai, json,os,pycountry,random
from typing import Optional, Dict, Any

ssl._create_default_https_context = ssl._create_unverified_context
from stt_project.repository.stt_repository import SttRepository

class SttRepositoryImpl(SttRepository):
    __instance = None
    load_dotenv()

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.bfloat16
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float16

        self.pipelines = {}
        self.gpt_model = None
        self.whisper_detection_model = None
        self.bark_processor = None
        self.bark_model = None
        self.get_model()
        print("--- SttRepositoryImpl: __init__ 최초 초기화 완료 ---")

    def get_model(self):
        model_names_str = os.getenv("STTMODELS", "openai/whisper-large-v3-turbo")
        model_names = [name.strip() for name in model_names_str.split(',')]

        for model_name in model_names:
            print(f"\n--- Whisper 모델 ({model_name}) 로드 시작 ---")
            try:
                processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=(self.device.type == "cpu"),
                    use_safetensors=True
                ).to(self.device)

                self.pipelines[model_name] = pipeline(
                    "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, torch_dtype=self.torch_dtype,
                    device=self.device, return_timestamps=True,
                )
                print(f"✅ Whisper 모델 ({model_name}) 및 ASR 파이프라인 생성 완료.")
            except Exception as e:
                print(f"❌ Whisper 모델 ({model_name}) 로드 중 오류: {e}")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.gpt_model = openai.OpenAI(api_key=openai_api_key)
            print("✅ OpenAI GPT 클라이언트 초기화 완료.")
        else:
            print("⚠️ OPENAI_API_KEY 환경 변수가 없어 OpenAI GPT 클라이언트를 초기화할 수 없습니다.")
            self.gpt_model = None

        print("\n--- (언어 감지용) Whisper 모델 로드 시작 ---")
        try:
            self.whisper_detection_model = whisper.load_model("base", device="cpu")
            print("✅ (언어 감지용) Whisper 모델 (base) 로드 완료.")
        except Exception as e:
            print(f"❌ (언어 감지용) Whisper 모델 로드 중 오류: {e}")

        print("\n--- Bark TTS 모델 (suno/bark-small) 로드 시작 ---")
        try:
            self.bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
            self.bark_model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=torch.float32).to(self.device)
            # 'eager' 모드에서도 torch.compile은 여전히 매우 효과적입니다.
            if hasattr(torch, 'compile'):
                print("PyTorch 2.0+ 감지됨. Bark 모델에 torch.compile을 적용합니다...")
                self.bark_model = torch.compile(self.bark_model)
                print("✅ torch.compile 적용 완료.")
            print("✅ Bark TTS 모델 및 프로세서 로드 완료.")
        except Exception as e:
            print(f"❌ Bark TTS 모델 로드 중 오류: {e}")
            self.bark_processor = None
            self.bark_model = None

        print("--- SttRepositoryImpl: get_model successfully ---")

    def universal_translation_sync(self, text_to_translate: str, source_lang: str, target_lang: str) -> str:
        if not self.gpt_model: return "GPT 번역 서비스 사용 불가"
        if not text_to_translate or not text_to_translate.strip(): return ""

        # Note: 이전 대화에서 제안드린 고품질 번역 프롬프트 대신, 확정해주신 버전의 프롬프트를 유지합니다.
        system_content = f"""You are a professional translator.
        Your task is to accurately translate the given text from {source_lang} to {target_lang}.
        Preserve the original formatting and meaning.
        Provide only the translated text, without any introductory phrases."""

        user_prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text_to_translate}"

        print(f"  (GPT 번역 요청: {source_lang} -> {target_lang} using gpt-4o-mini)")

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ]
        try:
            # Note: 이전 대화에서 제안드린 gpt-4o 대신, 확정해주신 gpt-4o-mini 모델을 유지합니다.
            completion = self.gpt_model.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0,
                                                                max_tokens=1024)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI GPT 번역 중 오류 발생: {e}")
            return f"GPT 번역 오류: {e}"

    def generate_llm_response_sync(self, text_input: str, original_language_code: str) -> dict:
        default_answers = {"answer_original_language": "답변 생성 실패", "answer_korean": "답변 생성 실패"}
        if not self.gpt_model:
            default_answers["answer_original_language"] = "OpenAI API 사용 불가"
            default_answers["answer_korean"] = "OpenAI API 사용 불가"
            return default_answers
        if not text_input or text_input.strip() == "":
            default_answers["answer_original_language"] = "입력 텍스트 없음"
            default_answers["answer_korean"] = "입력 텍스트 없음"
            return default_answers

        original_lang_name_en = original_language_code
        try:
            lang_obj = pycountry.languages.get(alpha_2=original_language_code.split('-')[0].lower())
            if lang_obj: original_lang_name_en = lang_obj.name
        except:
            pass

        system_prompt = f"""You are an AI expert specializing in South Korean public transportation. 
        Your primary role is to provide the most optimal travel route from a starting point to a destination within South Korea, in {original_lang_name_en}.

        Always follow these steps to structure your response:
        1.  Propose one "Recommended Route" that is the most efficient, considering a balance of travel time, number of transfers, and cost.
        2.  Describe each step of the Recommended Route in detail (e.g., walking, subway line, bus number, direction, travel time, getting off station, exit number).
        3.  You can also suggest one or two "Alternative Routes" that might be worth considering.
        4.  Conclude with a summary of the "Estimated Total Time" and "Estimated Fare".
        5.  Provide only the helpful answer, without any introductory phrases like "Of course, here is the route...".
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "How do I get from Gyeongbokgung Palace to the COEX Aquarium by subway?"},
            {"role": "assistant", "content": f"""Here is the most efficient subway route from Gyeongbokgung Palace to the COEX Aquarium.

        **Recommended Route: Line 3 → Line 9 (1 Transfer)**

        1.  **Walk**: From Gyeongbokgung Palace, walk about 5 minutes to **Gyeongbokgung Station (Line 3)**, Exit 5.
        2.  **Subway**: At Gyeongbokgung Station, take the **Line 3** train bound for **Ogeum** for 9 stops and get off at **Express Bus Terminal Station**. (Approx. 18 min)
        3.  **Transfer**: At Express Bus Terminal Station, transfer to the **Line 9 Express** train bound for **VHS Medical Center**.
        4.  **Subway**: Take the Line 9 Express train for 2 stops and get off at **Bongeunsa Station**. (Approx. 7 min)
        5.  **Walk**: Go out Exit 7 of Bongeunsa Station and follow the signs for about 2 minutes to the COEX Aquarium entrance.

        **Summary**
        * **Estimated Total Time**: Approx. 45 minutes (including walking and transfer time)
        * **Estimated Fare**: 1,500 KRW (with a transit card)

        **Alternative Route: Line 3 → Line 2 (1 Transfer)**
        * You can also take Line 3 from Gyeongbokgung Station to Euljiro 3-ga Station, then transfer to Line 2 and get off at Samseong (World Trade Center) Station. This requires a bit more walking from the station to the aquarium but can be a good alternative if you find the transfer at Express Bus Terminal complex."""},
            {"role": "user",
             "content": "What's the fastest way to get from Hongik University Station to the main gate of Seoul National University? I don't mind taking a bus or subway."},
            {"role": "assistant", "content": f"""The fastest way from Hongik University Station to the main gate of Seoul National University involves taking both the subway and a bus.

        **Recommended Route: Line 2 → Bus No. 5511**

        1.  **Subway**: At **Hongik Univ. Station (Line 2)**, take the Inner Circle Line train bound for **Sillim** for 8 stops and get off at **Seoul Nat'l Univ. Station**. (Approx. 16 min)
        2.  **Exit & Bus Stop**: Go out **Exit 3** of Seoul Nat'l Univ. Station and walk straight. The bus stop is right in front of you.
        3.  **Bus**: Take the **Green (Branch) Bus No. 5511** for 4 stops and get off at the **Seoul National University Main Gate** stop. (Approx. 10-15 min, depending on traffic)

        **Summary**
        * **Estimated Total Time**: Approx. 35-40 minutes (including walking and waiting time)
        * **Estimated Fare**: 1,500 KRW (with a transit card, including transfer discount)

        **Note**
        * During rush hour, Gwanak-ro (the road to the university) can be congested, which may increase the bus travel time slightly."""},
            {"role": "user", "content": text_input}
        ]
        try:
            print(f"\n--- [접근 방식 A] 1. 원본 언어({original_lang_name_en})로 답변 생성 ---")
            completion = self.gpt_model.chat.completions.create(model="gpt-4o", messages=messages,
                                                                max_tokens=1024,
                                                                temperature=0.7)
            original_language_answer = completion.choices[0].message.content.strip()
            print(f"  ✅ 생성된 원본 언어 답변: {original_language_answer[:100]}...")

            print(f"\n--- [접근 방식 A] 2. 생성된 답변을 한국어로 번역 ---")
            korean_answer = self.universal_translation_sync(original_language_answer, original_lang_name_en, "Korean")

            return {"answer_original_language": original_language_answer, "answer_korean": korean_answer}
        except Exception as e:
            print(f"❌ GPT 답변 생성 또는 번역 중 오류 발생: {e}")
            return default_answers

    def generate_speech_sync(self, text: str, lang_code: str) -> Optional[bytes]:
        if not self.bark_processor or not self.bark_model:
            print("⚠️ Bark 모델이 로드되지 않아 TTS 생성을 건너뜁니다.")
            return None
        if not text or not text.strip():
            return None

        try:
            # 1. Pandas로 Excel 파일 읽기
            file_path = "/Users/gim-yonghyeon/Downloads/bark.xlsx"  # 사용자가 지정한 경로
            df = pd.read_excel(file_path)

            # 2. 주어진 lang_code에 해당하는 프리셋 목록 필터링
            available_presets = df[df['lang_code'] == lang_code]['Prompt name'].tolist()

            if available_presets:
                # 3-1. 해당 언어의 프리셋이 있으면, 그 중 하나를 랜덤으로 선택
                voice_preset = random.choice(available_presets)
            else:
                print(f"⚠️ 언어 '{lang_code}'에 대한 Voice Preset을 찾을 수 없어 영어 기본값으로 대체합니다.")
                english_presets = df[df['lang_code'] == 'en']['Prompt name'].tolist()
                if english_presets:
                    voice_preset = 'en_speaker_6'  # 특정 영어 프리셋을 지정하거나
                else:
                    # 엑셀에 영어 프리셋조차 없을 경우를 대비한 최후의 기본값
                    voice_preset = 'en_speaker_0'

        except FileNotFoundError:
            print(f"❌ Voice Preset 파일 '{file_path}'을 찾을 수 없습니다. 기본 프리셋으로 진행합니다.")
            voice_preset = 'en_speaker_6'  # 파일을 못찾을 경우 기본값
        except Exception as e:
            print(f"❌ Voice Preset 처리 중 오류 발생: {e}. 기본 프리셋으로 진행합니다.")
            voice_preset = 'en_speaker_6'  # 그 외 오류 발생 시 기본값

        print(f"\n--- 🔊 TTS 생성 시작 (언어: {lang_code}, 프리셋: {voice_preset}) ---")
        try:
            inputs = self.bark_processor(text, voice_preset=voice_preset, return_tensors="pt")

            # 입력 텐서를 모델과 동일한 장치로 이동
            inputs = {k: v.to(self.bark_model.device) for k, v in inputs.items()}

            # 음성 생성 (do_sample=True는 더 자연스러운 결과를 위함)
            speech_values = self.bark_model.generate(**inputs, do_sample=True, fine_temperature=0.4,
                                                     coarse_temperature=0.8)

            sampling_rate = self.bark_model.generation_config.sample_rate
            audio_array = speech_values.cpu().numpy().squeeze()

            # 메모리 내에서 WAV 포맷으로 변환
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sampling_rate, format='WAV')
            buffer.seek(0)
            audio_bytes = buffer.read()

            print("✅ TTS 음성 생성 완료.")
            return audio_bytes

        except Exception as e:
            print(f"❌ Bark TTS 생성 중 오류 발생: {e}")
            return None

    async def transcription(self, audio_file: UploadFile, model_name: str) -> Dict[str, Any]:
        # --- ▼▼▼ 최종 출력 형식에 맞게 default_response 수정 ▼▼▼ ---
        default_response = {
            "text": "", "language_code": "unknown", "language_name": "Unknown",
            "language_probability": 0.0, "original_query": "", "korean_query": "",
            "original_language_answer": "", "korean_answer": "",
            "korean_answer_audio_base64": None,
            "original_language_answer_audio_base64": None
        }

        try:
            target_pipe = self.pipelines.get(model_name)
            if not target_pipe:
                error_message = f"요청한 모델 '{model_name}'을 사용할 수 없습니다. 사용 가능한 모델: {list(self.pipelines.keys())}"
                return {**default_response, "error": "ModelNotFoundError", "message": error_message}

            audio_bytes = await audio_file.read()
            try:
                print(" attempting to read audio with soundfile...")
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
                print("✅ Audio successfully read with soundfile.")
            except sf.LibsndfileError:
                print("⚠️ soundfile failed. Trying with pydub (likely m4a or mp3)...")
                try:
                    audio_file_like_object = io.BytesIO(audio_bytes)
                    audio_segment = AudioSegment.from_file(audio_file_like_object)
                    audio_segment = audio_segment.set_frame_rate(16000)
                    audio_segment = audio_segment.set_channels(1)
                    audio_array = np.array(audio_segment.get_array_of_samples())
                    if audio_array.dtype == np.int16:
                        audio_array = audio_array.astype(np.float32) / 32768.0
                    sampling_rate = audio_segment.frame_rate
                    print(f"✅ Audio successfully processed with pydub. Sample rate: {sampling_rate}")
                except Exception as e_pydub:
                    return {**default_response, "error": "AudioProcessingError",
                            "message": f"오디오 파일 처리 오류 (pydub): {e_pydub}"}

            if audio_array.dtype != np.float32: audio_array = audio_array.astype(np.float32)
            if audio_array.ndim > 1: audio_array = np.mean(audio_array, axis=1)

            loaded_audio_sample = {"array": audio_array, "sampling_rate": sampling_rate}
            print(f"✅ 오디오 데이터 준비 완료: {audio_file.filename}")

            print("\n--- 🔍 언어 감지 시작 (`openai-whisper` 라이브러리 사용) ---")
            language_info = {"code": "unknown", "name": "Unknown", "probability": 0.0}
            if self.whisper_detection_model:
                try:
                    trimmed_audio = whisper.pad_or_trim(audio_array)
                    mel = whisper.log_mel_spectrogram(trimmed_audio).to(self.whisper_detection_model.device)
                    _, probs = self.whisper_detection_model.detect_language(mel)
                    original_lang_code = max(probs, key=probs.get)
                    lang_probability = probs[original_lang_code]
                    lang_name = pycountry.languages.get(alpha_2=original_lang_code).name if pycountry.languages.get(
                        alpha_2=original_lang_code) else original_lang_code
                    language_info.update({
                        'code': original_lang_code,
                        'name': lang_name,
                        'probability': round(lang_probability, 2)
                    })
                    print(f"✅ 감지된 언어 코드: '{original_lang_code}' (확률: {lang_probability:.2f})")
                except Exception as e:
                    print(f"⚠️ `detect_language` 언어 감지 중 오류: {e}. STT는 자동 감지로 진행합니다.")
            else:
                print("⚠️ (언어 감지용) Whisper 모델이 로드되지 않아 언어 감지를 건너뜁니다.")

            print(f"\n--- 🗣️ STT 변환 작업 시작 (감지된 언어: {language_info['code']}) ---")
            generate_kwargs = {"language": language_info['code']} if language_info['code'] != 'unknown' else {}
            result = await run_in_threadpool(target_pipe, loaded_audio_sample.copy(), generate_kwargs=generate_kwargs)
            transcribed_text = result.get("text", "").strip()
            print(f"✅ STT 변환된 텍스트: {transcribed_text}")

            translated_question_korean = ""
            if transcribed_text and language_info['code'] != 'ko':
                translated_question_korean = await run_in_threadpool(self.universal_translation_sync, transcribed_text,
                                                                     language_info['name'], "Korean")
            elif language_info['code'] == 'ko':
                translated_question_korean = transcribed_text

            # --- Approach A만 실행 ---
            llm_answers = {"answer_original_language": "미실행", "answer_korean": "미실행"}
            if transcribed_text and language_info['code'] != 'unknown':
                llm_answers = await run_in_threadpool(self.generate_llm_response_sync, transcribed_text,
                                                      language_info['code'])
            korean_audio_base64 = None
            original_audio_base64 = None

            # 1. 한국어 답변 음성 생성
            korean_answer_text = llm_answers.get("answer_korean")
            if korean_answer_text and korean_answer_text not in ["미실행", "답변 생성 실패"]:
                korean_audio_bytes = await run_in_threadpool(self.generate_speech_sync, korean_answer_text, 'ko')
                if korean_audio_bytes:
                    korean_audio_base64 = base64.b64encode(korean_audio_bytes).decode('utf-8')

            original_answer_text = llm_answers.get("answer_original_language")
            if original_answer_text and original_answer_text not in ["미실행", "답변 생성 실패"]:
                original_audio_bytes = await run_in_threadpool(self.generate_speech_sync, original_answer_text,
                                                               language_info['code'])
                if original_audio_bytes:
                    original_audio_base64 = base64.b64encode(original_audio_bytes).decode('utf-8')

            final_response = {
                "original_query": transcribed_text,
                "language_code": language_info['code'],
                "language_name": language_info['name'],
                "language_probability": language_info['probability'],
                "korean_query": translated_question_korean,
                "original_language_answer": llm_answers.get("answer_original_language"),
                "korean_answer": llm_answers.get("answer_korean"),
                "korean_answer_audio_base64": korean_audio_base64,
                "original_language_answer_audio_base64": original_audio_base64
            }
            print(f"\n✅ 최종 응답 준비 완료 (음성 포함).")
            return final_response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {**default_response, "error": "TranscriptionProcessError", "message": f"음성 인식 처리 중 오류: {e}"}