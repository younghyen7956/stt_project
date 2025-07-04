import base64

import pandas as pd
from pydub import AudioSegment
from dotenv import load_dotenv
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModel
from transformers import pipeline
import soundfile as sf
import numpy as np
import io, ssl, whisper, torch, openai, json, os, pycountry, random
from typing import Optional, Dict, Any

# Kokoro TTS 라이브러리 임포트
try:
    from kokoro import KPipeline
except ImportError:
    print("WARNING: Kokoro library not found. Please install it to use Kokoro TTS.")
    KPipeline = None

ssl._create_default_https_context = ssl._unverified_context
from stt_project.repository.stt_repository import SttRepository

# --- Kokoro TTS 지원 언어 및 프리셋 정의 ---
# Kokoro가 사용하는 언어 코드 (예: 'a', 'b', 'e', 'f', 'h', 'i', 'j', 'p', 'z')
KOKORO_SUPPORTED_LANGS = ['a', 'b', 'e', 'f', 'h', 'i', 'j', 'p', 'z']

# 언어 코드별 기본 음성 프리셋 매핑
# 제공해주신 프리셋 목록을 바탕으로, 각 언어 코드에 맞는 대표 프리셋을 선정했습니다.
# 필요에 따라 더 다양하게 매핑하거나, 랜덤 선택 로직을 추가할 수 있습니다.
KOKORO_VOICE_PRESETS = {
    'a': ['am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa'],
    # American English
    'b': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
    # British English
    'e': ['ef_dora', 'em_alex', 'em_santa'],  # Spanish (es)
    'f': ['ff_siwis'],  # French (fr-fr)
    'h': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],  # Hindi (hi)
    'i': ['if_sara', 'im_nicola'],  # Italian (it)
    'j': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],  # Japanese (ja)
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],  # Brazilian Portuguese (pt-br)
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang']
    # Mandarin Chinese (zh)
}


# --- // Kokoro TTS 지원 언어 및 프리셋 정의 ---


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
        self.kokoro_tts_pipeline = None

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

        # Bark TTS 모델 로드 부분 제거
        self.bark_processor = None
        self.bark_model = None

        print("\n--- Kokoro TTS 모델 로드 시작 ---")
        if KPipeline:
            try:
                # Kokoro KPipeline 초기화. 가장 보편적인 'a' (American English)를 기본으로 설정.
                self.kokoro_tts_pipeline = KPipeline(lang_code='a')
                print("✅ Kokoro TTS KPipeline 로드 완료.")
            except Exception as e:
                print(f"❌ Kokoro TTS 모델 로드 중 오류: {e}")
                self.kokoro_tts_pipeline = None
        else:
            print("⚠️ Kokoro 라이브러리가 없어 Kokoro TTS 모델을 로드할 수 없습니다.")

        print("--- SttRepositoryImpl: get_model successfully ---")

    def universal_translation_sync(self, text_to_translate: str, source_lang: str, target_lang: str) -> str:
        if not self.gpt_model: return "GPT 번역 서비스 사용 불가"
        if not text_to_translate or not text_to_translate.strip(): return ""

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
            # pycountry를 사용하여 언어 이름 가져오기. 예: 'ko' -> 'Korean'
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
        if not self.kokoro_tts_pipeline:
            print("⚠️ Kokoro TTS 파이프라인이 로드되지 않아 TTS 생성을 건너뜁니다.")
            return None
        if not text or not text.strip():
            return None

        # Kokoro가 지원하는 언어 코드인지 확인
        # 감지된 언어 코드 (예: 'en', 'es', 'fr')를 Kokoro의 언어 코드 (예: 'a', 'e', 'f')로 매핑
        # pycountry.languages.get(alpha_2='en').alpha_1이 있다면 그걸 사용. 없으면 수동 매핑.
        kokoro_lang_code_map = {
            'en': 'a', 'es': 'e', 'fr': 'f', 'hi': 'h', 'it': 'i', 'ja': 'j', 'pt': 'p', 'zh': 'z'
        }
        # pycountry alpha_1이 없는 경우를 대비하여 직접 매핑 딕셔너리에서 가져옵니다.
        # 또한 'b' (British English)는 'a' (American English)로 대체합니다.
        kokoro_lang_char = kokoro_lang_code_map.get(lang_code.split('-')[0].lower())

        if kokoro_lang_char is None or kokoro_lang_char not in KOKORO_SUPPORTED_LANGS:
            print(f"⚠️ Kokoro TTS는 '{lang_code}' 언어에 대한 음성 생성을 지원하지 않습니다. (지원 언어: {', '.join(KOKORO_SUPPORTED_LANGS)})")
            return None

        # 해당 언어에 대한 사용 가능한 프리셋 목록 가져오기
        available_voice_presets = KOKORO_VOICE_PRESETS.get(kokoro_lang_char)

        if not available_voice_presets:
            print(f"⚠️ 언어 코드 '{kokoro_lang_char}'에 대한 Kokoro 음성 프리셋을 찾을 수 없습니다. 음성 생성을 건너뜁니다.")
            return None

        # 사용 가능한 프리셋 중 하나를 랜덤으로 선택
        voice_preset = random.choice(available_voice_presets)

        print(
            f"\n--- 🔊 Kokoro TTS 생성 시작 (텍스트 언어: {lang_code}, Kokoro 코드: '{kokoro_lang_char}', 사용 프리셋: {voice_preset}) ---")
        try:
            # KPipeline 인스턴스에 텍스트와 voice 프리셋 전달
            # 공식 문서 예시에서는 generator가 이터레이터를 반환하므로, 첫 번째 오디오 청크만 사용합니다.
            generator = self.kokoro_tts_pipeline(text, voice=voice_preset)

            # 첫 번째 오디오 청크를 가져옵니다.
            first_chunk = next(generator, None)

            if first_chunk is None:
                print("❌ Kokoro TTS 생성기에서 오디오 청크를 받지 못했습니다.")
                return None

            audio_array = first_chunk[2]
            sampling_rate = 24000  # Kokoro 공식 문서 예시에서 사용된 샘플링 레이트

            # 메모리 내에서 WAV 포맷으로 변환
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sampling_rate, format='WAV')
            buffer.seek(0)
            audio_bytes = buffer.read()

            print("✅ Kokoro TTS 음성 생성 완료.")
            return audio_bytes

        except Exception as e:
            print(f"❌ Kokoro TTS 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def transcription(self, audio_file: UploadFile, model_name: str) -> Dict[str, Any]:
        default_response = {
            "text": "", "language_code": "unknown", "language_name": "Unknown",
            "language_probability": 0.0, "original_query": "", "korean_query": "",
            "original_language_answer": "", "korean_answer": "",
            "korean_answer_audio_base64": None,  # 한국어 음성 반환 제거
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

            llm_answers = {"answer_original_language": "미실행", "answer_korean": "미실행"}
            if transcribed_text and language_info['code'] != 'unknown':
                llm_answers = await run_in_threadpool(self.generate_llm_response_sync, transcribed_text,
                                                      language_info['code'])

            korean_audio_base64 = None  # 한국어 음성 반환을 항상 None으로 설정
            original_audio_base64 = None

            # 원본 언어 답변 음성 생성 (한국어는 제외)
            original_answer_text = llm_answers.get("answer_original_language")
            if original_answer_text and original_answer_text not in ["미실행", "답변 생성 실패"]:
                # 한국어(ko)가 아닐 때만 음성 생성 시도
                if language_info['code'] != 'ko':
                    original_audio_bytes = await run_in_threadpool(self.generate_speech_sync, original_answer_text,
                                                                   language_info['code'])
                    if original_audio_bytes:
                        original_audio_base64 = base64.b64encode(original_audio_bytes).decode('utf-8')
                else:
                    print(f"⚠️ 원본 언어가 한국어이므로, 한국어 답변 음성 생성은 건너뜁니다.")

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