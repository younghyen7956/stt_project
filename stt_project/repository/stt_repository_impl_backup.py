from abc import ABC, abstractmethod
import os, pycountry

import openai, json
import torch
from dotenv import load_dotenv
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Gemma3ForCausalLM, AutoTokenizer
from transformers import pipeline
from stt_project.repository.stt_repository import SttRepository  # 이 부분은 실제 프로젝트 구조에 맞게 수정해야 할 수 있습니다.
import soundfile as sf
import numpy as np
import io
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import text as mp_text
from typing import Optional, Dict, Any

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

        self.model = None
        self.processor = None
        self.common_pipe = None
        self.pipelines = {}
        self.translator_model = None  # Gemma 모델 (기존 translation_task_sync용)
        self.translator_tokenizer = None  # Gemma 토크나이저
        self.gpt_model = None  # OpenAI 모델 (STT 결과 한국어 번역 및 LLM 답변 생성용)

        self.get_model()
        print("--- SttRepositoryImpl: __init__ 최초 초기화 완료 ---")

    def get_model(self):
        model_name = os.getenv("STTMODEL")
        trans_name = os.getenv("TRANSMODEL")  # Gemma 모델명
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        model_names_str = os.getenv("STTMODELS", "openai/whisper-large-v3-turbo")  # 기본값으로 large-v3-turbo 설정
        model_names = [name.strip() for name in model_names_str.split(',')]

        for model_name in model_names:
            print(f"\n--- Whisper 모델 ({model_name}) 로드 시작 ---")
            try:
                processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=(self.device.type == "cpu"),
                    use_safetensors=True
                ).to(self.device)

                # 해당 모델의 파이프라인을 생성하여 딕셔너리에 저장
                self.pipelines[model_name] = pipeline(
                    "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, torch_dtype=self.torch_dtype,
                    device=self.device, return_timestamps=True,
                )
                print(f"✅ Whisper 모델 ({model_name}) 및 ASR 파이프라인 생성 완료.")
            except Exception as e:
                print(f"❌ Whisper 모델 ({model_name}) 로드 중 오류: {e}")

        # ... (MediaPipe, OpenAI 클라이언트 초기화는 기존과 동일) ...
        print("\n--- get_model_and_client 완료 ---")

        print("✅ STT 파이프라인 생성 완료")

        # OpenAI GPT 모델 초기화
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.gpt_model = openai.OpenAI(api_key=openai_api_key)
            print("✅ OpenAI GPT 클라이언트 초기화 완료.")
        else:
            print("⚠️ OPENAI_API_KEY 환경 변수가 없어 OpenAI GPT 클라이언트를 초기화할 수 없습니다.")
            self.gpt_model = None

        self.mediapipe_language_detector = None
        try:
            language_detector_model_path = os.getenv("MEDIAPIPE_LANG_DETECTOR_MODEL_PATH", "models/detector.tflite")
            if not os.path.exists(language_detector_model_path):
                print(f"⚠️ 경고: MediaPipe 언어 감지 모델 파일({language_detector_model_path})을 찾을 수 없습니다.")
            else:
                base_options = mp_python.BaseOptions(model_asset_path=language_detector_model_path)
                options = mp_text.LanguageDetectorOptions(base_options=base_options)
                self.mediapipe_language_detector = mp_text.LanguageDetector.create_from_options(options)
                print("✅ MediaPipe Language Detector 초기화 완료.")
        except Exception as e:
            print(f"❌ MediaPipe Language Detector 초기화 중 오류 발생: {e}")

        print("--- SttRepositoryImpl: get_model successfully ---")

    def translate_text_to_korean_with_gpt_sync(self, text_to_translate: str) -> str:
        """
        OpenAI GPT 모델을 사용하여 주어진 텍스트를 한국어로 번역합니다.
        """
        if not self.gpt_model:
            print("⚠️ OpenAI 클라이언트(gpt_model)가 초기화되지 않아 GPT 한국어 번역을 수행할 수 없습니다.")
            return "GPT 번역 서비스 사용 불가 (클라이언트 미설정)"

        if not text_to_translate or not text_to_translate.strip():
            print("⚠️ 번역할 텍스트가 비어있어 GPT 한국어 번역을 건너<0xEB><0x81><0x84>니다.")
            return ""  # 빈 텍스트는 빈 텍스트로 반환

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that translates text into Korean. Provide only the Korean translation, without any introductory phrases or explanations."},
            {"role": "user", "content": f"Translate the following text into Korean:\n\n\"{text_to_translate}\""}
        ]

        try:
            print(f"  OpenAI GPT API 호출 (한국어 번역) - 입력 (앞 100자): {text_to_translate[:100]}...")
            completion = self.gpt_model.chat.completions.create(
                model="gpt-4o-mini",  # 필요시 모델 변경 (e.g., "gpt-3.5-turbo")
                messages=messages,
                temperature=0.1,  # 번역은 일관성이 중요하므로 낮은 온도로 설정
                max_tokens=1024,  # 번역 결과에 충분한 토큰 할당 (원본 텍스트 길이 고려)
                top_p=1.0,
            )
            translated_text = completion.choices[0].message.content.strip()
            print(f"  GPT 한국어 번역 결과 (첫 100자): {translated_text[:100]}...")
            return translated_text
        except Exception as e:
            print(f"❌ OpenAI GPT 한국어 번역 중 오류 발생: {e}")
            # import traceback; traceback.print_exc() # 디버깅용
            return f"GPT 한국어 번역 오류: {e}"

    def detect_language_with_mediapipe(self, text_to_detect: str) -> dict:
        default_result = {"code": "unknown", "name": "Unknown", "probability": 0.0}
        if not self.mediapipe_language_detector:
            print("⚠️ MediaPipe Language Detector가 초기화되지 않아 언어 감지를 수행할 수 없습니다.")
            return default_result
        if not text_to_detect or text_to_detect.strip() == "":
            print("⚠️ 언어 감지를 위한 텍스트가 비어있거나 공백입니다.")
            return default_result
        try:
            detection_result = self.mediapipe_language_detector.detect(text_to_detect)
            if detection_result.detections:
                best_detection = max(detection_result.detections, key=lambda d: d.probability)
                detected_lang_code = best_detection.language_code
                probability = round(best_detection.probability, 2)
                lang_name = "Unknown"
                code_to_lookup = detected_lang_code.split('-')[0].split('_')[0].lower()
                try:
                    lang_obj = None
                    if len(code_to_lookup) == 2:
                        lang_obj = pycountry.languages.get(alpha_2=code_to_lookup)
                    elif len(code_to_lookup) == 3:
                        lang_obj = pycountry.languages.get(alpha_3=code_to_lookup)

                    if lang_obj and hasattr(lang_obj, 'name'):
                        lang_name = lang_obj.name
                    # ... (기존 수동 매핑 로직 유지) ...
                    elif detected_lang_code.lower() == 'zh-hant':
                        lang_name = 'Chinese (Traditional)'
                    elif detected_lang_code.lower() == 'zh-hans':
                        lang_name = 'Chinese (Simplified)'
                    else:
                        lang_name = detected_lang_code
                except Exception:
                    lang_name = detected_lang_code
                result = {"code": detected_lang_code, "name": lang_name, "probability": probability}
                print(f"  [MediaPipe & pycountry] 감지된 언어 정보: {result}")
                return result
            else:
                print("  [MediaPipe] 감지된 언어가 없습니다.")
                return default_result
        except Exception as e:
            print(f"❌ MediaPipe 언어 감지 중 오류 발생: {e}")
            return default_result

    def generate_llm_response_sync(self, text_input: str, original_language_code: str) -> dict:
        default_answers = {
            "answer_original_language": "답변 생성 실패 (원본 언어)",
            "answer_korean": "답변 생성 실패 (한국어)"
        }
        if not self.gpt_model:
            print("⚠️ OpenAI 클라이언트(gpt_model)가 초기화되지 않아 GPT 답변 생성을 수행할 수 없습니다.")
            default_answers["answer_original_language"] = "OpenAI API 사용 불가"
            default_answers["answer_korean"] = "OpenAI API 사용 불가"
            return default_answers
        # ... (나머지 generate_llm_response_sync 함수 로직은 동일하게 유지) ...
        if not text_input or text_input.strip() == "":
            default_answers["answer_original_language"] = "입력 텍스트 없음"
            default_answers["answer_korean"] = "입력 텍스트 없음"
            return default_answers

        original_lang_name_en = original_language_code
        try:
            lang_code_for_lookup = original_language_code.split('-')[0].split('_')[0].lower()
            lang_obj = None
            if len(lang_code_for_lookup) == 2:
                lang_obj = pycountry.languages.get(alpha_2=lang_code_for_lookup)
            elif len(lang_code_for_lookup) == 3:
                lang_obj = pycountry.languages.get(alpha_3=lang_code_for_lookup)
            if lang_obj and hasattr(lang_obj, 'name'): original_lang_name_en = lang_obj.name
        except:
            pass

        system_prompt = """You are an AI expert specializing in South Korean transportation. Your primary role is to provide the most optimal travel route from a starting point to a destination within South Korea.

        When providing the route, consider factors like travel time, cost, and convenience. Suggest various methods like subway, bus, train (KTX, SRT), or taxi where appropriate.

        Your response MUST be a single JSON object. This JSON object must contain exactly two keys: "answer_in_original_language" and "answer_in_korean".

        - The value for "answer_in_original_language" must be your answer, written in the **same language as the user's input**.
        - The value for "answer_in_korean" must be the direct Korean translation of that same answer.

        Both answers must convey the exact same meaning. Do not include any other text, explanations, or conversational remarks outside of the JSON object.

        **Example:**

        User's question (in English):
        `"What's the best way to get from Incheon International Airport to Myeongdong Station?"`

        Your response:
        ```json
        {
          "answer_in_original_language": "The most efficient way is to take the AREX (Airport Railroad Express) All-Stop Train from Incheon Int'l Airport T1 to Seoul Station. At Seoul Station, transfer to subway Line 4 and go 2 stops to Myeongdong Station. The total travel time is approximately 1 hour and 10 minutes.",
          "answer_in_korean": "가장 효율적인 방법은 인천국제공항 T1에서 공항철도(AREX) 일반열차를 타고 서울역까지 가는 것입니다. 서울역에서 지하철 4호선으로 환승하여 2개 정거장을 이동하면 명동역에 도착합니다. 총 소요 시간은 약 1시간 10분입니다."
        }
        ```"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The user's text (in {original_lang_name_en}) is: \"{text_input}\""}
        ]
        try:
            print(
                f"  OpenAI API 호출 (gpt-4o-mini, JSON 모드) - 원본 언어: {original_lang_name_en}, 입력 (앞 100자): {text_input[:100]}...")
            completion = self.gpt_model.chat.completions.create(
                model="gpt-4o-mini", messages=messages, response_format={"type": "json_object"},
                max_tokens=512, temperature=0.7, top_p=0.9
            )
            response_content = completion.choices[0].message.content
            print(f"  OpenAI API JSON 응답 (Raw): {response_content[:200]}...")
            parsed_response = json.loads(response_content)
            llm_answer_original = parsed_response.get("answer_in_original_language", "JSON 파싱 실패 (원본 언어 답변)")
            llm_answer_korean = parsed_response.get("answer_in_korean", "JSON 파싱 실패 (한국어 답변)")
            return {
                "answer_original_language": llm_answer_original,
                "answer_korean": llm_answer_korean
            }
        except json.JSONDecodeError as e_json:
            print(
                f"GPT 응답 JSON 파싱 중 오류: {e_json}. Raw response: {response_content if 'response_content' in locals() else 'N/A'}")
            default_answers["answer_original_language"] = f"GPT 응답 형식 오류 (JSON 파싱 실패)"
            default_answers["answer_korean"] = f"GPT 응답 형식 오류 (JSON 파싱 실패)"
            return default_answers
        except Exception as e:
            print(f"GPT 답변 생성 중 오류 발생 (JSON 모드): {e}")
            default_answers["answer_original_language"] = f"GPT 답변 생성 중 오류 발생: {e}"
            default_answers["answer_korean"] = f"GPT 답변 생성 중 오류 발생: {e}"
            return default_answers

    async def transcription(self, audio_file: UploadFile, model_name: str) -> Dict[str, Any]:
        # --- 기본 응답 구조 ---
        default_response = {
            "text": "", "language_code": "unknown", "language_name": "Unknown",
            "language_probability": 0.0, "translated_text_korean": "",
            "llm_answer_original_language": "", "llm_answer_korean": ""
        }
        try:
            # --- 1. 요청된 모델 파이프라인 선택 ---
            print(f"음성 인식 요청 - 파일: {audio_file.filename}, 요청 모델: {model_name}")
            target_pipe = self.pipelines.get(model_name)
            if not target_pipe:
                available_models = list(self.pipelines.keys())
                error_message = f"요청한 모델 '{model_name}'을 사용할 수 없습니다. 사용 가능한 모델: {available_models}"
                print(f"❌ {error_message}")
                return {**default_response, "error": "ModelNotFoundError", "message": error_message}

            # --- 2. 오디오 파일 준비 ---
            audio_bytes = await audio_file.read()
            try:
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
            except sf.LibsndfileError as e_sf:
                return {**default_response, "error": "AudioProcessingError", "message": f"오디오 파일 오류: {e_sf}"}

            if audio_array.dtype != np.float32: audio_array = audio_array.astype(np.float32)
            if audio_array.ndim > 1 and audio_array.shape[1] > 1: audio_array = np.mean(audio_array, axis=1)

            loaded_audio_sample = {"array": audio_array, "sampling_rate": sampling_rate}
            print(f"✅ 오디오 데이터 준비 완료: {audio_file.filename}")

            # --- ✨ 수정된 부분: STT를 한 번만 실행하는 단순화된 로직 ✨ ---
            print("\n--- 🗣️ STT 변환 작업 시작 (오디오 기반 자동 언어 감지) ---")

            # 언어 옵션 없이 파이프라인을 호출하여 Whisper가 오디오에서 직접 언어를 감지하도록 함
            result = await run_in_threadpool(target_pipe, loaded_audio_sample.copy())
            transcribed_text = result.get("text", "").strip()
            print(f"✅ STT 변환된 텍스트: {transcribed_text}")

            # STT 결과 텍스트를 기반으로 어떤 언어였는지 확인
            language_info = {"code": "unknown", "name": "Unknown", "probability": 0.0}
            if transcribed_text:
                print("\n--- 🔍 MediaPipe 언어 확인 시작 ---")
                language_info = await run_in_threadpool(self.detect_language_with_mediapipe, transcribed_text)
            else:
                print("  ⚠️ STT 텍스트가 없어 언어 확인을 건너뜁니다.")

            original_lang_code = language_info.get('code', 'unknown')
            # --- ✨ 로직 수정 끝 ✨ ---

            # --- 4. GPT 한국어 번역 ---
            translated_text_korean_by_gpt = ""
            if transcribed_text and original_lang_code != 'ko':
                if self.gpt_model:
                    print(f"\n🌐 STT 텍스트를 한국어로 번역 시도 (GPT 사용)...")
                    translated_text_korean_by_gpt = await run_in_threadpool(
                        self.translate_text_to_korean_with_gpt_sync,
                        transcribed_text
                    )
                    print(f"  🇰🇷 GPT 번역 결과: {translated_text_korean_by_gpt[:100]}...")
                else:
                    translated_text_korean_by_gpt = "GPT 서비스 사용 불가"
            elif original_lang_code == 'ko':
                translated_text_korean_by_gpt = transcribed_text
            else:
                print("  ⚠️ STT 텍스트가 없어 GPT 한국어 번역을 건너뜁니다.")

            # --- 5. LLM 답변 생성 ---
            llm_answers = {"answer_original_language": "LLM 답변 미생성", "answer_korean": "LLM 답변 미생성"}
            if transcribed_text:
                if self.gpt_model:
                    if original_lang_code != 'unknown':
                        print(f"\n💬 STT 텍스트에 대한 LLM 답변 생성 시도 (원본 언어: {original_lang_code})...")
                        llm_answers = await run_in_threadpool(self.generate_llm_response_sync, transcribed_text,
                                                              original_lang_code)
                        print(f"  💡 LLM 생성 답변 (원본): {llm_answers.get('answer_original_language', '')[:100]}...")
                        print(f"  💡 LLM 생성 답변 (한국어): {llm_answers.get('answer_korean', '')[:100]}...")
                    else:
                        llm_answers["answer_original_language"] = llm_answers["answer_korean"] = "원본 언어 불명확"
                else:
                    llm_answers["answer_original_language"] = llm_answers["answer_korean"] = "GPT 서비스 사용 불가"
            else:
                llm_answers["answer_original_language"] = llm_answers["answer_korean"] = "STT 텍스트 없음"

            # --- 6. 최종 응답 반환 ---
            final_response = {
                "text": transcribed_text,
                "language_code": original_lang_code,
                "language_name": language_info.get('name', 'Unknown'),
                "language_probability": language_info.get('probability', 0.0),
                "translated_text_korean": translated_text_korean_by_gpt,
                "llm_answer_original_language": llm_answers.get("answer_original_language"),
                "llm_answer_korean": llm_answers.get("answer_korean")
            }
            print(f"\n✅ 최종 응답 준비 완료.")
            return final_response

        except Exception as e:
            import traceback
            print(f"❌ 음성 인식 및 처리 중 예상치 못한 오류 발생: {e} - 파일: {audio_file.filename}")
            traceback.print_exc()
            return {**default_response,
                    "error": "TranscriptionProcessError",
                    "message": f"음성 인식 및 처리 중 오류가 발생했습니다: {audio_file.filename}. ({e})"}