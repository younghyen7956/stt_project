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
import io, ssl, whisper, torch, openai, json, os, pycountry, random, re
from typing import Optional, Dict, Any
from kokoro import KPipeline
import time

ssl._create_default_https_context = ssl._create_unverified_context
from stt_project.repository.stt_repository import SttRepository

KOKORO_SUPPORTED_LANGS = ['a', 'b', 'e', 'f', 'h', 'i', 'j', 'p', 'z']
KOKORO_VOICE_PRESETS = {
    'a': ['am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa'],
    'b': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
    'e': ['ef_dora', 'em_alex', 'em_santa'], 'f': ['ff_siwis'], 'h': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'i': ['if_sara', 'im_nicola'], 'j': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang']
}


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
            self.device, self.torch_dtype = torch.device("mps"), torch.bfloat16
        elif torch.cuda.is_available():
            self.device, self.torch_dtype = torch.device("cuda"), torch.bfloat16
        else:
            self.device, self.torch_dtype = torch.device("cpu"), torch.float16

        self.pipelines = {}
        self.gpt_model = None
        self.whisper_detection_model = None
        self.kokoro_tts_pipelines = {}

        self.get_model()
        print("--- SttRepositoryImpl: __init__ 최초 초기화 완료 ---")

    def get_model(self):
        model_names_str = os.getenv("STTMODELS", "openai/whisper-large-v3-turbo")
        model_names = [name.strip() for name in model_names_str.split(',') if name.strip()]
        for model_name in model_names:
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
                print(f"✅ Whisper 모델 ({model_name}) 로드 완료.")
            except Exception as e:
                print(f"❌ Whisper 모델 ({model_name}) 로드 중 오류: {e}")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gpt_model = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        if self.gpt_model:
            print("✅ OpenAI GPT 클라이언트 초기화 완료.")
        else:
            print("⚠️ OPENAI_API_KEY 환경 변수가 없어 OpenAI GPT 클라이언트를 초기화할 수 없습니다.")

        try:
            self.whisper_detection_model = whisper.load_model("base", device="cpu")
            print("✅ (언어 감지용) Whisper 모델 로드 완료.")
        except Exception as e:
            print(f"❌ (언어 감지용) Whisper 모델 로드 중 오류: {e}")

        log_lang_map = {
            'a': 'English', 'b': 'Brazilian Portuguese', 'e': 'Spanish', 'f': 'French',
            'h': 'Hindi', 'i': 'Italian', 'j': 'Japanese', 'p': 'Portuguese', 'z': 'Chinese'
        }

        # KOKORO_SUPPORTED_LANGS 리스트에 있는 모든 언어 모델을 로드합니다.
        for kokoro_char in KOKORO_SUPPORTED_LANGS:
            if kokoro_char not in self.kokoro_tts_pipelines:
                try:
                    lang_name_for_log = log_lang_map.get(kokoro_char, f"Unknown ({kokoro_char})")
                    print(f"  - '{lang_name_for_log}' 모델 로드 중...")
                    self.kokoro_tts_pipelines[kokoro_char] = KPipeline(lang_code=kokoro_char)
                    print(f"  ✅ '{lang_name_for_log}' 모델 로드 완료.")
                except Exception as e:
                    print(f"  ❌ '{lang_name_for_log}' 모델 로드 실패: {e}")

        print("--- ✅ 모든 TTS 모델 미리 로드 완료 ---")
        print("--- SttRepositoryImpl: get_model successfully ---")

    def _clean_text_for_speech(self, text: str) -> str:
        """TTS 합성을 위해 텍스트에서 마크다운, 괄호, 특수 기호를 제거합니다."""
        if not text: return ""
        cleaned_text = text.replace('**', '').replace('*', '')
        cleaned_text = cleaned_text.replace('→', ' ')
        cleaned_text = re.sub(r'#+\s', '', cleaned_text)
        # 괄호와 그 안의 내용 (주로 영어 또는 부가 설명)을 통째로 제거
        cleaned_text = re.sub(r'\s*\(.*?\)\s*', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def universal_translation_sync(self, text_to_translate: str, source_lang: str, target_lang: str) -> str:
        if not self.gpt_model: return "GPT 번역 서비스 사용 불가"
        if not text_to_translate or not text_to_translate.strip(): return ""
        system_content = f"You are a professional translator. Your task is to accurately translate the given text from {source_lang} to {target_lang}. Preserve the original formatting and meaning. Provide only the translated text, without any introductory phrases."
        messages = [{"role": "system", "content": system_content}, {"role": "user",
                                                                    "content": f"Translate the following text from {source_lang} to {target_lang}:\n\n{text_to_translate}"}]
        try:
            completion = self.gpt_model.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.0)
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ OpenAI GPT 번역 중 오류: {e}")
            return f"GPT 번역 오류: {e}"

    def generate_llm_response_sync(self, text_input: str, original_language_code: str,
                                   enable_translation: bool = False) -> dict:
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
        6.  IMPORTANT for non-English responses: Provide the response purely in the target language ({original_lang_name_en}). Do not include English text, romanizations, or explanations in parentheses. For example, instead of "堂山 (Dangsan)", just write "堂山". Instead of writing durations like "(Approx. 10 min)", integrate the time naturally into the sentence in the target language.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_input}
        ]
        try:
            completion = self.gpt_model.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.7)
            original_language_answer = completion.choices[0].message.content.strip()

            korean_answer = "번역 비활성화됨"
            if enable_translation:
                print("  - LLM 답변 한국어 번역 중...")
                korean_answer = self.universal_translation_sync(original_language_answer, original_lang_name_en,
                                                                "Korean")

            return {"answer_original_language": original_language_answer, "answer_korean": korean_answer}
        except Exception as e:
            print(f"❌ OpenAI GPT 답변 생성/번역 중 오류: {e}")
            return default_answers

    def generate_speech_sync(self, text: str, lang_code: str) -> Optional[bytes]:
        if not KPipeline:
            print("⚠️ Kokoro 라이브러리가 없어 TTS 생성을 건너쌉니다.")
            return None
        if not text or not text.strip():
            return None

        kokoro_lang_code_map = {
            'en': 'a', 'es': 'e', 'fr': 'f', 'hi': 'h', 'it': 'i', 'ja': 'j', 'pt': 'p', 'zh': 'z'
        }
        kokoro_lang_char = kokoro_lang_code_map.get(lang_code.split('-')[0].lower())

        if kokoro_lang_char is None or kokoro_lang_char not in KOKORO_SUPPORTED_LANGS:
            print(f"⚠️ Kokoro TTS는 '{lang_code}' 언어에 대한 음성 생성을 지원하지 않습니다.")
            return None

        if kokoro_lang_char not in self.kokoro_tts_pipelines:
            print(f"🔧 Kokoro TTS 파이프라인 ('{kokoro_lang_char}' / {lang_code})을 처음 로드합니다...")
            try:
                self.kokoro_tts_pipelines[kokoro_lang_char] = KPipeline(lang_code=kokoro_lang_char)
                print(f"✅ Kokoro TTS 파이프라인 ('{kokoro_lang_char}') 로드 및 저장 완료.")
            except Exception as e:
                print(f"❌ Kokoro TTS 파이프라인 ('{kokoro_lang_char}') 로드 실패: {e}")
                return None

        tts_pipeline = self.kokoro_tts_pipelines[kokoro_lang_char]

        available_voice_presets = KOKORO_VOICE_PRESETS.get(kokoro_lang_char)
        if not available_voice_presets:
            print(f"⚠️ 언어 코드 '{kokoro_lang_char}'에 대한 Kokoro 음성 프리셋을 찾을 수 없습니다.")
            return None

        voice_preset = random.choice(available_voice_presets)
        print(f"\n--- 🔊 Kokoro TTS 생성 시작 (언어: {lang_code}, Kokoro 코드: '{kokoro_lang_char}', 프리셋: {voice_preset}) ---")
        try:
            # 정제된 텍스트를 TTS 엔진에 전달
            generator = tts_pipeline(text, voice=voice_preset)
            all_audio_arrays = [audio_chunk for _, _, audio_chunk in generator]

            if not all_audio_arrays:
                print("❌ Kokoro TTS에서 오디오 청크를 받지 못했습니다.")
                return None

            combined_audio_array = np.concatenate(all_audio_arrays)
            buffer = io.BytesIO()
            sf.write(buffer, combined_audio_array, 24000, format='WAV')
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            print(f"❌ Kokoro TTS 생성 중 오류: {e}")
            import traceback;
            traceback.print_exc()
            return None

    async def transcription(self, audio_file: UploadFile, model_name: str, enable_translation: bool = False) -> Dict[
        str, Any]:
        start_time = time.time()
        last_step_time = start_time
        print(f"\n--- 🚀 새로운 요청 처리 시작 (번역 활성화: {enable_translation}) 🚀 ---")

        default_response = {"text": "", "language_code": "unknown", "language_name": "Unknown",
                            "language_probability": 0.0, "original_query": "", "korean_query": "",
                            "original_language_answer": "", "korean_answer": "", "korean_answer_audio_base64": None,
                            "original_language_answer_audio_base64": None}
        try:
            target_pipe = self.pipelines.get(model_name)
            if not target_pipe:
                return {**default_response, "error": "ModelNotFoundError",
                        "message": f"요청한 모델 '{model_name}'을 사용할 수 없습니다."}

            audio_bytes = await audio_file.read()
            try:
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
            except sf.LibsndfileError:
                try:
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_frame_rate(16000).set_channels(
                        1)
                    audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                    sampling_rate = audio_segment.frame_rate
                except Exception as e_pydub:
                    return {**default_response, "error": "AudioProcessingError", "message": f"오디오 파일 처리 오류: {e_pydub}"}

            if audio_array.ndim > 1: audio_array = np.mean(audio_array, axis=1)
            loaded_audio_sample = {"array": audio_array.astype(np.float32), "sampling_rate": sampling_rate}

            current_time = time.time()
            print(f"⏱️  [1/7] 오디오 파일 처리 완료: {current_time - last_step_time:.2f}초")
            last_step_time = current_time

            language_info = {"code": "unknown", "name": "Unknown", "probability": 0.0}
            if self.whisper_detection_model:
                try:
                    trimmed_audio = whisper.pad_or_trim(loaded_audio_sample['array'])
                    mel = whisper.log_mel_spectrogram(trimmed_audio).to(self.whisper_detection_model.device)
                    _, probs = self.whisper_detection_model.detect_language(mel)
                    original_lang_code = max(probs, key=probs.get)
                    lang_name = pycountry.languages.get(alpha_2=original_lang_code).name if pycountry.languages.get(
                        alpha_2=original_lang_code) else original_lang_code
                    language_info.update({'code': original_lang_code, 'name': lang_name,
                                          'probability': round(probs[original_lang_code], 2)})
                    print(f"✅ 감지된 언어: '{original_lang_code}' (확률: {language_info['probability']})")
                except Exception as e:
                    print(f"⚠️ 언어 감지 중 오류: {e}")

            current_time = time.time()
            print(f"⏱️  [2/7] 언어 감지 완료: {current_time - last_step_time:.2f}초")
            last_step_time = current_time

            generate_kwargs = {"language": language_info['code']} if language_info['code'] != 'unknown' else {}
            result = await run_in_threadpool(target_pipe, loaded_audio_sample.copy(), generate_kwargs=generate_kwargs)
            transcribed_text = result.get("text", "").strip()

            current_time = time.time()
            print(f"⏱️  [3/7] STT (음성->텍스트) 변환 완료: {current_time - last_step_time:.2f}초")
            last_step_time = current_time

            translated_question_korean = "번역 비활성화됨"
            if enable_translation:
                if transcribed_text and language_info['code'] != 'ko':
                    translated_question_korean = await run_in_threadpool(self.universal_translation_sync,
                                                                         transcribed_text,
                                                                         language_info['name'], "Korean")
                elif language_info['code'] == 'ko':
                    translated_question_korean = transcribed_text
            else:
                if language_info['code'] == 'ko':
                    translated_question_korean = transcribed_text

            current_time = time.time()
            if enable_translation:
                print(f"⏱️  [4/7] 질문 번역 완료: {current_time - last_step_time:.2f}초")
            else:
                print(f"⏱️  [4/7] 질문 번역 건너뜀.")
            last_step_time = current_time

            llm_answers = {"answer_original_language": "미실행", "answer_korean": "미실행"}
            if transcribed_text and language_info['code'] != 'unknown':
                llm_answers = await run_in_threadpool(self.generate_llm_response_sync, transcribed_text,
                                                      language_info['code'], enable_translation)

            current_time = time.time()
            print(f"⏱️  [5/7] LLM 답변 생성 완료: {current_time - last_step_time:.2f}초")
            last_step_time = current_time

            cleaned_original_answer = self._clean_text_for_speech(llm_answers.get("answer_original_language", ""))
            cleaned_korean_answer = self._clean_text_for_speech(llm_answers.get("answer_korean", ""))

            current_time = time.time()
            print(f"⏱️  [6/7] 텍스트 정제 완료: {current_time - last_step_time:.2f}초")
            last_step_time = current_time

            original_audio_base64 = None
            if cleaned_original_answer and language_info['code'] != 'ko':
                original_audio_bytes = await run_in_threadpool(self.generate_speech_sync, cleaned_original_answer,
                                                               language_info['code'])
                if original_audio_bytes:
                    original_audio_base64 = base64.b64encode(original_audio_bytes).decode('utf-8')

            current_time = time.time()
            print(f"⏱️  [7/7] TTS (텍스트->음성) 생성 완료: {current_time - last_step_time:.2f}초")
            last_step_time = current_time

            final_response = {
                "original_query": transcribed_text,
                "language_code": language_info['code'],
                "language_name": language_info['name'],
                "language_probability": language_info['probability'],
                "korean_query": translated_question_korean,
                "original_language_answer": cleaned_original_answer,
                "korean_answer": cleaned_korean_answer,
                "korean_answer_audio_base64": None,
                "original_language_answer_audio_base64": original_audio_base64
            }

            total_time = time.time() - start_time
            print(f"✅ --- 요청 처리 성공! (총 소요 시간: {total_time:.2f}초) --- ✅")
            return final_response

        except Exception as e:
            import traceback
            traceback.print_exc()
            total_time = time.time() - start_time
            print(f"❌ --- 오류 발생! (총 소요 시간: {total_time:.2f}초) --- ❌")
            return {**default_response, "error": "TranscriptionProcessError", "message": f"음성 인식 처리 중 오류: {e}"}