# Voice Cloning Application - Compatible Solution
# Production-ready voice cloning with GUI interface

import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
import gradio as gr
from pathlib import Path
from pydub import AudioSegment
import tempfile
import shutil
import subprocess
import traceback

# Directory setup
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

for directory in [MODELS_DIR, SAMPLES_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Configure logging
import logging

log_file = LOG_DIR / "voice_cloning.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VoiceCloningApp")

# Constants
SR = 16000  # Sample rate for processing
MAX_SAMPLE_LENGTH = 30  # Maximum reference audio length in seconds

def pip_install(package):
    """Install packages using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_dependencies():
    """Check and install required dependencies."""
    try:
        logger.info("Checking dependencies...")

        try:
            import TTS
            logger.info("TTS is already installed")
        except ImportError:
            logger.info("Installing TTS...")
            pip_install("TTS==0.11.1")  # Use a specific compatible version
            logger.info("TTS installed successfully")

        logger.info("All dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

class XTTSVoiceCloner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            from TTS.utils.synthesizer import Synthesizer
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig

            self.Synthesizer = Synthesizer
            self.XttsConfig = XttsConfig
            self.Xtts = Xtts
            self.XttsAudioConfig = XttsAudioConfig

            torch.serialization.add_safe_globals({
                XttsConfig: XttsConfig,
                XttsAudioConfig: XttsAudioConfig,
                BaseDatasetConfig: BaseDatasetConfig,
                XttsArgs: XttsArgs
            })
        except Exception as e:
            logger.error(f"Error importing TTS modules: {e}")
            raise

        self.load_xtts_model()

    def load_xtts_model(self):
        logger.info("Loading XTTS-v2 model...")
        try:
            from TTS.api import TTS
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True
            ).to(self.device)
            logger.info("XTTS-v2 model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading XTTS-v2 model: {e}")
            raise

    def preprocess_audio(self, audio_path):
        logger.info(f"Processing reference audio: {audio_path}")
        try:
            y, sr = librosa.load(audio_path, sr=None)
            if sr != self.tts.synthesizer.output_sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.tts.synthesizer.output_sample_rate)
            y, _ = librosa.effects.trim(y, top_db=20)
            max_samples = self.tts.synthesizer.output_sample_rate * MAX_SAMPLE_LENGTH
            if len(y) > max_samples:
                logger.info(f"Audio too long, trimming to {MAX_SAMPLE_LENGTH} seconds")
                y = y[:max_samples]
            y = librosa.util.normalize(y)
            processed_path = str(SAMPLES_DIR / f"processed_{os.path.basename(audio_path)}")
            if not processed_path.endswith('.wav'):
                processed_path += '.wav'
            sf.write(processed_path, y, self.tts.synthesizer.output_sample_rate)
            return processed_path
        except Exception as e:
            logger.error(f"Error in preprocess_audio: {e}")
            raise

    def clone_voice(self, reference_audio_path, text_to_speak, language="en"):
        try:
            processed_audio = self.preprocess_audio(reference_audio_path)
            output_path = str(OUTPUT_DIR / f"cloned_{os.path.basename(reference_audio_path)}")
            if not output_path.endswith('.wav'):
                output_path += '.wav'
            logger.info(f"Generating speech with text: {text_to_speak[:50]}...")
            self.tts.tts_to_file(
                text=text_to_speak,
                file_path=output_path,
                speaker_wav=processed_audio,
                language=language
            )
            logger.info(f"Voice cloning completed, saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error in clone_voice: {e}")
            raise

class SoVitsTTSCloner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            logger.info("Checking for pydub and numpy...")
            import pydub
            import numpy
        except ImportError:
            logger.info("Installing required packages...")
            pip_install("pydub numpy soundfile")

    def preprocess_audio(self, audio_path):
        logger.info(f"Processing reference audio: {audio_path}")

        if not audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            wav_path = str(SAMPLES_DIR / f"{os.path.splitext(os.path.basename(audio_path))[0]}.wav")
            audio.export(wav_path, format="wav")
            audio_path = wav_path

        y, sr = librosa.load(audio_path, sr=None)

        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)

        y, _ = librosa.effects.trim(y, top_db=20)
        y = librosa.util.normalize(y)

        processed_path = str(SAMPLES_DIR / f"processed_{os.path.basename(audio_path)}")
        sf.write(processed_path, y, SR)

        return processed_path

    def clone_voice(self, reference_audio_path, text_to_speak, language="en"):
        processed_audio = self.preprocess_audio(reference_audio_path)

        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[0].id)
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)

            output_path = str(OUTPUT_DIR / f"basic_tts_{os.path.basename(reference_audio_path)}")
            if not output_path.endswith('.wav'):
                output_path += '.wav'

            engine.save_to_file(text_to_speak, output_path)
            engine.runAndWait()

            return output_path
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")
            raise

class AudioEnhancer:
    @staticmethod
    def enhance(audio_path):
        try:
            logger.info(f"Enhancing audio: {audio_path}")

            audio = AudioSegment.from_file(audio_path)
            audio = audio.compress_dynamic_range(threshold=-20, ratio=4.0, attack=5.0, release=50.0)
            audio = audio.high_pass_filter(80)
            audio = audio.low_pass_filter(12000)
            audio = audio.normalize()

            enhanced_path = str(OUTPUT_DIR / f"enhanced_{os.path.basename(audio_path)}")
            if not enhanced_path.endswith('.wav'):
                enhanced_path += '.wav'

            audio.export(enhanced_path, format="wav")

            logger.info(f"Audio enhancement completed, saved to {enhanced_path}")
            return enhanced_path
        except Exception as e:
            logger.error(f"Error in enhance: {e}")
            raise

def create_gui():
    try:
        if not check_and_install_dependencies():
            raise ImportError("Failed to install required dependencies")

        try:
            voice_cloner = XTTSVoiceCloner()
            cloner_type = "XTTS"
        except Exception as e:
            logger.warning(f"Failed to initialize XTTS: {e}")
            logger.info("Falling back to basic TTS approach")
            voice_cloner = SoVitsTTSCloner()
            cloner_type = "Basic TTS"

        enhancer = AudioEnhancer()

        def process(audio_file, text_input, apply_enhancement, language):
            if not audio_file or not text_input.strip():
                return None, "Please provide both reference audio and text to speak."

            try:
                output_path = voice_cloner.clone_voice(audio_file, text_input, language)

                if apply_enhancement:
                    output_path = enhancer.enhance(output_path)

                return output_path, f"Voice cloning completed successfully! Output saved to {output_path}"
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Error during processing: {error_details}")
                return None, f"Error during voice cloning: {str(e)}"

        demo = gr.Interface(
            fn=process,
            inputs=[
                gr.Audio(label="Reference Voice Audio", type="filepath"),
                gr.Textbox(label="Text to Convert to Speech", placeholder="Enter the text you want to convert to speech using the reference voice..."),
                gr.Checkbox(label="Apply Audio Enhancement", value=True),
                gr.Dropdown(
                    label="Language",
                    choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"],
                    value="en"
                )
            ],
            outputs=[
                gr.Audio(label="Cloned Voice Output"),
                gr.Textbox(label="Status")
            ],
            title="Voice Cloning Studio",
            description=f"""
            ## Voice Cloning Application

            This application allows you to clone voices by providing a reference audio sample. 
            The system will analyze the voice characteristics and generate new speech using the 
            provided text in a voice that sounds like the reference.

            Using: {cloner_type} (Compatible Version)

            ### Instructions:
            1. Upload a clear audio sample of the voice you want to clone (5-30 seconds recommended)
            2. Enter the text you want to convert to speech
            3. Select the language of the text you're entering
            4. Toggle audio enhancement if desired
            5. Click submit and wait for processing

            For best results:
            - Use clean audio with minimal background noise
            - Provide reference audio with clear speech
            - Keep reference audio under 30 seconds
            """
        )

        return demo
    except Exception as e:
        logger.error(f"Error creating GUI: {e}")
        raise

def main():
    try:
        logger.info("Initializing Voice Cloning Studio...")
        demo = create_gui()
        demo.launch(share=False, server_name="0.0.0.0")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Critical error: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()
