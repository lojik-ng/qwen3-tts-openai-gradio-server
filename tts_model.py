"""
Shared TTS Model Manager for Qwen3-TTS Server
Handles model loading and voice cloning with caching
"""

import os
import time
import torch
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
import numpy as np


class ClientDisconnected(Exception):
    """Raised when the caller vanished while queued for the GPU.

    The audio was never generated, so no synthesis work was wasted. Distinct from
    a failure: there is nobody left to return an error to.
    """


def _check_cuda_available() -> bool:
    """Check if CUDA is available with retries for initialization timing"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if torch.cuda.is_available():
                # Try to actually access the GPU to confirm it works
                torch.cuda.get_device_name(0)
                return True
        except Exception as e:
            print(f"CUDA check attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return False


class TTSModelManager:
    """Singleton manager for Qwen3-TTS model with voice prompt caching"""
    
    _instance: Optional["TTSModelManager"] = None
    _model = None
    _voice_prompts: Dict[str, any] = {}
    _device: str = "cpu"
    _lock = threading.Lock()
    
    MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    VOICES_DIR = "/app/voices"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the Qwen3-TTS Base model"""
        # Delayed import to allow CUDA to initialize
        from qwen_tts import Qwen3TTSModel
        
        print(f"Loading model: {self.MODEL_ID}")
        
        # Check CUDA availability with retries
        cuda_available = _check_cuda_available()
        
        if cuda_available:
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            self._device = "cuda:0"
            dtype = torch.bfloat16
        else:
            print("CUDA not available, using CPU")
            self._device = "cpu"
            dtype = torch.float32
        
        # Attention, best first, and every rung tried rather than two.
        #
        # flash_attention_2 stays at the head because it is the fastest where it
        # exists. It does not exist here: there is no usable flash-attn wheel for
        # an RTX 5090, compute capability 12.0, on CUDA 13, and the Dockerfile's
        # source build already falls back instead of succeeding.
        #
        # sdpa is the rung that was missing, and the only one actually available.
        # It ships with torch. Measured on this box, same model and voice:
        # 0.78x realtime against eager's 1.22x, flat from 46 to 796 character
        # requests in both cases. Eager does not degrade with length; it is
        # simply 56% dearer every time.
        #
        # That is not merely a speed figure. Generation is serialised behind
        # self._lock while the API accepts requests concurrently, so the time one
        # request takes is time every other request spends queueing, and a queue
        # longer than the caller's timeout is what "the server hung" means here.
        #
        # eager stays last. It is correct, just dear.
        candidates = ["flash_attention_2", "sdpa", "eager"] if cuda_available else ["eager"]

        last_error = None
        for attn_impl in candidates:
            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    self.MODEL_ID,
                    device_map=self._device,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )
                self._attn = attn_impl
                print(f"Model loaded successfully on {self._device} with {attn_impl}")
                return
            except Exception as e:
                last_error = e
                print(f"Could not load with {attn_impl}: {e}")

        # Every implementation failed on the card, so the card is the problem
        # rather than the attention. CPU is slow, but it answers.
        if self._device != "cpu":
            print(f"GPU loading failed, falling back to CPU: {last_error}")
            self._device = "cpu"
            self._model = Qwen3TTSModel.from_pretrained(
                self.MODEL_ID,
                device_map="cpu",
                dtype=torch.float32,
                attn_implementation="eager",
            )
            self._attn = "eager"
            print("Model loaded on CPU as fallback")
        else:
            raise last_error
    
    @property
    def model(self):
        return self._model
    
    @property
    def device(self) -> str:
        return self._device
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice names from the voices directory"""
        voices = []
        voices_path = Path(self.VOICES_DIR)
        
        if not voices_path.exists():
            print(f"Voices directory not found: {self.VOICES_DIR}")
            return voices
        
        for wav_file in voices_path.glob("*.wav"):
            voice_name = wav_file.stem
            txt_file = voices_path / f"{voice_name}.txt"
            if txt_file.exists():
                voices.append(voice_name)
        
        return sorted(voices)
    
    def get_voice_prompt(self, voice_name: str, force_reload: bool = False):
        """Get or create a cached voice prompt for the given voice name"""
        if voice_name in self._voice_prompts and not force_reload:
            return self._voice_prompts[voice_name]
        
        voices_path = Path(self.VOICES_DIR)
        wav_path = voices_path / f"{voice_name}.wav"
        txt_path = voices_path / f"{voice_name}.txt"
        
        if not wav_path.exists():
            raise FileNotFoundError(f"Voice audio not found: {wav_path}")
        if not txt_path.exists():
            raise FileNotFoundError(f"Voice text not found: {txt_path}")
        
        # Read reference text
        ref_text = txt_path.read_text(encoding="utf-8").strip()
        
        print(f"Creating voice prompt for '{voice_name}'...")
        
        # Create voice clone prompt
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=str(wav_path),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        
        self._voice_prompts[voice_name] = prompt
        print(f"Voice prompt cached for '{voice_name}'")
        
        return prompt
    
    # How often a queued request re-checks whether its client is still there.
    # 0.5s bounds how long an abandoned request keeps holding a place in line.
    QUEUE_POLL_INTERVAL = 0.5

    def _acquire(self, should_abort: Optional[Callable[[], bool]]) -> float:
        """Acquire the inference lock, giving up if the caller disconnects.

        On return the lock is held and the caller owns it; generate_speech's
        finally block releases it. Returns the queue wait in seconds.

        Raises ClientDisconnected without ever touching the model if the client
        goes away first, so the GPU slot goes to the next live request instead of
        being burnt on audio nobody will collect. On that path the lock is left
        exactly as it was found.

        Without should_abort (the Gradio path) this is the old plain blocking
        acquire. The polling loop is required: acquire() cannot be interrupted, and
        an abandoned thread that woke up later and grabbed the lock anyway would
        wedge every other request forever.
        """
        if should_abort is None:
            started = time.time()
            self._lock.acquire()
            return time.time() - started

        started = time.time()
        while not self._lock.acquire(timeout=self.QUEUE_POLL_INTERVAL):
            if should_abort():
                raise ClientDisconnected(
                    "client disconnected while queued for inference"
                )
        return time.time() - started

    def generate_speech(
        self,
        text: str,
        voice_name: str,
        language: str = "Auto",
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech for the given text using the specified voice
        
        Args:
            text: Text to synthesize
            voice_name: Name of the voice to use (from voices folder)
            language: Language code (Auto, English, Chinese, etc.)
            should_abort: Optional callable polled while queued; if it returns
                True the client is gone and ClientDisconnected is raised without
                generating anything. None means "no liveness signal" (Gradio).

        Raises:
            ClientDisconnected: caller vanished while waiting for the lock.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Get or create voice prompt, then run, serialised.
        #
        # Queue time is measured apart from run time on purpose. The API accepts
        # requests concurrently through asyncio.to_thread and they all meet this
        # one lock, so a caller that waited two minutes cannot tell whether the
        # server was slow or simply busy behind somebody else's line. Those are
        # different faults with different fixes.
        waited = self._acquire(should_abort)
        try:
            voice_prompt = self.get_voice_prompt(voice_name)

            started = time.time()
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_prompt,
            )
            took = time.time() - started

            audio = (len(wavs[0]) / sr) if sr else 0.0
            if audio:
                print(
                    f"[tts] {len(text):4d} chars  {self._attn}  queued {waited:5.1f}s  "
                    f"ran {took:5.1f}s  audio {audio:5.1f}s  {took / audio:4.2f}x",
                    flush=True,
                )
            else:
                print(
                    f"[tts] {len(text):4d} chars  {self._attn}  queued {waited:5.1f}s  "
                    f"ran {took:5.1f}s  NO AUDIO",
                    flush=True,
                )

            return wavs[0], sr
        finally:
            # Always release, including when generation raises. A leaked lock
            # would stall every future request until the container restarted.
            self._lock.release()

    def reload_voice(self, voice_name: str):
        """Force reload a voice prompt (e.g., if files were updated)"""
        if voice_name in self._voice_prompts:
            del self._voice_prompts[voice_name]
        self.get_voice_prompt(voice_name, force_reload=True)
    
    def clear_voice_cache(self):
        """Clear all cached voice prompts"""
        self._voice_prompts.clear()


# Global instance
_manager: Optional[TTSModelManager] = None


def get_tts_manager() -> TTSModelManager:
    """Get or create the global TTS manager instance"""
    global _manager
    if _manager is None:
        _manager = TTSModelManager()
    return _manager
