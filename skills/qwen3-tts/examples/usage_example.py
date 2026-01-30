"""
Simple example usage of the hardcoded Qwen3TTSClient.
"""
from scripts.qwen3_tts_client import Qwen3TTSClient

def run_example():
    client = Qwen3TTSClient()
    print("--- 🔍 Checking Health ---")
    print(client.health())

    print("\n--- 🔊 Generating Speech (Fixed Voice: Lojik) ---")
    result = client.generate(
        text="This is a test of the hardcoded Qwen3-TTS Agent Skill.",
        output_path="hardcoded_example.mp3"
    )
    print(result)

if __name__ == "__main__":
    run_example()
