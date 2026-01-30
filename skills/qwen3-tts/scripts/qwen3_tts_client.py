#!/usr/bin/env python3
import os
import sys
import argparse
import requests
import json

# ============================================================================
# Hardcoded Configuration
# ============================================================================
API_KEY = "sk-proj-9e7903fe-2096-4829-b8ce-13466446468c"
DEFAULT_VOICE = "Lojik"
DEFAULT_BASE_URL = "http://localhost:3011"

class Qwen3TTSClient:
    def __init__(self, base_url=None):
        # Allow override of URL via env var, but default to localhost
        self.base_url = base_url or os.environ.get("QWEN3_TTS_URL", DEFAULT_BASE_URL)
        
        # Ensure base_url doesn't end with slash
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    def _get_headers(self):
        return {"Authorization": f"Bearer {API_KEY}"}

    def health(self):
        try:
            resp = requests.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate(self, text, output_path, format="mp3", language="Auto"):
        payload = {
            "input": text,
            "voice": DEFAULT_VOICE,
            "response_format": format,
            "language": language
        }
        try:
            resp = requests.post(
                f"{self.base_url}/v1/audio/speech",
                headers=self._get_headers(),
                json=payload,
                stream=True
            )
            resp.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {"status": "success", "output_path": output_path, "voice": DEFAULT_VOICE}
        except Exception as e:
            return {"status": "error", "message": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS API Client (Fixed Voice & Key)")
    parser.add_argument("--action", choices=["health", "generate"], required=True)
    parser.add_argument("--text", help="Text to synthesize (for generate action)")
    parser.add_argument("--output", help="Output file path (for generate action)")
    parser.add_argument("--format", default="mp3", help="Audio format (default: mp3)")
    parser.add_argument("--lang", default="Auto", help="Language (default: Auto)")
    parser.add_argument("--url", help="Override API Base URL")

    args = parser.parse_args()
    
    client = Qwen3TTSClient(base_url=args.url)
    
    if args.action == "health":
        print(json.dumps(client.health(), indent=2))
    elif args.action == "generate":
        if not args.text or not args.output:
            print("Error: --text and --output are required for generate action.")
            sys.exit(1)
        result = client.generate(args.text, args.output, args.format, args.lang)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
