---
name: qwen3-tts
description: High-quality text-to-speech generation using Qwen3-TTS with a fixed "Lojik" voice profile. Use this skill to convert text into natural-sounding speech for the user.
---

# Qwen3-TTS Skill (Lojik Voice)

This skill provides a streamlined way for AI agents to generate speech using the fixed "Lojik" voice profile on the Qwen3-TTS server.

## Instructions

To generate speech, use the Python client script located at `scripts/qwen3_tts_client.py`.

### Prerequisites

- The Qwen3-TTS server must be running (usually on port 3011).

### Common Tasks

#### 1. Generate Speech

Convert text to speech using the default "Lojik" voice.

```bash
python3 scripts/qwen3_tts_client.py --action generate --text "Hello world" --output output.mp3
```

#### 2. Health Check

Verify the server is up and reachable.

```bash
python3 scripts/qwen3_tts_client.py --action health
```

## Examples

### Generating a greeting in MP3 format

```bash
python3 scripts/qwen3_tts_client.py --action generate --text "Welcome to the future of AI speech." --output greeting.mp3
```
