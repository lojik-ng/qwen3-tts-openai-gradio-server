# 🎙️ Qwen3-TTS Server

A Dockerized Text-to-Speech server powered by [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) with:

- **Gradio UI** - Beautiful web interface for voice generation
- **OpenAI-compatible API** - Drop-in replacement for OpenAI's TTS API
- **Custom Voice Cloning** - Use your own voice samples

## 🚀 Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA 12.1+ support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 1. Add Your Voices

Clone this repository and navigate to the directory.

Edit keys.json to set your API keys.

Create voice files in the `voices/` folder:

```
voices/
├── Lojik.wav      # Reference audio (5-30 seconds recommended)
├── Lojik.txt      # Transcript of the audio
├── sarah.wav
├── sarah.txt
└── ...
```

**Requirements for voice files:**

- `.wav` - Clear audio recording (16kHz+ sample rate recommended)
- `.txt` - Exact transcript of what's spoken in the audio

### 2. Start the Server

```bash
# Build and run with Docker Compose
docker-compose up -d --build

# Note: The server uses API Key authentication.
# Update keys.json with your own keys before deployment.

```

### 3. Access the Services

| Service | URL | Description |
|---------|-----|-------------|
| **Gradio UI** | <http://localhost:3010> | Web interface for TTS |
| **OpenAI API** | <http://localhost:3011> | REST API for TTS |
| **API Docs** | <http://localhost:3011/docs> | Swagger documentation |

## 🎨 Gradio UI

The web interface provides:

- 📝 Text input for synthesis
- 🎤 Voice selection dropdown (auto-populates from `voices/` folder)
- 🌍 Language selection
- 🔊 Audio playback and download

## 🔌 OpenAI-Compatible API

### Generate Speech

```bash
curl -X POST "http://localhost:3011/v1/audio/speech" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts",
    "input": "Hello, this is a test of the text to speech system.",
    "voice": "Lojik",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### List Available Voices

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:3011/v1/voices
```

### Python Example

```python
from openai import OpenAI

# Point to local server
client = OpenAI(
    base_url="http://localhost:3011/v1",
    api_key="YOUR_API_KEY"  # Key from keys.json
)

# Generate speech
response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Lojik",
    input="Welcome to Qwen3 TTS! This is an amazing voice cloning system."
)

# Save to file
response.stream_to_file("output.mp3")
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Generate speech from text |
| `/v1/voices` | GET | List available voices |
| `/v1/models` | GET | List available models |
| `/v1/voices/{name}/reload` | POST | Reload a specific voice |
| `/health` | GET | Health check |

### Request Body (POST /v1/audio/speech)

```json
{
  "model": "qwen3-tts",
  "input": "Text to synthesize",
  "voice": "voice_name",
  "response_format": "wav",
  "language": "Auto"
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `input` | Yes | - | Text to convert to speech |
| `voice` | Yes | - | Voice name from voices folder |
| `model` | No | `qwen3-tts` | Model identifier |
| `response_format` | No | `mp3` | Audio format: mp3, wav, flac, pcm |
| `language` | No | `Auto` | Language: Auto, English, Chinese, etc. |

## 📁 Project Structure

```
.
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Docker Compose configuration
├── keys.json            # API Keys configuration (JSON array)
├── requirements.txt     # Python dependencies
├── tts_model.py        # TTS model manager with caching
├── gradio_app.py       # Gradio web UI
├── openai_api.py       # OpenAI-compatible FastAPI server
├── server.py           # Combined server entry point
├── voices/             # Voice samples (mounted volume)
│   ├── voice.wav
│   └── voice.txt
└── README.md
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |
| `HF_TOKEN` | - | Hugging Face token (optional) |

### Ports

| Port | Service |
|------|---------|
| 3010 | Gradio UI |
| 3011 | OpenAI API |

### Volume Mounts

| Container Path | Host Path | Description |
|---------------|-----------|-------------|
| `/app/voices` | `./voices` | Voice samples (read-only) |
| `/app/keys.json` | `./keys.json` | API Keys (read-only) |

## 🔐 Authentication

The API is protected by API Key authentication.
You must provide a valid key in the `Authorization` header:

```http
Authorization: Bearer <YOUR_API_KEY>
```

**Configuration:**
Add your keys to `keys.json` in the project root:

```json
[
    "your-secret-key-1",
    "your-secret-key-2"
]
```

To reload keys without restarting, call:
`POST /v1/voices/reload` (requires a valid key if auth was already enabled)

## 🛠️ Development

### Run Without Docker

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Run server
python server.py

# Or run services separately
python gradio_app.py  # Port 3010
python openai_api.py  # Port 3011
```

### Build Docker Image Only

```bash
docker build -t qwen3-tts:latest .
```

### View Logs

```bash
docker-compose logs -f
```

## 📝 Creating Good Voice Samples

For best results:

1. **Recording Quality**: Use a quiet environment, good microphone
2. **Duration**: 5-30 seconds of clear speech
3. **Content**: Natural, conversational speech works best
4. **Transcript**: Must exactly match the spoken words
5. **Format**: WAV format, 16kHz+ sample rate

## 🔧 Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### Out of Memory

- Use a smaller batch size
- Reduce `max_new_tokens` in generation
- Use the 0.6B model instead of 1.7B

### Voice Not Found

- Ensure both `.wav` and `.txt` files exist
- Check file names match exactly (case-sensitive)
- Refresh voices in Gradio UI or call `/v1/voices/reload`

## 📄 License

This project uses the Qwen3-TTS model. Please refer to the [Qwen model license](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) for usage terms.
