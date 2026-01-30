Dockerise the following tts model as a server: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice>

Use the Qwen3-TTS-12Hz-0.6B-Base in a gradio ui and api.

All ui and api generation should accept a voice name e.g "lojik",  then voice should be loaded from the voices folder e.g "voices/lojik.wav" as reference voice and "voices/lojik.txt" as the reference Text.

The API must be openAI compatible API for voice generation.

The gradio ui should should read all the wav files in the voices folder and present them(without the .wav extension) in the ui for voice selection.

Do not copy the voices folder in the docker files but map the folder so that it is accessed on the host.

The ui and API should be served on port 3010,3011, 3012 etc as needed.

Aside from accepting voice parameter on api, it should also accept response_format: mp3 | wav | opus | aac | flac | pcm

It should support multiple requests at the same time by processing them in a queue.

It should support apiKey authentication in the header (openAI compatible). Api keys should be checked against the keys.json file. Do not copy the keys.json file in the docker files but map it so that it is accessed on the host.

Build the docker image and run it.
