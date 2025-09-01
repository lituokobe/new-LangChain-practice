from openai import OpenAI
from new_langchaing_practice.env_util import OPENAI_API_KEY

client = OpenAI(api_key = OPENAI_API_KEY,)
audio_file = open("/private/var/folders/sj/km6fyx_57pn857bnxsqvkj_80000gn/T/gradio/c179043481fe7f8836cadfaf9544ab512fd3f5e0076a125db9daeff88267c873/audio.wav", "rb")

transcription = client.audio.transcriptions.create(
    model="gpt-4o-transcribe",
    file=audio_file,
)
print(transcription)
print(transcription.text)