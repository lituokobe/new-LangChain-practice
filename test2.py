import base64
from langchain_core.messages import HumanMessage
from new_langchaing_practice.models import multimodal_llm

def transcribe_audio(audio_path):
    """使用Base64处理语音转为"""
    # 目前多模态大模型： 支持两个传参方式，1、base64（字符串）（本地）。2、网络访问的url地址（外网的服务器上） http://sxxxx.com/11.mp3
    try:
        with open(audio_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        # Qwen model format:
        audio_message = {  # 把音频文件，封装成一条消息
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/wav;base64,{audio_data}",
                # "duration": 30  # 单位：秒（帮助模型优化处理）
            }
        }
        return audio_message
    except Exception as e:
        print(e)
        return {}


voice_url = '/private/var/folders/sj/km6fyx_57pn857bnxsqvkj_80000gn/T/gradio/0f2572b97c210d7c16703725d3bf7a991b6b6eff6aa2ea6fe62c88556fe754c8/audio.wav'
msg = HumanMessage(content=[
    {"type": "text", "text": "transcribe this audio"},
    transcribe_audio(voice_url)
])
resp = multimodal_llm.invoke([msg])
print(resp.content)
