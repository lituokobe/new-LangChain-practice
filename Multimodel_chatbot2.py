import base64
import uuid
import io
import gradio as gr
from PIL import Image
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from new_langchaing_practice.models import multimodal_llm

# # test of the multimodal model
# resp = multimodal_llm.invoke([HumanMessage(content = "Do you know what is soccer?")])
# print(resp.content)

prompt = ChatPromptTemplate([
    ("system", "You are a multimodal AI assistant. You can take test, voice and image input."),
    MessagesPlaceholder(variable_name = "messages"),

])

chain = prompt | multimodal_llm

def get_session_history(session_id : str):
    return SQLChatMessageHistory(
        session_id = session_id,
        connection = "sqlite:///chat_history.db"
    )

chain_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
)

config = {"configurable" : {"session_id" : str(uuid.uuid4())}}

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


def transcribe_image(image_path):
    """
    将任意格式的图片转换为base64编码的data URL
    :param image_path: 图片路径
    :return: 包含base64编码的字典
    """
    with Image.open(image_path) as img:
        # 获取原始图片格式（如JPEG/PNG）
        img_format = img.format if img.format else 'JPEG'

        buffered = io.BytesIO()
        # 保留原始格式（避免JPEG强制转换导致透明通道丢失）
        img.save(buffered, format=img_format)

        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{img_format.lower()};base64,{image_data}",
                "detail": 'low'
            }
        }

def add_message(history, messages):
    """
    Add user's message to chat history
    """
    for m in messages["files"]:
        # print(m)
        history.append({"role": "user", "content": {"path": m}})
    if messages["text"] is not None:
        history.append({"role": "user", "content": messages["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def get_last_user_after_assistant(history):
    """Find last assistant's location and return all user messages after"""
    if not history:
        return None
    if history[-1]["role"] == "assistant":
        return None

    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    # if assistant is not found
    if last_assistant_idx == -1:
        return history
    else:
        # first user after assistant
        return history[last_assistant_idx + 1:]

def submit_messages(history):
    """
    Submit user input, generate chatbot reply
    """
    user_messages = get_last_user_after_assistant(history)
    # print(user_messages)
    content = []
    if user_messages:
        for x in user_messages:
            print(x)
            if isinstance(x["content"], str): # this means the content is text
                content.append({"type": "text", "text": x["content"]})
            elif isinstance(x["content"], tuple): #multimeida content
                file_path = x ["content"][0] # get the multimedia file's path
                if file_path.endswith(".wav") or file_path.endswith(".mp3"): # it is a audio file
                    file_message = transcribe_audio(file_path)
                elif file_path.endswith(".jpg") or file_path.endswith(".png") or file_path.endswith(".jpeg"):
                    file_message = transcribe_image(file_path)
                content.append(file_message)
            else:
                pass


    input_message = HumanMessage(content = content)
    resp = chain_history.invoke({"messages" : input_message}, config)

    history.append({"role" : "assistant", "content" : resp.content})

    return history

#TODO: use Gradio to develop a GUI for the chatbot
with gr.Blocks(title = "Multimodal Chatbot", theme = gr.themes.Soft()) as block:
    # chat history
    chatbot = gr.Chatbot(type = "messages", height = 500, label = "Chatbot")

    # multimodal input
    chat_input = gr.MultimodalTextbox(
        interactive = True,
        file_types=['image', '.wav', '.mp4'],
        file_count = "multiple",
        placeholder = "Please input information or upload files...",
        show_label = False,
        sources=["microphone", "upload"],
    )

    chat_input.submit(
        add_message,
        [chatbot, chat_input],
        [chatbot, chat_input]
    ).then(
        submit_messages,
        [chatbot],
        [chatbot]
    ).then(
        lambda: gr.MultimodalTextbox(interactive = True), # reset the chat window
        None,
        [chat_input]
    )

if __name__ == "__main__":
    block.launch()