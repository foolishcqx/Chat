#!/usr/bin/env python3
"""
可莉语音助手（ASR + LLM + TTS）
- 实时聊天记录
- 终端耗时打印
- 日志写入 chat.log
"""

import os
import json
import time
import datetime
import tempfile
import gradio as gr
import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
from openai import OpenAI
from indextts.infer import IndexTTS
import torchaudio
# -------------------- 全局变量 --------------------
chat_history = []      # 前端聊天记录
log_file   = "chat.log"
SetLogLevel(-1)        # 关闭 Vosk 冗余日志

# 模型初始化
asr_model = Model(r"D:\lasetTTS\code\vosk-model-small-cn-0.22")
tts_model = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")


# -------------------- 工具函数 --------------------
# 写入日志
def write_log(asr_text, reply, t_asr, t_llm, t_tts, audio_duration, duration_ratio):
    total_time = t_asr + t_llm + t_tts  # 计算总耗时
    with open("index_chat.log", "a", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"用户：{asr_text}\n")
        f.write(f"可莉：{reply}\n")
        f.write(f"耗时：ASR={t_asr:.2f}s | LLM={t_llm:.2f}s | TTS={t_tts:.2f}s\n")
        f.write(f"音频时长：{audio_duration:.2f}s\n")
        f.write(f"音频生成时间/音频时长：{duration_ratio:.2f}\n")
        f.write(f"总耗时：{total_time:.2f}s\n")  # 写入总耗时
# -------------------- 业务函数 --------------------
# def audio_recognize(wav_path):
#     t0 = time.time()
#     with wave.open(wav_path, "rb") as wf:
#         if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
#             raise ValueError("音频必须是单声道 16-bit PCM WAV")
#         rec = KaldiRecognizer(asr_model, wf.getframerate())
#         rec.SetWords(True)
#         audio_data = wf.readframes(wf.getnframes())
#         rec.AcceptWaveform(audio_data)
#         result = json.loads(rec.FinalResult())
#     text = result.get("text", "").replace(' ', '')
#     print(f"[ASR] {time.time()-t0:.2f}s → {text}")
#     return text, time.time() - t0

def audio_recognize(audio_data, sample_rate):
    t0 = time.time()
    
    # 检查音频数据是否为单声道 16-bit PCM
    if len(audio_data) % 2 != 0:
        raise ValueError("音频数据必须是 16-bit PCM 格式")
    
    rec = KaldiRecognizer(asr_model, sample_rate)
    rec.SetWords(True)
    
    # 将二进制音频数据传递给 KaldiRecognizer
    rec.AcceptWaveform(audio_data)
    result = json.loads(rec.FinalResult())
    
    text = result.get("text", "").replace(' ', '')
    print(f"[ASR] {time.time() - t0:.2f}s → {text}")
    return text, time.time() - t0

def llm_chat(content):
    t0 = time.time()
    try:
        client = OpenAI(
            api_key="ms-cb688843-be74-4cdf-8e0b-6237eda42d1e",  # ← 替换
            base_url="https://api-inference.modelscope.cn/v1/"
        )
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=[
                {"role": "system",
                 "content": "可莉是西风骑士团的火花骑士，最爱炸鱼和蹦蹦炸弹！说话要元气满满，偶尔提到琴团长会怕怕的。现在可莉要用可爱又爆炸的方式与旅行者交流哦！"},
                {"role": "user", "content": content + "。回复请限制在30个字以内"}
            ],
            stream=False,
            timeout=30
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"网络异常，可莉暂时离线：{e}"
    print(f"[LLM] {time.time()-t0:.2f}s → {reply}")
    return reply, time.time() - t0

def index_tts(text, refer_voice=r"D:\lasetTTS\code\keli\toknow-2.ogg", output_dir=r"D:\lasetTTS\code\temp_audio"):
    t0 = time.time()
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 用时间戳或 UUID 防止重名
    file_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}.wav"
    tmp_wav = os.path.join(output_dir, file_name)
    tts_model.infer(refer_voice, text, output_path=tmp_wav)
    print(f"[TTS] {time.time()-t0:.2f}s → {tmp_wav}")
    return tmp_wav, time.time() - t0

# -------------------- Gradio 流程 --------------------
def pipeline(audio, history):

    asr_text, t_asr = audio_recognize(audio)
    reply, t_llm = llm_chat(asr_text)
    tts_path, t_tts = index_tts(reply)

    # 计算音频时长
    audio_duration = torchaudio.info(tts_path).num_frames / torchaudio.info(tts_path).sample_rate
    duration_ratio = t_tts  / audio_duration if t_tts > 0 else 0

    # 追加到前端聊天记录
    history.append([asr_text, reply])
    write_log(asr_text, reply, t_asr, t_llm, t_tts, audio_duration, duration_ratio)

    return history, tts_path

# -------------------- 界面 --------------------
with gr.Blocks(title="可莉语音助手 🎤💥") as iface:
    gr.Markdown("### 🎤 按住说话 → 实时识别 → 可莉语音回复")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources="microphone", type="filepath", label="录音")
            send_btn    = gr.Button("发送", variant="primary")
        with gr.Column(scale=2):
            chatbot     = gr.Chatbot(label="聊天记录", height=300)
            audio_output = gr.Audio(label="可莉语音", autoplay=True)

    send_btn.click(
        pipeline,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, audio_output]
    )

# # 启动
# iface.launch(server_name="127.0.0.1", server_port=7861, share=True)