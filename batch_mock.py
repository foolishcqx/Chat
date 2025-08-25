import multiprocessing as mp
import time
import random
import os
import json
import torch
from itertools import islice
from openai import OpenAI
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method

# ========= 参考音频和提示 =========
id_to_refer = {
    "复杂zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\复杂zql_slicer_opt\复杂zql.mp3_0022588800_0022883200.wav",
    "霸总zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\霸总zql_slicer_opt\霸总zql.mp3_0007565120_0007871680.wav",
    "阳光zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\阳光zql_slicer_opt\阳光zql.mp3_0005888640_0006185920.wav",
    "bq": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\bq_slicer_opt\bq.mp4_0000351680_0000650880.wav",
    "lzy": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lzy_slicer_opt\lzy.mp4_0015106240_0015348160.wav",
    "xm": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\xm_slicer_opt\xm.mp4_0003723200_0003876480.wav",
    "lx": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lx_slicer_opt\bilibili_BV1Tc411b7BG_2_MP3.mp4_0000306880_0000608000.wav"
}

id_to_promt = {
    "复杂zql": "我相信你会做到的,我答应你,等着我,我会找到你的,既然你被剥夺了一物",
    "霸总zql": "不过后来因为一些事情离开了,他很擅长观察和模仿,如果不是十分熟悉的人,很容易被他骗过去,没什么",
    "阳光zql": "给你治愈的魔法道具啊,我该走了,好好睡一觉",
    "bq": "可以投扰,什么节目让你这么费心,说来听听,节目名字不错,听起来很有野心",
    "lzy": "不是不可以,不可以苏文聂的菜单,今天只看店长的心情,你说什么",
    "xm": "你联想到什么了,我做的可都是合法的研究",
    "lx": "再说我不是给你发过消息了吗,家里停电了没空调"
}

# ========= 环境配置 =========
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)
version = os.environ.get("version", "v2")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 初始化 =========
def init_ASR(model_path='./SenseVoiceSmall', device="cuda:0"):
    return pipeline(
        task=Tasks.auto_speech_recognition,
        model=model_path,
        device=device,
    )

def llm_chat(content):
    try:
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='你的ModelScope API Key',
        )
        extra_body = {"enable_thinking": False}
        response = client.chat.completions.create(
            model='Qwen/Qwen3-32B',
            messages=[
                {"role": "system", "content": "你是白起，短句深情，不超过30字。"},
                {"role": "user", "content": content}
            ],
            stream=False,
            extra_body=extra_body
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"网络异常: {e}"

def init_TTS(capacity=3):
    with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
        model_paths = json.load(f)
    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = device
    tts_config.is_half = is_half
    tts_config.version = version
    t2s_subset = dict(islice(model_paths["GPT"].items(), capacity))
    vits_subset = dict(islice(model_paths["SoVITS"].items(), capacity))
    tts_config.t2s_weights_path = t2s_subset
    tts_config.vits_weights_path = vits_subset
    if cnhubert_base_path:
        tts_config.cnhuhbert_base_path = cnhubert_base_path
    if bert_path:
        tts_config.bert_base_path = bert_path
    return TTS(tts_config, capacity)

def tts_inference(tts_pipeline, ids, text, ref_audio_path, prompt_text):
    inputs = {
        "text": text,
        "text_lang": "中文",
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": None,
        "prompt_text": prompt_text,
        "prompt_lang": "中文",
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": get_method("凑四句一切"),
        "batch_size": 20,
        "speed_factor": 1.0,
        "split_bucket": False,
        "return_fragment": False,
        "fragment_interval": 0.3,
        "seed": random.randrange(1 << 32),
    }
    for audio_bytes in tts_pipeline.run(ids, inputs):
        return audio_bytes

# ========= 多进程任务 =========
def asr_worker(asr_pipeline, q_audio, q_asr_llm, batch_size=4):
    buffer = []
    while True:
        audio_path = q_audio.get()
        if audio_path is None:
            if buffer:
                q_asr_llm.put([asr_pipeline(path)["text"] for path in buffer])
            break
        buffer.append(audio_path)
        if len(buffer) >= batch_size:
            q_asr_llm.put([asr_pipeline(path)["text"] for path in buffer])
            buffer.clear()

def llm_worker(q_asr_llm, q_llm_tts):
    while True:
        texts = q_asr_llm.get()
        if texts is None:
            break
        q_llm_tts.put([llm_chat(t) for t in texts])

def tts_worker(tts_pipeline, q_llm_tts, batch_size=4):
    buffer = []
    while True:
        texts = q_llm_tts.get()
        if texts is None:
            if buffer:
                _tts_batch(tts_pipeline, buffer)
            break
        buffer.extend(texts)
        while len(buffer) >= batch_size:
            _tts_batch(tts_pipeline, buffer[:batch_size])
            buffer = buffer[batch_size:]

def _tts_batch(tts_pipeline, texts):
    vid = random.choice(list(id_to_refer.keys()))
    audio = tts_inference(tts_pipeline, vid, texts, id_to_refer[vid], id_to_promt[vid])
        # fname = f"tts_{vid}_{time.time()}.wav"
        # with open(fname, "wb") as f:
        #     f.write(audio)
        # print(f"[TTS] 输出语音: {fname}")

# ========= 主程序入口 =========
if __name__ == "__main__":
    queue_audio = mp.Queue()
    queue_asr_llm = mp.Queue()
    queue_llm_tts = mp.Queue()

    asr_pipeline = init_ASR()
    tts_pipeline = init_TTS()
    total_start = time.time()
    p_asr = mp.Process(target=asr_worker, args=(asr_pipeline, queue_audio, queue_asr_llm))
    p_asr.start()

    p_llm_list = []
    for _ in range(4):
        p = mp.Process(target=llm_worker, args=(queue_asr_llm, queue_llm_tts))
        p.start()
        p_llm_list.append(p)

    p_tts = mp.Process(target=tts_worker, args=(tts_pipeline, queue_llm_tts))
    p_tts.start()

    # 随机输入音频
    input_audio = [
        r"D:\lasetTTS\code\demo\bq\audio1.wav",
        r"D:\lasetTTS\code\demo\bq\audio2.wav",
        r"D:\lasetTTS\code\demo\bq\audio3.wav",
        r"D:\lasetTTS\code\demo\bq\audio4.wav"
    ]
    for _ in range(12):
        queue_audio.put(random.choice(input_audio))

    # 结束信号
    queue_audio.put(None)
    queue_asr_llm.put(None)
    queue_llm_tts.put(None)

    p_asr.join()
    for p in p_llm_list:
        p.join()
    p_tts.join()

    total_end = time.time()
    print(f"整体耗时: {total_end - total_start:.2f} 秒")
