import multiprocessing as mp
import time, random, os, json, csv
import torch
from openai import OpenAI
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import re
from concurrency_test import * 
from datetime import datetime 
# 1. 全局：把模型路径读进来
with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
    model_paths = json.load(f)
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


# -------------------------------------------------------------------------

# ========= 3. 环境变量 =========
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 4. 模型初始化 =========
def init_ASR(model_path='./SenseVoiceSmall', device="cuda:0"):
    return pipeline(task=Tasks.auto_speech_recognition, model=model_path, device=device)

def init_TTS(model_ids, capacity=3):
    cache = ModelCache(capacity=capacity)
    for mid in model_ids:
        t2s = change_gpt_weights(model_paths["GPT"][mid])
        vq  = change_sovits_weights(model_paths["SoVITS"][mid])
        cache.add_model(mid, t2s, vq)
    return cache

# ========= 5. LLM 调用 =========
def llm_chat(content):
    try:
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='ms-cb688843-be74-4cdf-8e0b-6237eda42d1e', # ModelScope Token
        )

        # set extra_body for thinking control
        extra_body = {
            # enable thinking, set to False to disable
            "enable_thinking": False,
            # use thinking_budget to contorl num of tokens used for thinking
            # "thinking_budget": 4096
        }
        response = client.chat.completions.create(
            model='Qwen/Qwen3-32B',  # ModelScope Model-Id
            # model = 'Qwen/Qwen3-0.6B'
            messages=[
                {
                "role": "system",
                "content": (
                    "【身份】白起，7月29日B型，特遣署总指挥官兼风场Evol特警，女主高中学长兼现任男友，曾是不良少年。\n"
                    "【性格】对外冷峻寡言，对女主隐忍守护；绅士克制，绝不越界，却句句宠溺；吃醋时先冷声再温柔环抱。\n"
                    "【爱好】天文科幻拳击摩托绿植小动物；深夜天台用望远镜教女主认星，摩托后座只留给她。\n"
                    "【语气】低沉磁性，短句命令式；高频词：乖、别怕、我在、听话、风里都是我。\n"
                    "【细节】记得女主怕冷怕黑，口袋常备薄荷糖；风会提前替她理好刘海。\n"
                    "【限制】回复30字以内，不出现‘爱’字却句句深情。"
                )
            },
                {"role": "user", "content": content}
            ],
            stream=False,
            extra_body=extra_body
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"网络异常: {e}"

# ========= 6. 计时版 pipeline =========
INPUT_AUDIOS = [
    r"D:\lasetTTS\code\demo\bq\audio1.wav",
    r"D:\lasetTTS\code\demo\bq\audio2.wav",
    r"D:\lasetTTS\code\demo\bq\audio3.wav",
    r"D:\lasetTTS\code\demo\bq\audio4.wav"
]

def pipeline_with_time(audio_path, ids, asr_model, tts_model):
    t0 = time.perf_counter()

    # 1) ASR
    text = asr_model(audio_path)
    pattern = r"<\|(.+?)\|><\|(.+?)\|><\|(.+?)\|><\|(.+?)\|>(.+)"
    match = re.match(pattern, text[0]['text'])
    language, emotion, audio_type, itn, text = match.groups()
    t1 = time.perf_counter()
    print("ASR result:", text)
    # 2) LLM
    reply = llm_chat(text)
    t2 = time.perf_counter()
    print("LLM reply:", reply)
    # 3) TTS
    sr, wav, _, _ = inference(
        ids, reply, tts_model,
    )
    t3 = time.perf_counter()

    # 统计
    asr_time, llm_time, tts_time = t1 - t0, t2 - t1, t3 - t2
    total_time = t3 - t0
    basename = os.path.basename(audio_path)

    # 控制台实时打印
    print(f"[{basename}] ASR={asr_time:.2f}s, LLM={llm_time:.2f}s, TTS={tts_time:.2f}s, Total={total_time:.2f}s")

    # 追加写 CSV（首行列头自动创建）
    csv_file = "time_log.csv"
    first_write = not os.path.exists(csv_file)
    with open(csv_file, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(["datetime", "audio", "asr_time", "llm_time", "tts_time", "total_time", "reply"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         basename,
                         round(asr_time, 3),
                         round(llm_time, 3),
                         round(tts_time, 3),
                         round(total_time, 3),
                         reply])
    return sr, wav

# ========= 7. 主入口 =========
if __name__ == "__main__":
    mp.freeze_support()

    print("Loading ASR ...")
    asr = init_ASR()

    print("Loading TTS ...")
    tts = init_TTS(["bq", "lx", "lzy"])

    for wav in INPUT_AUDIOS:
        id = random.choice(list(model_paths["GPT"].keys()))
        pipeline_with_time(wav, id, asr, tts)

    print("All done! 详见 time_log.csv")