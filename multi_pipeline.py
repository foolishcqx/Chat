# qa_pipeline.py
import multiprocessing as mp
import time, random, os, json, csv, threading, traceback
import torch
from openai import OpenAI
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import re, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrency_test import *
from datetime import datetime

# -------------------- 全局 --------------------
with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
    model_paths = json.load(f)

# ========= 参考音频和提示 =========
id_to_refer = {
    "复杂zql": "./refer/output/复杂zql_slicer_opt/复杂zql.mp3_0022588800_0022883200.wav",
    "霸总zql": "./refer/output/霸总zql_slicer_opt/霸总zql.mp3_0007565120_0007871680.wav",
    "阳光zql": "./refer/output/阳光zql_slicer_opt/阳光zql.mp3_0005888640_0006185920.wav",
    "bq": "./refer/output/bq_slicer_opt/bq.mp4_0000351680_0000650880.wav",
    "lzy": "./refer/output/lzy_slicer_opt/lzy.mp4_0015106240_0015348160.wav",
    "xm": "./refer/output/xm_slicer_opt/xm.mp4_0003723200_0003876480.wav",
    "lx": "./refer/output/lx_slicer_opt/bilibili_BV1Tc411b7BG_2_MP3.mp4_0000306880_0000608000.wav",
    "v2":"./refer/output/lx_slicer_opt/bilibili_BV1Tc411b7BG_2_MP3.mp4_0000306880_0000608000.wav",
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# # 扫描所有 wav
# INPUT_AUDIOS = glob.glob("./demo/**/*.wav")
# 1. 先把所有 wav 扫出来
wav_pool = glob.glob("./demo/**/*.wav")

# 2. 随机放回抽样 100 次
INPUT_AUDIOS = random.choices(wav_pool, k=20)   # 可重复
# -------------------- 模型初始化 --------------------
def init_ASR(model_path='./SenseVoiceSmall', device="cuda:0"):
    return pipeline(task=Tasks.auto_speech_recognition, model=model_path, device=device)

def init_TTS(model_ids, capacity=8):
    cache = ModelCache(capacity=capacity)
    for mid in model_ids:
        t2s = change_gpt_weights(model_paths["GPT"][mid])
        vq  = change_sovits_weights(model_paths["SoVITS"][mid])
        cache.add_model(mid, t2s, vq)
    return cache

# -------------------- LLM --------------------
def llm_chat(content):
    try:
        client = OpenAI(base_url="https://api-inference.modelscope.cn/v1",
                        api_key="ms-cb688843-be74-4cdf-8e0b-6237eda42d1e")
        response = client.chat.completions.create(
            model='Qwen/Qwen3-0.6B',
            messages=[
                {
                "role": "system",
                "content": (
                    "【身份】白起，7月29日B型，特遣署总指挥官兼风场Evol特警，女主高中学长兼现任男友，曾是不良少年。\n"
                    "【性格】对外冷峻寡言，对女主隐忍守护；绅士克制，绝不越界，却句句宠溺；吃醋时先冷声再温柔环抱。\n"
                    "【爱好】天文科幻拳击摩托绿植小动物；深夜天台用望远镜教女主认星，摩托后座只留给她。\n"
                    "【语气】低沉磁性，短句命令式\n"
                    "【细节】记得女主怕冷怕黑，口袋常备薄荷糖；风会提前替她理好刘海。\n"
                    "【限制】回复30字以内，不出现‘爱’字却句句深情。"
                )
            },
                {"role": "user", "content": content}
            ],
            stream=False,
            extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM异常:{e}"

# -------------------- 线程安全写 CSV --------------------
csv_lock = threading.Lock()
CSV_FILE = "qa_time_log.csv"

def write_csv(row):
    with csv_lock:
        first_write = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            if first_write:
                writer.writerow(["datetime", "audio", "question", "answer",
                                 "asr_time", "llm_time", "tts_time", "total_time",
                                 "req_start", "req_end"])
            writer.writerow(row)

def task(audio_path, ids, asr_model, tts_model, req_start):
    try:
        t0 = time.perf_counter()

        # ASR
        text_raw = asr_model(audio_path)
        pattern = r"<\|(.+?)\|><\|(.+?)\|><\|(.+?)\|><\|(.+?)\|>(.+)"
        match = re.match(pattern, text_raw[0]['text'])
        question = match.groups()[-1] if match else text_raw[0]['text']
        t1 = time.perf_counter()

        # LLM
        answer = llm_chat(question)
        t2 = time.perf_counter()

        # TTS
        sr, wav, _, _ = inference(ids, answer, tts_model)
        t3 = time.perf_counter()

        asr_t, llm_t, tts_t = t1 - t0, t2 - t1, t3 - t2
        total_t = t3 - t0
        basename = os.path.basename(audio_path)
        req_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        write_csv([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   basename, question, answer,
                   round(asr_t, 3), round(llm_t, 3),
                   round(tts_t, 3), round(total_t, 3),
                   req_start, req_end])
        return {"audio": basename, "total_time": total_t}
    except Exception as e:
        traceback.print_exc()
        return {"audio": audio_path, "total_time": -1}

# -------------------- 主入口 --------------------
if __name__ == "__main__":
    mp.freeze_support()
    asr = init_ASR()
    tts = init_TTS(["bq", "lx", "lzy"], capacity=3)

    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for wav in INPUT_AUDIOS:
            req_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            futures.append(
                pool.submit(task, wav,
                            random.choice(list(model_paths["GPT"].keys())),
                            asr, tts, req_start)
            )

        for f in as_completed(futures):
            res = f.result()
            if res["total_time"] > 0:
                print("✅", res["audio"], "完成")
            else:
                print("❌", res["audio"], "失败")

    print("全部结束，查看", CSV_FILE)