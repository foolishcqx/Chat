# qa_dual_proc_final.py
import multiprocessing as mp
mp.set_start_method('spawn', force=True)   # 关键

import os, json, random, time, glob, csv, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import re
from openai import OpenAI
from concurrency_test import *
from datetime import datetime
def init_TTS(model_ids, capacity=8):
    cache = ModelCache(capacity=capacity)
    for mid in model_ids:
        t2s = change_gpt_weights(model_paths["GPT"][mid])
        vq  = change_sovits_weights(model_paths["SoVITS"][mid])
        cache.add_model(mid, t2s, vq)
    return cache

# ---------- 子进程 ----------
def _worker(gpu_id, wav_list, proc_id):
    """子进程：独占 GPU、加载模型、4 线程处理 wav_list"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)



    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = pipeline(task=Tasks.auto_speech_recognition, model='./SenseVoiceSmall', device=device)

    with open("./GPT_SoVITS/weight.json", encoding="utf-8") as f:
        model_paths = json.load(f)
    valid = [k for k, v in model_paths["GPT"].items() if os.path.isfile(v)]
    if not valid:
        return []

    tts = init_TTS(valid[:4], capacity=4)

    # 提示与参考音频（按需删减）
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

    csv_file = f"qa_time_log_proc{proc_id}.csv"
    lock = threading.Lock()

    def write_row(row):
        with lock:
            first = not os.path.exists(csv_file)
            with open(csv_file, "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                if first:
                    writer.writerow(["datetime", "audio", "question", "answer",
                                     "asr_time", "llm_time", "tts_time", "total_time",
                                     "req_start", "req_end"])
                writer.writerow(row)

    def llm_chat_local(text):
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
                    {"role": "user", "content": text}
                ],
                stream=False,
                extra_body={"enable_thinking": False}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM异常:{e}"

    def single_task(audio_path, ids):
        try:
            t0 = time.perf_counter()

            # ASR
            text_raw = asr(audio_path)
            pattern = r"<\|(.+?)\|><\|(.+?)\|><\|(.+?)\|><\|(.+?)\|>(.+)"
            match = re.match(pattern, text_raw[0]['text'])
            question = match.groups()[-1] if match else text_raw[0]['text']
            t1 = time.perf_counter()

            # LLM
            answer = llm_chat_local(question)
            t2 = time.perf_counter()

            # TTS
            sr, wav, _, _ = inference(ids, answer, tts)
            t3 = time.perf_counter()

            asr_t, llm_t, tts_t = t1 - t0, t2 - t1, t3 - t0
            total_t = t3 - t0
            basename = os.path.basename(audio_path)
            req_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            write_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       basename, question, answer,
                       round(asr_t, 3), round(llm_t, 3),
                       round(tts_t, 3), round(total_t, 3),
                       datetime.now().strftime("%Y-%m-%d %H:%M:%S"), req_end])
            return {"audio": basename, "total_time": total_t}
        except Exception as e:
            traceback.print_exc()
            return {"audio": audio_path, "total_time": -1}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(single_task, wav, random.choice(valid)) for wav in wav_list]
        return [f.result() for f in as_completed(futures)]

# ---------- 主入口 ----------
if __name__ == "__main__":
    wav_pool = glob.glob("./demo/**/*.wav")
    wavs = random.choices(wav_pool, k=60)
    chunk_size = 30
    chunks = [wavs[i:i + chunk_size] for i in range(0, len(wavs), chunk_size)]

    gpu_ids = [0, 1] if torch.cuda.device_count() > 1 else [0, 0]

    mp.freeze_support()
    with mp.Pool(processes=2) as pool:
        results = pool.starmap(_worker, [(gid, chk, pid) for pid, (gid, chk) in enumerate(zip(gpu_ids, chunks))])

    print("全部完成！结果：", sum(results, []))