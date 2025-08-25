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
import soundfile as sf
import re
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

id_to_prompt = {
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
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)
version = os.environ.get("version", "v2")
dict_language_v1 = {
    "中文": "all_zh",       # 全部按中文识别
    "英文": "en",           # 全部按英文识别
    "日文": "all_ja",       # 全部按日文识别
    "中英混合": "zh",        # 按中英混合识别
    "日英混合": "ja",        # 按日英混合识别
    "多语种混合": "auto",    # 多语种启动切分识别语种
}

dict_language_v2 = {
    "中文": "all_zh",              # 全部按中文识别
    "英文": "en",                  # 全部按英文识别
    "日文": "all_ja",              # 全部按日文识别
    "粤语": "all_yue",             # 全部按粤语识别
    "韩文": "all_ko",              # 全部按韩文识别
    "中英混合": "zh",               # 按中英混合识别
    "日英混合": "ja",               # 按日英混合识别
    "粤英混合": "yue",              # 按粤英混合识别
    "韩英混合": "ko",              # 按韩英混合识别
    "多语种混合": "auto",           # 多语种启动切分识别语种
    "多语种混合(粤语)": "auto_yue",  # 多语种启动切分识别语种
}


dict_language = dict_language_v1 if version =='v1' else dict_language_v2

cut_method = {
    "不切": "cut0",
    "凑四句一切": "cut1",
    "凑50字一切": "cut2",
    "按中文句号。切": "cut3",
    "按英文句号.切": "cut4",
    "按标点符号切": "cut5",
}
# ========= 初始化 =========
def init_ASR(model_path='./SenseVoiceSmall', device="cuda:0"):
    return pipeline(
        task=Tasks.auto_speech_recognition,
        model=model_path,
        device=device,
    )

def llm_chat(content):
    t0 = time.time()
    try:
        client = OpenAI(
            api_key="ms-cb688843-be74-4cdf-8e0b-6237eda42d1e",  # ← 替换
            base_url="https://api-inference.modelscope.cn/v1/"
        )
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages = [
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
            {
                "role": "user",
                "content": content
            }],
            stream=False,
            timeout=30
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"网络异常，可莉暂时离线~"
    print(f"[LLM] {time.time()-t0:.2f}s → {reply}")
    return reply

def init_TTS(capacity=3):
    with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
        model_paths = json.load(f)
    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = device
    tts_config.is_half = is_half
    t2s_subset = dict(islice(model_paths["GPT"].items(), capacity))
    vits_subset = dict(islice(model_paths["SoVITS"].items(), capacity))
    tts_config.t2s_weights_path = t2s_subset
    tts_config.vits_weights_path = vits_subset
    if cnhubert_base_path is not None:
        tts_config.cnhuhbert_base_path = cnhubert_base_path
    if bert_path is not None:
        tts_config.bert_base_path = bert_path
    return TTS(tts_config, capacity)

def tts_inference(tts_pipeline, ids, text, ref_audio_path, prompt_text):
    inputs = {
        "text": text,
        "text_lang":dict_language["中文"],
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": [],
        "prompt_text": prompt_text,
        "prompt_lang": dict_language["中文"],
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": cut_method["凑四句一切"],
        "batch_size": 20,
        "speed_factor": 1.0,
        "split_bucket": False,
        "return_fragment": False,
        "fragment_interval": 0.3,
        "seed": random.randrange(1 << 32),
    }
    for sr, wav in tts_pipeline.run(ids, inputs):
        return sr, wav
    
# 3. LLM 多进程处理
def llm_worker(text, return_dict, idx):
    return_dict[idx] = llm_chat(text)
# ========= 主流程 =========

if __name__ == "__main__":
    # 必须放在 __main__ 内，Windows 才能 spawn 成功
    mp.set_start_method('spawn', force=True)
    total_start = time.time()

    # 1. 初始化 ASR & TTS
    asr_pipeline = init_ASR()
    tts_pipeline = init_TTS()

    # 2. 四条音频 ASR（ModelScope 支持批量）
    input_audio = [
        r"D:\lasetTTS\code\demo\bq\audio1.wav",
        r"D:\lasetTTS\code\demo\bq\audio2.wav",
        r"D:\lasetTTS\code\demo\bq\audio3.wav",
        r"D:\lasetTTS\code\demo\bq\audio4.wav"
    ]
    res = asr_pipeline(input_audio)
    pattern = r"<\|(.+?)\|><\|(.+?)\|><\|(.+?)\|><\|(.+?)\|>(.+)"
    asr_results = [
        re.match(pattern, r['text']).group(5)
        for r in res
    ]
    print("ASR 结果：", asr_results)

    # 3. LLM 多进程并发
    manager = mp.Manager()
    return_dict = manager.dict()
    procs = [
        mp.Process(target=llm_worker, args=(txt, return_dict, i))
        for i, txt in enumerate(asr_results)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    llm_results = [return_dict[i] for i in range(len(asr_results))]
    print("LLM 回复：", llm_results)

    # 4. TTS 逐条合成（如需真正批量可自行修改）
    vid = "bq"
    wav_list = []
    sr, wav = tts_inference(tts_pipeline, vid, llm_results, id_to_refer[vid], id_to_prompt[vid])
    wav_list.append(wav)
    total_end = time.time()
    print(f"整体耗时: {total_end - total_start:.2f}秒")
    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    # 遍历每段音频
    try:
        for idx, audio_fragment in enumerate(wav):
            filename = os.path.join(output_dir, f"audio_{idx+1}.wav")
            sf.write(filename, audio_fragment, sr)
            print(f"Saved: {filename}")
    except:
        filename = os.path.join(output_dir, f"audio_{idx+1}.wav")
        sf.write(filename, wav, sr)
        print(f"Saved: {filename}")