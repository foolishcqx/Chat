'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import random
import time
import os, re, logging
import sys
import soundfile as sf
from itertools import islice
# now_dir = os.getcwd()
# sys.path.append(now_dir)
# # sys.path.append("%s/GPT_SoVITS" % (now_dir))

# logging.getLogger("markdown_it").setLevel(logging.ERROR)
# logging.getLogger("urllib3").setLevel(logging.ERROR)
# logging.getLogger("httpcore").setLevel(logging.ERROR)
# logging.getLogger("httpx").setLevel(logging.ERROR)
# logging.getLogger("asyncio").setLevel(logging.ERROR)
# logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
# logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch
import json
# try:
#     import gradio.analytics as analytics
#     analytics.version_check = lambda:None
# except:...
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)
version=os.environ.get("version","v2")

import gradio as gr
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method
from tools.i18n.i18n import I18nAuto, scan_language_list

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)


# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"

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

def inference(
    ids: str = "v2",
    text: str = "今天天气如何",
    text_lang: str = "中文",
    ref_audio_path: str = r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\bq_slicer_opt\bq.mp4_0000351680_0000650880.wav",
    aux_ref_audio_paths=None,
    prompt_text: str = "可以叨扰，什么节目让你这么费心，说来听听，节目名字不错，听起来很有野心",
    prompt_lang: str = "中文",
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    text_split_method: str = "凑四句一切",
    batch_size: int = 20,
    speed_factor: float = 1.0,
    ref_text_free: bool = False,
    split_bucket: bool = False,
    fragment_interval: float = 0.3,
    seed: int = -1,
    keep_random: bool = True,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    ):
    seed = -1 if keep_random else seed
    actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
    inputs={
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": [item.name for item in aux_ref_audio_paths] if aux_ref_audio_paths is not None else [],
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":False,
        "fragment_interval":fragment_interval,
        "seed":actual_seed,
    }
    results = []
    for item in tts_pipeline.run(ids, inputs):
        return item
    #     # return item, actual_seed
    # data = tts_pipeline.run(inputs)
    # return data

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


if __name__ == "__main__":
    with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
        model_paths = json.load(f)
    capacity = 3
    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = device
    tts_config.is_half = is_half
    tts_config.version = version
    t2s_subset = dict(islice(model_paths["GPT"].items(), capacity))
    vits_subset = dict(islice(model_paths["SoVITS"].items(), capacity))
    # 构造字典：{模型id: 对应权重路径}
    tts_config.t2s_weights_path  = {k: v for k, v in t2s_subset.items()}
    tts_config.vits_weights_path = {k: v for k, v in vits_subset.items()}
    if cnhubert_base_path is not None:
        tts_config.cnhuhbert_base_path = cnhubert_base_path
    if bert_path is not None:
        tts_config.bert_base_path = bert_path
    tts_pipeline = TTS(tts_config, capacity=5)
    t0 = time.time()
    text = ["本平台涵盖语料库管理、特征集构建、复杂度分析、可视化展示四大功能。语料库管理支持您建立自己的语料库，特征集构建支持您自由选择不同的特征",
            "语料库管理支持您建立自己的语料库，特征集构建支持您自由选择不同的特征",
            "复杂度分析帮助您进行文本特征的自动计算，可视化展示可以让您清晰地看到不同特征的变化轨迹。",
            "不同模块间的结合可以使您的文本分析高度定制化。"
            ]
    # text = "语料库管理支持您建立自己的语料库，特征集构建支持您自由选择不同的特征"
    id = "lx"
    text_lang = "中文"
    ref_audio_path = r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\bq_slicer_opt\bq.mp4_0000351680_0000650880.wav"
    aux_ref_audio_paths = None
    prompt_text = "可以叨扰，什么节目让你这么费心，说来听听，节目名字不错，听起来很有野心"
    prompt_lang = "中文"
    top_k = 5
    top_p = 1
    temperature = 1
    text_split_method = "凑四句一切"
    batch_size = 20
    speed_factor = 1
    ref_text_free = False
    split_bucket = False
    fragment_interval = 0.3
    seed = -1
    keep_random = True
    parallel_infer = True
    repetition_penalty = 1.35
    sample_steps = 32
    super_sampling = False
    sr, audio = inference(id,
    text,
    text_lang,
    ref_audio_path,
    aux_ref_audio_paths,
    prompt_text,
    prompt_lang,
    top_k,
    top_p,
    temperature,
    text_split_method,
    batch_size,
    speed_factor,
    ref_text_free,
    split_bucket,
    fragment_interval,
    seed,
    keep_random,
    parallel_infer,
    repetition_penalty)
    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    print("waste time:", time.time() - t0)
    # 遍历每段音频
    try:
        for idx, audio_fragment in enumerate(audio):
            filename = os.path.join(output_dir, f"audio_{idx+1}.wav")
            sf.write(filename, audio_fragment, sr)
            print(f"Saved: {filename}")
    except:
        filename = os.path.join(output_dir, f"audio_{idx+1}.wav")
        sf.write(filename, audio, sr)
        print(f"Saved: {filename}")