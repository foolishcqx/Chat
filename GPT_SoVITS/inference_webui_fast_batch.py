'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import random
import os, re, logging
import sys
import soundfile as sf
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch

try:
    import gradio.analytics as analytics
    analytics.version_check = lambda:None
except:...


infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
gpt_path = os.environ.get("gpt_path", None)
sovits_path = os.environ.get("sovits_path", None)
cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
bert_path = os.environ.get("bert_path", None)
version=os.environ.get("version","v2")

import gradio as gr
from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import get_method
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


tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
tts_config.version = version
if gpt_path is not None:
    tts_config.t2s_weights_path = gpt_path
if sovits_path is not None:
    tts_config.vits_weights_path = sovits_path
if cnhubert_base_path is not None:
    tts_config.cnhuhbert_base_path = cnhubert_base_path
if bert_path is not None:
    tts_config.bert_base_path = bert_path

print(tts_config)
tts_pipeline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path
version = tts_config.version

def inference(text, text_lang,
              ref_audio_path,
              aux_ref_audio_paths,
              prompt_text,
              prompt_lang, top_k,
              top_p, temperature,
              text_split_method, batch_size,
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,
              seed, keep_random, parallel_infer,
              repetition_penalty
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
    for item in tts_pipeline.run(inputs):
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


def change_choices():
    SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


pretrained_sovits_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "GPT_SoVITS/pretrained_models/s2G488k.pth"]
pretrained_gpt_name=["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]
_ =[[],[]]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
pretrained_gpt_name,pretrained_sovits_name = _

SoVITS_weight_root=["SoVITS_weights_v2","SoVITS_weights"]
GPT_weight_root=["GPT_weights_v2","GPT_weights"]
for path in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(path,exist_ok=True)

def get_weights_names(GPT_weight_root, SoVITS_weight_root):
    SoVITS_names = [i for i in pretrained_sovits_name]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [i for i in pretrained_gpt_name]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)



def change_sovits_weights(sovits_path,prompt_language=None,text_language=None):
    tts_pipeline.init_vits_weights(sovits_path)
    global version, dict_language
    dict_language = dict_language_v1 if tts_pipeline.configs.version =='v1' else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = {'__type__':'update'}, {'__type__':'update', 'value':prompt_language}
        else:
            prompt_text_update = {'__type__':'update', 'value':''}
            prompt_language_update = {'__type__':'update', 'value':i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
        else:
            text_update = {'__type__':'update', 'value':''}
            text_language_update = {'__type__':'update', 'value':i18n("中文")}
        return  {'__type__':'update', 'choices':list(dict_language.keys())}, {'__type__':'update', 'choices':list(dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update



if __name__ == "__main__":
    text = ["本平台涵盖语料库管理、特征集构建、复杂度分析、可视化展示四大功能。语料库管理支持您建立自己的语料库，特征集构建支持您自由选择不同的特征",
            "语料库管理支持您建立自己的语料库，特征集构建支持您自由选择不同的特征",
            "复杂度分析帮助您进行文本特征的自动计算，可视化展示可以让您清晰地看到不同特征的变化轨迹。",
            "不同模块间的结合可以使您的文本分析高度定制化。"
            ]
    # text = "语料库管理支持您建立自己的语料库，特征集构建支持您自由选择不同的特征"
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
    sr, audio = inference(
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