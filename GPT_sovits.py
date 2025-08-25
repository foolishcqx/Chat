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
import psutil
import os
import json
import logging
import os
import re
import sys
import traceback
import warnings
import time
import torch
import torchaudio
from text.LangSegmenter import LangSegmenter
import gradio as gr
import librosa
import numpy as np
from GPT_SoVITS.feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from time import time as ttime

from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from peft import LoraConfig, get_peft_model
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

from GPT_SoVITS.tools.assets import css, js, top_html
from GPT_SoVITS.tools.i18n.i18n import I18nAuto, scan_language_list

os.environ['no_proxy'] = "localhost, 127.0.0.1, ::1"

import random

from GPT_SoVITS.module.models import Generator, SynthesizerTrn, SynthesizerTrnV3
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)

version = model_version = os.environ.get("version", "v2")

from GPT_SoVITS.config import change_choices, get_weights_names, name2gpt_path, name2sovits_path
from GPT_SoVITS.config import pretrained_sovits_name
# -------------------- 全局变量 --------------------


def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# set_seed(42)


language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dict_language_v1 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("粤语"): "all_yue",  # 全部按中文识别
    i18n("韩文"): "all_ko",  # 全部按韩文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
    i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
}
dict_language = dict_language_v1 if version == "v1" else dict_language_v2




def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")





###todo:put them to process_ckpt and modify my_save func (save sovits weights), gpt save weights use my_save in process_ckpt
# symbol_version-model_version-if_lora_v3
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

v3v4set = {"v3", "v4"}


def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    print("load sovits=============================")
    if "！" in sovits_path or "!" in sovits_path:
        sovits_path = name2sovits_path[sovits_path]
    global hps, version, model_version, dict_language, if_lora_v3
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    print(sovits_path, version, model_version, if_lora_v3)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    if if_lora_v3 == True and is_exist == False:
        info = path_sovits + "SoVITS %s" % model_version + i18n("底模缺失，无法加载相应 LoRA 权重")
        gr.Warning(info)
        raise FileExistsError(info)
    dict_language = dict_language_v1 if version == "v1" else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = (
                {"__type__": "update"},
                {"__type__": "update", "value": prompt_language},
            )
        else:
            prompt_text_update = {"__type__": "update", "value": ""}
            prompt_language_update = {"__type__": "update", "value": i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {"__type__": "update"}, {"__type__": "update", "value": text_language}
        else:
            text_update = {"__type__": "update", "value": ""}
            text_language_update = {"__type__": "update", "value": i18n("中文")}
        if model_version in v3v4set:
            visible_sample_steps = True
            visible_inp_refs = False
        else:
            visible_sample_steps = False
            visible_inp_refs = True
    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    # print("sovits版本:",hps.model.version)
    if model_version not in v3v4set:
        if "Pro" not in model_version:
            model_version = version
        else:
            hps.model.version = model_version
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else:
        hps.model.version = model_version
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    if if_lora_v3 == False:
        print("loading sovits_%s" % model_version, vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(
            "loading sovits_%spretrained_G" % model_version,
            vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False),
        )
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print("loading sovits_%s_lora%s" % (model_version, lora_rank))
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        # torch.save(vq_model.state_dict(),"merge_win.pth")
        vq_model.eval()
    with open("./GPT_SoVITS/weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["SoVITS"][version] = sovits_path
    with open("./GPT_SoVITS/weight.json", "w") as f:
        f.write(json.dumps(data))
    return vq_model

# try:
#     next(change_sovits_weights(sovits_path))
# except:
#     pass


def change_gpt_weights(gpt_path):
    print("load GPT=============================")
    if "！" in gpt_path or "!" in gpt_path:
        gpt_path = name2gpt_path[gpt_path]
    global hz, max_sec, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    print(gpt_path, version, model_version)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./GPT_SoVITS/weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["GPT"][version] = gpt_path
    with open("./GPT_SoVITS/weight.json", "w") as f:
        f.write(json.dumps(data))
    return t2s_model


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch




def clean_hifigan_model():
    global hifigan_model
    if hifigan_model:
        hifigan_model = hifigan_model.cpu()
        hifigan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def clean_bigvgan_model():
    global bigvgan_model
    if bigvgan_model:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def clean_sv_cn_model():
    global sv_cn_model
    if sv_cn_model:
        sv_cn_model.embedding_model = sv_cn_model.embedding_model.cpu()
        sv_cn_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass


def init_bigvgan():
    global bigvgan_model, hifigan_model, sv_cn_model
    from GPT_SoVITS.BigVGAN import bigvgan

    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
        use_cuda_kernel=False,
    )  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    clean_hifigan_model()
    clean_sv_cn_model()
    if is_half == True:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)


def init_hifigan():
    global hifigan_model, bigvgan_model, sv_cn_model
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0,
        is_bias=True,
    )
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    state_dict_g = torch.load(
        "%s/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,),
        map_location="cpu",
        weights_only=False,
    )
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    clean_bigvgan_model()
    clean_sv_cn_model()
    if is_half == True:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)


from GPT_SoVITS.sv import SV


def init_sv_cn():
    global hifigan_model, bigvgan_model, sv_cn_model
    sv_cn_model = SV(device, is_half)
    clean_bigvgan_model()
    clean_hifigan_model()




def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


def get_spepc(hps, filename, dtype, device, is_v2pro=False):
    # audio = load_audio(filename, int(hps.data.sampling_rate))

    # audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
    # audio = torch.FloatTensor(audio)

    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load(filename)
    if sr0 != sr1:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        audio = resample(audio, sr0, sr1, device)
    else:
        audio = audio.to(device)
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)

    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro == True:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


def clean_text_inf(text, language, version):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text





def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


from text import chinese


def get_phones_and_bert(text, language, version, final=False):
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text,"zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text,"ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text,"ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(dtype), norm_text


from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch

spec_min = -12
spec_max = 2


def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)
mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


sr_model = None


def audio_sr(audio, sr):
    global sr_model
    if sr_model == None:
        from GPT_SoVITS.tools.audio_sr import AP_BWE

        try:
            sr_model = AP_BWE(device, DictToAttrRecursive)
        except FileNotFoundError:
            gr.Warning(i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


##ref_wav_path+prompt_text+prompt_language+text(单个)+text_language+top_k+top_p+temperature
# cache_tokens={}#暂未实现清理机制
cache = {}

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split("(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

def get_tts_wav(
    ref_wav_path,
    text,
    t2s_model,
    vq_model,
    text_language = "中文",
    prompt_language = "中文",
    prompt_text = None,
    how_to_cut="凑四句一切",
    top_k=15,
    top_p=1.0,
    temperature=1.0,
    ref_free=False,
    speed=1,
    if_freeze=False,
    inp_refs=None,
    sample_steps=8,
    if_sr=False,
    pause_second=0.3,
):
    with torch.no_grad():
        start_time = time.time()
        global cache
        if ref_wav_path:
            pass
        else:
            gr.Warning(i18n("请上传参考音频"))
        if text:
            pass
        else:
            gr.Warning(i18n("请填入推理文本"))
        t = []
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        if model_version in v3v4set:
            ref_free = False  # s2v3暂不支持ref_free
        else:
            if_sr = False
        if model_version not in {"v3", "v4", "v2Pro", "v2ProPlus"}:
            clean_bigvgan_model()
            clean_hifigan_model()
            clean_sv_cn_model()
        t0 = ttime()
        prompt_language = dict_language[prompt_language]
        text_language = dict_language[text_language]

        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_language != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
        text = text.strip("\n")
        # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

        print(i18n("实际输入的目标文本:"), text)
        zero_wav = np.zeros(
            int(hps.data.sampling_rate * pause_second),
            dtype=np.float16 if is_half == True else np.float32,
        )
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            zero_wav_torch = zero_wav_torch.to(device)
        if not ref_free:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                    gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
                    raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
                wav16k = torch.from_numpy(wav16k)
                if is_half == True:
                    wav16k = wav16k.half().to(device)
                else:
                    wav16k = wav16k.to(device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
                codes = vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(device)

        t1 = ttime()
        t.append(t1 - t0)

        if how_to_cut == i18n("凑四句一切"):
            text = cut1(text)
        elif how_to_cut == i18n("凑50字一切"):
            text = cut2(text)
        elif how_to_cut == i18n("按中文句号。切"):
            text = cut3(text)
        elif how_to_cut == i18n("按英文句号.切"):
            text = cut4(text)
        elif how_to_cut == i18n("按标点符号切"):
            text = cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        print(i18n("实际输入的目标文本(切句后):"), text)
        texts = text.split("\n")
        texts = process_text(texts)
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []
        ###s2v3暂不支持ref_free
        if not ref_free:
            phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

        for i_text, text in enumerate(texts):
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if text[-1] not in splits:
                text += "。" if text_language != "en" else "."
            print(i18n("实际输入的目标文本(每句):"), text)
            phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
            print(i18n("前端处理后的文本(每句):"), norm_text2)
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

            bert = bert.to(device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

            t2 = ttime()
            # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
            # print(cache.keys(),if_freeze)
            if i_text in cache and if_freeze == True:
                pred_semantic = cache[i_text]
            else:
                with torch.no_grad():
                    pred_semantic, idx = t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_len,
                        None if ref_free else prompt,
                        bert,
                        # prompt_phone_len=ph_offset,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=hz * max_sec,
                    )
                    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    cache[i_text] = pred_semantic
            t3 = ttime()
            is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
            # print(23333,is_v2pro,model_version)
            ###v3不存在以下逻辑和inp_refs
            if model_version not in v3v4set:
                refers = []
                if is_v2pro:
                    sv_emb = []
                    if sv_cn_model == None:
                        init_sv_cn()
                if inp_refs:
                    for path in inp_refs:
                        try:  #####这里加上提取sv的逻辑，要么一堆sv一堆refer，要么单个sv单个refer
                            refer, audio_tensor = get_spepc(hps, path.name, dtype, device, is_v2pro)
                            refers.append(refer)
                            if is_v2pro:
                                sv_emb.append(sv_cn_model.compute_embedding3(audio_tensor))
                        except:
                            traceback.print_exc()
                if len(refers) == 0:
                    refers, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device, is_v2pro)
                    refers = [refers]
                    if is_v2pro:
                        sv_emb = [sv_cn_model.compute_embedding3(audio_tensor)]
                if is_v2pro:
                    audio = vq_model.decode(
                        pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed, sv_emb=sv_emb
                    )[0][0]
                else:
                    audio = vq_model.decode(
                        pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed
                    )[0][0]
            else:
                refer, audio_tensor = get_spepc(hps, ref_wav_path, dtype, device)
                phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
                phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
                fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
                ref_audio, sr = torchaudio.load(ref_wav_path)
                ref_audio = ref_audio.to(device).float()
                if ref_audio.shape[0] == 2:
                    ref_audio = ref_audio.mean(0).unsqueeze(0)
                tgt_sr = 24000 if model_version == "v3" else 32000
                if sr != tgt_sr:
                    ref_audio = resample(ref_audio, sr, tgt_sr, device)
                # print("ref_audio",ref_audio.abs().mean())
                mel2 = mel_fn(ref_audio) if model_version == "v3" else mel_fn_v4(ref_audio)
                mel2 = norm_spec(mel2)
                T_min = min(mel2.shape[2], fea_ref.shape[2])
                mel2 = mel2[:, :, :T_min]
                fea_ref = fea_ref[:, :, :T_min]
                Tref = 468 if model_version == "v3" else 500
                Tchunk = 934 if model_version == "v3" else 1000
                if T_min > Tref:
                    mel2 = mel2[:, :, -Tref:]
                    fea_ref = fea_ref[:, :, -Tref:]
                    T_min = Tref
                chunk_len = Tchunk - T_min
                mel2 = mel2.to(dtype)
                fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
                cfm_resss = []
                idx = 0
                while 1:
                    fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                    if fea_todo_chunk.shape[-1] == 0:
                        break
                    idx += chunk_len
                    fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                    cfm_res = vq_model.cfm.inference(
                        fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
                    )
                    cfm_res = cfm_res[:, :, mel2.shape[2] :]
                    mel2 = cfm_res[:, :, -T_min:]
                    fea_ref = fea_todo_chunk[:, :, -T_min:]
                    cfm_resss.append(cfm_res)
                cfm_res = torch.cat(cfm_resss, 2)
                cfm_res = denorm_spec(cfm_res)
                if model_version == "v3":
                    if bigvgan_model == None:
                        init_bigvgan()
                else:  # v4
                    if hifigan_model == None:
                        init_hifigan()
                vocoder_model = bigvgan_model if model_version == "v3" else hifigan_model
                with torch.inference_mode():
                    wav_gen = vocoder_model(cfm_res)
                    audio = wav_gen[0][0]  # .cpu().detach().numpy()
            max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1:
                audio = audio / max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav_torch)  # zero_wav
            t4 = ttime()
            t.extend([t2 - t1, t3 - t2, t4 - t3])
            t1 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
        audio_opt = torch.cat(audio_opt, 0)  # np.concatenate
        if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
            opt_sr = 32000
        elif model_version == "v3":
            opt_sr = 24000
        else:
            opt_sr = 48000  # v4
        if if_sr == True and opt_sr == 24000:
            print(i18n("音频超分中"))
            audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr)
            max_audio = np.abs(audio_opt).max()
            if max_audio > 1:
                audio_opt /= max_audio
        else:
            audio_opt = audio_opt.cpu().detach().numpy()
        end_time = time.time()
        print("[TTS]:", end_time - start_time)
    return opt_sr, (audio_opt * 32767).astype(np.int16)




# -------------------- 业务函数 --------------------
def audio_recognize(wav_path):
    t0 = time.time()
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("音频必须是单声道 16-bit PCM WAV")
        rec = KaldiRecognizer(asr_model, wf.getframerate())
        rec.SetWords(True)
        audio_data = wf.readframes(wf.getnframes())
        rec.AcceptWaveform(audio_data)
        result = json.loads(rec.FinalResult())
    text = result.get("text", "").replace(' ', '')
    print(f"[ASR] {time.time()-t0:.2f}s → {text}")
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
    return reply, time.time() - t0

# def llm_chat(content):
#     t0 = time.time()
#     try:
#         client = OpenAI(
#             base_url='https://api-inference.modelscope.cn/v1',
#             api_key='ms-cb688843-be74-4cdf-8e0b-6237eda42d1e', # ModelScope Token
#         )

#         # set extra_body for thinking control
#         extra_body = {
#             # enable thinking, set to False to disable
#             "enable_thinking": False,
#             # use thinking_budget to contorl num of tokens used for thinking
#             # "thinking_budget": 4096
#         }

#         response = client.chat.completions.create(
#             model='Qwen/Qwen3-0.6B',  # ModelScope Model-Id
#             messages=[
#                 {"role": "system",
#                             "content": "可莉是西风骑士团的火花骑士，最爱炸鱼和蹦蹦炸弹！说话要元气满满，偶尔提到琴团长会怕怕的。现在可莉要用可爱又爆炸的方式与旅行者交流哦！"},
#                 {
#                 'role': 'user',
#                 'content': content + "。回复请限制在25个字以内"
#                 }
#             ],
#             stream=False,
#             extra_body=extra_body
#         )
#         reply = response.choices[0].message.content.strip()
#     except Exception as e:
#         reply = f"网络异常，可莉暂时离线~"
#     print(f"[LLM] {time.time()-t0:.2f}s → {reply}")
#     return reply, time.time() - t0

# -------------------- Gradio 流程 --------------------
def pipeline(audio, history, ref_audio =r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\阳光zql_slicer_opt\阳光zql.mp3_0017179840_0017435520.wav"):

    # 1. ASR 识别
    asr_text, t_asr = audio_recognize(audio)

    # 2. LLM 对话
    reply, t_llm = llm_chat(asr_text)

    # 3. TTS 合成
    tts_strat  = time.time()
    sr, wav = get_tts_wav(
        ref_wav_path=ref_audio,
        prompt_text=None,
        text=reply,
        text_language="中文",
    )
    tts_end = time.time()
    t_tts = tts_end - tts_strat  # TTS 花费时间

    # 保存生成的音频
    output_path = os.path.join("./temp_audio",f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
    write_wav(output_path, sr, wav)

    # 计算音频时长（秒）
    audio_duration = len(wav) / sr

    # 计算音频时长与花费时间的比值
    duration_ratio =   t_tts / audio_duration if t_tts > 0 else 0

    # 追加到前端聊天记录
    history.append([asr_text, reply])

    # 写入日志
    write_log(asr_text, reply, t_asr, t_llm, t_tts, audio_duration, duration_ratio)

    return history, output_path

# 保存 WAV 文件
def write_wav(path, sr, wav):
    from scipy.io.wavfile import write
    write(path, sr, wav)
    print(f"音频已保存到 {path}")

# 写入日志
def write_log(asr_text, reply, t_asr, t_llm, t_tts, audio_duration, duration_ratio):
    total_time = t_asr + t_llm + t_tts  # 计算总耗时
    with open("gpt_chat.log", "a", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"用户：{asr_text}\n")
        f.write(f"可莉：{reply}\n")
        f.write(f"耗时：ASR={t_asr:.2f}s | LLM={t_llm:.2f}s | TTS={t_tts:.2f}s\n")
        f.write(f"音频时长：{audio_duration:.2f}s\n")
        f.write(f"音频生成时间/音频时长：{duration_ratio:.2f}\n")
        f.write(f"总耗时：{total_time:.2f}s\n")  # 写入总耗时


SoVITS_names, GPT_names = get_weights_names()
chat_history = []      # 前端聊天记录
log_file   = "gpt_chat.log"
SetLogLevel(-1)        # 关闭 Vosk 冗余日志

# 模型初始化
asr_model = Model("vosk-model-small-cn-0.22")
path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)

if os.path.exists("./GPT_SoVITS/weight.json"):
    pass
else:
    with open("./GPT_SoVITS/weight.json", "w", encoding="utf-8") as file:
        json.dump({"GPT": {}, "SoVITS": {}}, file)

with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as file:
    weight_data = file.read()
    weight_data = json.loads(weight_data)
    gpt_path = os.environ.get("gpt_path", weight_data.get("GPT", {}).get(version, GPT_names[-1]))
    sovits_path = os.environ.get("sovits_path", weight_data.get("SoVITS", {}).get(version, SoVITS_names[0]))
    if isinstance(gpt_path, list):
        gpt_path = gpt_path[0]
    if isinstance(sovits_path, list):
        sovits_path = sovits_path[0]
cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
# is_half=False
punctuation = set(["!", "?", "…", ",", ".", "-", " "])
cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)
ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)
now_dir = os.getcwd()
bigvgan_model = hifigan_model = sv_cn_model = None
if model_version == "v3":
    init_bigvgan()
if model_version == "v4":
    init_hifigan()
if model_version in {"v2Pro", "v2ProPlus"}:
    init_sv_cn()

resample_transform_dict = {}
dtype = torch.float16 if is_half == True else torch.float32
