from copy import deepcopy
import math
import os, sys, gc
import random
import traceback
import time
from collections import OrderedDict
from tqdm import tqdm
now_dir = os.getcwd()
sys.path.append(now_dir)
import ffmpeg
import os
from typing import Generator, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForMaskedLM, AutoTokenizer
import json
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.feature_extractor.cnhubert import CNHubert
from GPT_SoVITS.module.models import SynthesizerTrn
import librosa
from time import time as ttime
from tools.i18n.i18n import I18nAuto, scan_language_list
from tools.my_utils import load_audio
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits
from GPT_SoVITS.TTS_infer_pack.TextPreprocessor import TextPreprocessor
from copy import deepcopy   
language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# configs/tts_infer.yaml
"""
custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
  version: v2
default:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/s2G488k.pth
  version: v1
default_v2:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cpu
  is_half: false
  t2s_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
  vits_weights_path: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
  version: v2
"""
with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
            path_map = json.load(f)
def set_seed(seed:int):
    seed = int(seed)
    seed = seed if seed != -1 else random.randrange(1 << 32)
    print(f"Set seed to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = True
            # 开启后会影响精度
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed

class TTS_Config:
    default_configs={
        "default":{
                "device": "cpu",
                "is_half": False,
                "version": "v1",
                "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "vits_weights_path": "GPT_SoVITS/pretrained_models/s2G488k.pth",
                "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            },
        "default_v2":{
                "device": "cpu",
                "is_half": False,
                "version": "v2",
                "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            },
    }
    configs:dict = None
    v1_languages:list = ["auto", "en", "zh", "ja",  "all_zh", "all_ja"]
    v2_languages:list = ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]
    languages:list = v2_languages
    # "all_zh",#全部按中文识别
    # "en",#全部按英文识别#######不变
    # "all_ja",#全部按日文识别
    # "all_yue",#全部按中文识别
    # "all_ko",#全部按韩文识别
    # "zh",#按中英混合识别####不变
    # "ja",#按日英混合识别####不变
    # "yue",#按粤英混合识别####不变
    # "ko",#按韩英混合识别####不变
    # "auto",#多语种启动切分识别语种
    # "auto_yue",#多语种启动切分识别语种

    def __init__(self, configs: Union[dict, str]=None):

        # 设置默认配置文件路径
        configs_base_path:str = "GPT_SoVITS/configs/"
        os.makedirs(configs_base_path, exist_ok=True)
        self.configs_path:str = os.path.join(configs_base_path, "tts_infer.yaml")

        if configs in ["", None]:
            if not os.path.exists(self.configs_path):
                self.save_configs()
                print(f"Create default config file at {self.configs_path}")
            configs:dict = deepcopy(self.default_configs)

        if isinstance(configs, str):
            self.configs_path = configs
            configs:dict = self._load_configs(self.configs_path)

        assert isinstance(configs, dict)
        version = configs.get("version", "v2").lower()
        assert version in ["v1", "v2"]
        self.default_configs["default"] = configs.get("default", self.default_configs["default"])
        self.default_configs["default_v2"] = configs.get("default_v2", self.default_configs["default_v2"])

        default_config_key = "default"if version=="v1" else "default_v2"
        self.configs:dict = configs.get("custom", deepcopy(self.default_configs[default_config_key]))


        self.device = self.configs.get("device", torch.device("cpu"))
        self.is_half = self.configs.get("is_half", False)
        self.version = version
        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)
        self.bert_base_path = self.configs.get("bert_base_path", None)
        self.cnhuhbert_base_path = self.configs.get("cnhuhbert_base_path", None)
        self.languages = self.v2_languages if self.version=="v2" else self.v1_languages


        # if (self.t2s_weights_path in [None, ""]) or (not os.path.exists(self.t2s_weights_path)):
        #     self.t2s_weights_path = self.default_configs[default_config_key]['t2s_weights_path']
        #     print(f"fall back to default t2s_weights_path: {self.t2s_weights_path}")
        # if (self.vits_weights_path in [None, ""]) or (not os.path.exists(self.vits_weights_path)):
        #     self.vits_weights_path = self.default_configs[default_config_key]['vits_weights_path']
        #     print(f"fall back to default vits_weights_path: {self.vits_weights_path}")
        if (self.bert_base_path in [None, ""]) or (not os.path.exists(self.bert_base_path)):
            self.bert_base_path = self.default_configs[default_config_key]['bert_base_path']
            print(f"fall back to default bert_base_path: {self.bert_base_path}")
        if (self.cnhuhbert_base_path in [None, ""]) or (not os.path.exists(self.cnhuhbert_base_path)):
            self.cnhuhbert_base_path = self.default_configs[default_config_key]['cnhuhbert_base_path']
            print(f"fall back to default cnhuhbert_base_path: {self.cnhuhbert_base_path}")
        self.update_configs()


        self.max_sec = None
        self.hz:int = 50
        self.semantic_frame_rate:str = "25hz"
        self.segment_size:int = 20480
        self.filter_length:int = 2048
        self.sampling_rate:int = 32000
        self.hop_length:int = 640
        self.win_length:int = 2048
        self.n_speakers:int = 300



    def _load_configs(self, configs_path: str)->dict:
        if os.path.exists(configs_path):
            ...
        else:
            print(i18n("路径不存在,使用默认配置"))
            self.save_configs(configs_path)
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        return configs

    def save_configs(self, configs_path:str=None)->None:
        configs=deepcopy(self.default_configs)
        if self.configs is not None:
            configs["custom"] = self.update_configs()

        if configs_path is None:
            configs_path = self.configs_path
        with open(configs_path, 'w') as f:
            yaml.dump(configs, f)

    def update_configs(self):
        self.config = {
            "device"             : str(self.device),
            "is_half"            : self.is_half,
            "version"            : self.version,
            "t2s_weights_path"   : self.t2s_weights_path,
            "vits_weights_path"  : self.vits_weights_path,
            "bert_base_path"     : self.bert_base_path,
            "cnhuhbert_base_path": self.cnhuhbert_base_path,
        }
        return self.config

    def update_version(self, version:str)->None:
        self.version = version
        self.languages = self.v2_languages if self.version=="v2" else self.v1_languages

    def __str__(self):
        self.configs = self.update_configs()
        string = "TTS Config".center(100, '-') + '\n'
        for k, v in self.configs.items():
            string += f"{str(k).ljust(20)}: {str(v)}\n"
        string += "-" * 100 + '\n'
        return string

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.configs_path)

    def __eq__(self, other):
        return isinstance(other, TTS_Config) and self.configs_path == other.configs_path
    
class ModelCache:
    def __init__(self, capacity=10):
        self.capacity = capacity  # 缓存容量
        self.cache = OrderedDict()  # 有序字典，用于记录模型及其最后访问时间
        self.model_update_log = []  # 记录模型更新的日志
        self.inference_locks = {}  # 用于标记模型是否正在推理

    def get_model(self, model_id):
        if model_id in self.cache:
            # 更新访问时间
            self.cache.move_to_end(model_id)
            return self.cache[model_id]
        else:
            return None

    def add_model(self, model_id, t2s_model, vq_model):
        if model_id in self.cache:
            self.cache.move_to_end(model_id)
            return

        # 缓存已满且全部被锁时，一直轮询
        while len(self.cache) >= self.capacity:
            evicted_model_id = next(iter(self.cache))   # 按 LRU 顺序
            if not self.inference_locks.get(evicted_model_id, False):
                # 真正淘汰
                self.cache.popitem(last=False)
                self.model_update_log.append((evicted_model_id, model_id))
                self.inference_locks.pop(evicted_model_id, None)
                break
            # 否则继续下一轮（CPU 睡眠 10 ms，避免 100% 占用）
            time.sleep(0.01)

        # 成功腾位置后加载
        self.cache[model_id] = {'t2s_model': t2s_model, 'vq_model': vq_model}
        self.inference_locks[model_id] = False

    def remove_model(self, model_id):
        if model_id in self.cache:
            del self.cache[model_id]
            self.inference_locks.pop(model_id, None)

    def lock_model_for_inference(self, model_id):
        if model_id in self.cache:
            self.inference_locks[model_id] = True

    def unlock_model_after_inference(self, model_id):
        if model_id in self.cache:
            self.inference_locks[model_id] = False

    def get_least_recently_used_model_id(self):
        if self.cache:
            return next(iter(self.cache))
        return None

    def get_model_update_log(self):
        return self.model_update_log


class TTS:
    def __init__(self, configs: Union[dict, str, TTS_Config], capacity=10):
        if isinstance(configs, TTS_Config):
            self.configs = configs
        else:
            self.configs:TTS_Config = TTS_Config(configs)
        self.t2s_model:Text2SemanticLightningModule = None
        self.vits_model:SynthesizerTrn = None
        self.model_cache = ModelCache(capacity=capacity)
        self.bert_tokenizer:AutoTokenizer = None
        self.bert_model:AutoModelForMaskedLM = None
        self.cnhuhbert_model:CNHubert = None

        self._init_models()

        self.text_preprocessor:TextPreprocessor = \
                            TextPreprocessor(self.bert_model,
                                            self.bert_tokenizer,
                                            self.configs.device)


        self.prompt_cache:dict = {
            "ref_audio_path" : None,
            "prompt_semantic": None,
            "refer_spec"     : [],
            "prompt_text"    : None,
            "prompt_lang"    : None,
            "phones"         : None,
            "bert_features"  : None,
            "norm_text"      : None,
            "aux_ref_audio_paths": [],
        }


        self.stop_flag:bool = False
        self.precision:torch.dtype = torch.float16 if self.configs.is_half else torch.float32

    def _init_models(self,):
        self.init_vits_weights(self.configs.vits_weights_path)
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)
        # self.enable_half_precision(self.configs.is_half)



    def init_cnhuhbert_weights(self, base_path: str):
        print(f"Loading CNHuBERT weights from {base_path}")
        self.cnhuhbert_model = CNHubert(base_path)
        self.cnhuhbert_model=self.cnhuhbert_model.eval()
        self.cnhuhbert_model = self.cnhuhbert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device)!="cpu":
            self.cnhuhbert_model = self.cnhuhbert_model.half()



    def init_bert_weights(self, base_path: str):
        print(f"Loading BERT weights from {base_path}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        self.bert_model=self.bert_model.eval()
        self.bert_model = self.bert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device)!="cpu":
            self.bert_model = self.bert_model.half()

    def init_vits_weights(self, weights_dict: dict):
        self.configs.vits_weights_path = weights_dict
        for model_id, weights_path in weights_dict.items():
            print(f"[init_vits_weights] Loading VITS for <{model_id}> from {weights_path}")

            dict_s2 = torch.load(weights_path, map_location=self.configs.device, weights_only=False)
            hps = dict_s2["config"]
            if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
                self.configs.update_version("v1")
            else:
                self.configs.update_version("v2")
            self.configs.save_configs()

            hps["model"]["version"] = self.configs.version
            self.configs.filter_length   = hps["data"]["filter_length"]
            self.configs.segment_size    = hps["train"]["segment_size"]
            self.configs.sampling_rate   = hps["data"]["sampling_rate"]
            self.configs.hop_length      = hps["data"]["hop_length"]
            self.configs.win_length      = hps["data"]["win_length"]
            self.configs.n_speakers      = hps["data"]["n_speakers"]
            self.configs.semantic_frame_rate = "25hz"

            kwargs = hps["model"]
            vits_model = SynthesizerTrn(
                self.configs.filter_length // 2 + 1,
                self.configs.segment_size // self.configs.hop_length,
                n_speakers=self.configs.n_speakers,
                **kwargs
            )
            if hasattr(vits_model, "enc_q"):
                del vits_model.enc_q
            vits_model = vits_model.to(self.configs.device).eval()
            vits_model.load_state_dict(dict_s2["weight"], strict=False)
            if self.configs.is_half and str(self.configs.device) != "cpu":
                vits_model = vits_model.half()

            # 👇 写进缓存（t2s_model=None 仅占位，后面 init_t2s_weights 会补）
            self.model_cache.add_model(model_id, t2s_model=None, vq_model=vits_model)

    def init_t2s_weights(self, weights_dict: dict):
        self.configs.t2s_weights_path = weights_dict
        self.configs.save_configs()
        self.configs.hz = 50
        for model_id, weights_path in weights_dict.items():
            print(f"[init_t2s_weights] Loading T2S for <{model_id}> from {weights_path}")

            dict_s1 = torch.load(weights_path, map_location=self.configs.device)
            config = dict_s1["config"]
            tmp_cfg = deepcopy(self.configs)
            tmp_cfg.max_sec = config["data"]["max_sec"]
            self.configs.max_sec = config["data"]["max_sec"]
            tmp_cfg.hz = 50

            t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
            t2s_model.load_state_dict(dict_s1["weight"])
            t2s_model = t2s_model.to(tmp_cfg.device).eval()
            if tmp_cfg.is_half and str(tmp_cfg.device) != "cpu":
                t2s_model = t2s_model.half()

            # 👇 更新缓存：如果 vits 已存在则保留，否则占位
            cached = self.model_cache.get_model(model_id)
            if cached is None:
                self.model_cache.add_model(model_id, t2s_model=t2s_model, vq_model=None)
            else:
                cached["t2s_model"] = t2s_model

    def enable_half_precision(self, enable: bool = True, save: bool = True):
        '''
            To enable half precision for the TTS model.
            Args:
                enable: bool, whether to enable half precision.

        '''
        if str(self.configs.device) == "cpu" and enable:
            print("Half precision is not supported on CPU.")
            return

        self.configs.is_half = enable
        self.precision = torch.float16 if enable else torch.float32
        if save:
            self.configs.save_configs()
        if enable:
            if self.t2s_model is not None:
                self.t2s_model =self.t2s_model.half()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.half()
            if self.bert_model is not None:
                self.bert_model =self.bert_model.half()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.half()
        else:
            if self.t2s_model is not None:
                self.t2s_model = self.t2s_model.float()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.float()
            if self.bert_model is not None:
                self.bert_model = self.bert_model.float()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.float()

    def set_ref_audio(self, ref_audio_path:str):
        '''
            To set the reference audio for the TTS model,
                including the prompt_semantic and refer_spepc.
            Args:
                ref_audio_path: str, the path of the reference audio.
        '''
        self._set_prompt_semantic(ref_audio_path)
        self._set_ref_spec(ref_audio_path)
        self._set_ref_audio_path(ref_audio_path)

    def _set_ref_audio_path(self, ref_audio_path):
        self.prompt_cache["ref_audio_path"] = ref_audio_path

    def _set_ref_spec(self, ref_audio_path):
        spec = self._get_ref_spec(ref_audio_path)
        if self.prompt_cache["refer_spec"] in [[],None]:
            self.prompt_cache["refer_spec"]=[spec]
        else:
            self.prompt_cache["refer_spec"][0] = spec

    def _get_ref_spec(self, ref_audio_path):
        audio = load_audio(ref_audio_path, int(self.configs.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx=audio.abs().max()
        if(maxx>1):audio/=min(2,maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.configs.filter_length,
            self.configs.sampling_rate,
            self.configs.hop_length,
            self.configs.win_length,
            center=False,
        )
        spec = spec.to(self.configs.device)
        if self.configs.is_half:
            spec = spec.half()
        return spec

    def _set_prompt_semantic(self, ref_wav_path:str):
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * 0.3),
            dtype=np.float16 if self.configs.is_half else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.configs.device)
            zero_wav_torch = zero_wav_torch.to(self.configs.device)
            if self.configs.is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            hubert_feature = self.cnhuhbert_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vits_model.extract_latent(hubert_feature)

            prompt_semantic = codes[0, 0].to(self.configs.device)
            self.prompt_cache["prompt_semantic"] = prompt_semantic

    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length:int=None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype:torch.dtype = seq.dtype
        pad_value = torch.tensor(pad_value, dtype=dtype)
        seq_lengths = [seq.shape[axis] for seq in sequences]
        if max_length is None:
            max_length = max(seq_lengths)
        else:
            max_length = max(seq_lengths) if max_length < max(seq_lengths) else max_length

        padded_sequences = []
        for seq, length in zip(sequences, seq_lengths):
            padding = [0] * axis + [0, max_length - length] + [0] * (ndim - axis - 1)
            padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            padded_sequences.append(padded_seq)
        batch = torch.stack(padded_sequences)
        return batch

    def to_batch(self, data:list,
                 prompt_data:dict=None,
                 batch_size:int=5,
                 threshold:float=0.75,
                 split_bucket:bool=True,
                 device:torch.device=torch.device("cpu"),
                 precision:torch.dtype=torch.float32,
                 ):
        _data:list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        batch_index_list = []
        if split_bucket:
            index_and_len_list.sort(key=lambda x: x[1])
            index_and_len_list = np.array(index_and_len_list, dtype=np.int64)

            batch_index_list_len = 0
            pos = 0
            while pos <index_and_len_list.shape[0]:
                # batch_index_list.append(index_and_len_list[pos:min(pos+batch_size,len(index_and_len_list))])
                pos_end = min(pos+batch_size,index_and_len_list.shape[0])
                while pos < pos_end:
                    batch=index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score=batch[(pos_end-pos)//2]/(batch.mean()+1e-8)
                    if (score>=threshold) or (pos_end-pos==1):
                        batch_index=index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end=pos_end-1

            assert batch_index_list_len == len(data)

        else:
            for i in range(len(data)):
                if i%batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)


        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            phones_len_list = []
            # bert_features_list = []
            all_phones_list = []
            all_phones_len_list = []
            all_bert_features_list = []
            norm_text_batch = []
            all_bert_max_len = 0
            all_phones_max_len = 0
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = torch.cat([prompt_data["bert_features"], item["bert_features"]], 1)\
                                                .to(dtype=precision, device=device)
                    all_phones = torch.LongTensor(prompt_data["phones"]+item["phones"]).to(device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    # norm_text = prompt_data["norm_text"]+item["norm_text"]
                else:
                    all_bert_features = item["bert_features"]\
                                            .to(dtype=precision, device=device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    all_phones = phones
                    # norm_text = item["norm_text"]

                all_bert_max_len = max(all_bert_max_len, all_bert_features.shape[-1])
                all_phones_max_len = max(all_phones_max_len, all_phones.shape[-1])

                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])

            phones_batch = phones_list
            all_phones_batch = all_phones_list
            all_bert_features_batch = all_bert_features_list


            max_len = max(all_bert_max_len, all_phones_max_len)
            # phones_batch = self.batch_sequences(phones_list, axis=0, pad_value=0, max_length=max_len)
            #### 直接对phones和bert_features进行pad。（padding策略会影响T2S模型生成的结果，但不直接影响复读概率。影响复读概率的主要因素是mask的策略）
            # all_phones_batch = self.batch_sequences(all_phones_list, axis=0, pad_value=0, max_length=max_len)
            # all_bert_features_batch = all_bert_features_list
            # all_bert_features_batch = torch.zeros((len(all_bert_features_list), 1024, max_len), dtype=precision, device=device)
            # for idx, item in enumerate(all_bert_features_list):
            #     all_bert_features_batch[idx, :, : item.shape[-1]] = item

            # #### 先对phones进行embedding、对bert_features进行project，再pad到相同长度，（padding策略会影响T2S模型生成的结果，但不直接影响复读概率。影响复读概率的主要因素是mask的策略）
            # all_phones_list = [self.t2s_model.model.ar_text_embedding(item.to(self.t2s_model.device)) for item in all_phones_list]
            # all_phones_list = [F.pad(item,(0,0,0,max_len-item.shape[0]),value=0) for item in all_phones_list]
            # all_phones_batch = torch.stack(all_phones_list, dim=0)

            # all_bert_features_list = [self.t2s_model.model.bert_proj(item.to(self.t2s_model.device).transpose(0, 1)) for item in all_bert_features_list]
            # all_bert_features_list = [F.pad(item,(0,0,0,max_len-item.shape[0]), value=0) for item in all_bert_features_list]
            # all_bert_features_batch = torch.stack(all_bert_features_list, dim=0)

            batch = {
                "phones": phones_batch,
                "phones_len": torch.LongTensor(phones_len_list).to(device),
                "all_phones": all_phones_batch,
                "all_phones_len": torch.LongTensor(all_phones_len_list).to(device),
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch,
                "max_len": max_len,
            }
            _data.append(batch)

        return _data, batch_index_list

    def recovery_order(self, data:list, batch_index_list:list)->list:
        '''
        Recovery the order of the audio according to the batch_index_list.

        Args:
            data (List[list(np.ndarray)]): the out of order audio .
            batch_index_list (List[list[int]]): the batch index list.

        Returns:
            list (List[np.ndarray]): the data in the original order.
        '''
        length = len(sum(batch_index_list, []))
        _data = [None]*length
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data

    def stop(self,):
        '''
        Stop the inference process.
        '''
        self.stop_flag = True

    @torch.no_grad()
    def run(self, id, inputs:dict):
        """
        Text to speech inference.

        Args:
            inputs (dict):
                {
                    "text": "",                   # str.(required) text to be synthesized
                    "text_lang: "",               # str.(required) language of the text to be synthesized
                    "ref_audio_path": "",         # str.(required) reference audio path
                    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                    "prompt_text": "",            # str.(optional) prompt text for the reference audio
                    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                    "top_k": 5,                   # int. top k sampling
                    "top_p": 1,                   # float. top p sampling
                    "temperature": 1,             # float. temperature for sampling
                    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
                    "batch_size": 1,              # int. batch size for inference
                    "batch_threshold": 0.75,      # float. threshold for batch splitting.
                    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                    "return_fragment": False,     # bool. step by step return the audio fragment.
                    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                    "seed": -1,                   # int. random seed for reproducibility.
                    "parallel_infer": True,       # bool. whether to use parallel inference.
                    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
                }
        returns:
            Tuple[int, np.ndarray]: sampling rate and audio data.
        """
        ########## variables initialization ###########
        self.stop_flag:bool = False
        text:str = inputs.get("text", "")
        text_lang:str = inputs.get("text_lang", "")
        ref_audio_path:str = inputs.get("ref_audio_path", "")
        aux_ref_audio_paths:list = inputs.get("aux_ref_audio_paths", [])
        prompt_text:str = inputs.get("prompt_text", "")
        prompt_lang:str = inputs.get("prompt_lang", "")
        top_k:int = inputs.get("top_k", 5)
        top_p:float = inputs.get("top_p", 1)
        temperature:float = inputs.get("temperature", 1)
        text_split_method:str = inputs.get("text_split_method", "cut0")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        actual_seed = set_seed(seed)
        parallel_infer = inputs.get("parallel_infer", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)
        models = self.model_cache.get_model(id)
        if models is None:
            # 未命中：先尝试现场加载，也可以直接 raise
            print(f"[run] 缓存未命中，现场加载 <{id}>")

            
            # 从 weight.json 里找路径
            
            vits_path = path_map["SoVITS"][id]
            t2s_path  = path_map["GPT"][id]
            self.init_vits_weights({id: vits_path})
            self.init_t2s_weights({id: t2s_path})
            models = self.model_cache.get_model(id)
            if models is None:
                raise RuntimeError(f"模型 <{id}> 加载失败")
        self.t2s_model = models["t2s_model"]
        self.vits_model = models["vq_model"]
        if parallel_infer:
            print(i18n("并行推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
        else:
            print(i18n("并行推理模式已关闭"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

        if return_fragment:
            print(i18n("分段返回模式已开启"))
            if split_bucket:
                split_bucket = False
                print(i18n("分段返回模式不支持分桶处理，已自动关闭分桶处理"))

        if split_bucket and speed_factor==1.0:
            print(i18n("分桶处理模式已开启"))
        elif speed_factor!=1.0:
            print(i18n("语速调节不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        else:
            print(i18n("分桶处理模式已关闭"))

        if fragment_interval<0.01:
            fragment_interval = 0.01
            print(i18n("分段间隔过小，已自动设置为0.01"))

        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True

        assert text_lang in self.configs.languages
        if not no_prompt_text:
            assert prompt_lang in self.configs.languages

        if ref_audio_path in [None, ""] and \
            ((self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spec"] in [None, []])):
            raise ValueError("ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()")

        ###### setting reference audio and prompt text preprocessing ########
        t0 = ttime()
        if (ref_audio_path is not None) and (ref_audio_path != self.prompt_cache["ref_audio_path"]):
            if not os.path.exists(ref_audio_path):
                raise ValueError(f"{ref_audio_path} not exists")
            self.set_ref_audio(ref_audio_path)

        aux_ref_audio_paths = aux_ref_audio_paths if aux_ref_audio_paths is not None else []
        paths = set(aux_ref_audio_paths)&set(self.prompt_cache["aux_ref_audio_paths"])
        if not (len(list(paths)) == len(aux_ref_audio_paths) == len(self.prompt_cache["aux_ref_audio_paths"])):
            self.prompt_cache["aux_ref_audio_paths"] = aux_ref_audio_paths
            self.prompt_cache["refer_spec"] = [self.prompt_cache["refer_spec"][0]]
            for path in aux_ref_audio_paths:
                if path in [None, ""]:
                    continue
                if not os.path.exists(path):
                    print(i18n("音频文件不存在，跳过："), path)
                    continue
                self.prompt_cache["refer_spec"].append(self._get_ref_spec(path))

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_lang != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
            if self.prompt_cache["prompt_text"] != prompt_text:
                self.prompt_cache["prompt_text"] = prompt_text
                self.prompt_cache["prompt_lang"] = prompt_lang
                phones, bert_features, norm_text = \
                    self.text_preprocessor.segment_and_extract_feature_for_text(
                                                                        prompt_text,
                                                                        prompt_lang,
                                                                        self.configs.version)
                self.prompt_cache["phones"] = phones
                self.prompt_cache["bert_features"] = bert_features
                self.prompt_cache["norm_text"] = norm_text




        ###### text preprocessing ########
        t1 = ttime()
        data:list = None
        if not return_fragment:
            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method, self.configs.version)
            if len(data) == 0:
                yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                            dtype=np.int16)
                return

            batch_index_list:list = None
            data, batch_index_list = self.to_batch(data,
                                prompt_data=self.prompt_cache if not no_prompt_text else None,
                                batch_size=batch_size,
                                threshold=batch_threshold,
                                split_bucket=split_bucket,
                                device=self.configs.device,
                                precision=self.precision
                                )
        else:
            print(f'############ {i18n("切分文本")} ############')
            texts = self.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
            data = []
            for i in range(len(texts)):
                if i%batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])

            def make_batch(batch_texts):
                batch_data = []
                print(f'############ {i18n("提取文本Bert特征")} ############')
                for text in tqdm(batch_texts):
                    phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(text, text_lang, self.configs.version)
                    if phones is None:
                        continue
                    res={
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    batch_data.append(res)
                if len(batch_data) == 0:
                    return None
                batch, _ = self.to_batch(batch_data,
                            prompt_data=self.prompt_cache if not no_prompt_text else None,
                            batch_size=batch_size,
                            threshold=batch_threshold,
                            split_bucket=False,
                            device=self.configs.device,
                            precision=self.precision
                            )
                return batch[0]


        t2 = ttime()
        try:
            print("############ 推理 ############")
            ###### inference ######
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            for item in data:
                t3 = ttime()
                if return_fragment:
                    item = make_batch(item)
                    if item is None:
                        continue

                batch_phones:List[torch.LongTensor] = item["phones"]
                # batch_phones:torch.LongTensor = item["phones"]
                batch_phones_len:torch.LongTensor = item["phones_len"]
                all_phoneme_ids:torch.LongTensor = item["all_phones"]
                all_phoneme_lens:torch.LongTensor  = item["all_phones_len"]
                all_bert_features:torch.LongTensor = item["all_bert_features"]
                norm_text:str = item["norm_text"]
                max_len = item["max_len"]

                print(i18n("前端处理后的文本(每句):"), norm_text)
                if no_prompt_text :
                    prompt = None
                else:
                    prompt = self.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.configs.device)


                pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,
                    all_bert_features,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.configs.hz * self.configs.max_sec,
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
                t4 = ttime()
                t_34 += t4 - t3

                refer_audio_spec:torch.Tensor = [item.to(dtype=self.precision, device=self.configs.device) for item in self.prompt_cache["refer_spec"]]


                batch_audio_fragment = []

                # ## vits并行推理 method 1
                # pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                # pred_semantic_len = torch.LongTensor([item.shape[0] for item in pred_semantic_list]).to(self.configs.device)
                # pred_semantic = self.batch_sequences(pred_semantic_list, axis=0, pad_value=0).unsqueeze(0)
                # max_len = 0
                # for i in range(0, len(batch_phones)):
                #     max_len = max(max_len, batch_phones[i].shape[-1])
                # batch_phones = self.batch_sequences(batch_phones, axis=0, pad_value=0, max_length=max_len)
                # batch_phones = batch_phones.to(self.configs.device)
                # batch_audio_fragment = (self.vits_model.batched_decode(
                #         pred_semantic, pred_semantic_len, batch_phones, batch_phones_len,refer_audio_spec
                #     ))

                if speed_factor == 1.0:
                    # ## vits并行推理 method 2
                    pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                    upsample_rate = math.prod(self.vits_model.upsample_rates)
                    audio_frag_idx = [pred_semantic_list[i].shape[0]*2*upsample_rate for i in range(0, len(pred_semantic_list))]
                    audio_frag_end_idx = [ sum(audio_frag_idx[:i+1]) for i in range(0, len(audio_frag_idx))]
                    all_pred_semantic = torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.configs.device)
                    _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(self.configs.device)
                    _batch_audio_fragment = (self.vits_model.decode(
                            all_pred_semantic, _batch_phones, refer_audio_spec, speed=speed_factor
                        ).detach()[0, 0, :])
                    audio_frag_end_idx.insert(0, 0)
                    batch_audio_fragment= [_batch_audio_fragment[audio_frag_end_idx[i-1]:audio_frag_end_idx[i]] for i in range(1, len(audio_frag_end_idx))]
                else:
                # ## vits串行推理
                    for i, idx in enumerate(idx_list):
                        phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                        _pred_semantic = (pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0))   # .unsqueeze(0)#mq要多unsqueeze一次
                        audio_fragment =(self.vits_model.decode(
                                _pred_semantic, phones, refer_audio_spec, speed=speed_factor
                            ).detach()[0, 0, :])
                        batch_audio_fragment.append(
                            audio_fragment
                        )  ###试试重建不带上prompt部分

                t5 = ttime()
                t_45 += t5 - t4
                if return_fragment:
                    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t4 - t3, t5 - t4))
                    yield self.audio_postprocess([batch_audio_fragment],
                                                    self.configs.sampling_rate,
                                                    None,
                                                    speed_factor,
                                                    False,
                                                    fragment_interval
                                                    )
                else:
                    audio.append(batch_audio_fragment)

                if self.stop_flag:
                    yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                            dtype=np.int16)
                    return

            if not return_fragment:
                print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
                if len(audio) == 0:
                    yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                                dtype=np.int16)
                    return
                yield self.audio_postprocess(audio,
                                                self.configs.sampling_rate,
                                                batch_index_list,
                                                speed_factor,
                                                split_bucket,
                                                fragment_interval
                                                )

        except Exception as e:
            traceback.print_exc()
            # 必须返回一个空音频, 否则会导致显存不释放。
            yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                            dtype=np.int16)
            # 重置模型, 否则会导致显存释放不完全。
            del self.t2s_model
            del self.vits_model
            self.t2s_model = None
            self.vits_model = None
            self.init_t2s_weights(self.configs.t2s_weights_path)
            self.init_vits_weights(self.configs.vits_weights_path)
            raise e
        finally:
            self.empty_cache()

    def empty_cache(self):
        try:
            gc.collect() # 触发gc的垃圾回收。避免内存一直增长。
            if "cuda" in str(self.configs.device):
                torch.cuda.empty_cache()
            elif str(self.configs.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass

    # def audio_postprocess(self,
    #                       audio:List[torch.Tensor],
    #                       sr:int,
    #                       batch_index_list:list=None,
    #                       speed_factor:float=1.0,
    #                       split_bucket:bool=True,
    #                       fragment_interval:float=0.3
    #                       )->Tuple[int, np.ndarray]:
    #     zero_wav = torch.zeros(
    #                     int(self.configs.sampling_rate * fragment_interval),
    #                     dtype=self.precision,
    #                     device=self.configs.device
    #                 )

    #     for i, batch in enumerate(audio):
    #         for j, audio_fragment in enumerate(batch):
    #             max_audio=torch.abs(audio_fragment).max()#简单防止16bit爆音
    #             if max_audio>1: audio_fragment/=max_audio
    #             audio_fragment:torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0)
    #             audio[i][j] = audio_fragment.cpu().numpy()


    #     if split_bucket:
    #         audio = self.recovery_order(audio, batch_index_list)
    #     else:
    #         # audio = [item for batch in audio for item in batch]
    #         audio = sum(audio, [])


    #     audio = np.concatenate(audio, 0)
    #     audio = (audio * 32768).astype(np.int16)

    #     # try:
    #     #     if speed_factor != 1.0:
    #     #         audio = speed_change(audio, speed=speed_factor, sr=int(sr))
    #     # except Exception as e:
    #     #     print(f"Failed to change speed of audio: \n{e}")

    #     return sr, audio
    def audio_postprocess(
        self,
        audio: List[torch.Tensor],
        sr: int,
        batch_index_list: list = None,
        speed_factor: float = 1.0,
        split_bucket: bool = True,
        fragment_interval: float = 0.3
    ) -> Tuple[int, List[np.ndarray]]:  # 返回每段音频列表
        zero_wav = torch.zeros(
            int(self.configs.sampling_rate * fragment_interval),
            dtype=self.precision,
            device=self.configs.device
        )

        processed_audio = []

        # 遍历每个 batch 和每个 fragment
        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                # 防止16bit爆音
                max_audio = torch.abs(audio_fragment).max()
                if max_audio > 1:
                    audio_fragment /= max_audio

                # 补零
                audio_fragment = torch.cat([audio_fragment, zero_wav], dim=0)

                # 转为 numpy int16
                audio_np = (audio_fragment.cpu().numpy() * 32768).astype(np.int16)
                processed_audio.append(audio_np)

        # 如果需要恢复原始顺序
        if split_bucket and batch_index_list is not None:
            processed_audio = self.recovery_order(processed_audio, batch_index_list)

        return sr, processed_audio




def speed_change(input_audio:np.ndarray, speed:float, sr:int):
    # 将 NumPy 数组转换为原始 PCM 流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio
