from GPT_sovits import get_tts_wav, change_gpt_weights, change_sovits_weights, write_wav
import torch
import time
import json
import random
from collections import OrderedDict
import os
import datetime
# 模型路径
with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
    model_paths = json.load(f)
# 参考音频路径
id_to_refer = {
    "复杂zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\复杂zql_slicer_opt\复杂zql.mp3_0022588800_0022883200.wav",
    "霸总zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\霸总zql_slicer_opt\霸总zql.mp3_0007565120_0007871680.wav",
    "阳光zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\阳光zql_slicer_opt\阳光zql.mp3_0005888640_0006185920.wav",
    "bq": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\bq_slicer_opt\bq.mp4_0000351680_0000650880.wav",
    "lzy": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lzy_slicer_opt\lzy.mp4_0015106240_0015348160.wav",
    "xm": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\xm_slicer_opt\xm.mp4_0003723200_0003876480.wav",
    "lx": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lx_slicer_opt\bilibili_BV1Tc411b7BG_2_MP3.mp4_0000306880_0000608000.wav",
    "v2":r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lx_slicer_opt\bilibili_BV1Tc411b7BG_2_MP3.mp4_0000306880_0000608000.wav",
}

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

def load_model(model_id, model_cache):
    if model_id not in model_cache.cache:
        # 加载模型
        t2s_model = change_gpt_weights(model_paths["GPT"][model_id])
        vq_model = change_sovits_weights(model_paths["SoVITS"][model_id])
        model_cache.add_model(model_id, t2s_model, vq_model)
    return model_cache.get_model(model_id)

def inference(model_id, text, model_cache):
    models = load_model(model_id, model_cache)
    if models is None:
        raise ValueError(f"Model {model_id} not found or failed to load.")
    t2s_model = models['t2s_model']
    vq_model = models['vq_model']
    
    # 锁定模型用于推理
    model_cache.lock_model_for_inference(model_id)
    
    # 重置显存峰值记录
    torch.cuda.reset_peak_memory_stats()
    
    # 进行推理
    start_time = time.time()
    sr, wav = get_tts_wav(
        ref_wav_path=id_to_refer[model_id],
        t2s_model=t2s_model,
        vq_model=vq_model,
        prompt_text=None,
        text=text,
        text_language="中文",
    )
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 获取推理过程中的显存峰值
    peak_memory_allocated = torch.cuda.max_memory_allocated()
    
    # 解锁模型
    model_cache.unlock_model_after_inference(model_id)
    
    return sr, wav, inference_time, peak_memory_allocated

if __name__ == "__main__":
    #测试逻辑
    # 初始化模型缓存
    model_nums = 5
    model_cache = ModelCache(capacity=model_nums)  # 初始缓存容量为3
    # 预加载3个模型
    initial_models = list(model_paths['GPT'].keys())[:model_nums]
    print(initial_models)
    for model_id in initial_models:
        t2s_model = change_gpt_weights(model_paths["GPT"][model_id])
        vq_model = change_sovits_weights(model_paths["SoVITS"][model_id])
        model_cache.add_model(model_id, t2s_model, vq_model)
        test_results = []
        all_model_ids = list(model_paths['GPT'].keys())
    for _ in range(20):
        model_id = random.choice(all_model_ids)
        text = "越长大，重要的人会变得越来越少；剩下的人，也就越来越重要。"
        start_time = time.time()
        sr, wav, inference_time, peak_memory_allocated = inference(model_id, text)
        # 保存生成的音频
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join("./demo/test", f"{model_id}_{timestamp}.wav")
        write_wav(output_path, sr, wav)
        end_time = time.time()
        total_time = end_time - start_time
        new_model_loaded = model_id not in initial_models
        test_results.append({
            "model_id": model_id,
            "inference_time": inference_time,
            "total_time": total_time,
            "new_model_loaded": new_model_loaded,
            "peak_memory_allocated": peak_memory_allocated / (1024 * 1024)  # 转换为 MB
        })

    # 打印测试结果
    log_file = "model_concurrency_test.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        for result in test_results:
            log_entry = (f"Model ID: {result['model_id']}, "
                        f"Inference Time: {result['inference_time']:.4f}s, "
                        f"Total Time: {result['total_time']:.4f}s, "
                        f"New Model Loaded: {result['new_model_loaded']}, "
                        f"Peak Memory Allocated: {result['peak_memory_allocated']:.2f} MB\n")
            f.write(log_entry)
            print(log_entry, end="")

        # 打印模型更新日志
        model_update_log = model_cache.get_model_update_log()
        f.write("\nModel Update Log:\n")
        for log in model_update_log:
            log_entry = f"Evicted Model: {log[0]}, Loaded Model: {log[1]}\n"
            f.write(log_entry)
            print(log_entry, end="")