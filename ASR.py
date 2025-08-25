from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import time
import numpy as np
import re
import soundfile as sf
import librosa
import soundfile as sf
import numpy as np

def convert_soundfile_to_librosa_format(audio_path):
    # 使用 soundfile 读取音频文件
    waveform, sample_rate = sf.read(audio_path)

    # 1. 如果音频是多声道的，转换为单声道
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # 2. 确保数据类型为 float32
    waveform = waveform.astype(np.float32)

    # 3. 归一化到 [-1, 1] 范围内
    if waveform.dtype == np.int16:
        waveform = waveform / np.iinfo(np.int16).max

    # 4. 如果需要特定的采样率，进行重采样
    target_sample_rate = 22050  # librosa 的默认采样率
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate

    return waveform, sample_rate

# 初始化 pipeline，使用本地模型
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='./SenseVoiceSmall',
    device="cuda:0",
)

# 读取音频文件
# audio_path = [r'D:\lasetTTS\code\demo\复杂zql\audio1.wav', r"D:\lasetTTS\code\demo\复杂zql\audio3.wav", r'D:\lasetTTS\code\demo\复杂zql\audio1.wav', r"D:\lasetTTS\code\demo\复杂zql\audio3.wav", r'D:\lasetTTS\code\demo\复杂zql\audio1.wav', r"D:\lasetTTS\code\demo\复杂zql\audio3.wav"]
id_to_refer = {
    "复杂zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\复杂zql_slicer_opt\复杂zql.mp3_0022588800_0022883200.wav",
    "霸总zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\霸总zql_slicer_opt\霸总zql.mp3_0007565120_0007871680.wav",
    "阳光zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\阳光zql_slicer_opt\阳光zql.mp3_0005888640_0006185920.wav",
    "bq": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\bq_slicer_opt\bq.mp4_0000351680_0000650880.wav",
    "lzy": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lzy_slicer_opt\lzy.mp4_0015106240_0015348160.wav",
    "xm": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\xm_slicer_opt\xm.mp4_0003723200_0003876480.wav",
    "lx": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lx_slicer_opt\bilibili_BV1Tc411b7BG_2_MP3.mp4_0000306880_0000608000.wav"
}
# audio_path = []
# for key, value in id_to_refer.items():
#     audio_path.append(value)
#     print(key)
audio_path = [r'D:\lasetTTS\code\recv.wav',r"D:\lasetTTS\code\demo\bq\audio1.wav"]
# 开始时间
start = time.time()
# waveform, sample_rate = librosa.load(audio_path,sr=None) 

# waveform, sample_rate = sf.read(audio_path)
# waveform2, sample_rate = librosa.load(audio_path) 
# 调用 pipeline 进行语音识别
rec_result = inference_pipeline(audio_path)
print(rec_result)
pattern = r"<\|(.+?)\|><\|(.+?)\|><\|(.+?)\|><\|(.+?)\|>(.+)"
match = re.match(pattern, rec_result[0]['text'])
language, emotion, audio_type, itn, text = match.groups()
# 结束时间
end = time.time()

# 输出识别结果和耗时
print("识别结果:", text)
print("耗时:", end - start, "秒")