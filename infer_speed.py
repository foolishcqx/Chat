from GPT_sovits import get_tts_wav, change_gpt_weights, change_sovits_weights, write_wav, pipeline
# import GPT_sovits
import datetime
import os
import json
import random
with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
    model_path = json.load(f)          # 这里就已经是 dict 了
def change_weigth(id):
    gpt_path   = model_path["GPT"].get(id, model_path["GPT"]["v2"])
    sovits_path = model_path["SoVITS"].get(id, model_path["SoVITS"]["v2"])
    vq_model = change_sovits_weights(sovits_path)
    t2s_model = change_gpt_weights(gpt_path)
    return vq_model, t2s_model
# 角色对应的参考音频路径
id_to_refer = {
    "复杂zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\复杂zql_slicer_opt\复杂zql.mp3_0022588800_0022883200.wav",
    "霸总zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\霸总zql_slicer_opt\霸总zql.mp3_0007565120_0007871680.wav",
    "阳光zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\阳光zql_slicer_opt\阳光zql.mp3_0005888640_0006185920.wav",
    "bq": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\bq_slicer_opt\bq.mp4_0000351680_0000650880.wav",
    "lzy": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lzy_slicer_opt\lzy.mp4_0015106240_0015348160.wav",
    "xm": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\xm_slicer_opt\xm.mp4_0003723200_0003876480.wav",
    "lx": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lx_slicer_opt\bilibili_BV1Tc411b7BG_2_MP3.mp4_0000306880_0000608000.wav"
}

# 真实音频文件夹路径
true_audio = {
    "复杂zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\复杂zql_slicer_opt",
    "霸总zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\霸总zql_slicer_opt",
    "阳光zql": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\阳光zql_slicer_opt",
    "bq": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\bq_slicer_opt",
    "lzy": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lzy_slicer_opt",
    "xm": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\xm_slicer_opt",
    "lx": r"D:\lasetTTS\GPT-SoVITS-v3lora-20250228\output\lx_slicer_opt"
}

# 文本列表
text = [
    "越长大，重要的人会变得越来越少；剩下的人，也就越来越重要。",
    "你的色彩，很特别，和其他人都不一样――至少，在我眼里。",
    "或许越过风雪的倔强和坚定，会凝练出令人敬佩的美丽。",
    "如果你将我描述为浪漫，那么我会将你描述为，浪漫的栖息之所。我愿栖息在你聆听到的所有声音里，愿意栖息在你眼中斑斓的颜色里。我愿栖息在你温热的肌肤，皮肤的触感里。愿意栖息在这个，始终将我视作一缕暖光的灵魂里。",
    "不过，适当满足自己的占有欲并不是一件坏事。欲望如果得不到适度的排解，最终会激烈地席卷原本平衡的爱。更何况，占有欲并不是疾病。它伴随着浓烈的爱而来，是人的本能。",
    "以前我觉得，夏天很闷热，冬天太寒冷，认识你之后，我觉得一年四季都过得很快。还没来得及感受到这些，时间就过去了。你让我分心，让我做出错误的判断。我该拿你怎么办才好？",
    "你越强迫自己努力奔跑，越容易没有力气。你可以慢慢地走，知道自己是一直在走就好。",
    "兴许是觉得风和日丽、天气晴好，想要写些悄悄话给你，又不好意思让大海知道。",
    "我的大脑总会不受控制地被你打扰，在第一朵花开的清晨，在暴雨结束的午后，在每一个黎明与黄昏降临，在每一场犹豫与渴慕背后，我无可奈何却又甘之如饴。"
]

# 遍历每个角色
for id in id_to_refer.keys():
    # 创建角色对应的文件夹
    id_folder = os.path.join("./demo", id)
    os.makedirs(id_folder, exist_ok=True)
    
    # 切换权重
    vq_model, t2s_model = change_weigth(id)
    
    # 随机抽取3条真实音频
    true_audio_files = random.sample(os.listdir(true_audio[id]), 3)
    true_audio_paths = [os.path.join(true_audio[id], file) for file in true_audio_files]
    
    # 随机抽取3条文本并生成音频
    selected_texts = random.sample(text, 3)
    generated_audio_paths = []
    for idx, t in enumerate(selected_texts):
        # 获取参考音频路径
        ref_audio = id_to_refer[id]
        
        # 生成音频
        sr, wav = get_tts_wav(
            ref_wav_path=ref_audio,
            vq_model=vq_model,
            t2s_model=t2s_model,
            prompt_text=None,
            text=t,
            text_language="中文",
        )
        
        # 保存生成的音频
        output_path = os.path.join(id_folder, f"generated_{idx}.wav")
        write_wav(output_path, sr, wav)
        generated_audio_paths.append(output_path)
    
    # 将真实音频和生成音频混合并随机打乱
    all_audio_paths = true_audio_paths + generated_audio_paths
    random.shuffle(all_audio_paths)
    
    # 保存混合后的音频
    for idx, audio_path in enumerate(all_audio_paths):
        output_path = os.path.join(id_folder, f"audio{idx + 1}.wav")
        os.rename(audio_path, output_path)
    
    # 生成对应的txt文件记录
    record_file_path = os.path.join(id_folder, "audio_records.txt")
    with open(record_file_path, "w", encoding="utf-8") as f:
        for idx, audio_path in enumerate(all_audio_paths):
            if "generated" in audio_path:
                text_idx = int(audio_path.split("_")[-1].split(".")[0])
                f.write(f"audio{idx + 1}.wav: 生成的音频，对应文本：{selected_texts[text_idx]}\n")
            else:
                f.write(f"audio{idx + 1}.wav: 真实的音频\n")