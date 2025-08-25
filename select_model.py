from GPT_sovits import get_tts_wav, change_gpt_weights, change_sovits_weights, write_wav, pipeline
# import GPT_sovits
import datetime
import os
import json
id_to_refer = {"zql":"","bq":"","lzy":"","xm":"","lx":""}
with open("./GPT_SoVITS/weight.json", "r", encoding="utf-8") as f:
    model_path = json.load(f)          # 这里就已经是 dict 了
def change_weigth(id):
    gpt_path   = model_path["GPT"].get(id, model_path["GPT"]["v2"])
    sovits_path = model_path["SoVITS"].get(id, model_path["SoVITS"]["v2"])
    change_sovits_weights(sovits_path)
    change_gpt_weights(gpt_path)
def chat(id, audio, history):
    change_weigth(id)
    history, output_path = pipeline(audio, history, ref_audio=id_to_refer[id])

