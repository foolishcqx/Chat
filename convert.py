import numpy as np
import wave
from pathlib import Path

def wav2pcm(wavfile, pcmfile, data_type=np.int16):
    with wave.open(str(wavfile), 'rb') as w:
        if w.getcomptype() != 'NONE':
            raise ValueError('仅支持无压缩 PCM')
        # 获取采样宽度，再决定 dtype
        width = w.getsampwidth()
        dtype_map = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}
        if width not in dtype_map:
            raise ValueError(f'不支持的位深: {width*8}-bit')
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=dtype_map[width])
    data.astype(data_type).tofile(pcmfile)
    print('已生成', pcmfile)

# 直接使用
wav2pcm('./demo/bq/audio4.wav', 'test.pcm')