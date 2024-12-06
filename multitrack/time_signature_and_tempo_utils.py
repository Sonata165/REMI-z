
import os
import sys
sys.path.append('..')
import yaml
from src_hf.utils import *

def main():
    # tide_up_time_signature()
    # test_tempo_converter()
    # generate_tempo_dict()
    test_ts_tempo_converter()


def test_ts_tempo_converter():
    ts = [(4, 4), (3, 4), (2, 4), (6, 8), (9, 8), (12, 8)]
    for t in ts:
        print(t, convert_time_signature_to_ts_token(t[0], t[1]))

    tempo = [60, 88, 96, 108, 128, 180, 200]
    for t in tempo:
        print(t, convert_tempo_to_tempo_token(t))


def convert_time_signature_to_ts_token(numerator, denominator):
    data = read_yaml('/home/longshen/work/MuseCoco/musecoco/midi_utils/ts_dict.yaml')
    valid = False
    for k, v in data.items():
        if v == '({}, {})'.format(numerator, denominator):
            valid = True
            return k
    if not valid:
        raise ValueError('Invalid time signature: {}/{}'.format(numerator, denominator))

def convert_tempo_to_tempo_token(bpm):
    data = read_yaml('/home/longshen/work/MuseCoco/musecoco/midi_utils/tempo_dict.yaml')
    valid = False
    for k, v in data.items():
        v = v[1:-1].split(', ')
        if int(v[0]) <= bpm <= int(v[1]):
            valid = True
            return k
    if not valid:
        raise ValueError('Invalid tempo: {}'.format(bpm))
        


def tide_up_time_signature():
    ts = read_json('/home/longshen/work/MuseCoco/musecoco/midi_utils/ts_dict.json')
    ts_new = {'s-{}'.format(k): '({}, {})'.format(v[0], v[1]) for k, v in ts.items()}
    save_yaml(ts_new, '/home/longshen/work/MuseCoco/musecoco/midi_utils/ts_dict.yaml')

def generate_tempo_dict():
    tempo_dict = {}
    tempo_token_bpm_range = {}
    for bpm in range(10, 280):
        tok = 't-{}'.format(convert_tempo_to_id(bpm))
        if tok not in tempo_token_bpm_range:
            tempo_token_bpm_range[tok] = [999, -1]
        tempo_token_bpm_range[tok][0] = min(tempo_token_bpm_range[tok][0], bpm)
        tempo_token_bpm_range[tok][1] = max(tempo_token_bpm_range[tok][1], bpm)
        # tempo_dict[tok] = bpm
        # print(bpm, tok)
    # print(tempo_token_bpm_range)
    res = {k: '({}, {})'.format(v[0], v[1]) for k, v in tempo_token_bpm_range.items()}
    save_yaml(res, '/home/longshen/work/MuseCoco/musecoco/midi_utils/tempo_dict.yaml')

    # for i in range(0, ct, '/home/longshen/work/MuseCoco/musecoco/midi_utils/tempo_dict.yaml')

def test_tempo_converter():
    print(convert_tempo_to_id(16))
    print(convert_tempo_to_id(256))
    print(convert_tempo_to_id(32))
    print(convert_tempo_to_id(64))
    print(convert_tempo_to_id(128))
    print(convert_tempo_to_id(256))

    for bpm in range(64, 150):
        print(bpm, convert_tempo_to_id(bpm), reverse_convert_tempo_to_id(convert_tempo_to_id(bpm)))
    
    '''
    Tempo token will fall into the range of t-0 ~ t-48
    bpm=16: t-0
    bpm=32: t-12
    bpm=64: t-24
    bpm=128: t-36
    bpm=256: t-48
    
    Starting from 16, bpm x 2 -> t-X + 12
    '''

def convert_tempo_to_id(x):
    import math
    x = max(x, 16) # min_tempo = 16
    x = min(x, 256) # max_tempo = 256
    x = x / 16
    e = round(math.log2(x) * 12) # tempo_quant = 12
    return e

def convert_id_to_tempo(x):
    import math
    e = x / 12
    x = 2 ** e
    x = round(x) * 16
    return x

def reverse_convert_tempo_to_id(e):
    # 反向量化
    log_value = e / 12
    
    # 反向对数变换：求2的对数值的幂，得到原始的缩放值
    scaled_value = 2 ** log_value
    
    # 反向缩放：乘以最小节奏得到原始的BPM值
    x = scaled_value * 16
    
    # 确保结果在原始函数定义的min_tempo和max_tempo之间
    x = max(x, 16)
    x = min(x, 256)
    
    return x

if __name__ == '__main__':
    main()