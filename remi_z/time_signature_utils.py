
import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(__file__))

from utils import read_yaml


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


class TimeSignatureUtil:
    @classmethod
    def convert_time_signature_to_ts_token(cls, numerator, denominator):
        if hasattr(cls, 'ts_dict'):
            data = cls.ts_dict  
        else:
            ts_fp = os.path.join(os.path.dirname(__file__), 'dict_time_signature.yaml')
            data = read_yaml(ts_fp)
            cls.ts_dict = data

        valid = False
        for k, v in data.items():
            if v == '({}, {})'.format(numerator, denominator):
                valid = True
                return k
            
        if not valid:
            raise ValueError('Invalid time signature: {}/{}'.format(numerator, denominator))
        
    @classmethod
    def convert_time_signature_token_to_tuple(cls, token):
        if hasattr(cls, 'ts_dict'):
            data = cls.ts_dict  
        else:
            ts_fp = os.path.join(os.path.dirname(__file__), 'dict_time_signature.yaml')
            data = read_yaml(ts_fp)
            cls.ts_dict = data

        if token in data:
            ret = data[token]
            t1, t2 = ret[1:-1].split(',')
            ret = int(t1), int(t2)
            return ret
        else:
            raise ValueError('Invalid time signature token: {}'.format(token))


def convert_time_signature_to_ts_token(numerator, denominator):
    ts_fp = os.path.join(os.path.dirname(__file__), 'dict_time_signature.yaml')
    data = read_yaml(ts_fp)
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



if __name__ == '__main__':
    main()