import os
import math
import yaml
import typing
import pickle
import miditoolkit
from copy import deepcopy
from remi_z import keys_normalization


class MidiEncoder(object):
    def __init__(self):
        key_profile_fp = os.path.join(os.path.dirname(__file__), 'key_profile.pickle')
        # key_profile_fp = '/Users/sonata/Code/REMI-z/multitrack/legacy/midiprocessor/key_profile.pickle'
        with open(key_profile_fp, 'rb') as f:
            self.key_profile = pickle.load(f) # [24, 12], former [12,12] for major, latter 12 for minor

        # Apply weighting to key_profile
        major_weight = [2, 0, 1.0, 0, 1.5, 1.0, 0, 1.5, 0, 1, 0, 1]
        minor_weight = [2, 0, 1, 1.5, 0, 1.0, 0, 1.5, 1.0, 0, 1.0, 0]

        adjusted_key_profile = deepcopy(self.key_profile).astype('float64')

        for i in range(24):
            if i < 12:
                # circular right shift major weight
                shifted_major_weight = major_weight[-i:] + major_weight[:-i]

                # Position-wise multiplication
                for k in range(12):
                    adjusted_key_profile[i][k] *= shifted_major_weight[k]
            else:
                j = i - 12
                # circular shift minor weight
                shifted_minor_weight = minor_weight[-j:] + minor_weight[:-j]

                # Position-wise multiplication
                for k in range(12):
                    adjusted_key_profile[i][k] *= shifted_minor_weight[k]
        self.key_profile = adjusted_key_profile

        self.DEFAULT_TS = (4, 4)
        self.DEFAULT_TEMPO = 120.0

        

    def time_to_pos(self, t, ticks_per_beat):
        pos_resolution = 12 # 12 positions per beat
        ret = round(t * pos_resolution / ticks_per_beat)
        return ret

    def reduce_time_signature(self, numerator, denominator):
        max_ts_denominator = 2 ** 6
        while denominator > max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
            denominator //= 2
            numerator //= 2
        # decomposition (when length of a bar exceed max_notes_per_bar)
        max_notes_per_bar = 2
        while numerator > max_notes_per_bar * denominator:
            for i in range(2, numerator + 1):
                if numerator % i == 0:
                    numerator //= i
                    break
        return numerator, denominator

    def collect_pos_info(self, midi_obj, trunc_pos=None, tracks=None, remove_same_notes=False, end_offset=0):
        if tracks is not None:
            from collections.abc import Iterable
            assert isinstance(tracks, (int, Iterable))
            if isinstance(tracks, str):
                tracks = int(tracks)
            if isinstance(tracks, int):
                if tracks < 0:
                    tracks = len(midi_obj.instruments) + tracks
                tracks = (tracks,)

        max_pos = 0
        for inst in midi_obj.instruments:
            for note in inst.notes:
                pos = self.time_to_pos(note.start, midi_obj.ticks_per_beat)
                max_pos = max(max_pos, pos)
        max_pos = max_pos + 1  # 最大global position
        if trunc_pos is not None:
            max_pos = min(max_pos, trunc_pos)

        pos_info = [
            [None, None, None, None, None]  # (bar, ts, local_pos, tempo, insts_notes)
            for _ in range(max_pos)
        ]
        pos_info: typing.List
        # bar: every pos
        # ts: only at pos where it changes, otherwise None
        # local_pos: every pos
        # tempo: only at pos where it changes, otherwise None
        # insts_notes: only at pos where the note starts, otherwise None

        ts_changes = midi_obj.time_signature_changes
        zero_pos_ts_change = False
        for ts_change in ts_changes:
            pos = self.time_to_pos(ts_change.time, midi_obj.ticks_per_beat)
            if pos >= max_pos:
                continue
            if pos == 0:
                zero_pos_ts_change = True
            ts_numerator = int(ts_change.numerator)
            ts_denominator = int(ts_change.denominator)
            
            ts_numerator, ts_denominator = self.reduce_time_signature(ts_numerator, ts_denominator)
            pos_info[pos][1] = (ts_numerator, ts_denominator)
        if not zero_pos_ts_change:
            pos_info[0][1] = self.DEFAULT_TS

        tempo_changes = midi_obj.tempo_changes
        zero_pos_tempo_change = False
        for tempo_change in tempo_changes:
            pos = self.time_to_pos(tempo_change.time, midi_obj.ticks_per_beat)
            if pos >= max_pos:
                continue
            if pos == 0:
                zero_pos_tempo_change = True
            pos_info[pos][3] = tempo_change.tempo
        if not zero_pos_tempo_change:
            pos_info[0][3] = self.DEFAULT_TEMPO

        insts = midi_obj.instruments
        for inst_idx, inst in enumerate(insts):
            if tracks is not None and inst_idx not in tracks:
                continue
            # if self.ignore_insts:
            #     inst_id = 0
            # else:
            inst_id = 128 if inst.is_drum else int(inst.program)
            notes = inst.notes
            for note in notes:
                pitch = int(note.pitch)
                velocity = int(note.velocity)
                start_time = int(note.start)
                end_time = int(note.end + end_offset)
                # assert end_time > start_time
                if end_time < start_time:
                    print('Warning: end_time < start_time')
                pos_start = self.time_to_pos(start_time, midi_obj.ticks_per_beat)
                pos_end = self.time_to_pos(end_time, midi_obj.ticks_per_beat)
                duration = pos_end - pos_start

                if pos_info[pos_start][4] is None:
                    pos_info[pos_start][4] = dict()
                if inst_id not in pos_info[pos_start][4]:
                    pos_info[pos_start][4][inst_id] = []
                note_info = [pitch, duration, velocity]
                if remove_same_notes:
                    if note_info in pos_info[pos_start][4][inst_id]:
                        continue
                pos_info[pos_start][4][inst_id].append([pitch, duration, velocity])

        cnt = 0
        bar = 0
        measure_length = None
        ts = self.DEFAULT_TS  # default MIDI time signature
        for j in range(max_pos):
            now_ts = pos_info[j][1]
            if now_ts is not None:
                if now_ts != ts:
                    ts = now_ts
            if cnt == 0:
                beat_note_factor = 4
                pos_resolution = 12
                measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
            pos_info[j][0] = bar
            pos_info[j][2] = cnt
            cnt += 1
            if cnt >= measure_length:
                assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(j)
                cnt = 0
                bar += 1

        return pos_info

    def normalize_pitch(self, pos_info):
        assert self.key_profile is not None, "Please load key_profile first, using load_key_profile method."

        pitch_shift, is_major, _, _ = keys_normalization.get_pitch_shift(
            pos_info,
            self.key_profile,
            normalize=True, use_duration=True, use_velocity=True,
            ensure_valid_range=True
        )
        pitch_shift = int(pitch_shift)
      
        return is_major, pitch_shift





def load_midi(file_path=None, file=None, midi_checker='default'):
    """
    Open and check MIDI file, return MIDI object by miditoolkit.
    :param file_path:
    :param midi_checker:
    :return:
    """
    midi_obj = miditoolkit.midi.parser.MidiFile(filename=file_path, file=file)

    if midi_checker is not None and midi_checker != 'none':
        if isinstance(midi_checker, str):
            if midi_checker == 'default':
                midi_checker = default_check_midi
            else:
                raise ValueError("midi checker does not support value: %s" % midi_checker)

        midi_checker(midi_obj)

    return midi_obj

def default_check_midi(midi_obj):
    # check abnormal values in parse result
    max_time_length = 2 ** 31
    assert all(0 <= j.start < max_time_length
               and 0 <= j.end < max_time_length
               for i in midi_obj.instruments for j in i.notes), 'Bad note time'
    assert all(0 < j.numerator < max_time_length and 0 < j.denominator < max_time_length for j in
               midi_obj.time_signature_changes), 'Bad time signature value'
    assert 0 < midi_obj.ticks_per_beat < max_time_length, 'Bad ticks per beat'

    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    assert midi_notes_count > 0, 'Blank note.'


def fill_pos_ts_and_tempo_(pos_info):
    '''
    Original item of pos_info: [bar, ts, local_pos, tempo, insts_notes]
    '''
    cur_ts = pos_info[0][1]
    cur_tempo = pos_info[0][3]
    assert cur_ts is not None
    assert cur_tempo is not None
    for idx in range(len(pos_info)):
        pos_item = pos_info[idx]
        if pos_item[1] is not None:
            cur_ts = pos_item[1]
        if pos_item[3] is not None:
            cur_tempo = pos_item[3]
        if pos_item[2] == 0:
            if pos_item[1] is None:
                pos_item[1] = cur_ts
            if pos_item[3] is None:
                pos_item[3] = cur_tempo
    return pos_info


def convert_tempo_to_id(x):
    '''
    Return: e, int, tempo id.
    '''
    x = max(x, 16) # min_tempo = 16
    x = min(x, 256) # max_tempo = 256
    tempo_quant = 12
    x = x / 16
    e = round(math.log2(x) * tempo_quant)
    return e


def convert_id_to_tempo(self, x):
    return 2 ** (x / self.tempo_quant) * self.min_tempo


