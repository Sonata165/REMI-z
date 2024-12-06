import os
import sys
sys.path.append("..")

import random
import numpy as np
# from utils_texture.texture_tools import (
#     get_time_function_from_remi_one_bar,
#     get_onset_density_of_a_bar_from_remi,
#     tokenize_onset_density_one_bar,
# )

from typing import List, Tuple, Dict
# from utils_instrument.inst_map import InstMapUtil
# from utils_chord.chord_detect_from_remi import generate_chord_notes


def from_remi_z_to_remi_plus(remi_seq):
    '''
    Convert a remi sequence to remi+ sequence
    '''
    # note_seq = []
    note_dic = {}
    pre_pos = None
    pre_inst = None
    find_p = False
    find_d = False
    for tok in remi_seq:
        if tok.startswith('o-'):
            cur_pos = tok
        elif tok.startswith('i-'):
            cur_inst = tok
        elif tok.startswith('p-'):
            cur_pitch = tok
            find_p = True
        elif tok.startswith('d-'):
            cur_dur = tok
            find_d = True

            if find_p and find_d:
                if cur_pos not in note_dic:
                    note_dic[cur_pos] = []
                note_dic[cur_pos].append((cur_inst, cur_pitch, cur_dur))

    # Sort the notes in same position by pitch id, reverse
    for pos in note_dic:
        note_dic[pos] = sorted(note_dic[pos], key=lambda x: int(x[1].split('-')[1]), reverse=True)

    # Sort note dic by position
    note_dic = dict(sorted(note_dic.items(), key=lambda x: int(x[0].split('-')[1])))

    remi_plus_seq = []
    for pos in note_dic:
        remi_plus_seq.append(pos)
        for note in note_dic[pos]:
            remi_plus_seq.append(note[0])
            remi_plus_seq.append(note[1])
            remi_plus_seq.append(note[2])

    # bar token
    remi_plus_seq.append('b-1')

    return remi_plus_seq


def from_remi_get_pitch_range(remi_seq):
    '''
    Get the pitch range of a remi sequence
    '''
    pitch_seq = [tok for tok in remi_seq if tok.startswith('p-')]
    pitch_ids = [int(tok.split('-')[1]) for tok in pitch_seq]
    pitch_range = max(pitch_ids) - min(pitch_ids) + 1
    return pitch_range 


def in_remi_shift_pitch_random(remi_seq):
    # Determine how many semitones to shift [-6, 6]
    shift = random.randint(-6, 6)

    ret = []
    for tok in remi_seq:
        if tok.startswith('p-'):
            pitch_id = int(tok.split('-')[1])
            pitch_id += shift
            pitch_id = max(0, pitch_id)
            pitch_id = min(127, pitch_id)
            ret.append('p-{}'.format(pitch_id))
        else:
            ret.append(tok)

    return ret


def from_remi_bar_get_two_chroma_feat_seq(remi_seq):
    '''
    compress all pitch tokens to the first octave.
    For previous half bar and later half bar

    '''
    pitch_of_pos_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=True)
    # 0, 12, 24, 36, 48
    pitch_seq_first_half = set()
    pitch_seq_later_half = set()
    for pos in pitch_of_pos_dict:
        if int(pos.split('-')[1]) < 24:
            pitch_seq_first_half.update(pitch_of_pos_dict[pos])
        else:
            pitch_seq_later_half.update(pitch_of_pos_dict[pos])

    pitch_seq_first_half = list(pitch_seq_first_half)
    pitch_seq_later_half = list(pitch_seq_later_half)

    # Convert to first octave
    pitch_seq_first_half = [convert_pitch_token_to_first_octave(tok) for tok in pitch_seq_first_half]
    pitch_seq_later_half = [convert_pitch_token_to_first_octave(tok) for tok in pitch_seq_later_half]

    # Deduplicate
    pitch_seq_first_half = list(set(pitch_seq_first_half))
    pitch_seq_later_half = list(set(pitch_seq_later_half))

    # Sort by pitch id
    # ret = sorted(ret, key=lambda x: int(x.split('-')[1]))
    pitch_seq_first_half = sorted(pitch_seq_first_half, key=lambda x: int(x.split('-')[1]))
    pitch_seq_later_half = sorted(pitch_seq_later_half, key=lambda x: int(x.split('-')[1]))

    return pitch_seq_first_half, pitch_seq_later_half


def from_remi_get_chroma_seq(remi_seq):
    '''
    compress all pitch tokens to the first octave.

    Deduplicate by default.
    '''
    ret = set()
    for tok in remi_seq:
        if tok.startswith('p-'):
            ret.add(convert_pitch_token_to_first_octave(tok))
    ret = list(ret)
    
    # Sort by pitch id
    ret = sorted(ret, key=lambda x: int(x.split('-')[1]))

    return ret


def from_remi_get_opd_seq_per_track(remi_seq, sort_by_avg_pitch=False, include_drum=True):
    '''
    NOTE: only works for a bar

    Extract all track-wise remi sequences from the multi track remi
    Return a dict of seqs for each track, key is instrument token
    Keys sort by program ID
    '''
    # Obtain all instruments of the bar
    insts = from_remi_get_insts(remi_seq, sort_inst=True) # All instruments, sorted by program id

    # Remove drum if required
    if include_drum is False:
        insts = [inst for inst in insts if inst != 'i-128']

    # Obtain track for each instrument
    ret = {}
    for inst in insts:
        track_seq = from_remi_get_opd_seq_of_track(remi_seq, inst)
        ret[inst] = track_seq

    if sort_by_avg_pitch is True:
        # Sort by average pitch of each track
        avg_pitch_of_track = {}
        has_drum = False
        for inst in ret:
            if inst == 'i-128':
                has_drum = True
                continue
            opd_seq = ret[inst]
            pitch_seq = [tok for tok in opd_seq if tok.startswith('p-')]
            avg_pitch = sum([int(tok.split('-')[1]) for tok in pitch_seq]) / len(pitch_seq)
            avg_pitch_of_track[inst] = avg_pitch

        insts = list(avg_pitch_of_track.keys())
        insts = sorted(insts, key=lambda x: avg_pitch_of_track[x], reverse=True)
        if has_drum:
            insts.append('i-128')
        ret = {inst: ret[inst] for inst in insts}

    return ret


def from_remi_mbar_get_opd_seq_of_track(remi_seq, inst):
    bar_indices = from_remi_get_bar_idx(remi_seq)
    ret = []
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_seq = remi_seq[bar_start_idx:bar_end_idx]
        t = from_remi_get_opd_seq_of_track(bar_seq, inst)
        if len(t) > 0:
            ret.append(inst)
            ret.extend(t)
        ret.append('b-1')
    return ret


def from_remi_get_opd_seq_of_track(remi_seq, inst):
    '''
    NOTE: only works for a bar

    Extract the track-wise remi sequence for a given instrument
    From a multi-track remi sequence
    '''
    ret = []
    in_seq = False
    cur_pos = 'o-0'
    cur_inst = None
    cur_pitch = None
    cur_dur = None
    pre_pos = None
    find_p, find_d = False, False
    for tok in remi_seq:
        if tok.startswith('o-'):
            cur_pos = tok
        elif tok.startswith('i-'):
            cur_inst = tok
        elif tok.startswith('p-'):
            cur_pitch = tok
            find_p = True
        elif tok.startswith('d-'):
            cur_dur = tok
            find_d = True

            if find_p and find_d:
                if cur_inst == inst:
                    if cur_pos != pre_pos:
                        ret.append(cur_pos)
                    ret.append(cur_pitch)
                    ret.append(cur_dur)

                    pre_pos = cur_pos

    return ret


def from_remi_get_opd_dict_of_track(remi_seq, inst):
    '''
    NOTE: only works for a bar

    Extract the track-wise remi sequence for a given instrument
    From a multi-track remi sequence
    '''
    ret = {}
    in_seq = False
    cur_pos = 'o-0'
    cur_inst = None
    cur_pitch = None
    cur_dur = None
    pre_pos = None
    find_p, find_d = False, False
    for tok in remi_seq:
        if tok.startswith('o-'):
            cur_pos = tok
        elif tok.startswith('i-'):
            cur_inst = tok
        elif tok.startswith('p-'):
            cur_pitch = tok
            find_p = True
        elif tok.startswith('d-'):
            cur_dur = tok
            find_d = True

            if find_p and find_d:
                if cur_inst == inst:
                    if cur_pos != pre_pos:
                        ret[cur_pos] = []
                    ret[cur_pos].append(cur_pitch)
                    ret[cur_pos].append(cur_dur)

                    pre_pos = cur_pos

    return ret


def from_remi_get_global_opd_seq(remi_seq):
    '''
    NOTE: only works for a bar
    '''
    opd_dict = from_remi_get_global_opd_dict(remi_seq)
    
    ret = []
    
    # sort pos
    pos_list = list(opd_dict.keys())
    pos_list = sorted(pos_list, key=lambda x: int(x.split('-')[1]))

    for pos in pos_list:
        ret.append(pos)
        ret.extend(opd_dict[pos])
    
    return ret


def from_remi_mbar_get_global_opd_seq(remi_seq):
    bar_indices = from_remi_get_bar_idx(remi_seq)
    ret = []
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_seq = remi_seq[bar_start_idx:bar_end_idx]
        t = from_remi_get_global_opd_seq(bar_seq)

        ret.extend(t)

        ret.append('b-1')
    return ret


def from_remi_get_global_opd_dict(remi_seq):
    '''
    NOTE: only works for a bar
    NOTE: drum is not included

    Extract the track-wise remi sequence for a given instrument
    From a multi-track remi sequence

    Notes in same position are merged, duration is the longest one
    '''
    ret = {}
    cur_pos = 'o-0'
    cur_pitch = None
    cur_dur = None
    pre_pos = None
    find_p, find_d = False, False
    for tok in remi_seq:
        if tok.startswith('o-'):
            cur_pos = tok
        elif tok.startswith('p-'):
            cur_pitch = tok
            find_p = True
        elif tok.startswith('d-'):
            cur_dur = tok
            find_d = True

            if find_p and find_d:
                if int(cur_pitch.split('-')[1]) >= 128:
                    continue
                if cur_pos not in ret:
                    ret[cur_pos] = {}
                if cur_pitch not in ret[cur_pos]:
                    ret[cur_pos][cur_pitch] = cur_dur
                else:
                    # If current duration is longer than the previous one, update
                    if int(cur_dur.split('-')[1]) > int(ret[cur_pos][cur_pitch].split('-')[1]):
                        ret[cur_pos][cur_pitch] = cur_dur

    # Flatten the inner dict
    for pos in ret:
        pd_seq = []
        pitches = list(ret[pos].keys())
        pitches = sorted(pitches, key=lambda x: int(x.split('-')[1]), reverse=True)
        for pitch in pitches:
            pd_seq.append(pitch)
            pd_seq.append(ret[pos][pitch])
        ret[pos] = pd_seq

    return ret


def remi_read_from_file(fp, split=True, remove_input=False):
    with open(fp) as f:
        remi_str = f.readline().strip()

    if remove_input:
        remi_str = remi_str.split(' <sep> ')[1]

    if split:
        remi_seq = remi_str.split(' ')
    else:
        remi_seq = remi_str
    return remi_seq

def remi_seq_save_to_file(remi_seq, fp):
    with open(fp, 'w') as f:
        f.write(' '.join(remi_seq))


def in_remi_multi_bar_replace_bar_tokens(remi_seq):
    '''
    Replace the bar line token from only b-1 to b-1, b-2, b-3, ...
    '''
    bar_indices = from_remi_get_bar_idx(remi_seq)
    for bar_id in bar_indices:
        bar_cnt = bar_id + 1
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        remi_seq[bar_end_idx-1] = 'b-{}'.format(bar_cnt)

    return remi_seq


def from_pitch_token_get_pitch_id(tok_p):
    '''
    Get pitch id in integer from pitch token
    '''
    return int(tok_p.split("-")[1])


def from_remi_get_bar_idx(remi_seq):
    # Get the starting token of each bar
    start_token_index_of_the_bar = 0
    bar_id = 0
    bar_indices = {}

    # bars_token_positions[bar_id] = (start token index of this bar, start token index of next bar)
    for idx, token in enumerate(remi_seq):
        if token == "b-1":
            start_token_index_of_next_bar = idx + 1
            bar_indices[bar_id] = (
                start_token_index_of_the_bar,
                start_token_index_of_next_bar,
            )

            # Go to the next bar
            start_token_index_of_the_bar = start_token_index_of_next_bar
            bar_id = bar_id + 1
    return bar_indices

def from_condition_get_bar_idx(condition_seq):
    '''
    Because in the condition, each bar is segmented also by the 'b-1' token, 
    Just call the get_bar_idx_from_remi function to do the work.
    '''
    ret = from_remi_get_bar_idx(condition_seq)
    return ret


def from_target_bar_obtain_features(out_seq):
    """
    Do the entire feature extraction + tokenization (convert to token format) process for sss model
    Obtain the input sequence from target remi sequence

    Prerequisite: output is one bar
    """
    # Get instrument
    inst_seq = from_remi_get_insts(out_seq)

    # Get pitch seq (without inst and duration info)
    pitch_seq_info = from_remi_get_pitch_of_pos_dict(out_seq, sort_pitch=True)
    pitch_seq = []
    for pos in pitch_seq_info:
        pitch_seq.append(pos)
        pitch_seq.extend(pitch_seq_info[pos])

    ret = {
        'pitch_seq': pitch_seq,
        'inst_seq': inst_seq,
    }    

    return ret

# def from_remi_get_melody_inst(remi_seq):
#     ''' Melody Keeping '''
#         if keep_melody is True:
#             mel_inst = self.__get_melody_inst(tgt_remi_seq, insts)
#             if mel_inst in insts_to_pred:
#                 insts_to_pred.remove(mel_inst)

def from_remi_get_condition_seq(
        remi_seq, 
        hist=False, 
        voice=False, 
        texture=False, 
        flatten_content=False, 
        remove_drum=False, 
        reorder=False,
    ):
    """
    Do the entire feature extraction + tokenization (convert to token format) process for sss model
    Obtain the input sequence from target remi sequence

    Prerequisite: remi_seq contains info strictly for two bars
    """
    b_1_indices = from_remi_get_bar_idx(remi_seq)
    num_bars = len(b_1_indices)
    
    if num_bars != 2:
        # To support inference with nohist model
        if num_bars == 1 and hist is False: # If only target bar is given
            remi_seq.insert(0, 'b-1')
            b_1_indices = from_remi_get_bar_idx(remi_seq)
        else:
            raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    # Get the target bar
    tgt_start_idx, tgt_end_idx = b_1_indices[1]
    tgt_remi_seq = remi_seq[tgt_start_idx:tgt_end_idx]
    
    # Remove empty positions
    tgt_remi_seq = in_remi_remove_empty_pos(tgt_remi_seq)
    # Drum removal
    if remove_drum is True:
        tgt_remi_seq = from_remi_bar_remove_drum(tgt_remi_seq)

    # Get the raw history bar
    hist_start_idx, hist_end_idx = b_1_indices[0]
    hist_seq = remi_seq[hist_start_idx:hist_end_idx]
    # Remove empty positions
    hist_seq = in_remi_remove_empty_pos(hist_seq)
    # Drum removal
    if remove_drum is True:
        hist_seq = from_remi_bar_remove_drum(hist_seq)

    # Get instrument (and possibly voice and texture)
    if voice is True and texture is True:
        # 1/5 of chance only get inst and voice, no texture
        # if random.random() < 0.2:
        #     inst_seq = from_remi_get_inst_and_voice(tgt_remi_seq)
        # else:
        inst_seq = from_remi_get_inst_voice_texture(tgt_remi_seq)
    elif voice is True and texture is False:
        inst_seq = from_remi_get_inst_and_voice(tgt_remi_seq)
    elif voice is False and texture is True:
        inst_seq = from_remi_get_inst_and_texture(tgt_remi_seq)
    else:
        inst_seq = from_remi_get_insts(tgt_remi_seq)

    # Get pitch seq (without inst and duration info)
    pos_pitch_seq_dict = from_remi_get_pitch_of_pos_dict(tgt_remi_seq, sort_pitch=True, flatten=flatten_content)
    pos_pitch_seq = []
    for pos in pos_pitch_seq_dict:
        pos_pitch_seq.append(pos)
        pos_pitch_seq.extend(pos_pitch_seq_dict[pos])
        
    # Get history
    if hist is not False:
        
        hist_seq = from_remi_hist_refine_history(
            hist_seq, 
            tgt_insts=[tok for tok in inst_seq if tok.startswith('i-')], # preserve the voice info in inst list
            hist_type=hist, 
            reorder_tgt=reorder,
            voice_control=voice,
        )

    ''' Tokenization for the segment '''
    condition_seq = []

    condition_seq.append('PITCH')
    condition_seq.extend(pos_pitch_seq)
    condition_seq.append("INS")
    condition_seq.extend(inst_seq)
    if hist is not False:
        condition_seq.append('HIST')
        condition_seq.extend(hist_seq)
    # no additional b-1 token

    return condition_seq, tgt_remi_seq

def in_remi_remove_empty_pos(remi_seq):
    ret = []
    for i, tok in enumerate(remi_seq):
        if tok.startswith('o-'):
            if i+1 < len(remi_seq) and (remi_seq[i+1].startswith('p-') or remi_seq[i+1].startswith('i-')):
                ret.append(tok)
        else:
            ret.append(tok)
    return ret

def from_remi_get_chord_seq(remi_seq):
    '''
    NOTE: only works for one bar
    NOTE: only accumulate chords at fixed position
    
    Detect the chord sequence from the remi sequence
    '''
    from utils_chord.chord_detect_from_remi import chord_detection_with_root
    pos_and_pitch_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=True)

    # Important position: 0, 12, 24, 36
    accumulate_pos = [0, 12, 24, 36, 48]

    # Get all pitch tokens, whose position is between each pair of important positions
    accumulated_pitch_tokens = [[], [], [], []]
    for pos in pos_and_pitch_dict:
        pos_id = int(pos.split('-')[1])
        for i in range(4):
            if pos_id >= accumulate_pos[i] and pos_id < accumulate_pos[i+1]:
                accumulated_pitch_tokens[i].extend(pos_and_pitch_dict[pos])
                break
    
    # Get the lowest note in each accumulated pitch tokens
    suspect_root_notes = [None, None, None, None]
    for i in range(4):
        if len(accumulated_pitch_tokens[i]) == 0:
            continue
        # Get pitch token with smallest pitch id from accumulated_pitch_tokens[i]
        suspect_root_notes[i] = min(accumulated_pitch_tokens[i], key=lambda x: int(x.split('-')[1]))

    # Detect chord for each segment
    chord_seq = []
    for i in range(4):
        note_list = [convert_pitch_token_to_first_octave(j) for j in accumulated_pitch_tokens[i]]
        root_note = convert_pitch_token_to_first_octave(suspect_root_notes[i]) if suspect_root_notes[i] is not None else None
        chord_seq.append(chord_detection_with_root(note_list, root_note, return_root_name=True))

    return chord_seq


def from_remi_get_chord_seq_2chord_1bar(remi_seq):
    '''
    NOTE: only works for one bar
    NOTE: only accumulate chords at fixed position
    
    Detect the chord sequence from the remi sequence
    '''
    from utils_chord.chord_detect_from_remi import chord_detection_with_root
    pos_and_pitch_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=True)

    # Important position: 0, 24
    accumulate_pos = [0, 24, 48]

    # Get all pitch tokens, whose position is between each pair of important positions
    accumulated_pitch_tokens = [[], []]
    for pos in pos_and_pitch_dict:
        pos_id = int(pos.split('-')[1])
        for i in range(2):
            if pos_id >= accumulate_pos[i] and pos_id < accumulate_pos[i+1]:
                accumulated_pitch_tokens[i].extend(pos_and_pitch_dict[pos])
                break
    
    # Get the lowest note in each accumulated pitch tokens
    suspect_root_notes = [None, None]
    for i in range(2):
        if len(accumulated_pitch_tokens[i]) == 0:
            continue
        # Get pitch token with smallest pitch id from accumulated_pitch_tokens[i]
        suspect_root_notes[i] = min(accumulated_pitch_tokens[i], key=lambda x: int(x.split('-')[1]))

    # Detect chord for each segment
    chord_seq = []
    for i in range(2):
        note_list = [convert_pitch_token_to_first_octave(j) for j in accumulated_pitch_tokens[i]]
        root_note = convert_pitch_token_to_first_octave(suspect_root_notes[i]) if suspect_root_notes[i] is not None else None
        chord_seq.append(chord_detection_with_root(note_list, root_note, return_root_name=True))

    return chord_seq


def from_remi_get_chord_seq_two_chord_a_bar(remi_seq):
    '''
    NOTE: only works for one bar
    NOTE: only accumulate chords at fixed position
    
    Detect the chord sequence from the remi sequence
    '''
    from utils_chord.chord_detect_from_remi import chord_detection_with_root
    pos_and_pitch_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=True)

    # Important position: 0, 12, 24, 36
    accumulate_pos = [0, 24, 48]

    # Get all pitch tokens, whose position is between each pair of important positions
    accumulated_pitch_tokens = [[], []]
    for pos in pos_and_pitch_dict:
        pos_id = int(pos.split('-')[1])
        for i in range(2):
            if pos_id >= accumulate_pos[i] and pos_id < accumulate_pos[i+1]:
                accumulated_pitch_tokens[i].extend(pos_and_pitch_dict[pos])
                break
    
    # Get the lowest note in each accumulated pitch tokens
    suspect_root_notes = [None, None]
    for i in range(2):
        if len(accumulated_pitch_tokens[i]) == 0:
            continue
        # Get pitch token with smallest pitch id from accumulated_pitch_tokens[i]
        suspect_root_notes[i] = min(accumulated_pitch_tokens[i], key=lambda x: int(x.split('-')[1]))

    # Detect chord for each segment
    chord_seq = []
    for i in range(2):
        note_list = [convert_pitch_token_to_first_octave(j) for j in accumulated_pitch_tokens[i]]
        root_note = convert_pitch_token_to_first_octave(suspect_root_notes[i]) if suspect_root_notes[i] is not None else None
        chord_seq.append(chord_detection_with_root(note_list, root_note, return_root_name=True))

    return chord_seq

def random_get_insts_list(remove_drum=False, sort_insts=False):
    inst_utils = InstMapUtil()
    supported_insts = inst_utils.slakh_get_supported_prog_ids()
    supported_insts = ['i-{}'.format(inst) for inst in supported_insts]

    if remove_drum is True:
        supported_insts.remove('i-128')

    # determine number of instruments (sample from lambda=5 possion)
    num_insts = np.random.poisson(5)
    num_insts = min(num_insts, len(supported_insts))
    num_insts = max(1, num_insts)

    insts = random.sample(supported_insts, num_insts)

    if sort_insts:
        insts = sorted(insts, key=lambda x: int(x.split("-")[1]))

    return insts


def from_remi_get_insts(remi_seq, sort_inst=True):
    '''
    Obtain all instrument in the input remi sequence
    Return a list of instrument tokens
    Sort by program id
    '''
    inst = set()
    for token in remi_seq:
        if token.startswith("i-"):
            inst.add(token)
    inst = list(inst)
    
    if sort_inst:
        inst = sorted(inst, key=lambda x: int(x.split("-")[1]))  # sort by inst id

    return inst


def from_remi_get_inst_and_texture(remi_seq):
    '''
    Obtain all instrument in the input remi sequence
    Return a list of instrument tokens
    Instrument on the left have higher average pitch than that on the right

    The instrument sequence contain texture control.
    - Three types of texture: 
        txt-0: lines
        txt-1: arpeggio
        txt-2: pad 
        txt-3: arpeggio-pad

    Format: each instrument token is followed by a texture token
    '''
    # Obtain pitch info of each instrument, save in a dict
    pos_and_pitch_of_track = from_remi_get_pos_and_pitch_seq_per_track(remi_seq)
    if 'i-128' in pos_and_pitch_of_track:
        has_drum = True
        pos_and_pitch_of_track.pop('i-128')
    else:
        has_drum = False
    
    # Compute playing style for each instrument
    style_of_track = {}
    for inst in pos_and_pitch_of_track:
        pos_and_pitch_seq = pos_and_pitch_of_track[inst]
        style = from_pitch_of_pos_seq_get_texture_of_the_track(pos_and_pitch_seq)
        style_of_track[inst] = style
    
    # Sort instruments by average pitch
    insts = []
    for inst in style_of_track:
        insts.append(inst)
        insts.append(style_of_track[inst])

    # Add drum to the end
    if has_drum:
        insts.append('i-128')
        
    return insts

def from_pitch_of_pos_seq_get_pitch_of_pos_dict(pos_and_pitch_seq):
    # Get the pitch token of each position
    pitch_of_pos = {}
    cur_pos = 'o-0'
    for tok in pos_and_pitch_seq:
        if tok.startswith('o-'):
            cur_pos = tok
            pitch_of_pos[cur_pos] = []
        elif tok.startswith('p-'):
            pitch_of_pos[cur_pos].append(tok)
        
    # # Sort pitch tokens of each position
    # for pos in pitch_of_pos:
    #     pitch_of_pos[pos] = sorted(pitch_of_pos[pos], key=lambda x: int(x.split('-')[1]), reverse=True)
    
    return pitch_of_pos

def from_pitch_of_pos_seq_get_texture_of_the_track(pos_and_pitch_seq):
    '''
    Compute the playing style of a track
    Return one of 
        txt-0: lines
        txt-1: arpeggio
        txt-2: pad 
        txt-3: arpeggio-pad
    '''
    # Get the pitch token of each position
    pitch_of_pos = {}
    cur_pos = 'o-0'
    for tok in pos_and_pitch_seq:
        if tok.startswith('o-'):
            cur_pos = tok
            pitch_of_pos[cur_pos] = []
        elif tok.startswith('p-'):
            pitch_of_pos[cur_pos].append(tok)

    # Determine if it's pad playing
    max_pitch_id = 0
    min_pitch_id = 128
    single_note_pos = False
    multi_note_pos = False
    for pos in pitch_of_pos:
        pos_and_pitch_seq = pitch_of_pos[pos]
        if len(pos_and_pitch_seq) == 1:
            single_note_pos = True
        else:
            multi_note_pos = True
        # Update max and min pitch
        for tok in pos_and_pitch_seq:
            pitch_id = int(tok.split('-')[1])
            if pitch_id > max_pitch_id:
                max_pitch_id = pitch_id
            if pitch_id < min_pitch_id:
                min_pitch_id = pitch_id
    pitch_range = max_pitch_id - min_pitch_id
    
    if multi_note_pos is True:
        # If all multi-note, pad playing
        if single_note_pos is False:
            txt = 'txt-2'
        # If both have, arpeggio-pad
        elif single_note_pos is True:
            txt = 'txt-3'
    # If all single-note, line or arpeggio playing
    elif single_note_pos is True:
        # txt = 'txt-1'
        if pitch_range >= 12: # If >= an octave, arpeggio
            txt = 'txt-1'
        else: # else, line
            txt = 'txt-0'
    else:
        # empty instrument, return txt-0
        print('Warning: illegal texture, might be empty instrument')
        txt = 'txt-0'
        # raise Exception('Illegal texture')

    # # 1/5 of chance, no control
    # if random.random() < 0.2:
    #     txt = 'txt-0'

    return txt

def from_remi_get_inst_and_voice(remi_seq):
    '''
    Can work for multiple bars

    Obtain all instrument in the input remi sequence
    Return a list of instrument tokens
    Instrument on the left have higher average pitch than that on the right
    '''
    # Obtain pitch info of each instrument, save in a dict
    pitch_of_track = from_remi_get_pitch_seq_per_track(remi_seq)
    if 'i-128' in pitch_of_track:
        has_drum = True
        pitch_of_track.pop('i-128')
    else:
        has_drum = False
    
    # filter empty inst
    insts = list(pitch_of_track.keys())
    for inst in insts:
        if len(pitch_of_track[inst]) == 0:
            pitch_of_track.pop(inst)

    # Compute average pitch for each instrument
    avg_pitch_of_track = {}
    for inst in pitch_of_track:
        pitch_seq = pitch_of_track[inst]
        avg_pitch = sum([int(tok.split('-')[1]) for tok in pitch_seq]) / len(pitch_seq)
        avg_pitch_of_track[inst] = avg_pitch

    # Sort instruments by average pitch
    insts = list(avg_pitch_of_track.keys())
    insts = sorted(insts, key=lambda x: avg_pitch_of_track[x], reverse=True)
    if has_drum:
        insts.append('i-128')
        
    return insts

def from_remi_get_inst_voice_texture(remi_seq):
    '''
    Obtain all instrument in the input remi sequence
    Return a list of instrument tokens
    Instrument on the left have higher average pitch than that on the right
    '''
    # Obtain pitch info of each instrument, save in a dict
    pitch_of_track = from_remi_get_pitch_seq_per_track(remi_seq)
    if 'i-128' in pitch_of_track:
        has_drum = True
        pitch_of_track.pop('i-128')
    else:
        has_drum = False
    
    # filter empty inst
    insts = list(pitch_of_track.keys())
    for inst in insts:
        if len(pitch_of_track[inst]) == 0:
            pitch_of_track.pop(inst)

    # Compute average pitch for each instrument
    avg_pitch_of_track = {}
    for inst in pitch_of_track:
        pitch_seq = pitch_of_track[inst]
        avg_pitch = sum([int(tok.split('-')[1]) for tok in pitch_seq]) / len(pitch_seq)
        avg_pitch_of_track[inst] = avg_pitch

    # Compute playing style for each instrument
    pos_and_pitch_of_track = from_remi_get_pos_and_pitch_seq_per_track(remi_seq)
    style_of_track = {}
    for inst in pos_and_pitch_of_track:
        pos_and_pitch_seq = pos_and_pitch_of_track[inst]
        style = from_pitch_of_pos_seq_get_texture_of_the_track(pos_and_pitch_seq)
        style_of_track[inst] = style

    # Sort instruments by average pitch
    insts = list(avg_pitch_of_track.keys())
    insts = sorted(insts, key=lambda x: avg_pitch_of_track[x], reverse=True)
    
    # Add texture token
    insts_and_texture = []
    for inst in insts:
        insts_and_texture.append(inst)
        insts_and_texture.append(style_of_track[inst])
    insts = insts_and_texture

    if has_drum:
        insts.append('i-128')
        
    return insts


def from_remi_get_pitch_of_pos_dict(bar_remi_seq, sort_pitch=True, flatten=False):
    """Obtain pitch of each position inside a same bar

    NOTE: Only works for one bar

    Args:
        bar_remi_seq (list): a list of remi tokens

    Raises:
        Exception: If multiple bars feeded as the parameter, will raise exception.
    """    
    if bar_remi_seq.count('b-1') > 1:
        # raise Exception('Multiple bars feed to this function: {} bars'.format(bar_remi_seq.count('b-1')))
        print('Warning: Multiple bars feed to this function: {} bars'.format(bar_remi_seq.count('b-1')))
    elif bar_remi_seq.count('b-1') < 1:
        bar_remi_seq.append('b-1')
    
    ret = {}
    cur_pos = None

    # Get pitch token of each position (after each position token)

    # Old version, may not suitable for reordered output
    # pitch_of_pos = {}
    # cur_pos = 'o-0'
    # for tok in bar_remi_seq:
    #     if tok.startswith('o-'):
    #         cur_pos = tok
    #         pitch_of_pos[cur_pos] = []
    #     elif tok.startswith('p-'):
    #         pitch_of_pos[cur_pos].append(tok)

    # 06-01 ver, suitable for reordered output
    # First get all positions
    pos_seq = from_remi_get_pos_seq(bar_remi_seq)
    pitch_of_pos = {pos: [] for pos in pos_seq}
    # Iterate through the remi sequence, add pitch to proper place
    for tok in bar_remi_seq:
        if tok in pos_seq:
            cur_pos = tok
        elif tok.startswith('p-'):
            if cur_pos is None:
                cur_pos = 'o-0'
                pitch_of_pos['o-0'] = []
            pitch_of_pos[cur_pos].append(tok)


    # Sort pitch tokens of each position
    if sort_pitch:
        for pos in pitch_of_pos:
            pitch_of_pos[pos] = sorted(pitch_of_pos[pos], key=lambda x: int(x.split('-')[1]), reverse=True)

    # Sort result dict by position id
    pitch_of_pos = dict(sorted(pitch_of_pos.items(), key=lambda x: int(x[0].split('-')[1])))

    if flatten is True:
        if sort_pitch is False:
            raise Exception('flatten is True, sort_pitch must be True')
        
        # Delete any repeat pitch tokens in each position, keep the order of pitch tokens within each position
        for pos in pitch_of_pos:
            pitch_of_pos[pos] = list(dict.fromkeys(pitch_of_pos[pos]))

    # Discard any empty position
    pitch_of_pos = {pos: pitch_of_pos[pos] for pos in pitch_of_pos if len(pitch_of_pos[pos]) > 0}

    return pitch_of_pos


def from_remi_get_pitch_of_pos_seq(remi_seq, flatten):
    '''
    NOTE: only works for one bar
    '''
    # Get pitch seq (without inst and duration info)
    pos_pitch_seq_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=flatten)
    pos_pitch_seq = []
    for pos in pos_pitch_seq_dict:
        pos_pitch_seq.append(pos)
        pos_pitch_seq.extend(pos_pitch_seq_dict[pos])

    return pos_pitch_seq


def from_remi_two_bar_split_hist_tgt_seq(remi_seq):
    b_1_indices = from_remi_get_bar_idx(remi_seq)
    num_bars = len(b_1_indices)
    
    if num_bars != 2:
        # To support inference with nohist model
        if num_bars == 1: # If only target bar is given
            remi_seq.insert(0, 'b-1')
            b_1_indices = from_remi_get_bar_idx(remi_seq)
        else:
            raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    # Get the target bar
    tgt_start_idx, tgt_end_idx = b_1_indices[1]
    tgt_remi_seq = remi_seq[tgt_start_idx:tgt_end_idx]
    
    # Get the raw history bar
    hist_start_idx, hist_end_idx = b_1_indices[0]
    hist_seq = remi_seq[hist_start_idx:hist_end_idx]

    return hist_seq, tgt_remi_seq


def from_remi_eight_bar_split_hist_tgt_seq(remi_seq):
    b_1_indices = from_remi_get_bar_idx(remi_seq)
    num_bars = len(b_1_indices)
    assert num_bars == 8, "Num bar issue: {} bars in the sample".format(num_bars)

    # Get the target bars
    tgt_start_idx = b_1_indices[4][0]
    tgt_end_idx = b_1_indices[7][1]
    tgt_remi_seq = remi_seq[tgt_start_idx:tgt_end_idx]
    
    # Get the raw history bar
    hist_start_idx = b_1_indices[0][0]
    hist_end_idx = b_1_indices[3][1]
    hist_seq = remi_seq[hist_start_idx:hist_end_idx]

    return hist_seq, tgt_remi_seq


def from_remi_get_melody_pitch_seq_highest_pitch(remi_seq):
    '''
    Get the melody sequence from a remi sequence
    Definition of melody: the highest note of each position
    i.e., the highest note of each position
    If there is only drum note in a position, discard the position

    remi_seq: a list of remi tokens. Can be multi-track or single track
    '''
    pitch_of_pos_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=False)

    # Discard any drum notes
    for pos in pitch_of_pos_dict:
        pitch_of_pos_dict[pos] = [pitch for pitch in pitch_of_pos_dict[pos] if from_pitch_token_get_pitch_id(pitch) < 128]

    # Discard any empty position
    pitch_of_pos_dict = {pos: pitch_of_pos_dict[pos] for pos in pitch_of_pos_dict if len(pitch_of_pos_dict[pos]) > 0}

    # Get the highest note of each position
    melody_seq = []
    for pos in pitch_of_pos_dict:
        highest_pitch = max(pitch_of_pos_dict[pos], key=lambda x: from_pitch_token_get_pitch_id(x))
        melody_seq.append(highest_pitch)

    return melody_seq


def from_remi_get_melody_pitch_seq_highest_track(remi_seq):
    '''
    Get the melody sequence from a remi sequence
    Definition of melody: the track that has highest average pitch
    i.e., the highest note of each position
    If there is only drum note in a position, discard the position

    remi_seq: a list of remi tokens. Can be multi-track or single track
    '''
    
    pitch_of_track_seqs = from_remi_get_pitch_seq_per_track(remi_seq)
    
    # Discard drum track
    if 'i-128' in pitch_of_track_seqs:
        pitch_of_track_seqs.pop('i-128')

    # Compute average pitch for each track
    avg_pitch_of_track = {}
    for inst in pitch_of_track_seqs:
        pitch_seq = pitch_of_track_seqs[inst]
        if len(pitch_seq) == 0:
            continue
        avg_pitch = sum([int(tok.split('-')[1]) for tok in pitch_seq]) / len(pitch_seq)
        avg_pitch_of_track[inst] = avg_pitch

    if len(avg_pitch_of_track) == 0:
        return []

    # Get the instrument with highest average pitch
    highest_pitch_track = max(avg_pitch_of_track, key=lambda x: avg_pitch_of_track[x])

    return pitch_of_track_seqs[highest_pitch_track]


def from_remi_get_melody_pos_and_pitch_seq_by_track(remi_seq, monophonic_only=True, top_note=False, return_inst=False):
    '''
    Recommended function for finding melody instruments with its melody sequence
    
    Get the melody sequence from a remi sequence
    i.e., the track with 
    (1) single note playing
    (2) highest average pitch

    There are two modes of melody extraction:
    (1) monophonic_only: the track with highest average pitch and only plays single note (recommended)
    (2) top_note: the highest note of the track with highest average pitch
    
    remi_seq: a list of remi tokens. Can be multi-track or single track
    '''
    assert monophonic_only is True or top_note is True, 'monophonic_only and top_note cannot be both False'

    pos_pitch_seq_of_tracks = from_remi_get_pos_and_pitch_seq_per_track(remi_seq)

    # Discard drum track
    if 'i-128' in pos_pitch_seq_of_tracks:
        pos_pitch_seq_of_tracks.pop('i-128')

    # Compute average pitch for each track
    avg_pitch_of_track = {}
    for inst in pos_pitch_seq_of_tracks:
        pos_pitch_seq = pos_pitch_seq_of_tracks[inst]
        pitch_seq = [tok for tok in pos_pitch_seq if tok.startswith('p-')]
        if len(pitch_seq) == 0:
            continue

        if monophonic_only is True:
            pos_seq = [tok for tok in pos_pitch_seq if tok.startswith('o-')]
            if len(pitch_seq) > len(pos_seq):
                continue

        avg_pitch = sum([int(tok.split('-')[1]) for tok in pitch_seq]) / len(pitch_seq)
        avg_pitch_of_track[inst] = avg_pitch

    if len(avg_pitch_of_track) == 0:
        if return_inst is False:
            return []
        else:
            return [], None
    
    # Get the instrument with highest average pitch
    highest_pitch_track = max(avg_pitch_of_track, key=lambda x: avg_pitch_of_track[x])
    highest_track_pos_pitch = pos_pitch_seq_of_tracks[highest_pitch_track]

    if top_note is True:
        # Get the highest note of each position
        ret = []
        cur_pos = None
        for tok in highest_track_pos_pitch:
            if tok.startswith('o-'):
                if cur_pos is not None:
                    ret.append(cur_pos)
                    ret.append(cur_pitch)

                cur_pos = tok
                cur_pitch = 'p-0'
            elif tok.startswith('p-'):
                pitch_id = from_pitch_token_get_pitch_id(tok)
                cur_pitch_id = from_pitch_token_get_pitch_id(cur_pitch) 
                if pitch_id > cur_pitch_id:
                    cur_pitch = tok
        ret.append(cur_pos)
        ret.append(cur_pitch)
    else:
        ret = highest_track_pos_pitch

    if return_inst is False:
        return ret
    else:
        return ret, highest_pitch_track


def from_melody_pos_pitch_seq_convert_to_dict(mel_seq):
    '''
    Convert a sequence of position and pitch to a dict, key is position, value is pitch
    '''
    pos_tokens = [tok for tok in mel_seq if tok.startswith('o-')]
    pos_tokens = sorted(pos_tokens, key=lambda x: int(x.split('-')[1]))

    ret = {}
    cur_pos = None
    for tok in mel_seq:
        if tok in pos_tokens:
            cur_pos = tok
        elif tok.startswith('p-') and cur_pos is not None:
            ret[cur_pos] = tok

    return ret


def from_remi_get_melody_pos_and_pitch_seq_by_pos(remi_seq):
    '''
    Get the melody sequence from a remi sequence
    i.e., the highest note of each position
    If there is only drum note in a position, discard the position

    remi_seq: a list of remi tokens. Can be multi-track or single track
    '''
    pitch_of_pos_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=False)

    # Discard any drum notes
    for pos in pitch_of_pos_dict:
        pitch_of_pos_dict[pos] = [pitch for pitch in pitch_of_pos_dict[pos] if from_pitch_token_get_pitch_id(pitch) < 128]

    # Discard any empty position
    pitch_of_pos_dict = {pos: pitch_of_pos_dict[pos] for pos in pitch_of_pos_dict if len(pitch_of_pos_dict[pos]) > 0}

    # Get the highest note of each position
    melody_seq = []
    for pos in pitch_of_pos_dict:
        highest_pitch = max(pitch_of_pos_dict[pos], key=lambda x: from_pitch_token_get_pitch_id(x))
        melody_seq.append(pos)
        melody_seq.append(highest_pitch)

    return melody_seq


def from_chord_seq_get_chord_note_seq(chord_seq):
    '''
    NOTE: Only works for one bar

    Convert a chord sequence to a sequence of chord notes
    '''
    # The value of each position is a tuple (one chord) or None
    chord_dict = {'o-0':chord_seq[0], 'o-12':chord_seq[1], 'o-24':chord_seq[2], 'o-36':chord_seq[3]}
    
    # Delete any empty position
    chord_dict = {pos: chord_dict[pos] for pos in chord_dict if chord_dict[pos] is not None}

    # Convert (root, type) tuple to a list of chord notes
    ret = []
    for pos in chord_dict:
        if chord_dict[pos] is not None:
            chord_notes = generate_chord_notes(*chord_dict[pos])
            chord_dict[pos] = chord_notes
            
            ret.append(pos)
            ret.extend(chord_notes)
        
    return ret


def from_chord_seq_get_chord_note_seq_2chord_per_bar(chord_seq):
    '''
    NOTE: Only works for one bar

    Convert a chord sequence to a sequence of chord notes
    '''
    # The value of each position is a tuple (one chord) or None
    chord_dict = {'o-0':chord_seq[0], 'o-24':chord_seq[1]}
    
    # Delete any empty position
    chord_dict = {pos: chord_dict[pos] for pos in chord_dict if chord_dict[pos] is not None}

    # Convert (root, type) tuple to a list of chord notes
    ret = []
    for pos in chord_dict:
        if chord_dict[pos] is not None:
            chord_notes = generate_chord_notes(*chord_dict[pos])
            chord_dict[pos] = chord_notes
            
            ret.append(pos)
            ret.extend(chord_notes)
            ret.append('d-24')
        
    return ret


def from_chord_seq_get_chord_note_opd_dict_2chord_per_bar(chord_seq):
    '''
    NOTE: Only works for one bar

    Convert a chord sequence to a sequence of chord notes
    '''
    # The value of each position is a tuple (one chord) or None
    chord_dict = {'o-0':chord_seq[0], 'o-24':chord_seq[1]}
    
    # Delete any empty position
    chord_dict = {pos: chord_dict[pos] for pos in chord_dict if chord_dict[pos] is not None}

    # Convert (root, type) tuple to a list of chord notes
    ret = {}
    for pos in chord_dict:
        if chord_dict[pos] is not None:
            chord_notes = generate_chord_notes(*chord_dict[pos])
            
            ret[pos] = []
            for chord_note in chord_notes:
                ret[pos].append(chord_note)
                ret[pos].append('d-24')
        
    return ret


def from_chord_seq_get_chord_token_seq(chord_seq):
    '''
    NOTE: Only works for one bar

    Convert a chord sequence to a sequence of chord notes
    '''
    # The value of each position is a tuple (one chord) or None
    chord_dict = {'o-0':chord_seq[0], 'o-12':chord_seq[1], 'o-24':chord_seq[2], 'o-36':chord_seq[3]}
    
    # Delete any empty position
    chord_dict = {pos: chord_dict[pos] for pos in chord_dict if chord_dict[pos] is not None}

    # Flatten to sequence
    ret = []
    for pos in chord_dict:
        ret.append(pos)
        ret.extend(chord_dict[pos])
    return ret



def from_remi_get_lead_sheet_seq(chord_seq, melody_seq):
    '''
    Get the lead sheet sequence from chord and melody sequences
    '''
    # The value of each position is a tuple (one chord) or None
    chord_dict = {'o-0':chord_seq[0], 'o-12':chord_seq[1], 'o-24':chord_seq[2], 'o-36':chord_seq[3]}
    
    # Delete any empty position
    chord_dict = {pos: chord_dict[pos] for pos in chord_dict if chord_dict[pos] is not None}

    # Convert (root, type) tuple to a list of chord notes
    for pos in chord_dict:
        if chord_dict[pos] is not None:
            chord_notes = generate_chord_notes(*chord_dict[pos])
            chord_dict[pos] = chord_notes

    # Convert melody seq to dict like
    melody_dict = from_remi_get_pitch_of_pos_dict(melody_seq)

    # Merge chord with melody
    lead_sheet_seq = []
    # Get the union of chord_dict and melody_dict's keys
    all_pos = list(set(chord_dict.keys()).union(set(melody_dict.keys())))
    # sort pos by position id
    all_pos = sorted(all_pos, key=lambda x: int(x.split('-')[1]))
    
    for pos in all_pos:
        lead_sheet_seq.append(pos)
        if pos in chord_dict:
            lead_sheet_seq.extend(chord_dict[pos])
        if pos in melody_dict:
            lead_sheet_seq.extend(melody_dict[pos])

    return lead_sheet_seq


def merge_chord_and_melody_opd_dict(chord_opd_dict, mel_opd_dict):
    all_pos = list(set(chord_opd_dict.keys()).union(set(mel_opd_dict.keys())))
    all_pos = sorted(all_pos, key=lambda x: int(x.split('-')[1]))

    ret = {}
    for pos in all_pos:
        ret[pos] = []
        if pos in chord_opd_dict:
            ret[pos].extend(chord_opd_dict[pos])
        if pos in mel_opd_dict:
            ret[pos].extend(mel_opd_dict[pos])
    return ret


def dict_to_seq(opd_dict):
    ret = []
    for pos in opd_dict:
        ret.append(pos)
        ret.extend(opd_dict[pos])
    return ret



def from_remi_get_drum_content_seq(remi_seq):
    return from_remi_get_pos_and_pitch_seq_of_track(remi_seq, 'i-128')

def from_remi_get_drum_pitch_seq(remi_seq):
    return from_remi_get_pitch_seq_of_track(remi_seq, 'i-128')

def from_remi_get_bass_pitch_seq(remi_seq):
    bass_insts = ['i-32', 'i-33', 'i-43', 'i-70']
    return from_remi_get_pitch_seq_of_multiple_insts(remi_seq, bass_insts)


def from_remi_mbar_remove_drum(remi_seq):
    bar_indices = from_remi_get_bar_idx(remi_seq)
    ret = []
    for bar_id in bar_indices:
        start_idx, end_idx = bar_indices[bar_id]
        bar_seq = remi_seq[start_idx:end_idx]

        # Get the opd dict without drum
        opd_seqs = from_remi_get_opd_seq_per_track(bar_seq, sort_by_avg_pitch=True, include_drum=False)
        for inst in opd_seqs:
            ret.append(inst)
            ret.extend(opd_seqs[inst])
        ret.append('b-1')

    return ret


def from_remi_bar_remove_drum(remi_seq):
    '''
    Remove drum notes from the target-side remi sequence (un-reoredered)
    NOTE: only works for a bar!
    '''
    # if len(remi_seq) > 2: # [06-12] now each bar has at least a time signature token and a bar line
    #     # ensure there is at least a token starts with 'd-'
    #     dur_token_exists = False
    #     for tok in remi_seq:
    #         if tok.startswith('d-'):
    #             dur_token_exists = True
    #             break
    #     assert dur_token_exists, 'No duration token in the input sequence, illegal remi seq'

    ret = []
    cur_pos = None
    cur_inst = None
    pre_pos = None
    pre_inst = None
    find_p = False
    find_d = False
    for tok in remi_seq:
        if tok.startswith('o'):
            cur_pos = tok
        elif tok.startswith('i'):
            cur_inst = tok
        elif tok.startswith('p'):
            cur_pitch = tok
            find_p = True
        elif tok.startswith('d'):
            cur_dur = tok
            find_d = True

            if find_p and find_d:
                if cur_inst != 'i-128':
                    if cur_pos != pre_pos:
                        ret.append(cur_pos)
                    # if cur_inst != pre_inst:
                    if cur_inst is None:
                        cur_inst = 'i-0'
                    ret.append(cur_inst)
                    ret.append(cur_pitch)
                    ret.append(cur_dur)

                    pre_pos = cur_pos
                    pre_inst = cur_inst
                    find_p, find_d = False, False
    ret.append('b-1')

    return ret

def from_remi_hist_refine_history(hist_seq, tgt_insts, hist_type='full', reorder_tgt=False, voice_control=False, hist_is_reordered=False):
    '''
    Prepare history to a better format from raw remi sequence
    
    Args:
        hist_seq (list): The history sequence in raw remi format.
        tgt_insts (list): The target instruments.
        hist_type (str, optional): The type of history to be prepared. Defaults to 'full'.
        reorder (bool, optional): Whether to reorder the history sequence. Defaults to False.
        voice_control (bool, optional): Whether to sort the history sequence by voice. Defaults to False.
        hist_is_reordered (bool, optional): Whether the history sequence is already reordered. Defaults to False.
    
    Returns:
        list: The refined history sequence in the desired format.
    '''
    
    tgt_has_drum = True if 'i-128' in tgt_insts else False
    
    hist_has_drum = True if 'i-128' in hist_seq else False
    if hist_type == 'full':
        new_hist_seq = hist_seq

        if reorder_tgt is True and hist_is_reordered is False:
            new_hist_seq = in_remi_bar_reorder_notes_by_inst(new_hist_seq, sort_by_voice=voice_control)

    else:
        new_hist_seq = []

        # For 'drum_pos' type, add position for each instrument to history
        if hist_type == 'drum_pos':
            # Add pitch range and position for each instrument to history
            insts_of_target = tgt_insts
            pos_of_track_seq = from_remi_get_pos_per_track_seq(hist_seq, keep_insts=insts_of_target, with_drum=True)
            new_hist_seq.extend(pos_of_track_seq)
        
        if hist_type == 'range_pos_with_drum_pos':
            # Add pitch range and position for each instrument to history
            insts_of_target = tgt_insts
            range_and_pos_of_track_seq = from_remi_get_range_and_pos_of_track_seq(hist_seq, with_drum=True, keep_insts=insts_of_target, drum_pos=True)
            new_hist_seq.extend(range_and_pos_of_track_seq)

        if hist_type == 'range_pos_with_drum_range':
            insts_of_target = tgt_insts
            range_and_pos_of_track_seq = from_remi_get_range_and_pos_of_track_seq(hist_seq, with_drum=True, keep_insts=insts_of_target, drum_range=True)
            new_hist_seq.extend(range_and_pos_of_track_seq)

        if hist_type == 'range_pos_with_drum_range_pos':
            insts_of_target = tgt_insts
            range_and_pos_of_track_seq = from_remi_get_range_and_pos_of_track_seq(hist_seq, with_drum=True, keep_insts=insts_of_target, drum_pos=True, drum_range=True)
            new_hist_seq.extend(range_and_pos_of_track_seq)

        # For 'drum_range_pos', 'range_pos' type, add pitch range and position for each instrument to history
        if hist_type in ['drum_range_pos', 'range_pos']:
            # Add pitch range and position for each instrument to history
            insts_of_target = tgt_insts
            range_and_pos_of_track_seq = from_remi_get_range_and_pos_of_track_seq(hist_seq, with_drum=False, keep_insts=insts_of_target, remi_reordered=hist_is_reordered)
            new_hist_seq.extend(range_and_pos_of_track_seq)

        # For 'drum_and_range', 'range', type, add pitch range for each instrument to history
        if hist_type in ['range', 'drum_and_range']:
            # Add pitch range for each instrument to history
            insts_of_target = tgt_insts
            pitch_range_of_track_seq = from_remi_get_range_of_track_seq(hist_seq, with_drum=False, keep_insts=insts_of_target)
            new_hist_seq.extend(pitch_range_of_track_seq)

        # For 'drum' and 'drum_and_range', 'drum_range_pos'
        if hist_type in ['drum', 'drum_and_range', 'drum_range_pos']:
            if hist_has_drum and tgt_has_drum:
                new_hist_seq.append('i-128')
                new_hist_seq.extend(from_remi_get_drum_content_seq(hist_seq))

    return new_hist_seq


def in_remi_bar_reorder_notes_by_inst(remi_seq, sort_by_voice=False):
    '''
    Re-order the target sequence, so that it become track-by-track, instead of mixing together
    
    Notes in remi seq can be either
    - o i p d
    - i p d
    - p d

    In return:
    - i o p d o p d ...  i o p d p d o p d
    '''
    insts = from_remi_get_insts(remi_seq) # Get inst, sort by program id

    if len(remi_seq) > 1 and len(insts) == 0:
        insts = ['i-0']

    opd_seq_of_tracks = from_remi_get_opd_seq_per_track(remi_seq)

    ret = []
    for inst in opd_seq_of_tracks:
        ret.append(inst)
        ret.extend(opd_seq_of_tracks[inst])

    if sort_by_voice:
        insts_from_high_to_low = from_remi_get_inst_and_voice(remi_seq)
        ret = []
        for inst in insts_from_high_to_low:
            assert inst in opd_seq_of_tracks, 'inst {} not in opd_seq_of_tracks'.format(inst)
            # if inst in opd_seq_of_tracks:
            ret.append(inst)
            ret.extend(opd_seq_of_tracks[inst])

    ret.append('b-1')

    return ret


def in_remi_multi_bar_delete_insts(remi_seq):
    '''
    remi_seq: already reordered by instruments
    '''
    # Determine inst to delete
    insts = from_remi_get_insts(remi_seq, sort_inst=False)
    if len(insts) <= 1:
        return remi_seq
    
    # Get insts to del from a poisson distribution
    lamb = max(len(insts) // 4, 1)
    num_insts_to_del = np.random.poisson(lamb)
    num_insts_to_del = min(num_insts_to_del, len(insts)-1)
    num_insts_to_del = max(0, num_insts_to_del)
    insts_to_del = set(random.sample(insts, num_insts_to_del))

    # Obtain bar positions
    remi_of_all_bars = []
    bar_indices = from_remi_get_bar_idx(remi_seq)
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_seq = remi_seq[bar_start_idx:bar_end_idx]

        time_sig = bar_seq[0]
        tempo = bar_seq[1]

        # Get opd seq of all insts
        opd_seq_of_tracks = from_remi_get_opd_seq_per_track(bar_seq, sort_by_avg_pitch=True)

        # Del insts
        opd_seq_new = {k:v for k,v in opd_seq_of_tracks.items() if k not in insts_to_del}

        # Reconstruct
        bar_seq_new = []
        bar_seq_new.append(time_sig)
        bar_seq_new.append(tempo)
        for inst in opd_seq_new:
            bar_seq_new.append(inst)
            bar_seq_new.extend(opd_seq_new[inst])
        bar_seq_new.append('b-1')

        remi_of_all_bars.extend(bar_seq_new)
    
    return remi_of_all_bars

def in_remi_multi_bar_delete_non_mel_insts(remi_seq):
    '''
    remi_seq: already reordered by instruments
    '''
    # Determine inst to delete
    insts = from_remi_get_insts(remi_seq, sort_inst=False)
    if len(insts) <= 1:
        return remi_seq
    
    ''' old implementation, polyphonic insts can be melody '''
    # Get the melody track, inst with highest average pitch
    pitch_seq_of_tracks = from_remi_get_pitch_seq_per_track(remi_seq)
    avg_pitch_of_tracks = {}
    for inst in pitch_seq_of_tracks:
        pitch_seq = pitch_seq_of_tracks[inst]
        avg_pitch = sum([int(tok.split('-')[1]) for tok in pitch_seq]) / len(pitch_seq)
        avg_pitch_of_tracks[inst] = avg_pitch
    melody_inst = max(avg_pitch_of_tracks, key=lambda x: avg_pitch_of_tracks[x])

    # # New implementation, same melody definition as the lead sheet model
    # _, melody_inst = from_remi_get_melody_pos_and_pitch_seq_by_track(
    #     remi_seq, return_inst=True, monophonic_only=False, top_note=True,
    # )

    # When no melody track found, the function above returns None as melody instrument
    if melody_inst is None:
        pass
    else:
        # Do not delete melody instrument
        insts.remove(melody_inst)

    # Get insts to del from a poisson distribution
    lamb = max(len(insts) // 4, 1)
    # lamb = 1
    num_insts_to_del = np.random.poisson(lamb)
    num_insts_to_del = min(num_insts_to_del, len(insts)-1)
    num_insts_to_del = max(0, num_insts_to_del)
    insts_to_del = set(random.sample(insts, num_insts_to_del))

    # Obtain bar positions
    remi_of_all_bars = []
    bar_indices = from_remi_get_bar_idx(remi_seq)
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_seq = remi_seq[bar_start_idx:bar_end_idx]

        # time_sig = bar_seq[0]
        # tempo = bar_seq[1]

        # Get opd seq of all insts
        opd_seq_of_tracks = from_remi_get_opd_seq_per_track(bar_seq, sort_by_avg_pitch=True)

        # Del insts
        opd_seq_new = {k:v for k,v in opd_seq_of_tracks.items() if k not in insts_to_del}

        # Reconstruct
        bar_seq_new = []
        # bar_seq_new.append(time_sig)
        # bar_seq_new.append(tempo)
        for inst in opd_seq_new:
            bar_seq_new.append(inst)
            bar_seq_new.extend(opd_seq_new[inst])
        bar_seq_new.append('b-1')

        remi_of_all_bars.extend(bar_seq_new)
    
    return remi_of_all_bars


def from_remi_get_avg_duration_per_track(remi_seq):
    '''
    Get the average duration of notes in each track, in integer
    '''
    # Obtain duration info for each instrument
    dur_of_track = from_remi_get_duration_per_track(remi_seq)

    # Compute the avg duration for each instrument
    avg_dur_of_track = {}
    for inst in dur_of_track:
        avg_dur = sum(dur_of_track[inst]) / len(dur_of_track[inst])
        avg_dur_of_track[inst] = int(avg_dur)

    return avg_dur_of_track


def from_remi_get_pos_per_track_seq(remi_seq, reorder_inst=False, with_drum=True, keep_insts=None):
    pos_per_track_dict = from_remi_get_pos_per_track_dict(remi_seq, remi_reordered=reorder_inst)
    ret = []
    for inst in pos_per_track_dict:
        if with_drum is False and inst == 'i-128':
            continue
        if keep_insts is not None and inst not in keep_insts:
            continue

        ret.append(inst)
        ret.extend(pos_per_track_dict[inst])
    return ret


def from_remi_get_pos_per_track_dict(remi_seq, remi_reordered=False, dedup=False):
    '''
    Get the position token of each instrument
    '''
    insts = from_remi_get_insts(remi_seq)
    pos_of_track = {inst: [] for inst in insts}

    if remi_reordered is True: # Reorder seems to be always false during the feature extraction phase. Only applied after augmentation
        cur_inst = None
        for tok in remi_seq:
            if tok.startswith('i-'):
                cur_inst = tok 
            elif tok.startswith('o-'):
                cur_inst = insts[0] if cur_inst is None else cur_inst
                pos_of_track[cur_inst].append(tok)
    else:
        cur_pos = None
        prev_pos, prev_inst = None, None
        for tok in remi_seq:
            if tok.startswith('o-'):
                cur_pos = tok
            elif tok.startswith('i-'):
                cur_inst = tok

                if cur_pos != prev_pos or cur_inst != prev_inst:
                    pos_of_track[cur_inst].append(cur_pos)
                    prev_pos = cur_pos
                    prev_inst = cur_inst

    if dedup:
        for inst in pos_of_track:
            pos_of_track[inst] = list(dict.fromkeys(pos_of_track[inst]))
            # Sort by position id
            pos_of_track[inst] = sorted(pos_of_track[inst], key=lambda x: int(x.split('-')[1]))

    
    return pos_of_track


def from_remi_get_pos_seq(remi_seq, reorder_inst=False, sort_pos=True):
    '''
    Get the position token of remi
    '''
    pos_seq = [tok for tok in remi_seq if tok.startswith('o-')]

    # Remove duplicate position tokens
    pos_seq = list(dict.fromkeys(pos_seq))

    # Sort by pos id
    if sort_pos:
        pos_seq = sorted(pos_seq, key=lambda x: int(x.split('-')[1]))

    return pos_seq


def from_remi_get_pos_seq_mbar(remi_seq, sort_pos=True):
    '''
    Get the position token of remi
    '''
    ret = []
    bar_indices = from_remi_get_bar_idx(remi_seq)
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_seq = remi_seq[bar_start_idx:bar_end_idx]

        pos_seq = [tok for tok in bar_seq if tok.startswith('o-')]

        # Remove duplicate position tokens
        pos_seq = list(dict.fromkeys(pos_seq))

        # Sort by pos id
        if sort_pos:
            pos_seq = sorted(pos_seq, key=lambda x: int(x.split('-')[1]))

        ret.extend(pos_seq)
        ret.append('b-1')

    return ret


def from_remi_get_duration_per_track(remi_seq):
    '''
    Obtain all duration token's value of each instrument
    '''
    insts = from_remi_get_insts(remi_seq)
    duration_of_track = {inst: [] for inst in insts}
    if len(insts) == 0:
        return {}

    cur_inst = insts[0] # There are bug with target sequence, inst is ommited sometimes
    for tok in remi_seq:
        if tok.startswith('i-'):
            cur_inst = tok
        elif tok.startswith('d-'):
            duration_of_track[cur_inst].append(int(tok.split('-')[1]))

    # Remove instruments with empty duration
    insts = list(duration_of_track.keys())
    for inst in insts:
        if len(duration_of_track[inst]) == 0:
            duration_of_track.pop(inst)
    
    return duration_of_track

def in_inst_list_sort_inst(inst_list):        
    sorted_list = sorted(inst_list, key=lambda x: int(x.split("-")[1]))
    return sorted_list


def in_condition_keep_only_specified_insts(input_seq, new_inst_list, has_texture=False):
    '''
    Delete any instrument tokens in input_seq that is not in new_inst_list
    '''
    new_inst = set(new_inst_list)

    inst_start_idx = input_seq.index('INS')
    inst_end_idx = input_seq.index('HIST') if 'HIST' in input_seq else len(input_seq)
    inst_spec = input_seq[inst_start_idx+1:inst_end_idx]

    new_inst_spec = []
    i = 0
    while i < len(inst_spec):
    # for inst in inst_spec:
        inst = inst_spec[i]
        if inst in new_inst:
            if has_texture is False or has_texture is True and inst == 'i-128':
                new_inst_spec.append(inst)
            else:
                new_inst_spec.append(inst)
                new_inst_spec.append(inst_spec[i+1])
                i += 1
        i += 1

    ret = input_seq[:inst_start_idx+1] + new_inst_spec + input_seq[inst_end_idx:]

    # ret = []

    # if 'HIST' in input_seq:
    #     hist_pos = input_seq.index('HIST')
    # else:
    #     hist_pos = 99999

    # for tok in input_seq[:hist_pos]:
    #     if tok.startswith('i-'):        # For each instrument token
    #         if tok not in new_inst:     # If we have deleted it from target sequence
    #             continue                # We then delete it from input sequence
    #     ret.append(tok)

    # # Recover history
    # ret.extend(input_seq[hist_pos:])
    
    return ret

def from_condition_get_inst_spec(condition_seq):
    '''
    Obtain the instrument specification from the condition sequence
    '''
    inst_start_idx = condition_seq.index('INS')
    inst_end_idx = condition_seq.index('HIST') if 'HIST' in condition_seq else len(condition_seq)
    inst_spec = condition_seq[inst_start_idx+1:inst_end_idx]
    return inst_spec

def from_condition_get_pos_and_pitch(condition_seq: List[str]) -> Tuple[int, int]:
    """Count the position and pitch tokens in the condition sequence

    Args:
        condition_seq (List[str]): A condition sequence

    Returns:
        Tuple[int, int]: number of position tokens, and number of pitch tokens
    """    
    pos_cnt = 0
    p_cnt = 0
    for tok in condition_seq:
        if tok.startswith('o-'):
            pos_cnt += 1
        elif tok.startswith('p-'):
            p_cnt += 1
    return pos_cnt, p_cnt

def from_pos_token_get_pos_id(pos_tok: str) -> int:
    """Obtain the integer part of the position token

    Args:
        pos_tok (str): A position token

    Returns:
        int: The integer part
    """    
    assert pos_tok.startswith('o-')
    ret = int(pos_tok.split('-')[1])
    return ret


def reorder_remi_bar(remi_seq, add_bar_token=False):
    '''
    Re-order the target sequence, so that it become track-by-track, instead of mixing together
    
    Notes in remi seq can be either
    - o i p d
    - i p d
    - p d

    In return:
    - i o p d o p d ...  i o p d p d o p d
    '''
    ret = []

    insts = from_remi_get_insts(remi_seq, sort_inst=False) # Get inst, sort by program id
    if len(insts) == 0:
        return remi_seq

    # for i in range(2):
    #     if not remi_seq[i].startswith('o'):
    #         ret.append(remi_seq[i])

    opd_seq_of_tracks = from_remi_get_opd_seq_per_track(remi_seq, sort_by_avg_pitch=True)

    
    for inst in opd_seq_of_tracks:
        ret.append(inst)
        ret.extend(opd_seq_of_tracks[inst])
    
    if add_bar_token is True:
        ret.append('b-1')

    return ret


def in_remi_replace_hist(remi_seq, hist_seq):
    '''
    Replace the HIST part inside a <condition> <sep> <target> sequence
    hist: a sequence of token
    '''
    hist_tok_pos = remi_seq.index('HIST')
    sep_pos = remi_seq.index('<sep>')
    ret = remi_seq[:hist_tok_pos+1] + hist_seq + remi_seq[sep_pos:]
    return ret


def in_condition_seq_replace_inst(condition_seq, inst_spec_seq):
    '''
    Replace the HIST part inside a <condition> <sep> <target> sequence
    hist: a sequence of token

    prerequisite: inst_spec sort by program id
    '''
    inst_tok_pos = condition_seq.index('INS')
    hist_tok_pos = condition_seq.index('HIST') if 'HIST' in condition_seq else len(condition_seq)
    inst_start_idx = inst_tok_pos + 1
    inst_end_idx = hist_tok_pos
    ret = condition_seq[:inst_start_idx] + inst_spec_seq + condition_seq[inst_end_idx:]
    return ret

def in_condition_collapse_chroma(condition_seq, deduplicate=True):
    '''
    Collapse the content sequence to chroma

    By compressing all non-melody pitch tokens into the first octave (pitch 0-11)
    '''
    # Get the position and pitch info from content
    pitch_tok_idx = condition_seq.index('PITCH')
    if condition_seq[pitch_tok_idx+1] == 'INS':
        return condition_seq
    
    content_start_idx = pitch_tok_idx + 1
    content_end_idx = condition_seq.index('INS')
    content_seq = condition_seq[content_start_idx:content_end_idx]

    # Organize content into a dict, key is all possible 'o-' tokens
    content_dict = {}
    cur_pos = None
    for tok in content_seq:
        if tok.startswith('o-'):
            cur_pos = tok
            content_dict[cur_pos] = []
        elif tok.startswith('p-'):
            content_dict[cur_pos].append(tok)
    
    # Collapse the content
    new_content_dict = {}
    for pos in content_dict:
        drum_notes = []
        instrument_notes = []
        inst_note_collapsed = []
        new_content_dict[pos] = []

        # Save drum notes to drum_notes list
        drum_notes = [pitch for pitch in content_dict[pos] if from_pitch_token_get_pitch_id(pitch) >= 128]
        # Remove drum notes from content_dict
        instrument_notes = [pitch for pitch in content_dict[pos] if from_pitch_token_get_pitch_id(pitch) < 128]

        pitch_tokens = instrument_notes
        # Preserve melody pitch and "counter melody pitch", which is the two highest pitch
        # The rest are compressed to the first octave
        if len(pitch_tokens) <= 2:
            inst_note_collapsed.extend(pitch_tokens)
        else:
            # Get the largest two pitch tokens in the position
            melody_pitch = max(pitch_tokens, key=lambda x: from_pitch_token_get_pitch_id(x))
            pitch_tokens.remove(melody_pitch)
            counter_melody_pitch = max(pitch_tokens, key=lambda x: from_pitch_token_get_pitch_id(x))
            pitch_tokens.remove(counter_melody_pitch)
            inst_note_collapsed.append(melody_pitch)
            inst_note_collapsed.append(counter_melody_pitch)

            for pitch_tok in pitch_tokens:
                if pitch_tok != melody_pitch and pitch_tok != counter_melody_pitch:
                    inst_note_collapsed.append(convert_pitch_token_to_first_octave(pitch_tok))

            # Sort tokens by pitch id in descending order
            inst_note_collapsed = sorted(inst_note_collapsed, key=lambda x: from_pitch_token_get_pitch_id(x), reverse=True)

        # Sort drum notes by pitch id, descending order
        drum_notes = sorted(drum_notes, key=lambda x: from_pitch_token_get_pitch_id(x), reverse=True)

        # Add drum notes to the end
        inst_note_collapsed.extend(drum_notes)

        # Deduplicate any repeated pitch tokens
        if deduplicate is True:
            inst_note_collapsed = list(dict.fromkeys(inst_note_collapsed))

        new_content_dict[pos] = inst_note_collapsed

    # Convert the dict back to a list
    new_content_seq = []
    for pos in new_content_dict:
        new_content_seq.append(pos)
        new_content_seq.extend(new_content_dict[pos])

    # Reconstruct the condition_seq
    ret = condition_seq[:content_start_idx] + new_content_seq + condition_seq[content_end_idx:]

    return ret
            

def convert_pitch_token_to_first_octave(pitch_tok):
    '''
    Map the pitch token to the first octave (id from 0 to 11)
    '''
    pitch_id = from_pitch_token_get_pitch_id(pitch_tok)
    new_pitch_id = pitch_id % 12
    new_pitch_tok = 'p-' + str(new_pitch_id)
    return new_pitch_tok


def song_remi_split_to_segments_8bar_hop4(remi_seq):
    '''
    Split the remi sequence of a song to a list of 2-bar segments
    NOTE: An additional blank is insert in the beginning.
    '''
    # Front pad empty bars to the beginning
    remi_seq_padded = ['b-1' for i in range(4)] + remi_seq

    ret = []
    bar_indices = from_remi_get_bar_idx(remi_seq_padded)
    
    seg_bars = 8
    hop_bars = 4

    # for tgt_begin_bar_id in range(4, len(bar_indices)-())
    tot_n_bars = len(bar_indices)
    max_bar_id = tot_n_bars - 1
    

    for seg_begin_bar_id in range(0, tot_n_bars-hop_bars, hop_bars):
        bar1_start_idx, bar1_end_idx = bar_indices[seg_begin_bar_id]
        end_bar_id = min(seg_begin_bar_id + (seg_bars - 1), max_bar_id)
        end_bar_start_idx, end_bar_end_idx = bar_indices[end_bar_id]
        seg_seq = remi_seq_padded[bar1_start_idx:end_bar_end_idx]

        # Pad seg_seq to 8 bars
        num_bars = seg_seq.count('b-1')
        if num_bars < seg_bars:
            for i in range(seg_bars - num_bars):
                seg_seq.append('b-1')

        ret.append(seg_seq)
        
    return ret


def song_remi_split_to_segments_2bar_hop1(remi_seq, ts_and_tempo=False):
    '''
    Split the remi sequence of a song to a list of 2-bar segments
    NOTE: An additional blank is insert in the beginning.
    '''
    if ts_and_tempo is False:
        t = ['b-1'] + remi_seq
    else:
        ts = remi_seq[0]
        tempo = remi_seq[1]
        t = [ts, tempo, 'b-1'] + remi_seq
    ret = []
    bar_indices = from_remi_get_bar_idx(t)
    for cur_bar_id in range(len(bar_indices)-1):
        bar1_start_idx, bar1_end_idx = bar_indices[cur_bar_id]
        next_bar_id = cur_bar_id + 1
        bar2_start_idx, bar2_end_idx = bar_indices[next_bar_id]
        ret.append(t[bar1_start_idx:bar2_end_idx])
    return ret


def from_remi_get_pos_and_pitch_seq_of_track(remi_seq, inst):
    '''
    Note: only works for a bar

    Extract the track-wise remi sequence for a given instrument
    From a multi-track remi sequence
    '''
    
    pos_pitch_dict = from_remi_get_pos_and_pitch_dict_of_track(remi_seq, inst)

    ret = []
    pos = list(pos_pitch_dict.keys())
    pos = sorted(pos, key=lambda x: int(x.split('-')[1]))

    for o in pos:
        ret.append(o)
        ret.extend(pos_pitch_dict[o])

    return ret

def from_remi_get_pos_and_pitch_dict_of_track(remi_seq, inst):
    '''
    NOTE: only works for reordered target

    Extract the track-wise remi sequence for a given instrument
    From a multi-track remi sequence
    '''
    ret = {}
    cur_pos = None
    cur_pitch = None
    cur_inst = None
    for tok in remi_seq:
        if tok.startswith('o-'):
            cur_pos = tok
        elif tok.startswith('i-'):
            cur_inst = tok
        elif tok.startswith('p-'):
            cur_pitch = tok

            if cur_inst == inst:
                if cur_pos is None or cur_pitch is None:
                    continue
                if cur_pos not in ret:
                    ret[cur_pos] = []
                ret[cur_pos].append(cur_pitch)

    return ret

def from_remi_get_pos_and_pitch_seq_per_track(remi_seq):
    '''
    NOTE: only guarentee to works for a bar
    Careful when using with multiple bars,
    where there is no separater between bars
    the output position will go back to o-0 for the second bar.

    Extract all track-wise remi sequences from the multi track remi
    Return a dict of seqs for each track, key is instrument token
    Keys sort by program ID
    '''
    # def extract_track_wise_pos_and_pitch_seq(remi_seq, inst):
        
    # Obtain all instruments of the bar
    insts = from_remi_get_insts(remi_seq) # All instruments, sorted by program id

    # Obtain track for each instrument
    ret = {}
    for inst in insts:
        track_seq = from_remi_get_pos_and_pitch_seq_of_track(remi_seq, inst)
        ret[inst] = track_seq

    return ret


def from_remi_reordered_opd_dict_merge_to_single_sequence(opd_dict):
    '''
    NOTE: Only work for a single bar

    Merge the position and pitch sequence of each track (dict format) 
    to a single position and pitch sequence
    Deduplicate repeated note by default
    '''
    # Flatten the input (and remove inst token)
    inp = []
    for inst in opd_dict:
        inp.extend(opd_dict[inst])
    
    all_pos = [tok for tok in inp if tok.startswith('o-')]
    all_pos = list(set(all_pos))
    all_pos = sorted(all_pos, key=lambda x: int(x.split('-')[1]))

    pitch_of_pos_dict = {k: [] for k in all_pos}
    for tok in inp:
        if tok.startswith('o-'):
            cur_pos = tok
        elif tok.startswith('p-'):
            cur_pitch = tok
        elif tok.startswith('d-'):
            cur_dur = tok

            pitch_of_pos_dict[cur_pos].append((cur_pitch, cur_dur))
    
    # Sort by pitch id
    ret = []
    for pos in pitch_of_pos_dict:
        note_of_pos = pitch_of_pos_dict[pos]

        # Remove repeated notes that have same pitch
        dur_of_notes = {} # k: pitch token    v: duration ID
        for note in note_of_pos:
            pitch = note[0]
            dur = note[1]
            dur_id = int(dur.split('-')[1])
            if pitch not in dur_of_notes:
                dur_of_notes[pitch] = dur_id
            else:
                dur_of_notes[pitch] = max(dur_of_notes[pitch], dur_id)

        # Convert duration back to token
        dur_of_notes = {k: 'd-' + str(v) for k,v in dur_of_notes.items()}

        # Sort by pitch id
        notes_sorted = [(k, dur_of_notes[k]) for k in sorted(dur_of_notes, key=lambda x: from_pitch_token_get_pitch_id(x), reverse=True)]
        ret.append(pos)
        for note in notes_sorted:
            ret.extend(note)

    return ret



def from_remi_get_pitch_set_of_track(remi_seq, inst, sort=True):
        '''
        Note: only works for a bar

        Extract the track-wise remi sequence for a given instrument
        From a multi-track remi sequence
        '''
        pitch_seq = from_remi_get_pitch_seq_of_track(remi_seq, inst)
        ret = list(set(pitch_seq))
        
        if sort is True:
            # Sort by pitch id, from small to large
            ret = sorted(ret, key=lambda x: int(x.split('-')[1]))

        return ret


def from_remi_mbar_get_pitch_seq_of_track(remi_seq, inst):
    '''
    Multiple-bar version of from_remi_get_pitch_seq_of_track
    '''
    bar_indices = from_remi_get_bar_idx(remi_seq)
    ret = []
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_seq = remi_seq[bar_start_idx:bar_end_idx]

        bar_pitch_seq = from_remi_get_pitch_seq_of_track(bar_seq, inst)
        ret.extend(bar_pitch_seq)
        ret.append('b-1')

    return ret


def from_remi_get_pitch_seq_of_track(remi_seq, inst):
        '''
        Note: only works for a bar

        Extract the track-wise remi sequence for a given instrument
        From a multi-track remi sequence
        '''
        ret = []
        in_seq = False
        for tok in remi_seq:
            if tok.startswith('i-'):
                if tok == inst: 
                    in_seq = True
                else:
                    in_seq = False
            elif tok.startswith('p-'):
                if in_seq:
                    ret.append(tok)

        return ret


def from_remi_get_pitch_seq_of_multiple_insts(remi_seq, insts):
        '''
        Note: only works for a bar

        Extract the track-wise remi sequence for a given instrument
        From a multi-track remi sequence
        '''
        ret = []
        in_seq = False
        for tok in remi_seq:
            if tok.startswith('i-'):
                if tok in insts: 
                    in_seq = True
                else:
                    in_seq = False
            elif tok.startswith('p-'):
                if in_seq:
                    ret.append(tok)

        return ret


def from_remi_get_pitch_seq_per_track(remi_seq, dedup=False):
    '''
    NOTE: Can work for multiple bars, but no bar line token returned

    Extract all track-wise remi sequences from the multi track remi
    Return a dict of seqs for each track, key is instrument token
    Keys sort by program ID
    '''

    # Obtain all instruments of the bar
    insts = from_remi_get_insts(remi_seq) # All instruments, sorted by program id

    # Obtain track for each instrument
    ret = {}
    for inst in insts:
        track_seq = from_remi_get_pitch_seq_of_track(remi_seq, inst)
        ret[inst] = track_seq

    if dedup is True:
        for inst in ret:
            ret[inst] = list(set(ret[inst]))
            # Sort by pitch id, from large to small
            ret[inst] = sorted(ret[inst], key=lambda x: int(x.split('-')[1]), reverse=True)

    return ret


def from_remi_get_range_and_pos_of_track_seq(remi_seq, keep_insts=None, with_drum=False, drum_pos=False, drum_range=False, remi_reordered=False):
    '''
    Get the pitch range and position of each instrument in the remi sequence
    e.g., i-0 p-20 p-50 o-0 o-2 i-2 ...
    '''
    if with_drum is True:
        assert drum_pos is True or drum_range is True
    
    if keep_insts is not None:
        assert 'i-128' not in remi_seq, 'Drum is not supported in keep_insts mode'

    range_of_track = from_remi_get_range_of_track_dict(remi_seq)
    pos_of_track = from_remi_get_pos_per_track_dict(remi_seq, remi_reordered=remi_reordered)
    ret = []
    is_drum = False
    # ensure range and pos are in the same order

    if keep_insts is None:
        for inst in range_of_track:
            if keep_insts is not None and inst not in keep_insts:
                continue

            if inst == 'i-128':
                is_drum = True
                if with_drum is False:
                    continue

            ret.append(inst)
            
            if is_drum:
                if drum_range is True:
                    # Get all pitch (not range here) of drum
                    drum_notes = from_remi_get_pitch_set_of_track(remi_seq, inst, sort=True)
                    ret.extend(drum_notes)
                if drum_pos is True:
                    ret.extend(pos_of_track[inst])
            
            else:
                ret.extend(range_of_track[inst])
                ret.extend(pos_of_track[inst])

            is_drum = False
    else:
        for inst in keep_insts:
            if inst in range_of_track:
                ret.append(inst)
                ret.extend(range_of_track[inst])
                ret.extend(pos_of_track[inst])
    
    return ret



def from_remi_get_range_of_track_seq(remi_seq, with_drum=False, keep_insts=None):
    '''
    When keep_insts is not none, return will be indexed by inst order in keep_insts
    '''
    pitch_range_of_track_dict = from_remi_get_range_of_track_dict(remi_seq)
    ret = []
    if keep_insts is None:
        for inst in pitch_range_of_track_dict:
            if inst == 'i-128':
                if with_drum is False:
                    continue

            ret.append(inst)
            ret.extend(pitch_range_of_track_dict[inst])    

    else:
        for inst in keep_insts:
            if inst in pitch_range_of_track_dict:
                ret.append(inst)
                ret.extend(pitch_range_of_track_dict[inst])

    return ret


def from_remi_get_range_of_track_dict(remi_seq, return_int=False):
    '''
    Get the pitch range of each instrument in the remi sequence
    The pitch range is a tuple with two elements, the min and max pitch id
    {
        inst: (min_pitch_id, max_pitch_id)
    }
    Save to a dictionary
    '''
    pitch_seq_of_inst = from_remi_get_pitch_seq_per_track(remi_seq)
    
    pitch_range_of_track = {inst: (256, 0) for inst in pitch_seq_of_inst}

    for inst in pitch_seq_of_inst:
        pitch_seq = pitch_seq_of_inst[inst]
        for pitch in pitch_seq:
            pitch_id = from_pitch_token_get_pitch_id(pitch)
            if pitch_id < pitch_range_of_track[inst][0]:
                pitch_range_of_track[inst] = (pitch_id, pitch_range_of_track[inst][1])
            if pitch_id > pitch_range_of_track[inst][1]:
                pitch_range_of_track[inst] = (pitch_range_of_track[inst][0], pitch_id)

    if return_int is False:
        # Convert tuple of int to tuple of inst token
        pitch_range_of_track = {inst: (f'p-{pitch_range_of_track[inst][0]}', f'p-{pitch_range_of_track[inst][1]}') for inst in pitch_range_of_track}
    
    return pitch_range_of_track


def from_remi_get_pitch_seq_global_mbar(remi_seq):
    '''
    Obtain pitch sequence from remi,
    I.e., a list of 'p-X' sequence from remi
    No further process is done
    '''
    ret = []

    bar_indices = from_remi_get_bar_idx(remi_seq)
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_seq = remi_seq[bar_start_idx:bar_end_idx]

        pitch_seq = from_remi_get_pitch_seq_global(bar_seq)
        ret.extend(pitch_seq)
        ret.append('b-1')

    return ret


def from_remi_get_pitch_seq_global(remi_seq):
    '''
    NOTE: Only works for one bar.

    Obtain pitch sequence from remi,
    I.e., a list of 'p-X' sequence from remi
    No further process is done
    '''
    # Old version, may not suitable for reordered output
    # ret = []
    # for tok in remi_seq:
    #     if tok.startswith('p-'):
    #         ret.append(tok)

    # 06-01 ver, suitable for reordered output
    pitch_of_pos_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=False)
    ret = []
    for pos in pitch_of_pos_dict:
        ret.extend(pitch_of_pos_dict[pos])

    return ret

def from_remi_get_pitch_seq_flattened(remi_seq):
    '''
    Obtain pitch sequence from remi,
    I.e., a list of 'p-X' sequence from remi
    No further process is done
    '''
    # 06-01 ver, suitable for reordered output
    pitch_of_pos_dict = from_remi_get_pitch_of_pos_dict(remi_seq, sort_pitch=True, flatten=True)
    ret = []
    for pos in pitch_of_pos_dict:
        ret.extend(pitch_of_pos_dict[pos])

    return ret
