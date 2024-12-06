import os
import sys
sys.path.append("..")

import random
import numpy as np
from utils_texture.texture_tools import (
    get_time_function_from_remi_one_bar,
    get_onset_density_of_a_bar_from_remi,
    tokenize_onset_density_one_bar,
)

from typing import List, Tuple, Dict

def get_pitch_id_from_pitch_token(tok_p):
    return int(tok_p.split("-")[1])

def read_remi(fp, split=True, remove_input=False):
    with open(fp) as f:
        remi_str = f.readline().strip()

    if remove_input:
        remi_str = remi_str.split(' <sep> ')[1]

    if split:
        remi_seq = remi_str.split(' ')
    else:
        remi_seq = remi_str
    return remi_seq

def save_remi_seq(remi_seq, fp):
    with open(fp, 'w') as f:
        f.write(' '.join(remi_seq))


def get_bar_idx_from_remi(remi_seq):
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

def get_bar_idx_from_condition(condition_seq):
    '''
    Because in the condition, each bar is segmented also by the 'b-1' token, 
    Just call the get_bar_idx_from_remi function to do the work.
    '''
    ret = get_bar_idx_from_remi(condition_seq)
    return ret

def get_bar_idx_from_remi_list(remi_seq):
    """
    Return the bar index in a list of tuples
    """

    # Get the starting token of each bar
    start_token_index_of_the_bar = 0
    bar_indices = []

    # bars_token_positions[bar_id] = (start token index of this bar, start token index of next bar)
    for idx, token in enumerate(remi_seq):
        if token == "b-1":
            start_token_index_of_next_bar = idx + 1
            bar_indices.append(
                (start_token_index_of_the_bar, start_token_index_of_next_bar)
            )

            # Go to the next bar
            start_token_index_of_the_bar = start_token_index_of_next_bar
    return bar_indices


def obtain_input_tokens_from_remi_sss_ipo_tf_hist(remi_seq):
    """
    Do the entire feature extraction + tokenization process for sss ipo tf hist model
    Obtain the input sequence from target remi sequence

    Prerequisite: remi_seq contains info strictly for two bars
    """
    input_tokens = []

    b_1_indices = get_bar_idx_from_remi(remi_seq)
    num_bars = len(b_1_indices)

    if num_bars != 2:
        raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    # Get history
    bar_start_idx, bar_end_idx = b_1_indices[0]
    hist = remi_seq[bar_start_idx:bar_end_idx]

    # Use the second bar as the target bar
    tgt_bar_indices = b_1_indices[1]
    bar_start_idx, bar_end_idx = tgt_bar_indices
    bar_remi_seq = remi_seq[bar_start_idx:bar_end_idx]

    inst_tokens = set()
    pitch_tokens = []

    """ collate instrument and pitch tokens """
    for tok in bar_remi_seq:
        if tok.startswith("i-"):
            inst_tokens.add(tok)
        elif tok.startswith("p-"):
            pitch_tokens.append(tok)

    # Convert inst token to list
    inst_tokens = list(inst_tokens)
    inst_tokens = sorted(
        inst_tokens, key=lambda x: int(x.split("-")[1])
    )  # sort by inst id

    input_tokens.append("INS")
    input_tokens.extend(inst_tokens)
    input_tokens.append("PITCH")
    input_tokens.extend(pitch_tokens)

    # Obtain rhythm pattern and add to result
    onset_density = get_onset_density_of_a_bar_from_remi(bar_remi_seq)
    txt_tokens = tokenize_onset_density_one_bar(onset_density, quantize=True)
    input_tokens.append("TF")
    input_tokens.extend(txt_tokens)

    # Add the history to result
    input_tokens.append("HIST")
    input_tokens.extend(hist)

    # No bar line token needed for the condition part
    # input_tokens.append("b-1")

    return input_tokens


def obtain_input_tokens_from_remi_target_bar_sss_ipo_tf_hist(remi_seq):
    """
    Do the entire feature extraction + tokenization process for sss ipo tf hist model
    Obtain the input sequence from target remi sequence

    Prerequisite: remi_seq contains info strictly for two bars
    """
    input_tokens = []

    bar_remi_seq = remi_seq

    inst_tokens = set()
    pitch_tokens = []

    """ collate instrument and pitch tokens """
    for tok in bar_remi_seq:
        if tok.startswith("i-"):
            inst_tokens.add(tok)
        elif tok.startswith("p-"):
            pitch_tokens.append(tok)

    # Convert inst token to list
    inst_tokens = list(inst_tokens)
    inst_tokens = sorted(
        inst_tokens, key=lambda x: int(x.split("-")[1])
    )  # sort by inst id

    input_tokens.append("INS")
    input_tokens.extend(inst_tokens)
    input_tokens.append("PITCH")
    input_tokens.extend(pitch_tokens)

    # Obtain rhythm pattern and add to result
    onset_density = get_onset_density_of_a_bar_from_remi(bar_remi_seq)
    txt_tokens = tokenize_onset_density_one_bar(onset_density, quantize=True)
    input_tokens.append("TF")
    input_tokens.extend(txt_tokens)

    return input_tokens


def obtain_target_tokens_from_remi_sss_ipo_tf_hist(remi_seq):
    """
    Get the target output part from a 2-bar remi_seq
    """
    b_1_indices = get_bar_idx_from_remi(remi_seq)
    num_bars = len(b_1_indices)

    if num_bars != 2:
        raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    # Use the second bar as the target bar
    tgt_bar_indices = b_1_indices[1]
    bar_start_idx, bar_end_idx = tgt_bar_indices
    bar_remi_seq = remi_seq[bar_start_idx:bar_end_idx]

    return bar_remi_seq

def obtain_input_tokens_from_remi_seg_for_sss(remi_seq):
    """
    Do the entire feature extraction + tokenization process for sss model
    Obtain the input sequence from target remi sequence

    Prerequisite: remi_seq contains info strictly for two bars
    """
    b_1_indices = get_bar_idx_from_remi(remi_seq)
    num_bars = len(b_1_indices)
    if num_bars != 2:
        raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    ''' Feature Extraction for the Segment '''
    seg_feat = []
    # Iterate over all bars
    for bar_id in b_1_indices:
        bar_feat = {}
        bar_start_idx, bar_end_idx = b_1_indices[bar_id]
        bar_remi_seq = remi_seq[bar_start_idx:bar_end_idx]

        """ Obtain Inst info """
        bar_inst_tokens = get_inst_in_remi(bar_remi_seq)
        bar_feat['inst_tokens'] = bar_inst_tokens

        ''' Obtain pitch seq (without inst and dur info) '''
        bar_pitch_of_each_pos = get_pitch_of_each_pos_for_a_bar_remi(bar_remi_seq, sort_pitch=True)
        bar_feat['pitch_dict'] = bar_pitch_of_each_pos

        seg_feat.append(bar_feat)
        
    ''' Tokenization for the segment '''
    seg_inp_seq = []
    for bar_feat in seg_feat:
        bar_inst_tokens = bar_feat['inst_tokens']
        bar_pitch_of_each_pos = bar_feat['pitch_dict']

        bar_pitch_seq = []
        # Convert pitch dict (of each pos) to pitch seq (flatten, sort by pos)
        for pos_tok in bar_pitch_of_each_pos:
            pitch_tok_of_the_pos = bar_pitch_of_each_pos[pos_tok]
            bar_pitch_seq.append(pos_tok)
            # May add some pitch re-ordering here, but not for now
            bar_pitch_seq.extend(pitch_tok_of_the_pos)

        seg_inp_seq.append("INS")
        seg_inp_seq.extend(bar_inst_tokens)
        seg_inp_seq.append('PITCH')
        seg_inp_seq.extend(bar_pitch_seq)
        seg_inp_seq.append('b-1')


    return seg_inp_seq

def obtain_features_from_output_bar_for_sss(out_seq):
    """
    Do the entire feature extraction + tokenization (convert to token format) process for sss model
    Obtain the input sequence from target remi sequence

    Prerequisite: output is one bar
    """
    # Get instrument
    inst_seq = get_inst_in_remi(out_seq)

    # Get pitch seq (without inst and duration info)
    pitch_seq_info = get_pitch_of_each_pos_for_a_bar_remi(out_seq, sort_pitch=True)
    pitch_seq = []
    for pos in pitch_seq_info:
        pitch_seq.append(pos)
        pitch_seq.extend(pitch_seq_info[pos])

    ret = {
        'pitch_seq': pitch_seq,
        'inst_seq': inst_seq,
    }    
    

    return ret

def obtain_input_tokens_from_remi_seg_for_sss_with_hist_and_texture(remi_seq):
    """
    Do the entire feature extraction + tokenization (convert to token format) process for sss model
    Obtain the input sequence from target remi sequence

    The instrument sequence contain texture control.
    - Three types of texture: 
        txt-0: lines
        txt-1: arpeggio
        txt-2: pad 
        txt-3: arpeggio-pad

    Prerequisite: remi_seq contains info strictly for two bars
    """
    b_1_indices = get_bar_idx_from_remi(remi_seq)
    num_bars = len(b_1_indices)
    if num_bars != 2:
        raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    # Get history
    hist_start_idx, hist_end_idx = b_1_indices[0]
    hist_seq = remi_seq[hist_start_idx:hist_end_idx]

    # Get the target bar
    tgt_start_idx, tgt_end_idx = b_1_indices[1]
    tgt_remi_seq = remi_seq[tgt_start_idx:tgt_end_idx]

    # Get instrument
    inst_seq = get_inst_in_remi_with_texture(tgt_remi_seq)

    # Get pitch seq (without inst and duration info)
    pitch_seq_info = get_pitch_of_each_pos_for_a_bar_remi(tgt_remi_seq, sort_pitch=True)
    pitch_seq = []
    for pos in pitch_seq_info:
        pitch_seq.append(pos)
        pitch_seq.extend(pitch_seq_info[pos])

        
    ''' Tokenization for the segment '''
    condition_seq = []

    condition_seq.append('PITCH')
    condition_seq.extend(pitch_seq)
    condition_seq.append("INS")
    condition_seq.extend(inst_seq)
    condition_seq.append('HIST')
    condition_seq.extend(hist_seq)
    # no additional b-1 token

    return condition_seq, tgt_remi_seq

def obtain_input_tokens_from_remi_seg_for_sss_with_hist_and_voice(remi_seq):
    """
    Do the entire feature extraction + tokenization (convert to token format) process for sss model
    Obtain the input sequence from target remi sequence

    The instrument sequence contain voice control. 
    - Instrument that is in front, has higher average pitch than later instruments.
    - Drum is always in the end.

    Prerequisite: remi_seq contains info strictly for two bars
    """
    b_1_indices = get_bar_idx_from_remi(remi_seq)
    num_bars = len(b_1_indices)
    if num_bars != 2:
        raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    # Get history
    hist_start_idx, hist_end_idx = b_1_indices[0]
    hist_seq = remi_seq[hist_start_idx:hist_end_idx]

    # Get the target bar
    tgt_start_idx, tgt_end_idx = b_1_indices[1]
    tgt_remi_seq = remi_seq[tgt_start_idx:tgt_end_idx]

    # Get instrument
    inst_seq = get_inst_in_remi_with_voice(tgt_remi_seq)

    # Get pitch seq (without inst and duration info)
    pitch_seq_info = get_pitch_of_each_pos_for_a_bar_remi(tgt_remi_seq, sort_pitch=True)
    pitch_seq = []
    for pos in pitch_seq_info:
        pitch_seq.append(pos)
        pitch_seq.extend(pitch_seq_info[pos])

        
    ''' Tokenization for the segment '''
    condition_seq = []

    condition_seq.append('PITCH')
    condition_seq.extend(pitch_seq)
    condition_seq.append("INS")
    condition_seq.extend(inst_seq)
    condition_seq.append('HIST')
    condition_seq.extend(hist_seq)
    # no additional b-1 token

    return condition_seq, tgt_remi_seq

def obtain_input_tokens_from_remi_seg_for_sss_with_hist(remi_seq):
    """
    Do the entire feature extraction + tokenization (convert to token format) process for sss model
    Obtain the input sequence from target remi sequence

    Prerequisite: remi_seq contains info strictly for two bars
    """
    b_1_indices = get_bar_idx_from_remi(remi_seq)
    num_bars = len(b_1_indices)
    if num_bars != 2:
        raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    # Get history
    hist_start_idx, hist_end_idx = b_1_indices[0]
    hist_seq = remi_seq[hist_start_idx:hist_end_idx]

    # Get the target bar
    tgt_start_idx, tgt_end_idx = b_1_indices[1]
    tgt_remi_seq = remi_seq[tgt_start_idx:tgt_end_idx]

    # Get instrument
    inst_seq = get_inst_in_remi(tgt_remi_seq)

    # Get pitch seq (without inst and duration info)
    pitch_seq_info = get_pitch_of_each_pos_for_a_bar_remi(tgt_remi_seq, sort_pitch=True)
    pitch_seq = []
    for pos in pitch_seq_info:
        pitch_seq.append(pos)
        pitch_seq.extend(pitch_seq_info[pos])

        
    ''' Tokenization for the segment '''
    condition_seq = []

    condition_seq.append('PITCH')
    condition_seq.extend(pitch_seq)
    condition_seq.append("INS")
    condition_seq.extend(inst_seq)
    condition_seq.append('HIST')
    condition_seq.extend(hist_seq)
    # no additional b-1 token

    return condition_seq, tgt_remi_seq

def obtain_input_tokens_from_remi_seg_for_sss_no_hist(remi_seq):
    """
    Do the entire feature extraction + tokenization (convert to token format) process for sss model
    Obtain the input sequence from target remi sequence

    Prerequisite: remi_seq contains info strictly for two bars
    """
    b_1_indices = get_bar_idx_from_remi(remi_seq)
    num_bars = len(b_1_indices)

    if num_bars == 1:
        remi_seq.insert(0, 'b-1')
        b_1_indices = get_bar_idx_from_remi(remi_seq)
    elif num_bars != 2:
        raise Exception("Num bar issue: {} bars in the sample".format(num_bars))

    # Get history
    hist_start_idx, hist_end_idx = b_1_indices[0]
    hist_seq = remi_seq[hist_start_idx:hist_end_idx]

    # Get the target bar
    tgt_start_idx, tgt_end_idx = b_1_indices[1]
    tgt_remi_seq = remi_seq[tgt_start_idx:tgt_end_idx]

    # Get instrument
    inst_seq = get_inst_in_remi(tgt_remi_seq)

    # Get pitch seq (without inst and duration info)
    pitch_seq_info = get_pitch_of_each_pos_for_a_bar_remi(tgt_remi_seq, sort_pitch=True)
    pitch_seq = []
    for pos in pitch_seq_info:
        pitch_seq.append(pos)
        pitch_seq.extend(pitch_seq_info[pos])

        
    ''' Tokenization for the segment '''
    condition_seq = []

    condition_seq.append('PITCH')
    condition_seq.extend(pitch_seq)
    condition_seq.append("INS")
    condition_seq.extend(inst_seq)
    # no additional b-1 token

    return condition_seq, tgt_remi_seq


def get_inst_in_remi(remi_seq):
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
    inst = sorted(inst, key=lambda x: int(x.split("-")[1]))  # sort by inst id
    return inst

def get_inst_in_remi_with_texture(remi_seq):
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
    pos_and_pitch_of_track = extract_track_wise_pos_and_pitch_seqs(remi_seq)
    if 'i-128' in pos_and_pitch_of_track:
        has_drum = True
        pos_and_pitch_of_track.pop('i-128')
    else:
        has_drum = False
    
    # Compute playing style for each instrument
    style_of_track = {}
    for inst in pos_and_pitch_of_track:
        pos_and_pitch_seq = pos_and_pitch_of_track[inst]
        style = get_playing_style_of_track(pos_and_pitch_seq)
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

def get_pitch_of_pos(pos_and_pitch_seq):
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

def get_playing_style_of_track(pos_and_pitch_seq):
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
        if pitch_range >= 12: # If >= an octave, arpeggio
            txt = 'txt-1'
        else: # else, line
            txt = 'txt-0'
    else:
        raise Exception('Illegal texture')

    return txt

def get_inst_in_remi_with_voice(remi_seq):
    '''
    Obtain all instrument in the input remi sequence
    Return a list of instrument tokens
    Instrument on the left have higher average pitch than that on the right
    '''
    # Obtain pitch info of each instrument, save in a dict
    pitch_of_track = extract_track_wise_pitch_seqs(remi_seq)
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


def get_pitch_of_each_pos_for_a_bar_remi(bar_remi_seq, sort_pitch=True):
    """Obtain pitch of each position inside a same bar

    Args:
        bar_remi_seq (list): a list of remi tokens

    Raises:
        Exception: If multiple bars feeded as the parameter, will raise exception.
    """    
    if bar_remi_seq.count('b-1') > 1:
        # print('bar num > 1')
        raise Exception('Multiple bars feed to this function: {} bars'.format(bar_remi_seq.count('b-1')))
    elif bar_remi_seq.count('b-1') < 1:
        bar_remi_seq.append('b-1')
    
    ret = {}
    cur_pos = None
    pitch_of_the_pos = []
    """ Only retain position, pitch, and bar line """
    for tok in bar_remi_seq:
        if tok.startswith("o-"):
            # Add the pich of previous position to pitch seq
            if len(pitch_of_the_pos) > 0:
                if sort_pitch:
                    pitch_of_the_pos = sorted(
                        pitch_of_the_pos, 
                        key=lambda x: int(x.split("-")[1]), 
                        reverse=True
                    )  # Sort pitch from high to low
                ret[cur_pos] = pitch_of_the_pos
                pitch_of_the_pos = []
            
            # Update current pos
            # cur_pos = int(tok.split('-')[-1])
            cur_pos = tok
        elif tok.startswith("p-"):
            # pitch_tokens.append(tok)
            pitch_of_the_pos.append(tok)
    return ret


def sort_inst(inst_list):        
    sorted_list = sorted(inst_list, key=lambda x: int(x.split("-")[1]))
    return sorted_list





def extract_condition_for_sss_ipo_tf_hist_from_all_remi_segments(remi_seq):
    """
    Extract the condition info for each segment in the remi
    for sss ipo tf with history model
    """
    bar_indices = get_bar_idx_from_remi(remi_seq)
    num_bars = len(bar_indices)
    if num_bars == 0:
        raise Exception("Bar num = 0")

    # Obtain the indices of each segment
    segment_bar_num = 2
    segment_indices = []
    hop_bar = 1
    for start_bar_id in range(0, num_bars - segment_bar_num + 1, hop_bar):  # 1 000 000
        bar_ids = [start_bar_id + i for i in range(segment_bar_num)]
        start_bar_id = bar_ids[0]
        end_bar_id = bar_ids[-1]
        segment_start_idx = bar_indices[start_bar_id][0]
        segment_end_idx = bar_indices[end_bar_id][1]
        segment_indices.append((segment_start_idx, segment_end_idx))

    all_seg_info = []
    for seg_start_idx, seg_end_idx in segment_indices:
        seg_remi = remi_seq[seg_start_idx:seg_end_idx]
        seg_info = {}
        seg_info["remi_seq"] = seg_remi

        # Obtain input tokens from segment remi
        from dataset_preparation.create_midi_only_dataset import (
            TwoBarDatasetPreparation,
        )

        input_tokens = obtain_input_tokens_from_remi_sss_ipo_tf_hist(seg_remi)
        seg_info["input_tokens"] = input_tokens

        all_seg_info.append(seg_info)

    return all_seg_info

def extract_condition_for_sss_for_all_segments(song_remi_seq, task_tokens=None):
    """
    Extract the condition info for each segments in the remi of the song
    for sss model

    Return: [
        {'remi_seq': ..., 'input_tokens': ...} # segment 1,
        ... # segment 2
    ]
    """
    bar_indices = get_bar_idx_from_remi(song_remi_seq)
    num_bars = len(bar_indices)
    if num_bars == 0:
        raise Exception("Bar num = 0")

    # Obtain the indices of each segment
    segment_bar_num = 2
    segment_indices = []
    hop_bar = 1
    for start_bar_id in range(0, num_bars - segment_bar_num + 1, hop_bar):  # 1 000 000
        bar_ids = [start_bar_id + i for i in range(segment_bar_num)]
        start_bar_id = bar_ids[0]
        end_bar_id = bar_ids[-1]
        segment_start_idx = bar_indices[start_bar_id][0]
        segment_end_idx = bar_indices[end_bar_id][1]
        segment_indices.append((segment_start_idx, segment_end_idx))

    # Loop over all segments
    all_seg_info = []
    for seg_start_idx, seg_end_idx in segment_indices:
        seg_remi = song_remi_seq[seg_start_idx:seg_end_idx]
        seg_info = {}
        seg_info["remi_seq"] = seg_remi

        # Obtain the input tokens for the remi of the segment
        input_tokens = obtain_input_tokens_from_remi_seg_for_sss(seg_remi)

        # Add task tokens if needed
        if task_tokens:
            input_tokens.extend(task_tokens)

        seg_info["input_tokens"] = input_tokens

        all_seg_info.append(seg_info)

    # 

    return all_seg_info


def filter_inst_in_condition(input_seq, new_inst_list):
    '''
    Delete any instrument tokens in input_seq that is not in new_inst_list
    '''
    new_inst = set(new_inst_list)
    ret = []

    if 'HIST' in input_seq:
        hist_pos = input_seq.index('HIST')
    else:
        hist_pos = 99999

    for tok in input_seq[:hist_pos]:
        if tok.startswith('i-'):        # For each instrument token
            if tok not in new_inst:     # If we have deleted it from target sequence
                continue                # We then delete it from input sequence
        ret.append(tok)

    # Recover history
    ret.extend(input_seq[hist_pos:])
    
    return ret

def get_pos_and_pitch_count_from_condition(condition_seq: List[str]) -> Tuple[int, int]:
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

def get_pos_id_from_pos_token(pos_tok: str) -> int:
    """Obtain the integer part of the position token

    Args:
        pos_tok (str): A position token

    Returns:
        int: The integer part
    """    
    assert pos_tok.startswith('o-')
    ret = int(pos_tok.split('-')[1])
    return ret

def split_condition_seq_segment(condition_seq: List[str]) -> List[Dict[str, List[str]]]:
    """Convert the format of the input condition sequence from list to a list of dict.

    Args:
        condition_seq (List[str]): The condition sequence of a 2-bar segment

    Returns:
        List[Dict[str, List[str]]]: The splitted condition seq. len(return) == num_bars. For each bar, the info is a dict, containing two keys: 'inst' and 'content'
    """    
    ret = []
    bar_indices = get_bar_idx_from_condition(condition_seq)
    for bar_id in bar_indices:
        bar_start_idx, bar_end_idx = bar_indices[bar_id]
        bar_condition = condition_seq[bar_start_idx:bar_end_idx]
        splitted_bar = split_condition_seq_bar(bar_condition)
        ret.append(splitted_bar)
    return ret

def split_condition_seq_bar(condition_seq: List[str]) -> Dict[str, List[str]]:
    """Convert the remi of a bar to a dict containing different aspect of the info

    Args:
        condition_seq (List[str]): remi of the bar

    Returns:
        Dict[str, List[str]]: splitted bar
    """    
    ret = {}
    pitch_seq_start_idx = condition_seq.index('PITCH')
    inst_subseq = condition_seq[:pitch_seq_start_idx]
    pitch_subseq = condition_seq[pitch_seq_start_idx:]
    ret = {
        'inst': inst_subseq,
        'content': pitch_subseq,
    }
    return ret

def unsplit_condition_seq_bar(bar_info: Dict[str, List[str]]) -> List[str]:
    """Recover the info of a bar from dict format to a sequence of token
    Undo the function split_condition_seq_bar

    Args:
        bar_info (Dict[str, List[str]]): info of a bar

    Returns:
        List[str]: corresponding condition sequence
    """    
    inst_subseq = bar_info['inst']
    pitch_subseq = bar_info['content']
    ret = inst_subseq + pitch_subseq
    return ret

def get_content_from_condition_seq(condition_seq: List[str]) -> List[str]:
    """Obtain the content part, i.e., the position and pitch subsequence, 

    Args:
        condition_seq (List[str]): _description_

    Returns:
        List[str]: _description_
    """    
    return None

def replace_inst_in_condition_seg(input_list: List[str], new_i_list: List[str]) -> List[str]:
    """Replace any instrument descriptors in the given condition to the new instrument list

    Args:
        input_list (List[str]): a list of condition tokens, of a segment containing multiple bars
        new_i_list (List[str]): a new instrument list to use

    Returns:
        List[str]: The condition sequence after modification. All bars' instrument part will be replaced to the desired new list.
    """    

    # 用于存储最终结果的列表
    result = []
    # 用于暂时存储子序列的列表
    temp_sequence = []
    # 标记是否处于INS和PITCH之间的序列
    in_sequence = False
    empty = True

    for i, element in enumerate(input_list):
        if element == "INS":
            # 遇到INS，开始记录序列
            in_sequence = True
            # 将INS加入到临时序列
            result.append(element)

            # If the bar is empty, i.e., no instruments
            if not input_list[i+1].startswith('i-'):
                empty=True
            else:
                empty=False
        elif element == "PITCH":
            # 遇到PITCH，结束当前序列的记录，但不包括PITCH本身
            in_sequence = False
            # 将处理过的序列添加到结果列表
            if not empty:
                result.extend(new_i_list)
            result.append(element)
            # 清空临时序列
            temp_sequence = []
        elif in_sequence:
            # 如果处于INS和PITCH之间，继续添加元素到临时序列
            # temp_sequence.append(element)
            pass
        else:
            # 如果不在任何子序列中，直接将元素添加到结果列表
            result.append(element)

    return result



def replace_hist(remi_seq, hist):
    '''
    Replace the HIST part inside a <condition> <sep> <target> sequence
    hist: a sequence of token
    '''
    hist_tok_pos = remi_seq.index('HIST')
    sep_pos = remi_seq.index('<sep>')
    ret = remi_seq[:hist_tok_pos+1] + hist + remi_seq[sep_pos:]
    return ret

def replace_inst(remi_seq, inst_spec):
    '''
    Replace the HIST part inside a <condition> <sep> <target> sequence
    hist: a sequence of token

    prerequisite: inst_spec sort by program id
    '''
    inst_tok_pos = remi_seq.index('INS')
    hist_tok_pos = remi_seq.index('HIST')
    ret = remi_seq[:inst_tok_pos+1] + inst_spec + remi_seq[hist_tok_pos:]
    return ret

def split_song_remi_to_segments(remi_seq):
    '''
    Split the remi sequence of a song to a list of 2-bar segments
    NOTE: An additional blank is insert in the beginning.
    '''
    t = ['b-1'] + remi_seq
    ret = []
    bar_indices = get_bar_idx_from_remi(t)
    for cur_bar_id in range(len(bar_indices)-1):
        bar1_start_idx, bar1_end_idx = bar_indices[cur_bar_id]
        next_bar_id = cur_bar_id + 1
        bar2_start_idx, bar2_end_idx = bar_indices[next_bar_id]
        ret.append(t[bar1_start_idx:bar2_end_idx])
    return ret

def extract_track_wise_pos_and_pitch_seqs(remi_seq):
    '''
    Note: only works for a bar

    Extract all track-wise remi sequences from the multi track remi
    Return a dict of seqs for each track, key is instrument token
    Keys sort by program ID
    '''
    # Obtain all instruments of the bar
    insts = get_inst_in_remi(remi_seq) # All instruments, sorted by program id

    # Obtain track for each instrument
    ret = {}
    for inst in insts:
        track_seq = extract_track_wise_pos_and_pitch_seq(remi_seq, inst)
        ret[inst] = track_seq

    return ret

def extract_track_wise_pos_and_pitch_seq(remi_seq, inst):
    '''
    Note: only works for a bar

    Extract the track-wise remi sequence for a given instrument
    From a multi-track remi sequence
    '''
    ret = []
    in_seq = False
    cur_pos = 'o-0'
    for tok in remi_seq:
        if tok.startswith('o-'):
            cur_pos = tok
        elif tok.startswith('i-'):
            if tok == inst: 
                in_seq = True
                ret.append(cur_pos)
            else:
                in_seq = False
        elif tok.startswith('p-'):
            if in_seq:
                ret.append(tok)

    return ret

def extract_track_wise_pitch_seqs(remi_seq):
    '''
    Note: only works for a bar

    Extract all track-wise remi sequences from the multi track remi
    Return a dict of seqs for each track, key is instrument token
    Keys sort by program ID
    '''
    # Obtain all instruments of the bar
    insts = get_inst_in_remi(remi_seq) # All instruments, sorted by program id

    # Obtain track for each instrument
    ret = {}
    for inst in insts:
        track_seq = extract_track_wise_pitch_seq(remi_seq, inst)
        ret[inst] = track_seq

    return ret

def extract_track_wise_pitch_seq(remi_seq, inst):
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

def extract_pitch_seq(remi_seq):
    '''
    Obtain pitch sequence from remi,
    I.e., a list of 'p-X' sequence from remi
    No further process is done
    '''
    ret = []
    for tok in remi_seq:
        if tok.startswith('p-'):
            ret.append(tok)
    return ret

class RemiAugment:
    '''
    This class define several modification operations to the remi sequence "<condition> <sep> <target>".
    Contains 5 different tasks, and 2 additional augmentation operations.
    '''

    def __init__(self) -> None:
        self.tasks = [
            self.task1_reconstruction,
            self.task2_content_simplification,
            self.task3_content_elaboration
        ]
        self.pitch_reorder = False
        self.pitch_shift = False


    def augment_remi(self, condition_seq, remi_seq):
        '''
        Conduct the task selection and augmentation
        '''

        # # For debugging
        # if len(remi_seq) > 2:
        #     a = 1

        # Modify input and output according one specific task
        task = random.choice(self.tasks)
        condition_seq, remi_seq = task(condition_seq, remi_seq)

        # Data augmentation
        # Aug 1: instrument aug
        t = random.uniform(0, 1)
        if t > 0.6666:
            condition_seq, remi_seq = self.aug_inst_add_one(condition_seq, remi_seq)
        elif t > 0.3333:
            condition_seq, remi_seq = self.aug_inst_del_one(condition_seq, remi_seq)
        else:
            pass # (1/3 chance input content same as output)

        # TODO: pitch reorder # May not be necessary ... ?

        # TODO: pitch shift

        return condition_seq, remi_seq
    
    def task1_reconstruction(self, condition_seq, remi_seq):
        # Append task tokens (to the end of condition sequence)
        task_tokens = ['X-0']
        condition_seq.extend(task_tokens)

        return condition_seq, remi_seq
    
    def deprecated_task2_inst_set_retrieval(self, condition_seq, remi_seq):
        # Track deletion, in target sequence (changes will reflect in condition seq by construct it again)
        inst = get_inst_in_remi(remi_seq)
        num_insts = len(inst)
        if num_insts == 0:
            return [], remi_seq

        # Determine new number of instruments
        # num_inst_del = min(np.random.poisson(lam=1), num_insts)  # number of instrument to remove    # Implementation from augtrack experiment
        
        num_inst_del = np.random.poisson(lam=num_insts//2)  # number of instrument to remove. In average, remove half of the instruments
        num_inst_del = min(num_inst_del, num_insts - 1) # delete num_insts - 1 instruments at most
        num_inst_del = max(num_inst_del, 1) # delete 1 instrument at least
        num_inst_new = num_insts - num_inst_del

        # Determine the instrument to retain
        new_inst = random.sample(inst, num_inst_new)

        # Modify the remi, delete notes do not needed.
        remi_seq = self.retain_specified_insts_in_remi(remi_seq, new_inst)
        new_inst = sort_inst(list(new_inst))

        # Modify inst in conditions according to track deletion augmentation
        condition_seq = filter_inst_in_condition(condition_seq, new_inst)

        # Append task tokens (to the end of condition sequence)
        task_tokens = ['A-0', 'X-2', 'C-1']
        condition_seq.extend(task_tokens)

        return condition_seq, remi_seq
    
    def deprecated_task3_inst_prediction(self, condition_seq, remi_seq):
        # Select an instrument to predict.
        insts = get_inst_in_remi(remi_seq)

        # When empty, return directly
        if len(insts) == 0:
            return condition_seq, remi_seq

        inst_to_pred = random.choice(insts)

        # Remove the instrument in target sequence, re-generate the condition seq
        tgt_seq_for_condition = self.remove_specified_insts_in_remi(remi_seq, [inst_to_pred])
        new_condition_seq = obtain_input_tokens_from_remi_seg_for_sss(tgt_seq_for_condition)
        new_condition_seq = replace_inst_in_condition_seg(new_condition_seq, [inst_to_pred])

        # Remove any other instrument from target sequence
        new_tgt_seq = self.retain_specified_insts_in_remi(remi_seq, [inst_to_pred])

        # Append task tokens (to the end of condition sequence)
        task_tokens = ['A-2', 'X-2', 'C-1']
        new_condition_seq.extend(task_tokens)

        return new_condition_seq, new_tgt_seq
    
    def task2_content_simplification(self, condition_seq: List[str], remi_seq: List[str]):
        """Adjust the input pitch sequence, adding more pitch and position tokens inside,
        So that the output content is simpler than input
        
        On average, 25% additional position tokens are added, each with additional pitch tokens that keeps the ratio of pitch:position
        Additionaly, 25% pitch tokens are randomly added to all positions

        NOTE: If condition is empty (not a single pitch), no modification will be maded. 
        
        Args:
            condition_seq (List[str]): condition sequence of a segment
            remi_seq (List[str]): remi sequence of the segment

        Returns:
            Modified condition_seq and original remi_seq
        """        
        new_segment_condition_seq = []
        bar_indices = get_bar_idx_from_condition(condition_seq)
        for bar_id in bar_indices:
            bar_start_idx, bar_end_idx = bar_indices[bar_id]
            bar_condition_seq = condition_seq[bar_start_idx:bar_end_idx]

            ''' Randomly add some position tokens: more complex rhythm '''
            # Achieved by introduce additional position tokens with a sequence of random pitch tokens.
            # Calculate the average pitch per position for the sample
            pos_cnt, pitch_cnt = get_pos_and_pitch_count_from_condition(condition_seq)
            if pitch_cnt == 0:
                new_segment_condition_seq.extend(bar_condition_seq)
                continue
            avg_p_per_o = int(pitch_cnt / pos_cnt) # should a value >= 1
            # Determine the number new positions: 
            avg_num_new_pos = max(1, pos_cnt // 4) # We expect 25% more positions added to the content
            num_new_pos = max(1, np.random.poisson(avg_num_new_pos))
            # For each new position,
            for i in range(num_new_pos):
                # Determine the locations of the new positions: random choice
                new_pos_tok = self.get_random_position_token()
                # Determine the number of pitch tokens for the new position
                num_pitch_token = max(1, np.random.poisson(lam=avg_p_per_o))
                # Prepare a subsequence of (o-X p-Y p-Z ...)
                p_subseq = self.get_random_pitch_tokens(n_tok=num_pitch_token)
                subseq = [new_pos_tok] + p_subseq
                # Insert the subsequence to the proper place in the input sequence
                bar_condition_seq = self.insert_subseq_to_condition_for_a_bar(bar_condition_seq, subseq)

            ''' Randomly add more pitch tokens: more complex harmony '''
            # Achieved by random adding pitch tokens to pitch seq in condition 
            # Determine the number of pitch to add
            avg_num_new_pitch = max(1, pitch_cnt // 4) # Random add 25% content to each position
            num_new_pitch = max(1, np.random.poisson(avg_num_new_pitch))
            
            # Obtain the location of the pitch sequence
            pitch_tok_idx = bar_condition_seq.index('PITCH')

            # For each additional token
            for i in range(num_new_pitch):
                # Random select a location in input
                idx = random.randint(pitch_tok_idx+1, len(bar_condition_seq)-1)
                # Insert it to the input sequence
                new_p_tok = self.get_random_pitch_token()
                bar_condition_seq.insert(idx, new_p_tok)

            new_segment_condition_seq.extend(bar_condition_seq)

        # Append task tokens (to the end of condition sequence)
        task_tokens = ['X-2']
        new_segment_condition_seq.extend(task_tokens)

        return new_segment_condition_seq, remi_seq
    
    def task3_content_elaboration(self, condition_seq, remi_seq):
        '''
        Adjust the input pitch sequence, delete some random content from pitch sequence
        So that the output content is more complex than input
        Keep in instrument prompt as-is. Keep the target sequence as-is.
        '''
        new_segment_condition_seq = []
        bar_indices = get_bar_idx_from_condition(condition_seq)
        for bar_id in bar_indices:
            bar_start_idx, bar_end_idx = bar_indices[bar_id]
            bar_condition_seq = condition_seq[bar_start_idx:bar_end_idx]
            _, pitch_cnt = get_pos_and_pitch_count_from_condition(bar_condition_seq)

            ''' Randomly delete pitch and position tokens '''
            content_start_idx = bar_condition_seq.index('PITCH') + 1
            content_end_idx = bar_condition_seq.index('b-1')

            if content_end_idx <= content_start_idx: # for empty bar, don't do anything
                new_segment_condition_seq.extend(bar_condition_seq)
                continue
            else: # If not empty, do the random deletion.
                # 截取指定索引范围的部分
                content_segment = bar_condition_seq[content_start_idx:content_end_idx]
                
                # 计算需要删除的元素数量
                avg_num_to_remove = pitch_cnt // 4
                num_to_remove = np.random.poisson(lam=avg_num_to_remove)
                num_to_remove = max(1, num_to_remove)
                num_to_remove = min(content_end_idx - content_start_idx, num_to_remove)
                
                # 随机选择要删除的元素索引
                indices_to_remove = random.sample(range(len(content_segment)), num_to_remove)
                
                # 删除选中的元素
                content_segment = [item for idx, item in enumerate(content_segment) if idx not in indices_to_remove]
                
                # 重建整个列表，保持其他部分不变
                new_bar_condition_seq = bar_condition_seq[:content_start_idx] + content_segment + bar_condition_seq[content_end_idx:]
                new_segment_condition_seq.extend(new_bar_condition_seq)

        # Append task tokens (to the end of condition sequence)
        task_tokens = ['X-1']
        new_segment_condition_seq.extend(task_tokens)

        return new_segment_condition_seq, remi_seq
    
    def deprecated_task6_content_elaboration_2(self, condition_seq, remi_seq):
        '''
        Adjust the input pitch sequence, delete content of a random instrument
        Keep in instrument prompt as-is. Keep the target sequence as-is.
        '''
        # Obtain the instruments in the target sequence (results sorted)
        insts = get_inst_in_remi(remi_seq)

        # When empty, return directly
        if len(insts) == 0:
            return condition_seq, remi_seq

        # Select an instrument to predict
        inst_to_pred = random.choice(insts)

        # Remove the instrument in target sequence, re-generate the condition seq
        tgt_seq_for_condition = self.remove_specified_insts_in_remi(remi_seq, [inst_to_pred])
        new_condition_seq = obtain_input_tokens_from_remi_seg_for_sss(tgt_seq_for_condition)

        # Recover the instrument prompt
        new_condition_seq = replace_inst_in_condition_seg(new_condition_seq, insts)

        # Append task tokens (to the end of condition sequence)
        task_tokens = ['A-1', 'X-1', 'C-0']
        new_condition_seq.extend(task_tokens)

        return new_condition_seq, remi_seq
    
    def aug_inst_add_one(self, condition_seq, remi_seq):
        '''
        Adjust condition so that the target has one more instrument than the input
        Delete the content from a certain instrument from input content
        '''
        # Obtain the instruments in the target sequence (results sorted)
        insts = get_inst_in_remi(remi_seq)

        # When empty, return directly
        if len(insts) == 0:
            return condition_seq, remi_seq

        # Select an instrument to predict
        inst_to_pred = random.choice(insts)

        # Remove the instrument in target sequence, re-generate the condition seq
        tgt_seq_for_condition = self.remove_specified_insts_in_remi(remi_seq, [inst_to_pred])
        new_condition_seq = obtain_input_tokens_from_remi_seg_for_sss(tgt_seq_for_condition)

        # Recover the instrument prompt
        new_condition_seq = replace_inst_in_condition_seg(new_condition_seq, insts)

        return new_condition_seq, remi_seq

    def aug_inst_del_one(self, condition_seq, remi_seq):
        '''
        Adjust the target sequence, delete one instrument from target
        Adjust the instrument prompt, delete corresponding instrument
        '''
        # Track deletion, in target sequence (changes will reflect in condition seq by construct it again)
        inst = get_inst_in_remi(remi_seq)
        num_insts = len(inst)
        if num_insts == 0:
            return [], remi_seq

        # Determine new number of instruments
        num_inst_del = 1
        num_inst_del = min(num_inst_del, num_insts - 1) # delete num_insts - 1 instruments at most
        num_inst_del = max(num_inst_del, 1) # delete 1 instrument at least
        num_inst_new = num_insts - num_inst_del

        # Determine the instrument to retain
        new_inst = random.sample(inst, num_inst_new)

        # Modify the remi, delete notes do not needed.
        remi_seq = self.retain_specified_insts_in_remi(remi_seq, new_inst) # TODO: might be buggy for 2-bar segment
        new_inst = sort_inst(list(new_inst))

        # Modify inst in conditions according to track deletion augmentation
        condition_seq = filter_inst_in_condition(condition_seq, new_inst)

        return condition_seq, remi_seq

    def retain_specified_insts_in_remi(self, remi_seq, inst_to_preserve: list):
        '''
        Modify a remi_seq so that it only contains notes from specified instruments
        '''

        new_inst = set(inst_to_preserve)

        # Retain notes only for selected instruments
        new_remi = []
        pos_buffer = []
        retain = True
        for tok in remi_seq:
            if tok.startswith("o-"):
                if len(pos_buffer) > 0:
                    new_remi.extend(pos_buffer)
                pos_buffer = [tok]
            elif tok.startswith("i-"):
                if tok in new_inst:  # continue until meet new instrument
                    retain = True
                    pos_buffer.append(tok)
                else:
                    retain = False
            elif tok.startswith("p-") or tok.startswith("d-"):
                if retain:
                    pos_buffer.append(tok)
            elif tok == "b-1":
                pos_buffer.append(tok)
                if len(pos_buffer) > 0:
                    new_remi.extend(pos_buffer)
                    pos_buffer = [] # When multiple bars in segment, need to clear the pos_buffer at the end of each bar

        # Filter out empty position tokens
        ret = []
        for i, tok in enumerate(new_remi):
            if tok.startswith('o-') and new_remi[i+1].startswith('o-'): # If we see a position token
                # And if it's followed by another position token
                continue
            else:
                ret.append(tok) # append to result in other cases

        return ret
    
    def remove_specified_insts_in_remi(self, remi_seq, inst_to_delete: list):
        '''
        Modify a remi_seq so that it only contains notes from specified instruments
        '''

        inst_to_delete = set(inst_to_delete)

        # Retain notes only for selected instruments
        new_remi = []
        pos_buffer = []
        retain = True
        for tok in remi_seq:
            if tok.startswith("o-"):
                if len(pos_buffer) > 0:
                    new_remi.extend(pos_buffer)
                pos_buffer = [tok]
            elif tok.startswith("i-"):
                if tok not in inst_to_delete:  # continue until meet new instrument
                    retain = True
                    pos_buffer.append(tok)
                else:
                    retain = False
            elif tok.startswith("p-") or tok.startswith("d-"):
                if retain:
                    pos_buffer.append(tok)
            elif tok == "b-1":
                pos_buffer.append(tok)
                if len(pos_buffer) > 0:
                    new_remi.extend(pos_buffer)
                    pos_buffer = [] # When multiple bars in segment, need to clear the pos_buffer at the end of each bar

        # Filter out empty position tokens
        ret = []
        for i, tok in enumerate(new_remi):
            if tok.startswith('o-') and new_remi[i+1].startswith('o-'): # If we see a position token
                # And if it's followed by another position token
                continue
            else:
                ret.append(tok) # append to result in other cases

        return ret

    def get_random_pitch_token(self) -> str:
        """Obtain a random pitch token

        Returns:
            str: A random pitch token in the supported vocab of MuseCoco (p-0 ~ p-255)
        """        
        p_value = random.randint(0, 255)
        ret = 'p-{}'.format(p_value)
        return ret
    
    def get_random_position_token(self) -> str:
        """Obtain a random position token

        Returns:
            str: A random position token in the supported vocab of MuseCoco (o-0 ~ o-47) (majority)
        """     
        o_value = random.randint(0, 47)
        ret = 'o-{}'.format(o_value)
        return ret

    def get_random_pitch_tokens(self, n_tok: int) -> List[str]:
        """Obtain a list of random pitch tokens

        Args:
            n_tok (int): the number of pitch tokens we want in the returned list.
        Returns:
            List[str]: a list of pitch tokens. len(return) == n_tok.
        """        
        ret = random.choices(range(256), k=n_tok)
        ret.sort(reverse=True)
        ret = ['p-{}'.format(i) for i in ret]
        return ret
    
    def insert_subseq_to_condition_for_a_bar(self, condition_seq: List[str], subseq: List[str]) -> List[str]:
        """Insert a subsequence of pitch tokens (with position) to the proper location within the original condition sequence

        Args:
            condition_seq (List[str]): the original condition sequence
            subseq (List[str]): the new position and pitch token sequence to be inserted

        Returns:
            List[str]: condition sequence after insertion
        """        
        # Get the position id of the subsequence
        pos_id = get_pos_id_from_pos_token(subseq[0])

        # Obtain the pitch sequence of the condition
        bar_info = split_condition_seq_bar(condition_seq)
        pitch_pos_seq = bar_info['content']

        # Find the index of the first position token in the condition sequence that is larger than the pos_id
        idx = None # index of the position token, before which to insert the sub sequence
        for i, tok in enumerate(pitch_pos_seq):
            if tok.startswith('o-'):
                cur_pos_id = get_pos_id_from_pos_token(tok)
                if cur_pos_id > pos_id:
                    idx = i
                    break
        
        # If find a bigger position, insert the subsequence before that position
        if idx != None:
            new_content = pitch_pos_seq[:idx] + subseq + pitch_pos_seq[idx:]
                
        # If not find a bigger position, append the subsequence to the original pitch sequence
        else:
            new_content = pitch_pos_seq + subseq

        # Reconstruct the bar condition seq from the bar_info dict
        bar_info['content'] = new_content
        ret = unsplit_condition_seq_bar(bar_info)

        return ret
    
    def reorder_tgt(self, remi_seq):
        '''
        Re-order the target sequence, so that it become track-by-track, instead of mixing together
        
        Notes in remi seq can be either
        - o i p d
        - i p d
        - p d

        In return:
        - i o p d o p d ...  i o p d p d o p d
        '''
        seq_of_inst = {}
        insts = get_inst_in_remi(remi_seq) # Get inst, sort by program id

        if len(remi_seq) > 1 and len(insts) == 0:
            insts = ['i-0']

        for inst in insts:
            seq_of_inst[inst] =  []
        
        
            # t = []

            # in_seq = False
            # cur_inst = None
            # for tok in remi_seq:


            # seq_of_inst[inst] = t


        pre_pos = None
        cur_pos = None
        pre_inst = None
        cur_inst = None
        cur_p = None
        cur_dur = None
        for tok in remi_seq:
            if tok.startswith('o-'):
                cur_pos = tok
            elif tok.startswith('i-'):
                cur_inst = tok
            elif tok.startswith('p-'):
                cur_p = tok
            elif tok.startswith('d-'):
                cur_dur = tok

                # If no instrument, set to the first instrument
                if cur_inst is None:
                    cur_inst = insts[0]

                # Add the note to its corresponding sequence
                if cur_inst != pre_inst and cur_inst is not None: # If for new inst
                    seq_of_inst[cur_inst].append(cur_pos)
                else: # If for a same instrument
                    if pre_pos is not None and cur_pos == pre_pos: # If for same position
                        pass # No need to add pos token
                    else:   # If for different position
                        seq_of_inst[cur_inst].append(cur_pos) # should add pos token
                seq_of_inst[cur_inst].append(cur_p)
                seq_of_inst[cur_inst].append(cur_dur)

                pre_pos = cur_pos
                pre_inst = cur_inst

        ret = []
        for inst in seq_of_inst:
            ret.append(inst)
            ret.extend(seq_of_inst[inst])

        return ret