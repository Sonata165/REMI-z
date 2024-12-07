# 
'''
Author: Longshen Ou

Fixed: 
- calculation based on weighted histogram, where 1, 3, 5 are 2x important.
- 0 weight for off-scale notes, instead of -1
'''

import numpy as np


def get_notes_from_pos_info(pos_info):
    notes = []
    min_pitch = 10000
    max_pitch = -1
    for bar, ts, pos, tempo, insts_notes in pos_info:
        if insts_notes is None:
            continue
        for inst_id in insts_notes:
            if inst_id == 128:  # ignore percussion
                continue
            inst_notes = insts_notes[inst_id]
            for note in inst_notes:
                notes.append(note)
                min_pitch = min(note[0], min_pitch)
                max_pitch = max(note[0], max_pitch)
    return notes, min_pitch, max_pitch


def get_pitch_class_histogram(notes, normalize=True, use_duration=True, use_velocity=True):
    weights = np.ones(len(notes))
    # Assumes that duration and velocity have equal weight
    # (pitch, duration, velocity)
    if use_duration:
        weights *= [note[1] for note in notes]  # duration
    if use_velocity:
        weights *= [note[2] for note in notes]  # velocity
    histogram, _ = np.histogram([note[0] % 12 for note in notes], bins=np.arange(13), weights=weights,
                                density=normalize)
    if normalize:
        histogram_sum = histogram.sum()
        histogram /= (histogram_sum + (histogram_sum == 0))
    return histogram


def get_pitch_shift(pos_info, key_profile, normalize=True, use_duration=True, use_velocity=True,
                    ensure_valid_range=True):
    '''
    Return:
    - Pitch shift value (If add this value to the pitch, the key will be C major or A minor)
    - Major or minor
    - Min pitch
    - Max pitch
    '''
    notes, min_pitch, max_pitch = get_notes_from_pos_info(pos_info)
    assert min_pitch >= 0 and max_pitch < 128
    if len(notes) == 0:
        return 0, None, None, None
    histogram = None
    key_candidate = None
    major_index = None
    minor_index = None

    use_duration = True
    use_velocity = True

    histogram = get_pitch_class_histogram(
        notes, 
        normalize=normalize,
        use_duration=use_duration, 
        use_velocity=use_velocity,
    )

    key_candidate = np.dot(key_profile, histogram) # [24,]
    major_key_candidate = key_candidate[:12]
    minor_key_candidate = key_candidate[12:]

    major_index = np.argmax(major_key_candidate)
    minor_index = np.argmax(minor_key_candidate)

    major_score = major_key_candidate[major_index]
    minor_score = minor_key_candidate[minor_index]
    if major_score < minor_score:
        key_number = minor_index  # 小调
        is_major = False
        real_key = key_number
        pitch_shift = -3 - real_key  # 小调
    else:
        key_number = major_index  # 大调
        is_major = True
        real_key = key_number
        pitch_shift = 0 - real_key  

    if ensure_valid_range:
        while pitch_shift + min_pitch < 0:
            pitch_shift += 12
        while pitch_shift + max_pitch >= 128:
            pitch_shift -= 12
        try:
            assert pitch_shift + min_pitch >= 0, \
                "Pitch value range (%d, %d) is too large to make the values valid after pitch shift."
        except AssertionError:
            pitch_shift = 0

    return pitch_shift, is_major, min_pitch, max_pitch
