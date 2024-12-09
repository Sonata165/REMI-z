# 
'''
Author: Longshen Ou

Fixed: 
- calculation based on weighted histogram, where 1, 3, 5 are 2x important.
- 0 weight for off-scale notes, instead of -1
'''

import numpy as np
from typing import List, Tuple


key_profile = np.array([
    [ 1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1], # C major
    [ 1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1],
    [-1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1],
    [ 1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1],
    [-1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1],
    [ 1, -1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1],
    [-1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1,  1],
    [ 1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1],
    [ 1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1],
    [-1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1],
    [ 1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1],
    [-1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1],

    [ 1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1], # A minor
    [-1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1],
    [ 1, -1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1],
    [-1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1,  1],
    [ 1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1],
    [ 1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1],
    [-1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1],
    [ 1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1],
    [-1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1,  1],
    [ 1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1],
    [ 1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1],
    [-1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1],
])


''' Apply weighting to key_profile '''
major_weight = [2, 0, 1.0, 0, 1.5, 1.0, 0, 1.5, 0, 1, 0, 1]
minor_weight = [2, 0, 1, 1.5, 0, 1.0, 0, 1.5, 1.0, 0, 1.0, 0]

for i in range(24):
    if i < 12:
        # circular right shift major weight
        shifted_major_weight = major_weight[-i:] + major_weight[:-i]

        # Position-wise multiplication
        for k in range(12):
            key_profile[i][k] *= shifted_major_weight[k]
    else:
        j = i - 12
        # circular shift minor weight
        shifted_minor_weight = minor_weight[-j:] + minor_weight[:-j]

        # Position-wise multiplication
        for k in range(12):
            key_profile[i][k] *= shifted_minor_weight[k]


def detect_key(note_list:List[Tuple[int, int, int, int]]):
    '''
    Determine the major/minor key and pitch shift needed for key normalization.

    Use this version instead. Others are deprecated.

    Args:
    - note_list: List of notes, each note is a tuple of (onset, pitch, duration, velocity)

    Returns:
    - is_major: True if major, False if minor
    - pitch_shift: The pitch shift needed for key normalization
    '''
    
    pitch_class_histogram = get_pitch_class_histogram(
        note_list, 
        normalize=True,
        use_duration=True, 
        use_velocity=True,
        note_has_onset=True,
    )

    pitches = [note[1] for note in note_list]
    min_pitch = min(pitches)
    max_pitch = max(pitches)

    pitch_shift, is_major = get_pitch_shift_from_pitch_histogram(
        pitch_class_histogram,
        min_pitch,
        max_pitch,
    )
    
    return is_major, pitch_shift


def get_pitch_class_histogram(
        notes:List[Tuple[int, int, int]], 
        normalize=True, 
        use_duration=True, 
        use_velocity=True,
        note_has_onset=False,
    ):
    '''
    Calculate the pitch class histogram of the notes.

    Args:
    - notes: List of notes, each note is a tuple of (pitch, duration, velocity)
    '''
    if note_has_onset:
        notes = [(note[1], note[2], note[3]) for note in notes]

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


def get_pitch_shift_from_pitch_histogram(histogram, min_pitch, max_pitch):
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

    # Make the shift to a nearer octave
    if pitch_shift > 6:
        pitch_shift -= 12
    elif pitch_shift < -6:
        pitch_shift += 12

    # Ensure the pitch shift is within the valid range
    while pitch_shift + min_pitch < 0:
        pitch_shift += 12
    while pitch_shift + max_pitch > 127:
        pitch_shift -= 12

    return pitch_shift, is_major