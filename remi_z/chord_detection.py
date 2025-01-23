from typing import List

note_id_to_note_name = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'D#',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'G#',
        9: 'A',
        10: 'A#',
        11: 'B'
    }
note_name_to_note_id = {v: k for k, v in note_id_to_note_name.items()}

# 和弦类型定义：和弦类型名称及其相对于根音的间隔（半音数）
chords = {
    'Major': [0, 4, 7],
    'Minor': [0, 3, 7],
    'Augmented': [0, 4, 8],
    'Diminished': [0, 3, 6],
    'Major7': [0, 4, 7, 11],      # Major 7th: root, major third, perfect fifth, major seventh
    'Minor7': [0, 3, 7, 10],      # Minor 7th: root, minor third, perfect fifth, minor seventh
    'Dominant7': [0, 4, 7, 10],   # Dominant 7th: root, major third, perfect fifth, minor seventh
    'Sus4': [0, 5, 7],            # Suspended 4th: root, perfect fourth, perfect fifth
    'Sus2': [0, 2, 7]             # Suspended 2nd: root, major second, perfect fifth
}


def main():
    pitches = [48, 52, 55]

    res = detect_chord_from_pitch_list(pitches, return_root_name=False)
    print(res) # (0, 'Major')

    res = detect_chord_from_pitch_list(pitches, return_root_name=True)
    print(res) # ('C', 'Major')

    pitches = [53, 60, 64, 69]

    res = detect_chord_from_pitch_list(pitches, return_root_name=False)
    print(res) # (0, 'Major')

    res = detect_chord_from_pitch_list(pitches, return_root_name=True)
    print(res) # (0, 'Major')
    exit(10)


def detect_chords_from_note_name_list(suspect_root, note_names: list):
    suspect_root = note_name_to_note_token(suspect_root)
    note_tokens = note_name_list_to_note_token_list(note_names)
    
    res = chord_detection_with_root(note_tokens, suspect_root, return_root_name=True)
    return res


def note_name_to_note_token(note_name):
    """ 将音符名称（如 'C'）转化为音符表示（如 'p-0'）"""
    return 'p-{}'.format(note_name_to_note_id[note_name])


def note_name_list_to_note_token_list(note_names):
    """ 将音符名称列表（如 ['C', 'E', 'G']）转化为音符表示列表（如 ['p-0', 'p-4', 'p-7']）。"""
    return [note_name_to_note_token(n) for n in note_names]


def note_token_to_note_id(note):
    """ 将音符表示（如 'p-0'）转化为整数表示（0-11）。"""
    return int(note.split('-')[1])


def detect_chord_from_pitch_list(note_list: List[int], return_root_name=False):
    
    """ 
    Detect the best matching chord type from a set of notes with a suspected root note. 

    notes: a list of note tokens,
        prerequisite: all in first octave
    suspected_root_note: a note token
        prerequisite: in first octave

    Return: 
        (root_name, chord_type) of best matching chord
        None if no chord is detected, or len(notes) < 2
    """
    # Convert notes to integer semitones
    note_ints = set(n%12 for n in note_list)

    # Find the suspected root note
    suspected_root_pitch = min(note_list)
    suspected_root_int = suspected_root_pitch % 12
    
    if len(note_ints) < 2:
        return None
    
    best_match = None
    max_match_size = 0
    
    # Check the suspected root note first
    for chord_type, pattern in chords.items():
        chord_notes = set((suspected_root_int + p) % 12 for p in pattern)
        match_size = len(note_ints & chord_notes)
        if match_size > max_match_size:
            best_match = (suspected_root_int, chord_type)
            max_match_size = match_size

    # Check all other roots only if needed
    if max_match_size < len(note_ints):  # Only search other roots if not all notes matched
        for chord_type, pattern in chords.items():
            for root in range(12):
                if root == suspected_root_int:
                    continue  # Skip the already checked root
                chord_notes = set((root + p) % 12 for p in pattern)
                match_size = len(note_ints & chord_notes)
                if match_size > max_match_size:
                    best_match = (root, chord_type)
                    max_match_size = match_size
    
    if return_root_name:
        if best_match:
            root_name, chord_type = best_match
            return note_id_to_note_name[root_name], chord_type

    return best_match


def chord_detection_with_root(note_list: List[str], suspected_root_note: str, return_root_name=False):
    
    """ 
    Detect the best matching chord type from a set of notes with a suspected root note. 

    notes: a list of note tokens,
        prerequisite: all in first octave
    suspected_root_note: a note token
        prerequisite: in first octave

    Return: 
        (root_name, chord_type) of best matching chord
        None if no chord is detected, or len(notes) < 2
    """
    if suspected_root_note is None:
        return None    

    if suspected_root_note not in note_list:
        raise ValueError('Suspected root note not in note list: {}'.format(suspected_root_note))

    # Convert notes to integer semitones
    note_ints = set(note_token_to_note_id(n) for n in note_list)
    suspected_root_int = note_token_to_note_id(suspected_root_note)
    
    if len(note_ints) < 2:
        return None
    
    best_match = None
    max_match_size = 0
    
    # Check the suspected root note first
    for chord_type, pattern in chords.items():
        chord_notes = set((suspected_root_int + p) % 12 for p in pattern)
        match_size = len(note_ints & chord_notes)
        if match_size > max_match_size:
            best_match = (suspected_root_int, chord_type)
            max_match_size = match_size

    # Check all other roots only if needed
    if max_match_size < len(note_ints):  # Only search other roots if not all notes matched
        for chord_type, pattern in chords.items():
            for root in range(12):
                if root == suspected_root_int:
                    continue  # Skip the already checked root
                chord_notes = set((root + p) % 12 for p in pattern)
                match_size = len(note_ints & chord_notes)
                if match_size > max_match_size:
                    best_match = (root, chord_type)
                    max_match_size = match_size
    
    if return_root_name:
        if best_match:
            root_name, chord_type = best_match
            return note_id_to_note_name[root_name], chord_type

    return best_match


def generate_chord_notes(chord_root_name, chord_type):
    root_id = note_name_to_note_id[chord_root_name]
    pattern = chords[chord_type]

    octave = 2

    note_tokens = ['p-{}'.format(root_id + p + octave*12) for p in pattern]
    
    return note_tokens

def chord_to_id(detected_chord):
    if detected_chord is None:
        return 12, 9
    chord_root_name, chord_type = detected_chord
    root_id = note_name_to_note_id[chord_root_name]
    type_id = list(chords.keys()).index(chord_type)
    return root_id, type_id


if __name__ == "__main__":
    main()