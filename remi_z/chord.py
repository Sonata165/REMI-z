from typing import List, Optional


# Mapping between semitone index and note name
note_id_to_note_name = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B',
}
note_name_to_note_id = {v: k for k, v in note_id_to_note_name.items()}

# Chord type definitions: name → semitone intervals relative to root
chords = {
    '':     [0, 4, 7],       # Major triad
    'm':    [0, 3, 7],       # Minor triad
    'dim':  [0, 3, 6],       # Diminished triad
    'aug':  [0, 4, 8],       # Augmented triad
    'sus4': [0, 5, 7],       # Suspended 4th
    'sus2': [0, 2, 7],       # Suspended 2nd
    'M7':   [0, 4, 7, 11],   # Major 7th
    'm7':   [0, 3, 7, 10],   # Minor 7th
    '7':    [0, 4, 7, 10],   # Dominant 7th
    'o':    [0, 3, 6, 9],    # Diminished 7th
    'm7b5': [0, 3, 6, 10],   # Half-diminished 7th
}


class Chord:
    """
    A single chord represented by a root note and a quality (e.g., root='C', quality='M7').
    Quality strings match the keys of the `chords` dict (e.g., '', 'm', 'M7', 'm7b5').
    """

    def __init__(self, root: str, quality: str):
        self.root = root
        self.quality = quality

    def __str__(self) -> str:
        return f"{self.root}{self.quality}"

    def __repr__(self) -> str:
        return self.__str__()


class ChordSeq:
    """
    A sequence of Chord objects.

    Typically contains two chords per bar (one per half-bar), as produced by Bar.get_chord().
    This is a wrapper around a list of Chord objects, intended to be extended with
    higher-level harmony operations.
    """

    def __init__(self, chord_list: List[Chord]):
        self.chord_list = chord_list

    def __str__(self):
        return " ".join(str(chord) for chord in self.chord_list) + " | "

    def __repr__(self):
        return self.__str__()


def detect_chord_from_pitch_list(note_list: List[int]) -> Optional[Chord]:
    """
    Detect the best matching chord from a list of MIDI pitches.

    Uses the lowest pitch as the suspected root and finds the chord type
    whose pattern has the most overlap with the given notes. If no match
    is found (fewer than 2 distinct pitch classes), returns None.

    Parameters
    ----------
    note_list : List[int]
        MIDI pitch values. Octave is ignored; only pitch classes (mod 12) matter.

    Returns
    -------
    Chord or None
        The best matching Chord (root name + quality), or None if detection fails.
    """
    if len(note_list) < 2:
        return None

    note_ints = set(n % 12 for n in note_list)

    suspected_root_int = min(note_list) % 12

    if len(note_ints) < 2:
        return None

    best_match = None
    max_match_size = 0

    # Check suspected root first
    for chord_type, pattern in chords.items():
        chord_notes = set((suspected_root_int + p) % 12 for p in pattern)
        match_size = len(note_ints & chord_notes)
        if match_size > max_match_size:
            best_match = (suspected_root_int, chord_type)
            max_match_size = match_size

    # Search all other roots only if not all notes matched
    if max_match_size < len(note_ints):
        for chord_type, pattern in chords.items():
            for root in range(12):
                if root == suspected_root_int:
                    continue
                chord_notes = set((root + p) % 12 for p in pattern)
                match_size = len(note_ints & chord_notes)
                if match_size > max_match_size:
                    best_match = (root, chord_type)
                    max_match_size = match_size

    if best_match is None:
        return None

    root_int, chord_type = best_match
    return Chord(root=note_id_to_note_name[root_int], quality=chord_type)


def generate_chord_notes(chord_root_name: str, chord_type: str) -> List[str]:
    """
    Generate REMI-z pitch tokens for a chord.

    Parameters
    ----------
    chord_root_name : str
        Root note name, e.g. 'C', 'F#'.
    chord_type : str
        Chord quality string matching a key in `chords`, e.g. '', 'm', 'M7'.

    Returns
    -------
    List[str]
        List of pitch tokens like ['p-36', 'p-40', 'p-43'].
    """
    root_id = note_name_to_note_id[chord_root_name]
    pattern = chords[chord_type]
    octave = 2
    return [f'p-{root_id + p + octave * 12}' for p in pattern]
