import re
from typing import List, Tuple


def midi_pitch_to_note_name(pitch: int) -> str:
    """
    Convert a MIDI pitch number (0–127) to a note name string like 'C4'.
    """
    if not (0 <= pitch <= 127):
        raise ValueError("MIDI pitch must be between 0 and 127")

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    note = note_names[pitch % 12]
    octave = (pitch // 12) - 1  # MIDI note 0 is C-1
    return f"{note}{octave}"


def note_name_to_midi_pitch(note_name: str) -> int:
    """
    Convert a note name string like 'C4' or 'F#3' to a MIDI pitch number (0–127).
    """
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    match = re.match(r"^([A-G]#?)(-?\d+)$", note_name)
    if not match:
        raise ValueError(f"Invalid note name: {note_name}")
    note, octave = match.groups()
    if note not in note_names:
        raise ValueError(f"Invalid note name: {note_name}")
    octave = int(octave)
    pitch = note_names.index(note) + (octave + 1) * 12
    if not (0 <= pitch <= 127):
        raise ValueError("Resulting MIDI pitch must be between 0 and 127")
    return pitch


class Note:
    def __init__(self, onset: int, duration: int, pitch: int, velocity: int = 64):
        """
        Create an instance of a Note object.

        Parameters
        ----------
        onset : int
            The onset time of the note, in positions (1/12 of a beat, i.e. a 48th note).
            In a 4/4 bar there are 48 positions; position 0 is the downbeat.
            Range: [0, 127].
        duration : int
            The duration of the note, in positions (1/12 of a beat).
            Values below 1 are rounded up to 1; values above 127 are clamped to 127.
            Range: [1, 127].
        pitch : int
            The MIDI pitch of the note.
            Range: [0, 127]. Drum notes use the same range; the +128 offset for drum
            pitches in the REMI-z token sequence is applied during tokenization, not here.
        velocity : int
            The MIDI velocity of the note. Default: 64.
            Range: [0, 127].
        """
        assert isinstance(onset, int), "onset must be an integer"
        assert isinstance(duration, int), "duration must be an integer"
        assert isinstance(pitch, int), "pitch must be an integer"
        assert isinstance(velocity, int), "velocity must be an integer"
        assert 0 <= onset <= 127, f"onset must be in the range of [0, 127], got {onset}"
        assert 0 <= pitch <= 127, f"pitch must be in the range of [0, 127], got {pitch}"
        assert (
            0 <= velocity <= 127
        ), f"velocity must be in the range of [0, 127], got {velocity}"

        # Round the values
        duration = min(
            max(1, duration), 127
        )  # duration must be in the range of [1, 127]

        self.onset = onset
        self.duration = duration
        self.pitch = pitch
        self.velocity = velocity

    def get_note_name(self):
        return midi_pitch_to_note_name(self.pitch)

    def __str__(self) -> str:
        return f"(o:{self.onset},p:{self.pitch},d:{self.duration},v:{self.velocity})"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other):
        """
        When comparing two notes
        - If the onset is different, the note with smaller onset will be placed at the front
        - If the onset is the same, the note with higher pitch will be placed at the front
        - If onset and pitch are same, note with longer duration will be placed at the front
        - If onset, pitch, and duration are same, note with larger velocity will be placed at the front
        """
        if self.onset != other.onset:
            return self.onset < other.onset
        elif self.pitch != other.pitch:
            return self.pitch > other.pitch
        elif self.duration != other.duration:
            return self.duration > other.duration
        else:
            return self.velocity > other.velocity


class NoteSeq:
    """
    A sequence of Note objects
    Can be used to represent the notes of a single instrument within a bar.

    This is a wrapper around a list of Note objects, intended to be extended with
    higher-level music operations (e.g., transposition, quantization, pattern matching).

    Notes are not guaranteed to be sorted on construction; call sorted() or .notes.sort()
    if order matters.
    """

    def __init__(self, note_list: List[Note]):
        self.notes = note_list

    def __str__(self):
        return (
            "NoteSeq: [" + " ".join([note.get_note_name() for note in self.notes]) + "]"
        )

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.notes[idx]
        elif isinstance(idx, slice):
            return NoteSeq(self.notes[idx])

    def get_note_name_list(self):
        ret = [note.get_note_name() for note in self.notes]
        return ret

    def get_pitch_range(self):
        """
        Return the pitch range of the NoteSeq object.
        In format of (lowest_pitch, highest_pitch)

        When there are no notes in the NoteSeq, return None.
        """
        if len(self.notes) == 0:
            return None
        if len(self.notes) == 1:
            return (self.notes[0].pitch, self.notes[0].pitch)

        pitch = [note.pitch for note in self.notes]
        l_pitch = min(pitch)
        h_pitch = max(pitch)
        return (l_pitch, h_pitch)
