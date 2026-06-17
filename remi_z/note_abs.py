from typing import List, Tuple

from .note import midi_pitch_to_note_name


class NoteAbs:
    def __init__(self, onset: float, duration: float, pitch: int, velocity: int = 96):
        """
        Parameters
        ----------
        onset : float
            Note onset time in seconds.
        duration : float
            Note duration in seconds. Must be > 0.
        pitch : int
            MIDI pitch [0, 127].
        velocity : int
            MIDI velocity [0, 127]. Default: 96.
        """
        assert isinstance(pitch, int), "pitch must be an integer"
        assert isinstance(velocity, int), "velocity must be an integer"
        assert onset >= 0, f"onset must be >= 0, got {onset}"
        assert duration > 0, f"duration must be > 0, got {duration}"
        assert 0 <= pitch <= 127, f"pitch must be in [0, 127], got {pitch}"
        assert 0 <= velocity <= 127, f"velocity must be in [0, 127], got {velocity}"

        self.onset = float(onset)
        self.duration = float(duration)
        self.pitch = pitch
        self.velocity = velocity

    @property
    def offset(self) -> float:
        return self.onset + self.duration

    def get_note_name(self) -> str:
        return midi_pitch_to_note_name(self.pitch)

    def __str__(self) -> str:
        return f"(o:{self.onset:.3f}s,p:{self.pitch},d:{self.duration:.3f}s,v:{self.velocity})"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: "NoteAbs") -> bool:
        if self.onset != other.onset:
            return self.onset < other.onset
        elif self.pitch != other.pitch:
            return self.pitch > other.pitch
        elif self.duration != other.duration:
            return self.duration > other.duration
        else:
            return self.velocity > other.velocity


class NoteAbsSeq:
    """
    A sequence of NoteAbs objects (absolute timing in seconds).
    """

    def __init__(self, note_list: List[NoteAbs]):
        self.notes = note_list

    @classmethod
    def from_midi(cls, path: str, instrument_idx: int = 0) -> "NoteAbsSeq":
        """
        Load a NoteAbsSeq from a MIDI file.

        Parameters
        ----------
        path : str
            Path to the MIDI file.
        instrument_idx : int
            Index of the instrument track to load. Default: 0.
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(path)
        instrument = midi.instruments[instrument_idx]
        notes = []
        for n in instrument.notes:
            onset = round(n.start, 3)
            offset = round(n.end, 3)
            duration = round(offset - onset, 3)
            notes.append(NoteAbs(onset=onset, duration=duration, pitch=n.pitch, velocity=n.velocity))
        notes.sort()
        return cls(notes)

    @classmethod
    def from_triplet_list(cls, triplets: List[List]) -> "NoteAbsSeq":
        """
        Create a NoteAbsSeq from a list of [onset, offset, pitch] triplets.

        Parameters
        ----------
        triplets : list of [onset, offset, pitch]
            onset and offset are in seconds; duration is derived as offset - onset.
            Times are rounded to the nearest millisecond (0.001 s).
        """
        if triplets and len(triplets) > 0:
            # Ensure pitch is int
            assert isinstance(triplets[0][2], int), "pitch must be an integer"
        
        notes = []
        for onset, offset, pitch in triplets:
            onset = round(float(onset), 3)
            offset = round(float(offset), 3)
            duration = round(offset - onset, 3)
            notes.append(NoteAbs(onset=onset, duration=duration, pitch=pitch))
        return cls(notes)

    def __str__(self) -> str:
        return f"NoteAbsSeq of {len(self.notes)} notes: [" + " ".join([note.get_note_name() for note in self.notes]) + "]"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.notes[idx]
        elif isinstance(idx, slice):
            return NoteAbsSeq(self.notes[idx])

    def get_note_name_list(self) -> List[str]:
        return [note.get_note_name() for note in self.notes]

    def get_pitch_range(self) -> Tuple[int, int] | None:
        """
        Return (lowest_pitch, highest_pitch), or None if empty.
        """
        if not self.notes:
            return None
        pitches = [note.pitch for note in self.notes]
        return (min(pitches), max(pitches))

    def get_onset_list(self) -> List[float]:
        return [note.onset for note in self.notes]

    def get_offset_list(self) -> List[float]:
        return [note.offset for note in self.notes]

    def to_triplet_list(self) -> List[List]:
        """Return [[onset, offset, pitch], ...] for each note."""
        return [[note.onset, round(note.offset, 3), note.pitch] for note in self.notes]

    def to_midi(self, path: str, program: int = 0, tempo: float = 120.0):
        """
        Write the sequence to a MIDI file.

        Parameters
        ----------
        path : str
            Output file path.
        program : int
            General MIDI program number [0, 127]. Default: 0 (Acoustic Grand Piano).
        tempo : float
            Tempo in BPM. Default: 120.0.
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=10000)
        instrument = pretty_midi.Instrument(program=program)
        for note in self.notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.onset,
                    end=note.offset,
                )
            )
        midi.instruments.append(instrument)
        midi.write(path)
