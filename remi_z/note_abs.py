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


class NoteStream:
    """
    A sequence of NoteAbs objects (absolute timing in seconds).
    Can be used to handle performance MIDI, i.e., absolute timing, no time signature, tempo, downbeat information

    Assume all notes are from a single instrument.
    Support multi-channel MIDI. Notes from different channels can be merged into a single NoteStream, as long as they are from the same instrument. 
    """

    def __init__(self, note_list: List[NoteAbs], inst_id: int = 0):
        self.inst_id = inst_id
        self.notes = note_list
        self.is_drum = inst_id == 128  # Use 128 to indicate drum instrument

    @classmethod
    def from_midi(cls, path: str, merge_tracks: bool = False, skip_drums: bool = True, dedup: bool = False) -> "NoteStream":
        """
        Load a NoteStream from a MIDI file.

        Parameters
        ----------
        path : str
            Path to the MIDI file.
        merge_tracks : bool
            If True, merge tracks with the same instrument ID into a single track. Default: False
        skip_drums : bool
            If True, skip drum tracks in merge. Default: True
        dedup : bool
            If True, remove duplicate notes after merging. Default: False
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(path)

        if len(midi.instruments) == 0:
            raise ValueError(f"No notes found in MIDI file {path}")
        elif len(midi.instruments) > 1:
            if not merge_tracks:
                raise ValueError(f"Multiple instruments found in MIDI file {path}. Please specify instrument_idx.")
            else:
                # Merge all instruments into a single instrument
                merged_instrument = pretty_midi.Instrument(program=0)
                for inst in midi.instruments:
                    if skip_drums and inst.is_drum:
                        continue
                    merged_instrument.notes.extend(inst.notes)
                midi.instruments = [merged_instrument]

        instrument_idx = 0
        instrument = midi.instruments[instrument_idx]
        prog_id = instrument.program
        if instrument.is_drum:
            prog_id = 128  # Use 128 to indicate drum instrument

        notes = []
        for n in instrument.notes:
            onset = round(n.start, 3)
            offset = round(n.end, 3)
            duration = round(offset - onset, 3)
            notes.append(NoteAbs(onset=onset, duration=duration, pitch=n.pitch, velocity=n.velocity))
        
        if dedup:
            # Remove notes with same onset and pitch
            unique_notes = {}
            for note in notes:
                key = (note.onset, note.pitch)
                if key not in unique_notes:
                    unique_notes[key] = note
                else:
                    # If duplicate, keep the one with longer duration
                    if note.duration > unique_notes[key].duration:
                        unique_notes[key] = note
            notes = list(unique_notes.values())
        
        notes.sort()
        return cls(notes, inst_id=prog_id)

    @classmethod
    def from_triplet_list(cls, triplets: List[List]) -> "NoteStream":
        """
        Create a NoteStream from a list of [onset, offset, pitch] triplets.

        Parameters
        ----------
        triplets : list of [onset, offset, pitch]
            onset and offset are in seconds; duration is derived as offset - onset.
            Times are rounded to the nearest millisecond (0.001 s).
        """
        if triplets and len(triplets) > 0:
            # Ensure pitch is int
            if not isinstance(triplets[0][2], int):
                raise ValueError("Pitch must be an integer in triplets.")
        
        notes = []
        for onset, offset, pitch in triplets:
            onset = round(float(onset), 3)
            offset = round(float(offset), 3)
            duration = round(offset - onset, 3)
            notes.append(NoteAbs(onset=onset, duration=duration, pitch=pitch))
        notes.sort()
        return cls(notes)

    def __str__(self) -> str:
        return f"NoteStream of {len(self.notes)} notes: [" + " ".join([note.get_note_name() for note in self.notes]) + "]"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.notes[idx]
        elif isinstance(idx, slice):
            return NoteStream(self.notes[idx])

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


def _dedup_by_onset_pitch(notes: List[NoteAbs]) -> List[NoteAbs]:
    """Drop notes sharing the same (onset, pitch), keeping the longest duration."""
    unique = {}
    for note in notes:
        key = (note.onset, note.pitch)
        if key not in unique or note.duration > unique[key].duration:
            unique[key] = note
    return list(unique.values())


class MultiStream:
    """A multi-track performance: an ordered collection of ``NoteStream`` tracks.

    Each track is a ``NoteStream`` that carries its own General MIDI program in
    ``inst_id`` (128 = drums).  Tracks are stored as an ordered **list**, not a
    dict keyed by program, because a piece can contain several tracks that share
    the same program (e.g. two violins, or multiple piano tracks).  The list
    index is the stable per-track key and it preserves the MIDI track order; use
    :meth:`by_program` when you want program-grouped access.
    """

    def __init__(self, streams: List[NoteStream]):
        self.streams: List[NoteStream] = list(streams)

    @classmethod
    def from_midi(cls, path: str, skip_drums: bool = False,
                  dedup: bool = False) -> "MultiStream":
        """Load a multi-track MIDI: one ``NoteStream`` per instrument track.

        Parameters
        ----------
        path : str
            Path to the MIDI file.
        skip_drums : bool
            If True, drop drum tracks. Default: False.
        dedup : bool
            If True, within each track drop notes sharing the same (onset, pitch),
            keeping the longest. Default: False.

        Empty tracks (no playable notes) are skipped.
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(path)
        if len(midi.instruments) == 0:
            raise ValueError(f"No instruments found in MIDI file {path}")

        streams: List[NoteStream] = []
        for inst in midi.instruments:
            if skip_drums and inst.is_drum:
                continue
            prog_id = 128 if inst.is_drum else int(inst.program)   # 128 marks drums

            notes = []
            for n in inst.notes:
                onset = round(n.start, 3)
                offset = round(n.end, 3)
                duration = round(offset - onset, 3)
                if duration <= 0:        # skip zero/negative-length notes
                    continue
                notes.append(NoteAbs(onset=onset, duration=duration,
                                     pitch=n.pitch, velocity=n.velocity))
            if dedup:
                notes = _dedup_by_onset_pitch(notes)
            if not notes:
                continue
            notes.sort()
            streams.append(NoteStream(notes, inst_id=prog_id))

        if not streams:
            raise ValueError(f"No playable notes found in MIDI file {path}")
        return cls(streams)

    @classmethod
    def from_note_streams(cls, streams: List[NoteStream]) -> "MultiStream":
        """Create a MultiStream from a list of NoteStream objects (order preserved)."""
        streams = list(streams)
        if not all(isinstance(s, NoteStream) for s in streams):
            raise TypeError("from_note_streams expects a list of NoteStream objects")
        return cls(streams)

    def by_program(self) -> dict:
        """Group tracks by program id -> list of NoteStream (handles duplicates)."""
        groups: dict = {}
        for st in self.streams:
            groups.setdefault(st.inst_id, []).append(st)
        return groups

    @property
    def programs(self) -> List[int]:
        """Program id of each track, in track order (128 = drums)."""
        return [st.inst_id for st in self.streams]

    def flatten(self, include_drum: bool = False) -> NoteStream:
        """Collapse all tracks into a single program-0 NoteStream.

        Instrument information is dropped.  Notes sharing the same (onset, pitch)
        across tracks are deduplicated, keeping the one with the longest duration.

        Parameters
        ----------
        include_drum : bool
            If False (default), drum tracks (inst_id == 128) are excluded.
        """
        notes = [note for st in self.streams
                 if include_drum or not st.is_drum
                 for note in st.notes]
        notes = _dedup_by_onset_pitch(notes)
        notes.sort()
        return NoteStream(notes, inst_id=0)

    def __len__(self) -> int:
        return len(self.streams)

    def __iter__(self):
        return iter(self.streams)

    def __getitem__(self, idx) -> NoteStream:
        return self.streams[idx]

    def __str__(self) -> str:
        n_notes = sum(len(s) for s in self.streams)
        return (f"MultiStream of {len(self.streams)} tracks, {n_notes} notes "
                f"(programs: {self.programs})")

    def __repr__(self) -> str:
        return self.__str__()

    def to_midi(self, path: str, tempo: float = 120.0):
        """Write all tracks to a multi-track MIDI file.

        Each track becomes one instrument, restoring its program from ``inst_id``
        (``inst_id == 128`` is written back as a drum instrument).
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=10000)
        for st in self.streams:
            is_drum = st.inst_id == 128
            program = 0 if is_drum else st.inst_id
            instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)
            for note in st.notes:
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


# Backward-compatibility alias: NoteStream was previously named NoteAbsSeq.
NoteAbsSeq = NoteStream
