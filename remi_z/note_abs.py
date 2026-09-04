import collections
from typing import List, Tuple

from .note import midi_pitch_to_note_name

CHROMATIC_PERCUSSION_PROG_IDS = {8, 9, 10, 11, 12, 13, 14, 15}

def _apply_sustain_control_changes(instrument, sustain_num: int = 64) -> None:
    """Extend note offsets over the sustain pedal (CC64), in place.

    While the pedal is held (CC ``sustain_num`` value >= 64), a released key keeps
    sounding until either the pedal is lifted or the same pitch is struck again --
    the standard Onsets & Frames / ``note_seq.apply_sustain_control_changes``
    convention used by piano-AMT. Only ``note.end`` is mutated; onsets/pitches are
    untouched. Operates on one ``pretty_midi.Instrument`` (uses its
    ``control_changes``). Verified to reproduce MAPS' pedal-extended .txt offsets
    to <1 ms.
    """
    # Time-ordered events; ties break by priority so at an equal timestamp a
    # note-off is settled before pedal toggles / note-ons. 1:off 2:on 3:sus-on 4:sus-off.
    events = []
    for n in instrument.notes:
        events.append((n.start, 2, n))
        events.append((n.end, 1, n))
    for cc in instrument.control_changes:
        if cc.number == sustain_num:
            events.append((cc.time, 3 if cc.value >= 64 else 4, cc))
    events.sort(key=lambda e: (e[0], e[1]))

    held = collections.defaultdict(list)  # pitch -> notes released while pedal down
    sustain = False
    for t, typ, ev in events:
        if typ == 3:  # sustain on
            sustain = True
        elif typ == 4:  # sustain off: release everything held
            sustain = False
            for p in list(held):
                for n in held[p]:
                    n.end = t
                del held[p]
        elif typ == 2:  # note on: retrigger cuts a held same pitch
            if sustain and ev.pitch in held:
                for n in held[ev.pitch]:
                    n.end = t
                del held[ev.pitch]
        else:  # note off: hold if pedal down, else ends now
            if sustain:
                held[ev.pitch].append(ev)


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
    def from_midi(
        cls,
        path: str,
        merge_tracks: bool = False,
        skip_drums: bool = True,
        dedup: bool = False,
        pedal_extend: bool = False,
        normalize_chrom_perc_dur: float | None = None,
    ) -> "NoteStream":
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
        pedal_extend : bool
            If True, extend each note's offset over the sustain pedal (CC64) before
            reading it (standard piano-AMT convention); only offsets change, not
            onsets/pitches. Applied per source instrument (using that instrument's
            own control changes) prior to any merge. Default: False
        normalize_chrom_perc_dur : float | None
            If not None, normalize all note durations to this value for chromatic percussion. Default: None
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(path)

        if pedal_extend:
            for inst in midi.instruments:
                _apply_sustain_control_changes(inst)

        if len(midi.instruments) == 0:
            raise ValueError(f"No notes found in MIDI file {path}")
        elif len(midi.instruments) > 1:
            if not merge_tracks:
                raise ValueError(
                    f"Multiple instruments found in MIDI file {path}. Please specify instrument_idx."
                )
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

        if prog_id in CHROMATIC_PERCUSSION_PROG_IDS:
            normalize_dur = normalize_chrom_perc_dur
        else:
            normalize_dur = None

        notes = []
        for n in instrument.notes:
            onset = round(n.start, 3)
            offset = round(n.end, 3)
            duration = max(
                round(offset - onset, 3), 0.001
            )  # Ensure duration is at least 1 ms
            if normalize_dur is not None:
                duration = normalize_dur
            notes.append(
                NoteAbs(
                    onset=onset, duration=duration, pitch=n.pitch, velocity=n.velocity
                )
            )

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

        # Ensure no note overlap if duration is normalized
        if normalize_dur is not None:
            notes = _adjust_offset_overlap(notes)

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
            duration = max(
                round(offset - onset, 3), 0.001
            )  # Ensure duration is at least 1 ms
            notes.append(NoteAbs(onset=onset, duration=duration, pitch=pitch))
        notes.sort()
        return cls(notes)

    def __str__(self) -> str:
        return (
            f"NoteStream of {len(self.notes)} notes: ["
            + " ".join([note.get_note_name() for note in self.notes])
            + "]"
        )

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

    def to_midi(
        self,
        path: str,
        program: int = 0,
        tempo: float = 120.0,
        play_rate: float = 1.0,
        pitch_shift: int = 0,
    ) -> None:
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
        play_rate : float
            Playback rate. Default: 1.0.
        pitch_shift : int
            Pitch shift in semitones. Default: 0.
        """
        import pretty_midi

        # resolution 500 ticks/beat = 1 ms at 120 BPM (note times are ms-rounded),
        # low enough that long recordings stay under pretty_midi's 10M-tick limit.
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=500)
        instrument = pretty_midi.Instrument(program=program)
        for note in self.notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch + pitch_shift,
                    start=note.onset / play_rate,
                    end=note.offset / play_rate,
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


def _adjust_offset_overlap(notes: List[NoteAbs], eps: float = 0.001) -> List[NoteAbs]:
    """Return copies of ``notes`` with same-pitch offset overlaps removed.

    MIDI Note-On/Note-Off events carry only ``(channel, pitch)`` -- they do not
    tag which Note-Off closes which Note-On.  So two notes of the same pitch whose
    intervals overlap (in particular one fully nested inside another) cannot be
    round-tripped through a MIDI file: on parse the offsets get rematched and the
    intended pairing is lost.  To keep a ``NoteStream`` always faithfully
    serialisable, this shortens each note's offset to at most ``eps`` seconds
    before the onset of the *next* same-pitch note, so same-pitch intervals never
    overlap.

    Only offsets are shortened, never lengthened; onsets, pitches and velocities
    are untouched, and the input objects are not mutated (new ``NoteAbs`` are
    returned).  Notes are assumed ms-aligned and already deduplicated by
    ``(onset, pitch)`` (so same-pitch onsets are distinct and differ by >= eps).
    In the degenerate case where the next same-pitch onset is only ``eps`` later,
    the offset is set equal to that onset (a zero-gap "touch"), which is still
    unambiguous and keeps the duration >= eps.
    """
    by_pitch = collections.defaultdict(list)
    for note in notes:
        by_pitch[note.pitch].append(note)

    adjusted: List[NoteAbs] = []
    for group in by_pitch.values():
        group.sort(key=lambda n: n.onset)
        for i, cur in enumerate(group):
            duration = cur.duration
            if i + 1 < len(group):
                nxt_onset = group[i + 1].onset
                cap = round(nxt_onset - eps, 3)  # eps before next same-pitch onset
                if cap <= cur.onset:  # onsets only eps apart: touch instead
                    cap = nxt_onset
                if cur.offset > cap:
                    duration = round(cap - cur.onset, 3)
            adjusted.append(
                NoteAbs(
                    onset=cur.onset,
                    duration=duration,
                    pitch=cur.pitch,
                    velocity=cur.velocity,
                )
            )
    return adjusted


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
    def from_midi(
        cls, path: str, skip_drums: bool = False, dedup: bool = False
    ) -> "MultiStream":
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
            prog_id = 128 if inst.is_drum else int(inst.program)  # 128 marks drums

            notes = []
            for n in inst.notes:
                onset = round(n.start, 3)
                offset = round(n.end, 3)
                duration = round(offset - onset, 3)
                if duration <= 0:  # skip zero/negative-length notes
                    continue
                notes.append(
                    NoteAbs(
                        onset=onset,
                        duration=duration,
                        pitch=n.pitch,
                        velocity=n.velocity,
                    )
                )
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

    def flatten(
        self, include_drum: bool = False, adjust_offset_overlap: bool = True
    ) -> NoteStream:
        """Collapse all tracks into a single program-0 NoteStream.

        Instrument information is dropped.  Notes sharing the same (onset, pitch)
        across tracks are deduplicated, keeping the one with the longest duration.

        Parameters
        ----------
        include_drum : bool
            If False (default), drum tracks (inst_id == 128) are excluded.
        adjust_offset_overlap : bool
            If True (default), after dedup shorten offsets so that no note
            overlaps a later note of the same pitch (see
            :func:`_adjust_offset_overlap`).  This guarantees the resulting
            NoteStream can be written to MIDI and read back unchanged; merging
            tracks routinely produces same-pitch overlaps that MIDI cannot
            represent faithfully.
        """
        notes = [
            note
            for st in self.streams
            if include_drum or not st.is_drum
            for note in st.notes
        ]
        notes = _dedup_by_onset_pitch(notes)
        if adjust_offset_overlap:
            notes = _adjust_offset_overlap(notes)
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
        return (
            f"MultiStream of {len(self.streams)} tracks, {n_notes} notes "
            f"(programs: {self.programs})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def to_midi(self, path: str, tempo: float = 120.0):
        """Write all tracks to a multi-track MIDI file.

        Each track becomes one instrument, restoring its program from ``inst_id``
        (``inst_id == 128`` is written back as a drum instrument).
        """
        import pretty_midi

        # resolution 500 ticks/beat = 1 ms at 120 BPM (note times are ms-rounded),
        # low enough that long recordings stay under pretty_midi's 10M-tick limit.
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=500)
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
