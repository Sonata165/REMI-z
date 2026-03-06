import numpy as np
from typing import List, Dict, Tuple

from .note import Note
from .track import Track
from .midi_encoding import convert_tempo_to_id, convert_id_to_tempo
from .time_signature_utils import TimeSignatureUtil
from .chord import detect_chord_from_pitch_list


def deduplicate_notes(notes: List[Note]) -> List[Note]:
    """
    Remove repeated notes with same onset and pitch.
    NOTE: Ensure the notes are sorted before calling this function.

    Args:
        notes: List of Note objects

    Returns:
        List of Note objects with repeated notes removed
        If note has same onset and pitch, only the first note (with longest duration) will be kept.
        If note has same onset, pitch, and duration, only the first note (with highest velocity) will be kept.
    """
    notes_dedup = []
    prev_onset = None
    prev_pitch = None
    for note in notes:
        if note.onset == prev_onset and note.pitch == prev_pitch:
            continue
        notes_dedup.append(note)
        prev_onset = note.onset
        prev_pitch = note.pitch
    return notes_dedup


class Bar:
    """
    A single bar of music containing one or more instrument tracks.

    Tracks are stored in a dict keyed by instrument ID (GM program number 0–127,
    or 128 for drums) and ordered by descending average pitch so that higher-pitched
    voices appear first.

    Attributes
    ----------
    bar_id : int
        Index of this bar within its parent MultiTrack (or -1 if standalone).
    tracks : Dict[int, Track]
        Mapping from instrument ID to Track, sorted by descending average pitch.
    time_signature : Tuple[int, int]
        Numerator and denominator of the time signature (default ``(4, 4)``).
    tempo : float
        Tempo in BPM, rounded to two decimal places (default ``120.0``).
    """

    def __init__(
        self,
        id,
        notes_of_insts: Dict[int, Dict[int, List]],
        time_signature: Tuple[int, int] = None,
        tempo: float = None,
    ):
        """
        Parameters
        ----------
        id : int
            Bar index.
        notes_of_insts : Dict[int, Dict[int, List]]
            Mapping from instrument ID (or ``(prog_id, track_id)`` tuple) to a
            dict of ``{onset: [(pitch, duration, velocity), ...]}`` note data.
            Pass an empty dict ``{}`` for a bar with no notes.
        time_signature : Tuple[int, int], optional
            ``(numerator, denominator)``; defaults to ``(4, 4)``.
        tempo : float, optional
            BPM; defaults to ``120.0``.

        Notes
        -----
        Tracks are sorted so that the instrument with the highest average pitch
        is placed first (index 0).
        """
        if time_signature:
            assert isinstance(time_signature, tuple), "time_signature must be a tuple"
        else:
            time_signature = (4, 4)
        if tempo:
            assert isinstance(tempo, (int, float)), "tempo must be an integer or float"
        else:
            tempo = 120.0

        # Round tempo to 0.01
        tempo = round(tempo, 2)

        self.bar_id = id

        # Parse notes_of_insts
        if notes_of_insts is not None:
            track_list = []
            for inst_id, notes in notes_of_insts.items():
                if isinstance(inst_id, int):
                    prog_id = inst_id
                    track_id = inst_id
                elif isinstance(inst_id, tuple):
                    assert len(inst_id) == 2, "inst_id tuple must have length 2"
                    prog_id, track_id = inst_id
                track = Track(prog_id, notes, track_id)
                track_list.append(track)
            track_list.sort()

            self.tracks: Dict[int, Track] = {}
            for track in track_list:
                track_id = track.track_id
                self.tracks[track_id] = track
        else:
            self.tracks = {}

        self.time_signature = time_signature
        self.tempo = tempo

    def __len__(self):
        return len(self.tracks)

    def __str__(self) -> str:
        return f"Bar {self.bar_id}: {len(self.tracks)} insts"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_tracks(cls, bar_id, track_list, time_signature=(4, 4), tempo=120.0):
        """
        Create a Bar object from a list of Track objects.
        """
        assert isinstance(track_list, list), "track_list must be a list"
        # return cls(id=bar_id, notes_of_insts={track.inst_id:track.get_all_notes() for track in track_list}, time_signature=time_signature, tempo=tempo)

        if time_signature:
            assert isinstance(time_signature, tuple), "time_signature must be a tuple"
        else:
            time_signature = (4, 4)
        if tempo:
            assert isinstance(tempo, (int, float)), "tempo must be an integer or float"
        else:
            tempo = 120.0

        # Round tempo to 0.01
        tempo = round(tempo, 2)

        # Create an empty Bar object
        bar = cls(
            id=bar_id, notes_of_insts={}, time_signature=time_signature, tempo=tempo
        )

        # Add tracks to the Bar object
        track_list.sort()
        for track in track_list:
            inst_id = track.inst_id
            bar.tracks[inst_id] = track

        return bar

    @classmethod
    def from_piano_roll(
        cls, piano_roll, pos_per_bar=16, time_signature=(4, 4), tempo=120.0
    ):
        """
        Create a Bar from a piano roll matrix.

        The piano roll is a 2-D array of shape ``(pos_per_bar, 128)`` where
        ``piano_roll[position, pitch]`` holds the note duration (in piano-roll
        positions, 0 = silent).  All notes are assigned to instrument 0 (piano).

        Parameters
        ----------
        piano_roll : np.ndarray
            2-D integer array of shape ``(pos_per_bar, 128)``.
        pos_per_bar : int, optional
            Number of grid positions per bar used by the piano roll (default 16).
            Each piano-roll position maps to ``48 // pos_per_bar`` REMI-z positions.
        time_signature : Tuple[int, int], optional
            ``(numerator, denominator)``; defaults to ``(4, 4)``.
        tempo : float, optional
            BPM; defaults to ``120.0``.

        Returns
        -------
        Bar
            A new Bar whose ``bar_id`` is ``-1`` and contains a single
            instrument-0 track built from the piano roll data.
        """
        coeff = 48 // pos_per_bar

        notes = {}
        for p_pos in range(piano_roll.shape[0]):
            for pitch in range(piano_roll.shape[1]):
                if piano_roll[p_pos, pitch] > 0:
                    p_dur = piano_roll[p_pos, pitch]
                    dur = int((p_dur * coeff).item())
                    onset = p_pos * coeff
                    # note = Note(onset=onset, duration=dur, pitch=pitch)
                    note = [pitch, dur, 64]  # pitch, duration, velocity

                    # Add note to notes
                    if onset not in notes:
                        notes[onset] = []
                    notes[onset].append(note)

        return cls(
            id=-1, notes_of_insts={0: notes}, time_signature=time_signature, tempo=tempo
        )

    @classmethod
    def from_remiz_seq(cls, bar_seq: List[str]):
        """
        Create a Bar from a REMI-z token sequence.

        The sequence must represent exactly one bar (i.e. contain a single
        ``b-1`` end token).

        Parameters
        ----------
        bar_seq : List[str]
            Tokenized REMI-z sequence for one bar, e.g.
            ``['i-0', 'o-0', 'p-60', 'd-12', 'v-64', 'b-1']``.

        Returns
        -------
        Bar
        """
        from .multitrack import MultiTrack
        mt = MultiTrack.from_remiz_seq(bar_seq)
        assert len(mt) == 1, "Only support single-bar remiz seq"
        return mt[0]

    @classmethod
    def from_remiz_str(cls, bar_str: str):
        """
        Create a Bar from a whitespace-delimited REMI-z string.

        Convenience wrapper around :meth:`from_remiz_seq` that splits the
        string before parsing.

        Parameters
        ----------
        bar_str : str
            A single-bar REMI-z string, e.g.
            ``'i-0 o-0 p-60 d-12 v-64 b-1'``.

        Returns
        -------
        Bar
        """
        remiz_seq = bar_str.strip().split()
        return cls.from_remiz_seq(remiz_seq)

    def flatten(self) -> "Bar":
        """
        Merge all non-drum tracks into a single instrument-0 track.

        Returns a new Bar object; the original is not modified.
        Drum tracks are excluded from the merged result.

        Returns
        -------
        Bar
            A new Bar with one track (instrument 0) containing all
            deduplicated notes from every non-drum track.
        """
        # assert len(self.tracks) > 0, "Bar has no tracks to flatten"
        all_notes = self.get_all_notes(include_drum=False, deduplicate=True)

        track = Track.from_note_list(inst_id=0, note_list=all_notes)

        # Create a new Bar object with a single track
        new_bar = Bar.from_tracks(
            bar_id=self.bar_id,
            track_list=[track],
            time_signature=self.time_signature,
            tempo=self.tempo,
        )
        return new_bar

    def to_remiz_str(
        self, with_ts=False, with_tempo=False, with_velocity=False, include_drum=False
    ):
        """
        Serialize the Bar to a whitespace-delimited REMI-z string.

        Parameters
        ----------
        with_ts : bool, optional
            Prepend a time-signature token (default ``False``).
        with_tempo : bool, optional
            Prepend a tempo token (default ``False``).
        with_velocity : bool, optional
            Include ``v-X`` velocity tokens for each note (default ``False``).
        include_drum : bool, optional
            Include drum tracks in the output (default ``False``).

        Returns
        -------
        str
            Space-joined REMI-z token string ending with ``b-1``.
        """
        remiz_seq = self.to_remiz_seq(
            with_ts=with_ts,
            with_tempo=with_tempo,
            with_velocity=with_velocity,
            include_drum=include_drum,
        )
        remiz_str = " ".join(remiz_seq)
        return remiz_str

    def to_remiz_seq(
        self, with_ts=False, with_tempo=False, with_velocity=False, include_drum=False
    ):
        """
        Serialize the Bar to a list of REMI-z tokens.

        Tokens are emitted per track in voice order (highest average pitch
        first), each track using the format ``i-X o-X p-X d-X [v-X] ...``,
        followed by a ``b-1`` bar-end token.

        Parameters
        ----------
        with_ts : bool, optional
            Prepend a time-signature token (default ``False``).
        with_tempo : bool, optional
            Prepend a tempo token (default ``False``).
        with_velocity : bool, optional
            Include ``v-X`` velocity tokens (default ``False``).
        include_drum : bool, optional
            Include drum tracks (default ``False``).

        Returns
        -------
        List[str]
            REMI-z token list ending with ``'b-1'``.
        """
        bar_seq = []

        # Add time signature
        if with_ts:
            # time_sig = bar.time_signature.strip()[1:-1]
            num, den = self.time_signature
            ts_token = TimeSignatureUtil.convert_time_signature_to_ts_token(
                int(num), int(den)
            )
            bar_seq.append(ts_token)

        if with_tempo:
            tempo_id = convert_tempo_to_id(self.tempo)
            tempo_tok = f"t-{tempo_id}"
            bar_seq.append(tempo_tok)

        for inst_id, track in self.tracks.items():
            if include_drum is False and track.is_drum:
                continue

            track_seq = track.to_remiz_seq(with_velocity=with_velocity)

            bar_seq.extend(track_seq)
        bar_seq.append("b-1")

        return bar_seq

    def to_remiplus_seq(
        self, with_ts=False, with_tempo=False, with_velocity=False, include_drum=False
    ):
        """
        Serialize the Bar to a REMI+ style token list.

        Unlike REMI-z, REMI+ interleaves notes from all tracks sorted by onset
        and uses the token order ``o-X i-X p-X d-X``.  Drum pitches are offset
        by +128 in the ``p-X`` token.  The sequence ends with ``b-1``.

        Parameters
        ----------
        with_ts : bool, optional
            Prepend a time-signature token (default ``False``).
        with_tempo : bool, optional
            Prepend a tempo token (default ``False``).
        with_velocity : bool, optional
            Included for API symmetry; not yet used (default ``False``).
        include_drum : bool, optional
            Include drum tracks (default ``False``).

        Returns
        -------
        List[str]
            REMI+ token list ending with ``'b-1'``.
        """
        bar_seq = []

        # Add time signature
        if with_ts:
            # time_sig = bar.time_signature.strip()[1:-1]
            num, den = self.time_signature
            ts_token = TimeSignatureUtil.convert_time_signature_to_ts_token(
                int(num), int(den)
            )
            bar_seq.append(ts_token)

        if with_tempo:
            tempo_id = convert_tempo_to_id(self.tempo)
            tempo_tok = f"t-{tempo_id}"
            bar_seq.append(tempo_tok)

        all_notes_oipd = []
        for inst_id, track in self.tracks.items():
            if include_drum is False and track.is_drum:
                continue

            notes = track.get_all_notes()

            for note in notes:
                if track.is_drum:
                    pitch_id = note.pitch + 128
                else:
                    pitch_id = note.pitch

                all_notes_oipd.append((note.onset, inst_id, pitch_id, note.duration))

        all_notes_oipd.sort()

        for onset, inst_id, pitch, dur in all_notes_oipd:
            bar_seq.extend(
                [
                    f"o-{onset}",
                    f"i-{inst_id}",
                    f"p-{pitch}",
                    f"d-{dur}",
                ]
            )
        bar_seq.append("b-1")

        return bar_seq

    def to_piano_roll(self, of_insts: List[int] = None, pos_per_bar=16) -> np.ndarray:
        """
        Convert the Bar to a piano roll matrix.

        Returns a 2-D integer array of shape ``(pos_per_bar, 128)`` where
        ``result[position, pitch]`` holds the note duration in piano-roll
        positions (0 = silent).  Drum tracks are excluded.

        Parameters
        ----------
        of_insts : List[int], optional
            Instrument IDs to include.  ``None`` (default) includes all
            non-drum instruments.
        pos_per_bar : int, optional
            Grid resolution in positions per bar (default 16).

        Returns
        -------
        np.ndarray
            Integer array of shape ``(pos_per_bar * beats_per_bar / 4, 128)``.

        Notes
        -----
        Quantize the Bar to 16th-note resolution before calling this method
        to avoid rounding artefacts.
        """
        # Create a piano roll matrix
        # [pos, pitch] = duration
        n_pitch = 128
        coeff = 48 / pos_per_bar

        pos_per_beat = pos_per_bar // 4
        beats_per_bar = self.time_signature[0]
        pos_per_bar = pos_per_beat * beats_per_bar
        piano_roll = np.zeros((pos_per_bar, n_pitch), dtype=int)

        # Get valid instruments
        all_insts = self.get_unique_insts()
        if of_insts is None:
            insts = all_insts
        else:
            insts = set(all_insts).intersection(of_insts)

        # Obtain notes to be added to the piano roll
        notes = self.get_all_notes(include_drum=False, of_insts=insts)

        # Deduplicate notes
        notes = deduplicate_notes(notes)

        # Add notes to the piano roll
        # NOTE: the pos in piano roll is 1/3 of note.onset
        for note in notes:
            onset_pos = min(round(note.onset / coeff), pos_per_bar - 1)
            dur = round(note.duration / coeff)
            pitch = note.pitch
            piano_roll[onset_pos, pitch] = dur

        return piano_roll

    def to_midi(self, midi_fp: str, tempo: float = None):
        """
        Write the Bar to a MIDI file.

        Parameters
        ----------
        midi_fp : str
            Destination file path (e.g. ``'output.mid'``).
        tempo : float, optional
            Override tempo in BPM.  If ``None``, the Bar's own tempo is used.
        """
        from .multitrack import MultiTrack
        mt = MultiTrack.from_bars([self])
        mt.to_midi(midi_fp, tempo=tempo)

    def get_all_notes(
        self, include_drum=True, of_insts: List[int] = None, deduplicate=False
    ) -> List[Note]:
        """
        Return all notes in the Bar sorted by onset, pitch, duration, velocity.

        Parameters
        ----------
        include_drum : bool, optional
            Whether to include notes from drum tracks (default ``True``).
        of_insts : List[int], optional
            Restrict to these instrument IDs.  ``None`` (default) returns
            notes from all instruments.
        deduplicate : bool, optional
            If ``True``, remove duplicate notes that share the same onset and
            pitch, keeping the first occurrence (longest duration) per the
            sorted order (default ``False``).

        Returns
        -------
        List[Note]
            Sorted, optionally deduplicated list of Note objects.
        """
        assert (
            isinstance(of_insts, (list, set)) or of_insts is None
        ), "of_insts must be a list or None"

        if of_insts is None:
            of_insts = list(self.tracks.keys())

        all_notes = []
        for inst_id in of_insts:
            if inst_id not in self.tracks:
                continue
            # assert inst_id in self.tracks, f"of_inst {inst_id} not found in the bar"
            track = self.tracks[inst_id]
            if not include_drum and track.is_drum:
                continue
            all_notes.extend(track.get_all_notes())

        # Sort the notes
        all_notes.sort()

        if deduplicate:
            # Remove repeated notes with same onset and pitch, keep one with largest duration
            all_notes = deduplicate_notes(all_notes)

        return all_notes

    def get_content_seq(self, include_drum=False, of_insts=None, with_dur=True):
        """
        Convert the Bar object to a content sequence.
        Including information about all notes being played
        Without instrument information.

        Args:
            include_drum: Whether to include drum tracks
            of_insts: A list of instrument IDs to extract the content sequence. None means all instruments.
            with_dur: Whether to include duration information in the content sequence.
        """
        assert (
            include_drum is False
        ), "include_drum in content sequence is not supported yet"

        notes = self.get_all_notes(include_drum=include_drum, of_insts=of_insts)

        # Remove repeated notes with same onset and pitch, keep one with largest duration
        notes = deduplicate_notes(notes)

        # Convert to content sequence (containing only o-X, p-X, d-X)
        bar_seq = []
        prev_pos = -1
        for note in notes:
            if note.onset > prev_pos:
                bar_seq.append(f"o-{note.onset}")
                prev_pos = note.onset

            bar_seq.extend(
                [
                    f"p-{note.pitch}",
                ]
            )
            if with_dur:
                bar_seq.append(f"d-{note.duration}")

        bar_seq.append("b-1")

        return bar_seq

    def get_drum_content_seq(self, with_dur=True):
        """
        Convert the Bar object to a content sequence.
        Including information about all drum notes being played

        Args:
            with_dur: Whether to include duration information in the content sequence.
        """

        notes = self.get_all_notes(
            include_drum=True,
            of_insts=[128],
        )

        # Remove repeated notes with same onset and pitch, keep one with largest duration
        notes = deduplicate_notes(notes)

        # Convert to content sequence (containing only o-X, p-X, d-X)
        bar_seq = []
        prev_pos = -1
        for note in notes:
            if note.onset > prev_pos:
                bar_seq.append(f"o-{note.onset}")
                prev_pos = note.onset

            pitch_id = note.pitch + 128
            bar_seq.extend(
                [
                    f"p-{pitch_id}",
                ]
            )
            if with_dur:
                bar_seq.append(f"d-{note.duration}")

        bar_seq.append("b-1")

        return bar_seq

    def get_unique_insts(self, sort_by_voice=True, include_drum=True) -> List[int]:
        """
        Return the instrument IDs present in this Bar.

        Parameters
        ----------
        sort_by_voice : bool, optional
            Must be ``True`` (only voice-sorted order is currently supported).
        include_drum : bool, optional
            Whether to include the drum track (instrument 128) in the result
            (default ``True``).

        Returns
        -------
        List[int]
            Instrument IDs in voice order (highest average pitch first).
        """
        assert sort_by_voice is True, "sort_by_voice must be True"

        all_insts = []
        for inst_id in self.tracks.keys():
            if include_drum is False and inst_id == 128:
                continue
            all_insts.append(inst_id)

        return all_insts

    def get_pitch_range(self, of_insts: List[int] = None):
        """
        Calculate the range of the notes in the Bar.
        Will return max_pitch - min_pitch + 1
        If no notes found, return -1.
        """
        assert (
            isinstance(of_insts, (list, set)) or of_insts is None
        ), "of_insts must be a list or None"
        all_insts = self.get_unique_insts()
        if of_insts is None:
            insts = all_insts
        else:
            all_insts = set(all_insts)
            insts = all_insts.intersection(of_insts)

        notes = self.get_all_notes(include_drum=False, of_insts=list(insts))
        if len(notes) == 0:
            return -1

        min_pitch = 128
        max_pitch = -1
        for note in notes:
            min_pitch = min(min_pitch, note.pitch)
            max_pitch = max(max_pitch, note.pitch)
        pitch_range = max_pitch - min_pitch
        pitch_range = int(pitch_range)
        return pitch_range + 1

    def get_melody(self, mel_def: str) -> List[Note]:
        """
        Extract melody notes from the Bar.

        Parameters
        ----------
        mel_def : str
            Strategy for melody extraction:

            - ``'hi_track'`` — return all notes from the track with the
              highest average pitch.
            - ``'hi_note'`` — return the highest-pitched note at each
              onset position across all non-drum tracks.

        Returns
        -------
        List[Note]
        """
        assert mel_def in ["hi_track", "hi_note"]

        if mel_def == "hi_track":
            track_list = list(self.tracks.values())
            track_list.sort()
            melody_track = track_list[0]
            return melody_track.get_all_notes()
        elif mel_def == "hi_note":
            all_notes = self.get_all_notes(include_drum=False)
            melody_notes = []
            cur_pos = -1
            for note in all_notes:
                if note.onset != cur_pos:
                    cur_pos = note.onset
                    melody_notes.append(note)
            return melody_notes

    def get_chord(self):
        """
        Detect the two half-bar chords for this bar.

        Splits all non-drum notes into the first half (onset < 24) and the
        second half (onset >= 24) and runs chord detection on each group.

        Returns
        -------
        List[Optional[Chord]]
            A two-element list ``[chord_1, chord_2]``.  Each element is a
            :class:`~remi_z.chord.Chord` instance, or ``None`` if no chord
            could be detected for that half.

            Example: ``[Chord('C', ''), Chord('D', 'm7')]``

        Notes
        -----
        Only 4/4 bars are supported (position 24 is the half-bar boundary).
        """
        notes = self.get_all_notes(include_drum=False)

        p_list_1 = [note.pitch for note in notes if note.onset < 24]
        p_list_2 = [note.pitch for note in notes if note.onset >= 24]

        chord_1 = detect_chord_from_pitch_list(p_list_1)
        chord_2 = detect_chord_from_pitch_list(p_list_2)
        return [chord_1, chord_2]

    def get_phrases(self, with_bar_end=False) -> List[str]:
        """
        Return each track's REMI-z token sequence as a list of strings.

        Each element is the space-joined token sequence for one track
        (without the ``b-1`` bar-end token unless ``with_bar_end=True``).

        Parameters
        ----------
        with_bar_end : bool, optional
            If ``True``, append a ``'b-1'`` string as the final element
            (default ``False``).

        Returns
        -------
        List[str]
            One string per track, plus optionally ``'b-1'`` at the end.
        """
        res = []
        for track_id, track in self.tracks.items():
            track_seq = track.to_remiz_seq(with_velocity=False)
            res.append(" ".join(track_seq))
        if with_bar_end:
            res.append("b-1")
        return res

    def has_drum(self):
        """
        Check if the Bar has drum tracks.
        """
        for inst_id, track in self.tracks.items():
            if track.is_drum_track():
                return True
        return False

    def has_piano(self):
        """
        Check if the Bar has any piano tracks.
        """
        piano_ids = set([0, 1, 2, 3, 4, 5, 6, 7])
        for inst_id, track in self.tracks.items():
            if inst_id in piano_ids:
                return True
        return False

    def filter_tracks(self, insts: List[int]):
        """
        Filter the tracks in the Bar object. Only keep the tracks in the insts list.
        """
        new_tracks = {}
        for inst_id in insts:
            if inst_id in self.tracks:
                new_tracks[inst_id] = self.tracks[inst_id]
        self.tracks = new_tracks

    def change_instrument(self, old_inst_id: int, new_inst_id: int):
        """
        Change the instrument ID of a track in the Bar object.
        """
        # assert old_inst_id in self.tracks, f"old_inst_id {old_inst_id} not found in the Bar"
        if old_inst_id not in self.tracks:
            return

        track = self.tracks.pop(old_inst_id)
        track.set_inst_id(new_inst_id)
        self.tracks[new_inst_id] = track

        # Re-sort the tracks
        track_list = list(self.tracks.values())
        track_list.sort()
        self.tracks = {}
        for track in track_list:
            self.tracks[track.inst_id] = track
