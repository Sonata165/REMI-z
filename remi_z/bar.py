import operator
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterable

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


def check_inst_id(inst_id) -> int:
    """
    Validate an instrument ID and return it as a plain int.

    An instrument ID is a General MIDI program number (0–127), or 128 for drums.
    Integer-like values such as numpy integers are accepted.
    """
    try:
        inst_id = operator.index(inst_id)
    except TypeError:
        raise TypeError(
            f"Instrument ID must be an integer (GM program 0-127, or 128 for drums), got {inst_id!r}. "
            "A bar holds at most one track per program; to keep same-program MIDI tracks apart, "
            "remap them with MultiTrack.from_midi(program_map=...)."
        ) from None
    if not 0 <= inst_id <= 128:
        raise ValueError(f"Instrument ID must be in the range of [0, 128], got {inst_id}")
    return inst_id


class Bar:
    """
    A single bar of music containing one or more instrument tracks.

    A bar holds at most one track per instrument: tracks are stored in a dict
    keyed by instrument ID (GM program number 0–127, or 128 for drums), and the
    key always equals the track's ``inst_id``.

    Attributes
    ----------
    bar_id : int
        Index of this bar within its parent MultiTrack (or -1 if standalone).
    tracks : Dict[int, Track]
        Mapping from instrument ID to Track, in voice order.
    voice : List[int]
        Instrument IDs ordered from the track with the highest average pitch to
        the lowest, computed when the bar is constructed. It is the default track
        order for serialization and is not recomputed when notes are edited later;
        ``filter_tracks`` and ``change_instrument`` keep it in sync.
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
            Mapping from instrument ID to a dict of
            ``{onset: [(pitch, duration, velocity), ...]}`` note data.
            Pass an empty dict ``{}`` for a bar with no notes.
        time_signature : Tuple[int, int], optional
            ``(numerator, denominator)``; defaults to ``(4, 4)``.
        tempo : float, optional
            BPM; defaults to ``120.0``.
        """
        if time_signature is None:
            time_signature = (4, 4)
        assert isinstance(time_signature, tuple), "time_signature must be a tuple"
        if tempo is None:
            tempo = 120.0
        assert isinstance(tempo, (int, float)), "tempo must be an integer or float"
        assert tempo > 0, f"tempo must be positive, got {tempo}"

        self.bar_id = id
        self.time_signature = time_signature
        self.tempo = round(tempo, 2)

        tracks = []
        if notes_of_insts is not None:
            for inst_id, notes in notes_of_insts.items():
                tracks.append(Track(check_inst_id(inst_id), notes))
        self._set_tracks(tracks)

    def _set_tracks(self, tracks: List[Track]):
        """
        Replace all tracks and recompute the voice. Instrument IDs must be unique.
        """
        ordered = sorted(tracks)
        self.tracks: Dict[int, Track] = {track.inst_id: track for track in ordered}
        assert len(self.tracks) == len(ordered), "a bar can hold only one track per instrument"
        self.voice: List[int] = [track.inst_id for track in ordered]

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

        Tracks with the same ``inst_id`` are merged into one new Track; other
        Track objects are used as they are, not copied. ``track_list`` itself
        is not modified.
        """
        assert isinstance(track_list, list), "track_list must be a list"

        tracks_of_insts: Dict[int, Track] = {}
        for track in track_list:
            assert isinstance(track, Track), "track_list must contain Track objects"
            inst_id = check_inst_id(track.inst_id)
            if inst_id in tracks_of_insts:
                merged_notes = list(tracks_of_insts[inst_id].notes) + list(track.notes)
                tracks_of_insts[inst_id] = Track.from_note_list(inst_id, merged_notes)
            else:
                tracks_of_insts[inst_id] = track

        bar = cls(id=bar_id, notes_of_insts={}, time_signature=time_signature, tempo=tempo)
        bar._set_tracks(list(tracks_of_insts.values()))
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
                    dur = int(piano_roll[p_pos, pitch]) * coeff
                    onset = p_pos * coeff
                    notes.setdefault(onset, []).append([pitch, dur, 64])  # pitch, duration, velocity

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

        Only notes sharing the same onset and pitch are deduplicated; same-pitch
        offset overlaps are kept, since notes held into the next bar can only be
        fixed at song level (see ``MultiTrack.flatten`` / ``MultiTrack.adjust_offset_overlap``).

        Returns
        -------
        Bar
            A new Bar with one track (instrument 0) containing all
            deduplicated notes from every non-drum track.
        """
        all_notes = self.get_all_notes(include_drum=False, deduplicate=True)

        track = Track.from_note_list(inst_id=0, note_list=all_notes)

        new_bar = Bar.from_tracks(
            bar_id=self.bar_id,
            track_list=[track],
            time_signature=self.time_signature,
            tempo=self.tempo,
        )
        return new_bar

    def _resolve_voice(self, voice: Optional[Iterable[int]]) -> List[int]:
        """
        Return the instrument IDs of this bar's tracks in the order given by
        ``voice`` (``self.voice`` if None).

        ``voice`` may list instruments that are not in this bar (they are
        skipped), so one song-level order can be used for every bar; it must
        cover every track of the bar.
        """
        voice = list(self.voice if voice is None else voice)
        if len(set(voice)) != len(voice):
            raise ValueError(f"voice must not contain duplicate instrument IDs, got {voice}")
        missing = [inst_id for inst_id in self.tracks if inst_id not in voice]
        if missing:
            raise ValueError(
                f"voice {voice} does not cover instrument(s) {missing} of bar {self.bar_id}"
            )
        return [inst_id for inst_id in voice if inst_id in self.tracks]

    def _get_meta_tokens(self, with_ts: bool, with_tempo: bool) -> List[str]:
        """
        Return the optional time-signature and tempo tokens that start a bar.
        """
        tokens = []
        if with_ts:
            num, den = self.time_signature
            tokens.append(
                TimeSignatureUtil.convert_time_signature_to_ts_token(int(num), int(den))
            )
        if with_tempo:
            tokens.append(f"t-{convert_tempo_to_id(self.tempo)}")
        return tokens

    def to_remiz_str(
        self,
        with_ts=False,
        with_tempo=False,
        with_velocity=False,
        include_drum=False,
        voice: Optional[List[int]] = None,
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
        voice : List[int], optional
            Track order as a list of instrument IDs; see :meth:`to_remiz_seq`.

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
            voice=voice,
        )
        return " ".join(remiz_seq)

    def to_remiz_seq(
        self,
        with_ts=False,
        with_tempo=False,
        with_velocity=False,
        include_drum=False,
        voice: Optional[List[int]] = None,
    ):
        """
        Serialize the Bar to a list of REMI-z tokens.

        Tokens are emitted per track in voice order, each track using the format
        ``i-X o-X p-X d-X [v-X] ...``, followed by a ``b-1`` bar-end token.

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
        voice : List[int], optional
            Track order as a list of instrument IDs. Defaults to ``self.voice``.
            Instruments not in this bar are skipped; every track of the bar must
            be listed.

        Returns
        -------
        List[str]
            REMI-z token list ending with ``'b-1'``.
        """
        bar_seq = self._get_meta_tokens(with_ts, with_tempo)
        for inst_id in self._resolve_voice(voice):
            track = self.tracks[inst_id]
            if not include_drum and track.is_drum:
                continue
            bar_seq.extend(track.to_remiz_seq(with_velocity=with_velocity))
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
        bar_seq = self._get_meta_tokens(with_ts, with_tempo)

        all_notes_oipd = []
        for inst_id, track in self.tracks.items():
            if not include_drum and track.is_drum:
                continue
            pitch_offset = 128 if track.is_drum else 0
            for note in track.notes:
                all_notes_oipd.append(
                    (note.onset, inst_id, note.pitch + pitch_offset, note.duration)
                )

        all_notes_oipd.sort()

        for onset, inst_id, pitch, dur in all_notes_oipd:
            bar_seq.extend([f"o-{onset}", f"i-{inst_id}", f"p-{pitch}", f"d-{dur}"])
        bar_seq.append("b-1")

        return bar_seq

    def to_piano_roll(self, of_insts: List[int] = None, pos_per_bar=16) -> np.ndarray:
        """
        Convert the Bar to a piano roll matrix.

        Returns a 2-D integer array where ``result[position, pitch]`` holds the
        note duration in piano-roll positions (0 = silent).  Drum tracks are
        excluded.

        Parameters
        ----------
        of_insts : List[int], optional
            Instrument IDs to include.  ``None`` (default) includes all
            non-drum instruments.
        pos_per_bar : int, optional
            Grid resolution in positions per 4/4 bar, i.e. per whole note
            (default 16).

        Returns
        -------
        np.ndarray
            Integer array of shape ``(pos_per_bar * num // den, 128)`` for a
            ``num/den`` time signature, e.g. 16 rows for 4/4, 12 for 3/4 and 6/8.

        Notes
        -----
        Quantize the Bar to 16th-note resolution before calling this method
        to avoid rounding artefacts.
        """
        n_pitch = 128
        coeff = 48 / pos_per_bar

        num, den = self.time_signature
        n_pos = pos_per_bar * num // den
        piano_roll = np.zeros((n_pos, n_pitch), dtype=int)

        all_insts = self.get_unique_insts()
        if of_insts is None:
            insts = all_insts
        else:
            insts = set(all_insts).intersection(of_insts)

        notes = self.get_all_notes(include_drum=False, of_insts=insts)
        notes = deduplicate_notes(notes)

        for note in notes:
            onset_pos = min(round(note.onset / coeff), n_pos - 1)
            dur = round(note.duration / coeff)
            piano_roll[onset_pos, note.pitch] = dur

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
            track = self.tracks[inst_id]
            if not include_drum and track.is_drum:
                continue
            all_notes.extend(track.get_all_notes())

        all_notes.sort()

        if deduplicate:
            all_notes = deduplicate_notes(all_notes)

        return all_notes

    @staticmethod
    def _notes_to_content_seq(notes: List[Note], with_dur: bool, pitch_offset: int = 0) -> List[str]:
        """
        Convert sorted notes to a content sequence (``o-X p-X [d-X] ... b-1``),
        removing repeated notes with the same onset and pitch first.
        """
        bar_seq = []
        prev_pos = -1
        for note in deduplicate_notes(notes):
            if note.onset > prev_pos:
                bar_seq.append(f"o-{note.onset}")
                prev_pos = note.onset
            bar_seq.append(f"p-{note.pitch + pitch_offset}")
            if with_dur:
                bar_seq.append(f"d-{note.duration}")
        bar_seq.append("b-1")
        return bar_seq

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
        return self._notes_to_content_seq(notes, with_dur)

    def get_drum_content_seq(self, with_dur=True):
        """
        Convert the Bar object to a content sequence.
        Including information about all drum notes being played

        Args:
            with_dur: Whether to include duration information in the content sequence.
        """
        notes = self.get_all_notes(include_drum=True, of_insts=[128])
        return self._notes_to_content_seq(notes, with_dur, pitch_offset=128)

    def get_unique_insts(
        self, sort_by_voice=True, include_drum=True, voice: Optional[List[int]] = None
    ) -> List[int]:
        """
        Return the instrument IDs present in this Bar.

        Parameters
        ----------
        sort_by_voice : bool, optional
            Must be ``True`` (only voice-sorted order is currently supported).
        include_drum : bool, optional
            Whether to include the drum track (instrument 128) in the result
            (default ``True``).
        voice : List[int], optional
            Track order as a list of instrument IDs; see :meth:`to_remiz_seq`.

        Returns
        -------
        List[int]
            Instrument IDs in voice order (``self.voice`` unless ``voice`` is given).
        """
        assert sort_by_voice is True, "sort_by_voice must be True"

        return [
            inst_id
            for inst_id in self._resolve_voice(voice)
            if include_drum or inst_id != 128
        ]

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
            insts = set(all_insts).intersection(of_insts)

        notes = self.get_all_notes(include_drum=False, of_insts=list(insts))
        if len(notes) == 0:
            return -1

        pitches = [note.pitch for note in notes]
        return int(max(pitches) - min(pitches)) + 1

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

    def get_phrases(self, with_bar_end=False, voice: Optional[List[int]] = None) -> List[str]:
        """
        Return each track's REMI-z token sequence as a list of strings.

        Each element is the space-joined token sequence for one track
        (without the ``b-1`` bar-end token unless ``with_bar_end=True``).

        Parameters
        ----------
        with_bar_end : bool, optional
            If ``True``, append a ``'b-1'`` string as the final element
            (default ``False``).
        voice : List[int], optional
            Track order as a list of instrument IDs; see :meth:`to_remiz_seq`.

        Returns
        -------
        List[str]
            One string per track, plus optionally ``'b-1'`` at the end.
        """
        res = []
        for inst_id in self._resolve_voice(voice):
            track_seq = self.tracks[inst_id].to_remiz_seq(with_velocity=False)
            res.append(" ".join(track_seq))
        if with_bar_end:
            res.append("b-1")
        return res

    def has_drum(self):
        """
        Check if the Bar has drum tracks.
        """
        return 128 in self.tracks

    def has_piano(self):
        """
        Check if the Bar has any piano tracks.
        """
        piano_ids = set([0, 1, 2, 3, 4, 5, 6, 7])
        return any(inst_id in piano_ids for inst_id in self.tracks)

    def filter_tracks(self, insts: List[int]):
        """
        Filter the tracks in the Bar object. Only keep the tracks in the insts list.
        The voice order of the kept tracks is unchanged.
        """
        insts = set(insts)
        self.tracks = {
            inst_id: track for inst_id, track in self.tracks.items() if inst_id in insts
        }
        self.voice = [inst_id for inst_id in self.voice if inst_id in insts]

    def change_instrument(self, old_inst_id: int, new_inst_id: int):
        """
        Change the instrument ID of a track in the Bar object.

        The track keeps its place in the voice order. If the bar already has a
        track of ``new_inst_id``, the two tracks are merged into that one, at
        its place in the voice order. Does nothing if ``old_inst_id`` is not
        in the bar.
        """
        new_inst_id = check_inst_id(new_inst_id)
        if old_inst_id not in self.tracks or old_inst_id == new_inst_id:
            return

        track = self.tracks.pop(old_inst_id)
        track.set_inst_id(new_inst_id)
        if new_inst_id in self.tracks:
            self.tracks[new_inst_id].merge_with(track)
            self.voice.remove(old_inst_id)
        else:
            self.tracks[new_inst_id] = track
            self.voice[self.voice.index(old_inst_id)] = new_inst_id

        self.tracks = {inst_id: self.tracks[inst_id] for inst_id in self.voice}
