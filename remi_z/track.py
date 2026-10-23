from typing import Dict, Iterable, List, Tuple

from .note import Note, NoteSeq


class Track:
    """
    All notes of a single instrument within one bar.

    Tracks are ordered by average pitch (descending) when sorted, so that
    higher-pitched tracks (typically melody) appear first.

    Attributes
    ----------
    inst_id : int
        General MIDI program number (0–127), or 128 for drums.
    is_drum : bool
        True when inst_id == 128. Read-only; derived from inst_id.
    notes : NoteSeq
        All notes in this track, always sorted by onset (ascending), then pitch
        (descending), then duration (descending), then velocity (descending).
        Assigning a list or NoteSeq replaces the notes with a sorted copy.
    non_empty_pos : List[int]
        Sorted list of onset positions at which at least one note starts.
        Read-only; derived from notes.
    avg_pitch : float
        Mean MIDI pitch across all notes. 0 if the track is empty; -1 if
        the track is a drum track. Read-only; derived from notes.
    """

    def __init__(self, inst_id: int, notes: Dict[int, List[Tuple[int, int, int]]]):
        """
        Create a Track from a position-indexed note dictionary.

        Parameters
        ----------
        inst_id : int
            General MIDI program number (0–127), or 128 for drums.
        notes : Dict[int, List[Tuple[int, int, int]]]
            Mapping from onset position to a list of raw note tuples
            (pitch, duration, velocity) at that position.
        """
        self.inst_id = inst_id
        self.notes = [
            Note(onset=pos, duration=duration, pitch=pitch, velocity=velocity)
            for pos, notes_of_pos in notes.items()
            for pitch, duration, velocity in notes_of_pos
        ]

    @classmethod
    def from_note_list(cls, inst_id: int, note_list: List[Note]):
        """
        Create a Track from a list of Note objects.

        The input list is copied and will not be modified. The Note objects
        themselves are shared with the input, not copied.

        Parameters
        ----------
        inst_id : int
            General MIDI program number (0–127), or 128 for drums.
        note_list : List[Note]
            Notes to include in this track.
        """
        assert isinstance(note_list, list), "note_list must be a list"

        ret = cls(inst_id=inst_id, notes={})
        ret.notes = note_list
        return ret

    @property
    def notes(self) -> NoteSeq:
        return self._notes

    @notes.setter
    def notes(self, notes: Iterable[Note]):
        self._notes = NoteSeq(notes)

    @property
    def is_drum(self) -> bool:
        return self.inst_id == 128

    @property
    def avg_pitch(self) -> float:
        if self.is_drum:
            return -1
        return self._notes.get_avg_pitch()

    @property
    def non_empty_pos(self) -> List[int]:
        return self._notes.get_onsets()

    def __str__(self) -> str:
        return f"Inst {self.inst_id}: {len(self.notes)} notes, avg_pitch={self.avg_pitch:.02f}"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return len(self.notes)

    def __lt__(self, other):
        """
        Order by avg_pitch descending so that higher-pitched tracks sort first.
        """
        return self.avg_pitch > other.avg_pitch

    def set_inst_id(self, inst_id: int):
        """
        Change the instrument ID. is_drum and avg_pitch follow automatically.
        """
        self.inst_id = inst_id

    def get_note_list(self) -> List[Tuple[int, int, int, int]]:
        """
        Get all notes in the Track.

        Returns:
            List of tuples (onset, pitch, duration, velocity)
        """
        all_notes = []
        for note in self.notes:
            all_notes.append((note.onset, note.pitch, note.duration, note.velocity))
        return all_notes

    def get_avg_pitch(self) -> float:
        """
        Return the average MIDI pitch of all notes.
        Returns 0 for an empty track, -1 for a drum track.
        """
        return self.avg_pitch

    def get_all_notes(self) -> List[Note]:
        """
        Return the Note objects in this track as a new list, sorted by onset.

        Changing the returned list does not change the track, but the Note
        objects in it are shared with the track.
        """
        return self._notes.notes

    def to_remiz_seq(self, with_velocity=False) -> List[str]:
        """
        Convert this track to a REMI-z token sequence.

        The sequence starts with an instrument token (e.g. 'i-24'), followed
        by interleaved onset ('o-X'), pitch ('p-X'), and duration ('d-X') tokens.
        Drum pitches are offset by +128 in the token (e.g. pitch 36 → 'p-164').

        Parameters
        ----------
        with_velocity : bool
            If True, a velocity token ('v-X') is appended after each note.
        """
        track_seq = [f"i-{self.inst_id}"]
        prev_pos = -1
        for note in self.notes:
            if note.onset > prev_pos:
                track_seq.append(f"o-{note.onset}")
                prev_pos = note.onset

            if self.is_drum:
                pitch_id = note.pitch + 128
            else:
                pitch_id = note.pitch
            track_seq.extend(
                [
                    f"p-{pitch_id}",
                    f"d-{note.duration}",
                ]
            )

            if with_velocity:
                track_seq.append(f"v-{note.velocity}")

        return track_seq

    def to_remiz_str(self, with_velocity=False) -> str:
        """
        Convert this track to a REMI-z token string (space-separated tokens).

        Parameters
        ----------
        with_velocity : bool
            If True, a velocity token ('v-X') is appended after each note.
        """
        track_seq = self.to_remiz_seq(with_velocity=with_velocity)
        track_str = " ".join(track_seq)
        return track_str

    def is_drum_track(self) -> bool:
        """
        Check if the Track is a drum track.
        """
        return self.is_drum

    def merge_with(self, other):
        """
        Merge another Track into this one in place.

        Notes from `other` are inserted in sorted order; avg_pitch and
        non_empty_pos follow automatically. Both tracks must share the same
        inst_id.
        """
        assert isinstance(other, Track), "other must be a Track object"
        assert (
            self.inst_id == other.inst_id
        ), "inst_id of the two tracks must be the same"

        self._notes.extend(other.notes)
