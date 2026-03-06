from typing import List, Dict, Tuple

from .note import Note


class Track:
    """
    All notes of a single instrument within one bar.

    Tracks are ordered by average pitch (descending) when sorted, so that
    higher-pitched tracks (typically melody) appear first.

    Attributes
    ----------
    inst_id : int
        General MIDI program number (0–127), or 128 for drums.
    track_id : int or None
        Optional secondary identifier used when multiple tracks share the
        same program ID (e.g., two piano parts).
    is_drum : bool
        True when inst_id == 128.
    notes : List[Note]
        All notes in this track, sorted by onset (ascending), then pitch
        (descending), then duration (descending), then velocity (descending).
    non_empty_pos : List[int]
        Sorted list of onset positions at which at least one note starts.
    avg_pitch : float
        Mean MIDI pitch across all notes. 0 if the track is empty; -1 if
        the track is a drum track.
    """

    def __init__(self, inst_id, notes: Dict[int, List[Note]], track_id=None):
        """
        Create a Track from a position-indexed note dictionary.

        Parameters
        ----------
        inst_id : int
            General MIDI program number (0–127), or 128 for drums.
        notes : Dict[int, List[Note]]
            Mapping from onset position to a list of raw note tuples
            (pitch, duration, velocity) at that position.
        track_id : int or None
            Optional secondary identifier for disambiguation when multiple
            tracks share the same inst_id.
        """
        self.inst_id = inst_id
        self.track_id = track_id
        self.non_empty_pos = list(notes.keys())
        self.non_empty_pos.sort()

        if inst_id == 128:
            self.is_drum = True
        else:
            self.is_drum = False

        self.notes = []
        for pos, notes_of_pos in notes.items():
            for note in notes_of_pos:
                pitch, duration, velocity = note
                note_instance = Note(
                    onset=pos, duration=duration, pitch=pitch, velocity=velocity
                )
                self.notes.append(note_instance)
        self.notes.sort()

        # Calculate the average pitch
        if self.is_drum:
            self.avg_pitch = -1
        else:
            pitches = [note.pitch for note in self.notes]
            if len(pitches) == 0:
                self.avg_pitch = 0
            else:
                self.avg_pitch = sum(pitches) / len(pitches)

    @classmethod
    def from_note_list(cls, inst_id: int, note_list: List[Note]):
        """
        Create a Track from a list of Note objects.

        The input list is copied and will not be modified.

        Parameters
        ----------
        inst_id : int
            General MIDI program number (0–127), or 128 for drums.
        note_list : List[Note]
            Notes to include in this track.
        """
        assert isinstance(note_list, list), "note_list must be a list"

        ret = cls(inst_id=inst_id, notes={})
        ret.notes = list(note_list)
        ret.notes.sort()
        ret.inst_id = inst_id
        ret.non_empty_pos = list(set([note.onset for note in note_list]))
        ret.non_empty_pos.sort()

        # Calculate the average pitch
        if ret.is_drum:
            ret.avg_pitch = -1
        else:
            pitches = [note.pitch for note in ret.notes]
            if len(pitches) == 0:
                ret.avg_pitch = 0
            else:
                ret.avg_pitch = sum(pitches) / len(pitches)

        return ret

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
        Change the instrument ID and update is_drum accordingly.
        """
        self.inst_id = inst_id
        if inst_id == 128:
            self.is_drum = True
        else:
            self.is_drum = False

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
        Return the list of Note objects in this track, sorted by onset.
        """
        return self.notes

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

        Notes from `other` are appended, the combined list is re-sorted, and
        avg_pitch and non_empty_pos are updated. Both tracks must share the
        same inst_id.
        """
        assert isinstance(other, Track), "other must be a Track object"
        assert (
            self.inst_id == other.inst_id
        ), "inst_id of the two tracks must be the same"

        # Merge the notes
        self.notes.extend(other.notes)
        self.notes.sort()

        # Update non_empty_pos
        self.non_empty_pos = sorted(set(note.onset for note in self.notes))

        # Update the average pitch
        if self.is_drum:
            self.avg_pitch = -1
        else:
            pitches = [note.pitch for note in self.notes]
            if len(pitches) == 0:
                self.avg_pitch = 0
            else:
                self.avg_pitch = sum(pitches) / len(pitches)
