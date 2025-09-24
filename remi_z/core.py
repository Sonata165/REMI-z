import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(__file__))

import miditoolkit
import pretty_midi
import numpy as np
from utils import read_yaml
from typing import List, Dict, Tuple
from midi_encoding import MidiEncoder, load_midi, fill_pos_ts_and_tempo_, convert_tempo_to_id, convert_id_to_tempo
from time_signature_utils import TimeSignatureUtil
from keys_normalization import detect_key
from chord_detection import detect_chord_from_pitch_list

class Note:
    def __init__(self, onset:int, duration:int, pitch:int, velocity:int=64, is_drum=False):
        '''
        Create an instance of a Note object.

        Parameters
        ----------
        onset : int
            The onset time of the note.
            NOTE: The time unit is "position", which is 48th note (1/12 of a beat). Usually, one bar has 48 positions.
            Range: [0, 127] The maximum allowed onset position is 127, e.g., at the end of 127*1/12 = 10.58 beats.
        duration : int
            The duration of the note.
            NOTE: The time unit is "position", which is 48th note (1/12 of a beat).
            And, the duration of a note should be greater than 1 (smaller value will be rounded to 1).
            The maximum allowed duration is 127, e.g., has length of 127*1/12 = 10.58 beats.
        pitch : int
            The MIDI pitch of the note.
            The pitch value should be in the range of [0, 255].
                0~127: MIDI pitch
        velocity : int
            The velocity of the note.
        '''
        assert isinstance(onset, int), "onset must be an integer"
        assert isinstance(duration, int), "duration must be an integer"
        assert isinstance(pitch, int), "pitch must be an integer"
        assert isinstance(velocity, int), "velocity must be an integer"
        assert 0 <= onset <= 127, f"onset must be in the range of [0, 127], got {onset}"
        assert 0 <= pitch <= 255, f"pitch must be in the range of [0, 255], got {pitch}"
        assert 0 <= velocity <= 127, f"velocity must be in the range of [0, 127], got {velocity}"
        # if is_drum:
        #     assert 128 <= pitch <= 255, "Drum pitch must be in the range of [128, 255]"
        # else:
        assert 0 <= pitch <= 127, "MIDI pitch must be in the range of [0, 127]"

        # Round the values
        duration = min(max(1, duration), 127) # duration must be in the range of [1, 127]

        self.onset = onset
        self.duration = duration
        self.pitch = pitch
        self.velocity = velocity

    def get_note_name(self):
        return midi_pitch_to_note_name(self.pitch)

    def __str__(self) -> str:
        return f'(o:{self.onset},p:{self.pitch},d:{self.duration},v:{self.velocity})'
    
    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other):
        '''
        When comparing two notes
        - If the onset is different, the note with smaller onset will be placed at the front
        - If the onset is the same, the note with higher pitch will be placed at the front
        - If onset and pitch are same, note with longer duration will be placed at the front
        - If onset, pitch, and duration are same, note with larger velocity will be placed at the front
        '''
        if self.onset != other.onset:
            return self.onset < other.onset
        elif self.pitch != other.pitch:
            return self.pitch > other.pitch
        elif self.duration != other.duration:
            return self.duration > other.duration
        else:
            return self.velocity > other.velocity


class NoteSeq:
    def __init__(self, note_list: List[Note]):
        self.notes = note_list

    def __str__(self):
        return 'NoteSeq: [' + ' '.join([note.get_note_name() for note in self.notes]) + ']'
    
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
        '''
        Return the pitch range of the NoteSeq object.
        In format of (lowest_pitch, highest_pitch)

        When there are note notes in the NoteSeq, return None.
        '''
        if len(self.notes) == 0:
            return None
        if len(self.notes) == 1:
            return (self.notes[0].pitch, self.notes[0].pitch)
        

        pitch = [note.pitch for note in self.notes]
        l_pitch = min(pitch)
        h_pitch = max(pitch)
        return (l_pitch, h_pitch)
    

class ChordSeq:
    def __init__(self, chord_list:List[Tuple[str, str]]):
        self.chord_list = chord_list

    def __str__(self):
        return f'{self.chord_list[0][0]} {self.chord_list[0][1]} {self.chord_list[1][0]} {self.chord_list[1][1]} | '
    
    def __repr__(self):
        return self.__str__()
    

class Track:
    '''
    This class save all notes for a same track within a bar.
    '''
    def __init__(self, inst_id, notes:Dict[int, List[Note]], track_id=None):
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
                note_instance = Note(onset=pos, duration=duration, pitch=pitch, velocity=velocity, is_drum=self.is_drum)
                self.notes.append(note_instance)
        self.notes.sort()

        # Calculate the average pitch
        if self.is_drum:
            self.avg_pitch = -1
        else:
            pitches = [note.pitch for note in self.notes]
            if len(pitches) == 0:
                self.avg_pitch = -1
            else:
                self.avg_pitch = sum(pitches) / len(pitches)

    @classmethod
    def from_note_list(cls, inst_id:int, note_list:List[Note]):
        '''
        Create a Track object from a list of Note objects.
        '''
        assert isinstance(note_list, list), "note_list must be a list"

        ret = cls(inst_id=inst_id, notes={})
        ret.notes = note_list
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
                ret.avg_pitch = -1
            else:
                ret.avg_pitch = sum(pitches) / len(pitches)
            
        return ret

    def __str__(self) -> str:
        return f'Inst {self.inst_id}: {len(self.notes)} notes, avg_pitch={self.avg_pitch:.02f}'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __len__(self):
        return len(self.notes)
    
    def __lt__(self, other):
        '''
        Track with higher pitch will be placed at the front (more important)
        '''
        return self.avg_pitch > other.avg_pitch

    def set_inst_id(self, inst_id:int):
        '''
        Set the instrument ID of the Track.
        '''
        self.inst_id = inst_id
        if inst_id == 128:
            self.is_drum = True
        else:
            self.is_drum = False
    
    def get_note_list(self) -> List[Tuple[int, int, int, int]]:
        '''
        Get all notes in the Track.

        Returns:
            List of tuples (onset, pitch, duration, velocity)
        '''
        all_notes = []
        for note in self.notes:
            all_notes.append((note.onset, note.pitch, note.duration, note.velocity))
        return all_notes
    
    def get_avg_pitch(self) -> float:
        '''
        Get the average pitch of the Track.
        If the Track is a drum track, return -1.
        '''
        return self.avg_pitch
    
    def get_all_notes(self) -> List[Note]:
        '''
        Get all notes in the Track.
        '''
        return self.notes
    
    def is_drum_track(self) -> bool:
        '''
        Check if the Track is a drum track.
        '''
        return self.is_drum
    
    def merge_with(self, other):
        '''
        Merge the current Track with another Track.
        '''
        assert isinstance(other, Track), "other must be a Track object"
        assert self.inst_id == other.inst_id, "inst_id of the two tracks must be the same"

        # Merge the notes
        self.notes.extend(other.notes)
        self.notes.sort()

        # Update the average pitch
        if self.is_drum:
            self.avg_pitch = -1
        else:
            pitches = [note.pitch for note in self.notes]
            if len(pitches) == 0:
                self.avg_pitch = -1
            else:
                self.avg_pitch = sum(pitches) / len(pitches)


class Bar:
    def __init__(self, 
                 id, 
                 notes_of_insts:Dict[int, Dict[int, List]], 
                 time_signature:Tuple[int, int]=None, 
                 tempo:float=None
                 ):
        '''
        NOTE: The instrument with higher average pitch will be placed at the front.
        '''
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
        return f'Bar {self.bar_id}: {len(self.tracks)} insts'

    def __repr__(self) -> str:
        return self.__str__()
    
    @classmethod
    def from_tracks(cls, bar_id, track_list, time_signature=(4, 4), tempo=120.0):
        '''
        Create a Bar object from a list of Track objects.
        '''
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
        bar = cls(id=bar_id, notes_of_insts={}, time_signature=time_signature, tempo=tempo)

        # Add tracks to the Bar object
        track_list.sort()
        for track in track_list:
            inst_id = track.inst_id
            bar.tracks[inst_id] = track

        return bar

    @classmethod
    def from_piano_roll(cls, piano_roll, pos_per_bar=16, time_signature=(4, 4), tempo=120.0):
        '''
        Convert the Bar object to a piano roll matrix.
        '''
        coeff = 48 // pos_per_bar

        notes = {}
        for p_pos in range(piano_roll.shape[0]):
            for pitch in range(piano_roll.shape[1]):
                if piano_roll[p_pos, pitch] > 0:
                    p_dur = piano_roll[p_pos, pitch]
                    dur = int((p_dur * coeff).item())
                    onset = p_pos * coeff
                    # note = Note(onset=onset, duration=dur, pitch=pitch)
                    note = [pitch, dur, 64] # pitch, duration, velocity

                    # Add note to notes
                    if onset not in notes:
                        notes[onset] = []
                    notes[onset].append(note)

        return cls(id=-1, notes_of_insts={0:notes}, time_signature=time_signature, tempo=tempo)

    def flatten(self) -> 'Bar':
        '''
        Flatten all tracks into a same one
        Save to a new Bar object.
        Remove drum tracks.
        '''        
        assert len(self.tracks) > 0, "Bar has no tracks to flatten"
        all_notes = self.get_all_notes(include_drum=False, deduplicate=True)

        track = Track.from_note_list(inst_id=0, note_list=all_notes)

        # Create a new Bar object with a single track
        new_bar = Bar.from_tracks(
            bar_id=self.bar_id,
            track_list=[track],
            time_signature=self.time_signature,
            tempo=self.tempo
        )
        return new_bar

    def to_remiz_seq(self, with_ts=False, with_tempo=False, with_velocity=False, include_drum=False):
        bar_seq = []

        # Add time signature
        if with_ts:
            # time_sig = bar.time_signature.strip()[1:-1]
            num, den = self.time_signature
            ts_token = TimeSignatureUtil.convert_time_signature_to_ts_token(int(num), int(den))
            bar_seq.append(ts_token)

        if with_tempo:
            tempo_id = convert_tempo_to_id(self.tempo)
            tempo_tok = f't-{tempo_id}'
            bar_seq.append(tempo_tok)
            
        for inst_id, track in self.tracks.items():
            if include_drum is False and track.is_drum:
                continue

            track_seq = [f'i-{inst_id}']
            prev_pos = -1
            for note in track.notes:
                if note.onset > prev_pos:
                    track_seq.append(f'o-{note.onset}')
                    prev_pos = note.onset

                if track.is_drum:
                    pitch_id = note.pitch + 128
                else:
                    pitch_id = note.pitch
                track_seq.extend([
                    f'p-{pitch_id}',
                    f'd-{note.duration}',
                ])

                if with_velocity:
                    track_seq.append(f'v-{note.velocity}')
            bar_seq.extend(track_seq)
        bar_seq.append('b-1')

        return bar_seq
    
    def to_remiplus_seq(self, with_ts=False, with_tempo=False, with_velocity=False, include_drum=False):
        bar_seq = []

        # Add time signature
        if with_ts:
            # time_sig = bar.time_signature.strip()[1:-1]
            num, den = self.time_signature
            ts_token = TimeSignatureUtil.convert_time_signature_to_ts_token(int(num), int(den))
            bar_seq.append(ts_token)

        if with_tempo:
            tempo_id = convert_tempo_to_id(self.tempo)
            tempo_tok = f't-{tempo_id}'
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

                all_notes_oipd.append(
                    (note.onset, inst_id, pitch_id, note.duration)
                )

        all_notes_oipd.sort()

        for onset, inst_id, pitch, dur in all_notes_oipd:
            bar_seq.extend([
                f'o-{onset}',
                f'i-{inst_id}',
                f'p-{pitch}',
                f'd-{dur}',
            ])
        bar_seq.append('b-1')

        return bar_seq
    
    def to_piano_roll(self, of_insts: List[int]=None, pos_per_bar=16) -> np.ndarray:
        '''
        Convert the Bar object to a piano roll matrix.

        NOTE: Always first quantize the MultiTrack to 16th notes before use this function

        Args:
            of_insts: List of instrument IDs to be included in the piano roll. None means all instruments.
        '''
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
            insts = all_insts.intersection(of_insts)

        # Obtain notes to be added to the piano roll
        notes = self.get_all_notes(
            include_drum=False,
            of_insts=insts
        )

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

    def to_midi(self, midi_fp: str):
        mt = MultiTrack.from_bars([self])
        mt.to_midi(midi_fp)

    def get_all_notes(self, include_drum=True, of_insts:List[int]=None, deduplicate=False) -> List[Note]:
        '''
        Get all notes in the Bar 
        NOTE: Results are sorted by onset, pitch, duration, and velocity.
        '''
        assert isinstance(of_insts, (list, set)) or of_insts is None, "of_insts must be a list or None"
        
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
        '''
        Convert the Bar object to a content sequence.
        Including information about all notes being played
        Without instrument information.

        Args:
            include_drum: Whether to include drum tracks
            of_insts: A list of instrument IDs to extract the content sequence. None means all instruments.
            with_dur: Whether to include duration information in the content sequence.
        '''
        assert include_drum is False, "include_drum in content sequence is not supported yet"

        notes = self.get_all_notes(
            include_drum=include_drum,
            of_insts=of_insts
        )

        # Remove repeated notes with same onset and pitch, keep one with largest duration
        notes = deduplicate_notes(notes)

        # Convert to content sequence (containing only o-X, p-X, d-X)
        bar_seq = []
        prev_pos = -1
        for note in notes:
            if note.onset > prev_pos:
                bar_seq.append(f'o-{note.onset}')
                prev_pos = note.onset

            bar_seq.extend([
                f'p-{note.pitch}',
            ])
            if with_dur:
                bar_seq.append(f'd-{note.duration}')

        bar_seq.append('b-1')

        return bar_seq

    def get_drum_content_seq(self, with_dur=True):
        '''
        Convert the Bar object to a content sequence.
        Including information about all drum notes being played

        Args:
            with_dur: Whether to include duration information in the content sequence.
        '''

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
                bar_seq.append(f'o-{note.onset}')
                prev_pos = note.onset

            pitch_id = note.pitch + 128
            bar_seq.extend([
                f'p-{pitch_id}',
            ])
            if with_dur:
                bar_seq.append(f'd-{note.duration}')

        bar_seq.append('b-1')

        return bar_seq

    def get_unique_insts(self, sort_by_voice=True, include_drum=True) -> List[int]:
        '''
        Get all unique instruments in the MultiTrack object.
        '''
        assert sort_by_voice is True, "sort_by_voice must be True"

        all_insts = []
        for inst_id in self.tracks.keys():
            if include_drum is False and inst_id == 128:
                continue
            all_insts.append(inst_id)

        return all_insts

    def get_pitch_range(self, of_insts:List[int]=None):
        '''
        Calculate the range of the notes in the Bar.
        Will return max_pitch - min_pitch + 1
        If no notes found, return -1.
        '''
        assert isinstance(of_insts, (list, set)) or of_insts is None, "of_insts must be a list or None"
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
    
    def get_melody(self, mel_def:str) -> List[Note]:
        '''
        Get melody notes
        mel_def = 'hi_track': content of tarck with highest avg pitch
        mel_def = 'hi_note': highest note of each position
        '''
        assert mel_def in ['hi_track', 'hi_note']

        if mel_def == 'hi_track':
            track_list = list(self.tracks.values())
            track_list.sort()
            melody_track = track_list[0]
            return melody_track.get_all_notes()
        elif mel_def == 'hi_note':
            all_notes = self.get_all_notes(include_drum=False)
            melody_notes = []
            cur_pos = -1
            for note in all_notes:
                if note.onset != cur_pos:
                    cur_pos = note.onset
                    melody_notes.append(note)
            return melody_notes
        
    def get_chord(self):
        '''
        Calculate the chord of this bar
        Return a list contains two chords, like below
            [('C', 'Major'), ('D', 'Minor7')]
        If no notes inside, return [None, None]

        NOTE: this function only support 4/4 bars for now
        '''
        notes = self.get_all_notes(include_drum=False)
        
        p_list_1 = [note.pitch for note in notes if note.onset < 24]
        p_list_2 = [note.pitch for note in notes if note.onset >= 24]

        chord_1 = detect_chord_from_pitch_list(p_list_1, return_root_name=True)
        chord_2 = detect_chord_from_pitch_list(p_list_2, return_root_name=True)
        return [chord_1, chord_2]
        
    
    def has_drum(self):
        '''
        Check if the Bar has drum tracks.
        '''
        for inst_id, track in self.tracks.items():
            if track.is_drum_track():
                return True
        return False

    def has_piano(self):
        '''
        Check if the Bar has any piano tracks.
        '''
        piano_ids = set([0, 1, 2, 3, 4, 5, 6, 7])
        for inst_id, track in self.tracks.items():
            if inst_id in piano_ids:
                return True
        return False
    
    def filter_tracks(self, insts:List[int]):
        '''
        Filter the tracks in the Bar object. Only keep the tracks in the insts list.
        '''
        new_tracks = {}
        for inst_id in insts:
            if inst_id in self.tracks:
                new_tracks[inst_id] = self.tracks[inst_id]
        self.tracks = new_tracks
    

class MultiTrack:
    def __init__(self, bars:List[Bar]):
        '''
        Args:
            bars: List of Bar objects
            pitch_shift: The pitch shift value. None means not detected.
            is_major: The major/minor key information. None means not detected.
        '''
        self.bars = bars

        self.update_ts_and_tempo()

        # Load the time signature dictionary
        ts_fp = os.path.join(os.path.dirname(__file__), 'dict_time_signature.yaml')
        self.ts_dict = read_yaml(ts_fp)

    def update_ts_and_tempo(self):
        # Collate time signature and tempo info
        self.time_signatures = set()
        self.tempos = set()
        for bar in self.bars:
            self.time_signatures.add(bar.time_signature)
            self.tempos.add(bar.tempo)
        self.time_signatures = list(self.time_signatures)
        self.tempos = list(self.tempos)

    def __len__(self):
        return len(self.bars)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            bar_subset = [self.bars[idx]]
            return self.bars[idx]
        elif isinstance(idx, slice):
            bar_subset = self.bars[idx]
            return MultiTrack(bars=bar_subset)
    
    def __str__(self) -> str:
        return f'MultiTrack: {len(self.bars)} bars'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def set_tempo(self, tempo:float):
        '''
        Set the tempo for all bars in the MultiTrack object.
        '''
        assert isinstance(tempo, (int, float)), "tempo must be an integer or float"
        tempo = round(tempo, 2)
        for bar in self.bars:
            bar.tempo = tempo

        # Update the tempo in the tempo set
        self.update_ts_and_tempo()
    
    def normalize_pitch(self):
        '''
        Normalize the pitch of all notes in the MultiTrack object.
        '''

        ''' Detect major/minor key and pitch shift needed for the key normalization '''
        is_major, pitch_shift = self.detect_key()

        ''' Apply the pitch shift to the notes '''
        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                if track.is_drum:
                    continue
                else:
                    for note in track.notes:
                        note.pitch += pitch_shift
                        
                        # If the pitch is out of range, adjust it
                        if note.pitch < 0:
                            note.pitch += 12

    def shift_pitch(self, pitch_shift):
        '''
        Shift pitch for all notes
        '''

        ''' Apply the pitch shift to the notes '''
        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                if track.is_drum:
                    continue
                else:
                    for note in track.notes:
                        note.pitch += pitch_shift
                        
                        # If the pitch is out of range, adjust it
                        if note.pitch < 0:
                            note.pitch += 12
    
    def quantize_to_16th(self):
        '''
        Quantize the MultiTrack object to 16th notes. 
        On both onset and duration.
        '''
        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                for note in track.notes:
                    note.onset = round(note.onset / 3) * 3
                    note.duration = max(1, round(note.duration / 3)) * 3

                    if note.duration > 20:
                        a = 3
    
    def detect_key(self):
        '''
        Determine the major/minor key and pitch shift needed for key normalization.

        Returns:
        - is_major: True if major, False if minor
        - pitch_shift: The pitch shift needed for key normalization
        '''
        note_list = self.get_note_list(with_drum=False)
        is_major, pitch_shift = detect_key(note_list)
        return is_major, pitch_shift

    def get_unique_insts(self):
        '''
        Get all unique instruments in the MultiTrack object.
        '''
        all_insts = set()
        for bar in self.bars:
            for inst_id in bar.tracks.keys():
                all_insts.add(inst_id)
        return all_insts

    @classmethod
    def from_bars(cls, bars:List[Bar]):
        '''
        Create a MultiTrack object from a list of Bar objects.
        '''
        assert isinstance(bars, list), "bars must be a list"
        return cls(bars=bars)

    @classmethod
    def from_note_seqs(cls, note_seq:List[NoteSeq], program_id:int=0):
        '''
        Create a MultiTrack object from a list of NoteSeq objects.
        Each NoteSeq will be converted to a Bar with a single instrument.
        '''
        assert isinstance(note_seq, list), "note_seq must be a list"
        assert len(note_seq) > 0, "note_seq must not be empty"
        assert isinstance(program_id, int), "program_id must be an integer"
        assert 0 <= program_id <= 127, "program_id must be in the range of [0, 127]"

        # Create a list of Bar objects
        bars = []
        for bar_id, seq in enumerate(note_seq):
            assert isinstance(seq, NoteSeq), "note_seq must be a list of NoteSeq objects"
            notes_of_insts = {program_id: {note.onset: [[note.pitch, note.duration, note.velocity]] for note in seq.notes}}
            bar = Bar(id=bar_id, notes_of_insts=notes_of_insts)
            bars.append(bar)
        
        return cls(bars=bars)

    @classmethod
    def from_midi(cls, midi_fp:str, support_same_program_multi_instance=False):
        '''
        Create a MultiTrack object from a MIDI file.
        '''
        assert isinstance(midi_fp, str), "midi_fp must be a string"
        assert os.path.exists(midi_fp), "midi_fp does not exist"

        midi_encoder = MidiEncoder()
        midi_obj = load_midi(midi_fp)

        # Obtain information for each position (a dense representation)
        pos_info = midi_encoder.collect_pos_info(
            midi_obj, 
            trunc_pos=None, 
            tracks=None, 
            remove_same_notes=False, 
            end_offset=0,
            multi_instance_same_program_support=support_same_program_multi_instance, # support multiple instances of the same program ID
        )
        # bar: every pos
        # ts: only at pos where it changes, otherwise None
        # in-bar position
        # tempo: only at pos where it changes, otherwise None
        # insts_notes: only at pos where the note starts, otherwise None

        # Fill time signature and tempo info to the first pos of each bar
        pos_info = fill_pos_ts_and_tempo_(pos_info)

        # # Determine pitch normalization and major/minor info
        # _, is_major, pitch_shift = midi_encoder.normalize_pitch(pos_info) # Can make error some times. Not sure about direction of pitch shift yet.
        # # If apply this pitch shift to the original MIDI, the key will be C major or A minor

        # Generate bar sequences and note sequences
        bar_seqs = []
        bar_id_prev_pos = -1
        cur_ts = None
        cur_tempo = None
        for i in range(len(pos_info)):
            bar_id, ts, pos, tempo, insts_notes = pos_info[i]

            # Update time signature and tempo
            if ts is not None:
                cur_ts = ts
            if tempo is not None:
                cur_tempo = tempo
            
            # Determine if this is a new bar
            if bar_id > bar_id_prev_pos:
                notes_of_tracks = {}

            # Add the note info
            if insts_notes is not None:
                for inst_id, notes in insts_notes.items():
                    # each note contain [pitch, duration, velocity]
                    if inst_id not in notes_of_tracks:
                        notes_of_tracks[inst_id] = {}
                    if pos not in notes_of_tracks[inst_id]:
                        notes_of_tracks[inst_id][pos] = []
                    notes_of_tracks[inst_id][pos].extend(notes)

            # Determine if this is the last position of a bar
            last_pos_of_bar = False
            if i == len(pos_info) - 1:
                last_pos_of_bar = True
            else:
                next_bar_id, _, next_pos, _, _ = pos_info[i+1]
                if next_bar_id != bar_id:
                    last_pos_of_bar = True
            
            # Add the bar info
            if last_pos_of_bar:
                bar_instance = Bar(
                    id=bar_id, 
                    notes_of_insts=notes_of_tracks,
                    time_signature=cur_ts,
                    tempo=cur_tempo,
                )
                bar_seqs.append(bar_instance)
                
            bar_id_prev_pos = bar_id

        return cls(bars=bar_seqs)
    
    @classmethod
    def from_remiz_seq(cls, remiz_seq:List[str]):
        '''
        Create a MultiTrack object from a remiz sequence.
        '''
        assert isinstance(remiz_seq, list), "remiz_seq must be a list"

        remiz_str = ' '.join(remiz_seq)
        return cls.from_remiz_str(remiz_str)

    @classmethod
    def from_remiz_str(cls, remiz_str:str):
        '''
        Create a MultiTrack object from a remiz string.
        '''
        assert isinstance(remiz_str, str), "remiz_str must be a string"
        if 'b-1' not in remiz_str:
            print('WARNING: remiz_str does not contain any bar information. Adding a bar at the end.')
            remiz_str += ' b-1'
        if 'v' in remiz_str:
            with_velocity = True
        else:
            with_velocity = False

        bar_seqs = []

        # Split to bars
        bar_strs = remiz_str.split('b-1')
        bar_strs.pop()
        for bar_id, bar_str in enumerate(bar_strs):
            bar_seq = bar_str.strip().split()
            
            inst_id = None
            time_sig = None
            tempo = None
            need_create_note = False
            notes_of_instruments = {}
            for tok in bar_seq:
                if tok.startswith('s-'):
                    time_sig = TimeSignatureUtil.convert_time_signature_token_to_tuple(tok)
                elif tok.startswith('t-'):
                    tempo_id = int(tok[2:])
                    tempo = convert_id_to_tempo(tempo_id)
                elif tok.startswith('i-'):
                    inst_id = int(tok[2:])
                elif tok.startswith('o-'):
                    pos = int(tok[2:])
                elif tok.startswith('p-'):
                    pitch = int(tok[2:])
                    if pitch >= 128:
                        pitch -= 128
                elif tok.startswith('d-'):
                    duration = int(tok[2:])
                    if not with_velocity:
                        velocity = 64
                        need_create_note = True
                elif tok.startswith('v-'):
                    velocity = int(tok[2:])
                    need_create_note = True

                if need_create_note:
                    if inst_id is None:
                        inst_id = 0
                    if inst_id not in notes_of_instruments:
                        notes_of_instruments[inst_id] = {}
                    if pos not in notes_of_instruments[inst_id]:
                        notes_of_instruments[inst_id][pos] = []
                    notes_of_instruments[inst_id][pos].append([pitch, duration, velocity])
                    need_create_note = False
            
            # Create a Bar instance
            bar_instance = Bar(
                id=bar_id, 
                notes_of_insts=notes_of_instruments,
                time_signature=time_sig,
                tempo=tempo,
            )
            bar_seqs.append(bar_instance)
                
        return cls(bars=bar_seqs)

    def to_remiz_seq(self, with_ts=False, with_tempo=False, with_velocity=False):
        '''
        Convert the MultiTrack object to a REMI-z sequence of tokens.
        '''
        ret = []
        for bar in self.bars:
            bar_seq = []

            bar_seq = bar.to_remiz_seq(with_ts=with_ts, with_tempo=with_tempo, with_velocity=with_velocity)
            ret.extend(bar_seq) 
        return ret
    
    def to_remiz_str(self, with_ts=False, with_tempo=False, with_velocity=False):
        '''
        Convert the MultiTrack object to a REMI-z string.
        '''
        ret = self.to_remiz_seq(with_ts=with_ts, with_tempo=with_tempo, with_velocity=with_velocity)
        return ' '.join(ret)

    def to_midi(self, midi_fp: str):
        """
        Create a MIDI file from the MultiTrack object using miditoolkit.
        """
        assert isinstance(midi_fp, str), "midi_fp must be a string"
        
        # 创建一个空的 MidiFile 对象
        # 默认 ticks_per_beat 是480，你可以根据需要修改
        midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=480)
        ticks_per_beat = midi_obj.ticks_per_beat

        # 如果有小节，则获取初始速度，否则用默认值120 BPM
        if len(self.bars) > 0:
            initial_tempo = self.bars[0].tempo
        else:
            initial_tempo = 120.0

        # 初始化时间线计数（以ticks为单位）
        cumulative_bar_ticks = 0

        # 插入初始速度与拍号（如果有小节）
        if len(self.bars) > 0:
            # 初始拍号
            numerator, denominator = self.bars[0].time_signature
            # TimeSignature 与 TempoChange 用 ticks 来定位事件位置
            midi_obj.time_signature_changes.append(
                miditoolkit.midi.containers.TimeSignature(
                    numerator=numerator,
                    denominator=denominator,
                    time=0  # 第一个小节拍号从0 tick开始
                )
            )
            
            # 初始速度
            midi_obj.tempo_changes.append(
                miditoolkit.midi.containers.TempoChange(
                    tempo=initial_tempo,
                    time=0  # 初始速度从0 tick开始生效
                )
            )

        # 乐器映射表：inst_id -> Instrument对象
        instrument_map = {}

        last_time_signature = self.bars[0].time_signature if len(self.bars) > 0 else None
        last_tempo = initial_tempo

        # 遍历每个小节
        for bar_index, bar in enumerate(self.bars):
            # 如果拍号变了，插入新的 TimeSignature 事件
            if bar.time_signature != last_time_signature:
                numerator, denominator = bar.time_signature
                midi_obj.time_signature_changes.append(
                    miditoolkit.midi.containers.TimeSignature(
                        numerator=numerator,
                        denominator=denominator,
                        time=cumulative_bar_ticks
                    )
                )
                last_time_signature = bar.time_signature

            # 如果速度变了，插入新的 TempoChange 事件
            if bar.tempo != last_tempo:
                midi_obj.tempo_changes.append(
                    miditoolkit.midi.containers.TempoChange(
                        tempo=bar.tempo,
                        time=cumulative_bar_ticks
                    )
                )
                last_tempo = bar.tempo

            # 计算本小节的长度（以拍为单位）
            # 拍数 = 分子 * (4 / 分母)
            # 与之前一样的计算方法
            beats_per_bar = bar.time_signature[0] * (4.0 / bar.time_signature[1])
            # 小节长度(以ticks为单位)
            bar_length_ticks = int(beats_per_bar * ticks_per_beat)

            # 为当前小节中的音符计算相对时间（转换为ticks）
            for track_id, track in bar.tracks.items():
                # 获取或创建Instrument
                if track_id not in instrument_map:
                    prog_id = track.inst_id
                    program = 0 if prog_id == 128 else prog_id
                    # 创建乐器（Instrument）
                    # miditoolkit不强制要求不同instrument_id映射到特定音色，你可以根据实际需要调整program值。
                    instrument = miditoolkit.midi.containers.Instrument(
                        program=program,
                        is_drum=(prog_id == 128),  # 若为打击乐
                        name=f"Instrument_{prog_id}" # Set Track name to Instrument_{inst_id} # Note CA v2 need {inst_id}
                        # name=f"{inst_id}" # Use with Composer's Assistant
                    )
                    instrument_map[track_id] = instrument
                    midi_obj.instruments.append(instrument)
                else:
                    instrument = instrument_map[track_id]

                for note in track.notes:
                    # onset和duration是以某种beats为单位（如之前为12分音符换算）
                    onset_time_beats = note.onset / 12.0
                    duration_beats = note.duration / 12.0

                    # 转换为ticks
                    note_start = cumulative_bar_ticks + int(onset_time_beats * ticks_per_beat)
                    note_end = note_start + int(duration_beats * ticks_per_beat)

                    midi_note = miditoolkit.midi.containers.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note_start,
                        end=note_end
                    )
                    instrument.notes.append(midi_note)

            # 更新累计的ticks数，以便下一个小节从正确的时间点开始
            cumulative_bar_ticks += bar_length_ticks

        # 写入MIDI文件
        midi_obj.dump(midi_fp)
        print(f"MIDI file successfully written to {midi_fp}")


    def to_midi_prettymidi(self, midi_fp: str):
        """
        Create a MIDI file from the MultiTrack object.

        Deprecated: Use to_midi() instead. Because this version cannot handle tempo changes.
        """
        assert isinstance(midi_fp, str), "midi_fp must be a string"
        
        import numpy as np
        import pretty_midi

        # Initialize instrument map
        instrument_map = {}

        # Track the cumulative time for each bar
        cumulative_bar_time = 0.0
        last_time_signature = None

        # Initialize arrays for tempo changes
        tempo_change_times = []
        tempi = []

        # Get initial tempo from the first bar, if available
        if len(self.bars) > 0:
            initial_tempo = self.bars[0].tempo
        else:
            # If no bars, fallback to a default tempo (e.g. 120 bpm)
            initial_tempo = 120.0

        # Set the initial tempo event at time zero
        tempo_change_times.append(0.0)
        tempi.append(60_000_000 / initial_tempo)
        last_tempo = initial_tempo

        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(
            initial_tempo=initial_tempo
        )

        for bar in self.bars:
            # Handle time signature changes
            if bar.time_signature != last_time_signature:
                numerator, denominator = bar.time_signature
                midi.time_signature_changes.append(
                    pretty_midi.TimeSignature(numerator, denominator, cumulative_bar_time)
                )
                last_time_signature = bar.time_signature

            # Handle tempo changes
            if bar.tempo != last_tempo:
                tempo_in_microseconds = 60_000_000 / bar.tempo
                tempo_change_times.append(cumulative_bar_time)
                tempi.append(tempo_in_microseconds)
                last_tempo = bar.tempo

            # Iterate over tracks in the current bar
            for inst_id, track in bar.tracks.items():
                # Ensure instrument_map is used correctly
                if inst_id not in instrument_map:
                    program = 0 if inst_id == 128 else inst_id
                    instrument = pretty_midi.Instrument(program=program, is_drum=(inst_id == 128))
                    instrument_map[inst_id] = instrument
                    midi.instruments.append(instrument)
                else:
                    instrument = instrument_map[inst_id]

                for note in track.notes:
                    onset_time_beats = note.onset / 12.0
                    onset_time_seconds = cumulative_bar_time + (onset_time_beats * (60.0 / bar.tempo))
                    duration_beats = note.duration / 12.0
                    duration_seconds = duration_beats * (60.0 / bar.tempo)

                    midi_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=onset_time_seconds,
                        end=onset_time_seconds + duration_seconds
                    )
                    instrument.notes.append(midi_note)

            # Calculate beats per bar using the current bar's time signature
            beats_per_bar = bar.time_signature[0] * (4 / bar.time_signature[1])  # Numerator * (4 / Denominator)
            bar_duration = beats_per_bar * (60.0 / bar.tempo)
            cumulative_bar_time += bar_duration

        # Set tempo changes in PrettyMIDI
        midi._tempo_change_times = np.array(tempo_change_times)
        midi._tempi = np.array(tempi)

        # Write MIDI file
        midi.write(midi_fp)
        print(f"MIDI file successfully written to {midi_fp}")

    def convert_time_signature_to_ts_token(self, numerator, denominator):
        ts_dict = self.ts_dict
        valid = False
        for k, v in ts_dict.items():
            if v == '({}, {})'.format(numerator, denominator):
                valid = True
                return k
        if not valid:
            raise ValueError('Invalid time signature: {}/{}'.format(numerator, denominator))

    def get_note_list(self, with_drum=False) -> List[Tuple[int, int, int, int]]:
        '''
        Get all notes in the MultiTrack.

        Returns:
            List of tuples (onset, pitch, duration, velocity)
        '''
        all_notes = []
        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                if not with_drum and track.is_drum:
                    continue
                all_notes.extend(track.get_note_list())
        return all_notes

    def flatten(self) -> "MultiTrack":
        '''
        Flatten the content of MultiTrack object to a single track, but still save to a MultiTrack object.
        Keep all info the same, such as bars, time signature, tempo, etc.
        Remove drum tracks, if any.
        
        This will merge all tracks into a single track.
        '''
        assert len(self.bars) > 0, "MultiTrack must have at least one bar"
        
        new_bars = []
        for bar in self.bars:
            t = bar.flatten()
            new_bars.append(t)
            # for inst_id, track in bar.tracks.items():
            #     if inst_id not in merged_bar.tracks:
            #         merged_bar.tracks[inst_id] = track
            #     else:
            #         merged_bar.tracks[inst_id].notes.extend(track.notes)
        
        return MultiTrack(bars=new_bars)

    def get_all_notes(self, include_drum=True, of_insts:List[int]=None) -> List[Note]:
        '''
        Get all notes in the MultiTrack.
        '''
        all_notes = []
        for bar in self.bars:
            all_notes.extend(bar.get_all_notes(include_drum=include_drum, of_insts=of_insts))
        return all_notes
    
    def get_all_notes_by_bar(self, include_drum=True, of_insts:List[int]=None) -> List[List[Note]]:
        '''
        Get all notes in the MultiTrack.
        '''
        all_notes = []
        for bar in self.bars:
            all_notes.append(bar.get_all_notes(include_drum=include_drum, of_insts=of_insts))
        return all_notes
    
    def get_content_seq(self, include_drum=False, of_insts=None, with_dur=True, return_str=False):
        '''
        Convert the MultiTrack object to a content sequence.
        Including information about all notes being played
        Without instrument information.
        '''
        content_seq = []
        for bar in self.bars:
            content_seq.extend(bar.get_content_seq(
                include_drum=include_drum, 
                of_insts=of_insts, 
                with_dur=with_dur
            ))

        if return_str:
            return ' '.join(content_seq)
        else:
            return content_seq
        
    def get_pitch_range(self, return_range=False):
        '''
        Calculate the range of the notes in the Bar.
        Will return max_pitch - min_pitch + 1
        If no notes found, return -1.
        '''
        all_insts = self.get_unique_insts()
        
        notes = self.get_all_notes(include_drum=False)
        if len(notes) == 0:
            return -1
        
        min_pitch = 128
        max_pitch = -1
        for note in notes:
            min_pitch = min(min_pitch, note.pitch)
            max_pitch = max(max_pitch, note.pitch)
        pitch_range = max_pitch - min_pitch
        pitch_range = int(pitch_range)

        if return_range:
            return min_pitch, max_pitch
    
        return pitch_range + 1
    
    def get_melody_of_song(self, mel_def:str) -> List[List[Note]]:
        '''
        Get melody notes for the entire MultiTrack object.
        NOTE: This algorithm calculate melody for each bar independently.

        hi_track: The track with the highest average pitch.
        
        '''
        assert mel_def in ['hi_track'], "mel_def must be 'hi_track'"
        
        # Collate the average pitch of each track from all bars
        track_avg_pitches = {}
        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                if track.is_drum:
                    continue
                avg_pitch = track.get_avg_pitch()
                if inst_id not in track_avg_pitches:
                    track_avg_pitches[inst_id] = []
                track_avg_pitches[inst_id].append(avg_pitch)
        # Calculate the average pitch for each track
        for inst_id in track_avg_pitches:
            avg_pitch = sum(track_avg_pitches[inst_id]) / len(track_avg_pitches[inst_id])
            track_avg_pitches[inst_id] = avg_pitch
        # Sort the tracks by average pitch
        sorted_tracks = sorted(track_avg_pitches.items(), key=lambda x: x[1], reverse=True)
        # Get the track with the highest average pitch
        highest_track_id = sorted_tracks[0][0]
        
        # Get the melody notes from the highest track, from each bar
        mel_notes = []
        for bar in self.bars:
            if highest_track_id in bar.tracks:
                mel_track = bar.tracks[highest_track_id]
                mel_notes.append(mel_track.get_all_notes())
            else:
                mel_notes.append([])

        return mel_notes

    def get_melody(self, mel_def):
        '''
        NOTE: This algorithm calculate melody for each bar independently.
        '''
        mel_notes = []
        for bar in self.bars:
            mel_notes.append(bar.get_melody(mel_def))
        return mel_notes
        
    def insert_empty_bars_at_front(self, num_bars):
        '''
        Insert empty bars at the front of the MultiTrack object.
        '''
        assert isinstance(num_bars, int), "num_bars must be an integer"
        assert num_bars >= 0, "num_bars must be a non-negative integer"

        ts = self.bars[0].time_signature
        tempo = self.bars[0].tempo

        empty_bars = []
        for i in range(1, num_bars+1):
            empty_bars.insert(0, Bar(id=-i, notes_of_insts={}, time_signature=ts, tempo=tempo))
        self.bars = empty_bars + self.bars
    
    def merge_with(self, other:"MultiTrack", other_prog_id) -> "MultiTrack":
        '''
        Merge two MultiTrack objects.
        Both MultiTrack objects must have the same number of bars, time signature, and tempo.
        '''
        assert isinstance(other, MultiTrack), "other must be a MultiTrack object"
        assert len(self.bars) == len(other.bars), "Both MultiTrack objects must have the same number of bars"

        new_bars = []
        for bar1, bar2 in zip(self.bars, other.bars):
            # Merge the two bars
            merged_bar = Bar(id=bar1.bar_id, notes_of_insts={}, time_signature=bar1.time_signature, tempo=bar1.tempo)
            for inst_id, track in bar1.tracks.items():
                merged_bar.tracks[inst_id] = track
            for inst_id, track in bar2.tracks.items():
                track.set_inst_id(other_prog_id)
                if other_prog_id not in merged_bar.tracks:
                    merged_bar.tracks[other_prog_id] = track
                else:
                    merged_bar.tracks[other_prog_id].notes.extend(track.notes)
            new_bars.append(merged_bar)
        
        return MultiTrack(bars=new_bars)

def deduplicate_notes(notes:List[Note]) -> List[Note]:
    '''
    Remove repeated notes with same onset and pitch.
    NOTE: Ensure the notes are sorted before calling this function.

    Args:
        notes: List of Note objects

    Returns:
        List of Note objects with repeated notes removed
        If note has same onset and pitch, only the first note (with longest duration) will be kept.
        If note has same onset, pitch, and duration, only the first note (with highest velocity) will be kept.
    '''
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


def midi_pitch_to_note_name(pitch: int) -> str:
    """
    Convert a MIDI pitch number (0–127) to a note name string like 'C4'.
    """
    if not (0 <= pitch <= 127):
        raise ValueError("MIDI pitch must be between 0 and 127")
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    note = note_names[pitch % 12]
    octave = (pitch // 12) - 1  # MIDI note 0 is C-1
    return f"{note}{octave}"

def note_name_to_midi_pitch(note_name: str) -> int:
    """
    Convert a note name string like 'C4' or 'F#3' to a MIDI pitch number (0–127).
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    import re
    match = re.match(r'^([A-G]#?)(-?\d+)$', note_name)
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