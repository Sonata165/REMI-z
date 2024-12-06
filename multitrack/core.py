import os
from typing import List, Dict
from . import midiprocessor as mp
from .midi_data_extractor.midi_processing import get_midi_pos_info, convert_pos_info_to_tokens
from .midi_data_extractor.data_extractor import get_bar_positions, get_bars_insts, ChordDetector
from .midi_data_extractor.utils.pos_process import fill_pos_ts_and_tempo_
import pretty_midi


class Note:
    def __init__(self, onset:int, duration:int, pitch:int, velocity:int, is_drum=False):
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
                128~255: Drum pitch
        velocity : int
            The velocity of the note.
        '''
        assert isinstance(onset, int), "onset must be an integer"
        assert isinstance(duration, int), "duration must be an integer"
        assert isinstance(pitch, int), "pitch must be an integer"
        assert isinstance(velocity, int), "velocity must be an integer"
        assert 0 <= onset <= 127, "onset must be in the range of [0, 127]"
        assert 0 <= pitch <= 255, "pitch must be in the range of [0, 255]"
        assert 0 <= velocity <= 127, "velocity must be in the range of [0, 127]"
        if is_drum:
            assert 128 <= pitch <= 255, "Drum pitch must be in the range of [128, 255]"
        else:
            assert 0 <= pitch <= 127, "MIDI pitch must be in the range of [0, 127]"

        # Round the values
        duration = min(max(1, duration), 127) # duration must be in the range of [1, 127]

        self.onset = onset
        self.duration = duration
        self.pitch = pitch
        self.velocity = velocity

    def __str__(self) -> str:
        return f'(o:{self.onset},d:{self.duration},p:{self.pitch},v:{self.velocity})'
    
    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other):
        if self.onset != other.onset:
            return self.onset < other.onset
        else:
            return self.pitch > other.pitch


class Track:
    '''
    This class save all notes for a same track within a bar.
    '''
    def __init__(self, inst_id, notes:Dict[int, List[Note]]):
        self.inst_id = inst_id
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
        pitches = [note.pitch for note in self.notes]
        self.avg_pitch = sum(pitches) / len(pitches)

    def __str__(self) -> str:
        return f'Inst {self.inst_id}: {len(self.notes)} notes'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __len__(self):
        return len(self.notes)
    
    def __lt__(self, other):
        '''
        Track with higher pitch will be placed at the front (more important)
        '''
        return self.avg_pitch > other.avg_pitch


class Bar:
    def __init__(self, id, notes_of_insts:List[Note]):
        '''
        NOTE: The instrument with higher average pitch will be placed at the front.
        '''
        self.bar_id = id
        track_list = []
        self.tracks = {}
        for inst_id, notes in notes_of_insts.items():
            track = Track(inst_id, notes)
            track_list.append(track)
        track_list.sort()
        for track in track_list:
            inst_id = track.inst_id
            self.tracks[inst_id] = track

    def __len__(self):
        return len(self.tracks)
    
    def __str__(self) -> str:
        return f'Bar {self.bar_id}: {len(self.tracks)} insts'

    def __repr__(self) -> str:
        return self.__str__()
    

class MultiTrack:
    def __init__(self, bars:List[Bar]):
        self.bars = bars 

    def __len__(self):
        return len(self.bars)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            bar_subset = [self.bars[idx]]
        elif isinstance(idx, slice):
            bar_subset = self.bars[idx]
        return MultiTrack(bars=bar_subset)
    
    def __str__(self) -> str:
        return f'MultiTrack: {len(self.bars)} bars'
    
    def __repr__(self) -> str:
        return self.__str__()
    
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
    def from_midi(cls, midi_fp:str):
        '''
        Create a MultiTrack object from a MIDI file.
        '''
        assert isinstance(midi_fp, str), "midi_fp must be a string"
        assert os.path.exists(midi_fp), "midi_fp does not exist"

        encoding_method = "REMIGEN2"
        midi_encoder = mp.MidiEncoder(encoding_method)

        midi_obj = mp.midi_utils.load_midi(midi_fp)
        pos_info = get_midi_pos_info(midi_encoder, midi_path=None, midi_obj=midi_obj, remove_empty_bars=False)
        # bar: every pos
        # ts: only at pos where it changes, otherwise None
        # in-bar position
        # tempo: only at pos where it changes, otherwise None
        # insts_notes: only at pos where the note starts, otherwise None

        # Longshen: don't remove empty bars to facilitate

        pos_info = fill_pos_ts_and_tempo_(pos_info)

        normalize_pitch = True

        # Obtain pitch shift and major/minor info
        # Deep copy pos_info
        pos_info_copy = pos_info.copy()
        _, is_major, pitch_shift = midi_encoder.normalize_pitch(pos_info) # Can make error some times. Not sure about direction of pitch shift yet.

        # Get bar token index, and instruments inside each bar
        bars_positions = get_bar_positions(pos_info)
        bars_instruments = get_bars_insts(pos_info, bars_positions)
        num_bars = len(bars_positions)
        assert num_bars == len(bars_instruments)

        # Generate bar sequences and note sequences
        bar_seqs = []
        seen_bars = set()
        bar_id_prev_pos = -1
        for i in range(len(pos_info)):
            bar_id, ts, pos, tempo, insts_notes = pos_info[i]
            
            # Determine if this is a new bar
            if bar_id > bar_id_prev_pos:
                notes_of_instruments = {}

            # Add the note info
            if insts_notes is not None:
                for inst_id, notes in insts_notes.items():
                    # each note contain [pitch, duration, velocity]
                    if inst_id not in notes_of_instruments:
                        notes_of_instruments[inst_id] = {}
                    if pos not in notes_of_instruments[inst_id]:
                        notes_of_instruments[inst_id][pos] = []
                    notes_of_instruments[inst_id][pos].extend(notes)

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
                bar_instance = Bar(id=bar_id, notes_of_insts=notes_of_instruments)
                bar_seqs.append(bar_instance)
                
            bar_id_prev_pos = bar_id

        return cls(bars=bar_seqs)
    
    def to_remiz_seq(self):
        '''
        Convert the MultiTrack object to a REMI-z sequence of tokens.
        '''
        ret = []
        for bar in self.bars:
            bar_seq = []
            for inst_id, track in bar.tracks.items():
                track_seq = [f'i-{inst_id}']
                prev_pos = -1
                for note in track.notes:
                    if note.onset > prev_pos:
                        track_seq.append(f'o-{note.onset}')
                        prev_pos = note.onset
                    track_seq.extend([
                        f'p-{note.pitch}',
                        f'd-{note.duration}',
                    ])
                bar_seq.extend(track_seq)
            bar_seq.append('b-1')
            ret.extend(bar_seq) 
        return ret
    
    def to_remiz_str(self):
        '''
        Convert the MultiTrack object to a REMI-z string.
        '''
        ret = self.to_remiz_seq()
        return ' '.join(ret)
    
    def to_midi(self, midi_fp: str):
        """
        Create a MIDI file from the MultiTrack object.
        """
        assert isinstance(midi_fp, str), "midi_fp must be a string"
        
        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI()

        # Iterate over each bar in the MultiTrack
        instrument_map = {}  # Map to keep track of instruments

        # Calculate seconds per beat from BPM
        seconds_per_beat = 60.0 / 120.0  # Assume 120 BPM

        # Track the cumulative time for each bar
        cumulative_bar_time = 0.0

        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                # Check if instrument already exists, otherwise create a new one
                if inst_id not in instrument_map:
                    # You can modify this to assign an appropriate program or instrument type
                    program = inst_id  # General MIDI Acoustic Grand Piano as default
                    instrument = pretty_midi.Instrument(program=program, name=f'Instrument {inst_id}')
                    instrument_map[inst_id] = instrument
                    midi.instruments.append(instrument)
                else:
                    instrument = instrument_map[inst_id]
                
                # Iterate over all notes in the current track
                for note in track.notes:
                    # Convert onset and duration from '1/48 note' (1/12 beat) to beats, then to seconds
                    onset_time_beats = note.onset / 12.0  # Convert to beats
                    onset_time_seconds = cumulative_bar_time + (onset_time_beats * seconds_per_beat)

                    duration_beats = note.duration / 12.0  # Convert to beats
                    duration_seconds = duration_beats * seconds_per_beat

                    # Create PrettyMIDI Note object
                    midi_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=onset_time_seconds,
                        end=onset_time_seconds + duration_seconds
                    )

                    # Add the note to the corresponding instrument
                    instrument.notes.append(midi_note)

            # Update cumulative time for the next bar
            # Assuming each bar is 4 beats (a full measure in 4/4 time)
            cumulative_bar_time += 4 * seconds_per_beat

        # Write the PrettyMIDI object to the specified MIDI file path
        midi.write(midi_fp)

        print(f"MIDI file successfully written to {midi_fp}")