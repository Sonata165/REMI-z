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
        assert 0 <= onset <= 127, "onset must be in the range of [0, 127]"
        assert 0 <= pitch <= 255, "pitch must be in the range of [0, 255]"
        assert 0 <= velocity <= 127, "velocity must be in the range of [0, 127]"
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
        if self.is_drum:
            self.avg_pitch = -1
        else:
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
    
    def get_all_notes(self) -> List[Note]:
        '''
        Get all notes in the Track.
        '''
        return self.notes


class Bar:
    def __init__(self, id, notes_of_insts:Dict[int, Dict[int, List]], time_signature=None, tempo=None):
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
        track_list = []
        self.tracks: Dict[int, Track] = {}
        for inst_id, notes in notes_of_insts.items():
            track = Track(inst_id, notes)
            track_list.append(track)
        track_list.sort()
        for track in track_list:
            inst_id = track.inst_id
            self.tracks[inst_id] = track

        self.time_signature = time_signature
        self.tempo = tempo

    def __len__(self):
        return len(self.tracks)
    
    def __str__(self) -> str:
        return f'Bar {self.bar_id}: {len(self.tracks)} insts'

    def __repr__(self) -> str:
        return self.__str__()
    
    @classmethod
    def from_piano_roll(cls, piano_roll, time_signature=(4, 4), tempo=120.0):
        '''
        Convert the Bar object to a piano roll matrix.
        '''
        notes = {}
        for p_pos in range(piano_roll.shape[0]):
            for pitch in range(piano_roll.shape[1]):
                if piano_roll[p_pos, pitch] > 0:
                    p_dur = piano_roll[p_pos, pitch]
                    dur = (p_dur * 3).item()
                    onset = p_pos * 3
                    # note = Note(onset=onset, duration=dur, pitch=pitch)
                    note = [pitch, dur, 64] # pitch, duration, velocity

                    # Add note to notes
                    if onset not in notes:
                        notes[onset] = []
                    notes[onset].append(note)

        return cls(id=-1, notes_of_insts={0:notes}, time_signature=time_signature, tempo=tempo)

    def to_remiz_seq(self, with_ts=False, with_tempo=False, with_velocity=False):
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
    
    def to_piano_roll(self, of_inst=None):
        '''
        Convert the Bar object to a piano roll matrix.
        '''
        pos_per_beat = 4
        beats_per_bar = self.time_signature[0]
        pos_per_bar = pos_per_beat * beats_per_bar

        # Create a piano roll matrix
        piano_roll = np.zeros((pos_per_bar, 128), dtype=int)

        # Obtain notes to be added to the piano roll
        if of_inst:
            assert isinstance(of_inst, int), "of_inst must be an integer"
            assert of_inst in self.tracks, "of_inst not found in the bar"
            notes = self.tracks[of_inst].get_all_notes()
        else:
            notes = self.get_all_notes(include_drum=False)

        # Add notes to the piano roll
        # NOTE: the pos in piano roll is 1/3 of note.onset
        for note in notes:
            onset_pos = min(round(note.onset / 3), pos_per_bar - 1)
            dur = round(note.duration / 3)
            pitch = note.pitch
            piano_roll[onset_pos, pitch] = dur

        return piano_roll

    def to_midi(self, midi_fp: str):
        mt = MultiTrack.from_bars([self])
        mt.to_midi(midi_fp)

    def get_all_notes(self, include_drum) -> List[Note]:
        '''
        Get all notes in the Bar.
        '''
        all_notes = []
        for inst_id, track in self.tracks.items():
            if not include_drum and track.is_drum:
                continue
            all_notes.extend(track.get_all_notes())
        return all_notes
    
    def get_content_seq(self, include_drum=False, of_inst=None):
        '''
        Convert the Bar object to a content sequence.
        Including information about all notes being played
        Without instrument information.

        Args:
            include_drum: Whether to include drum tracks
            of_inst: The instrument ID to extract the content sequence. None means all instruments.
        '''
        if of_inst is not None:
            assert isinstance(of_inst, int), "of_inst must be an integer"
            assert of_inst in self.tracks, "of_inst not found in the bar"
            track = self.tracks[of_inst]
            notes = track.get_all_notes()
        else:
            notes = self.get_all_notes(include_drum=include_drum)
        notes.sort()

        # Remove repeated notes with same onset and pitch
        notes_dedup = []
        prev_onset = None
        prev_pitch = None
        for note in notes:
            if note.onset == prev_onset and note.pitch == prev_pitch:
                continue
            notes_dedup.append(note)
            prev_onset = note.onset
            prev_pitch = note.pitch
        notes = notes_dedup

        # Convert to content sequence (containing only o-X, p-X, d-X)
        bar_seq = []
        for note in notes:
            bar_seq.extend([
                f'o-{note.onset}',
                f'p-{note.pitch}',
                f'd-{note.duration}',
            ])

        bar_seq.append('b-1')

        return bar_seq

class MultiTrack:
    def __init__(self, bars:List[Bar]):
        '''
        Args:
            bars: List of Bar objects
            pitch_shift: The pitch shift value. None means not detected.
            is_major: The major/minor key information. None means not detected.
        '''
        self.bars = bars 

        # Load the time signature dictionary
        ts_fp = os.path.join(os.path.dirname(__file__), 'dict_time_signature.yaml')
        self.ts_dict = read_yaml(ts_fp)

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
        return self
    
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
    def from_midi(cls, midi_fp:str):
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
            end_offset=0
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
                bar_instance = Bar(
                    id=bar_id, 
                    notes_of_insts=notes_of_instruments,
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
        assert 'b-1' in remiz_str, "remiz_str must contain at least one bar"
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
            for inst_id, track in bar.tracks.items():
                # 获取或创建Instrument
                if inst_id not in instrument_map:
                    program = 0 if inst_id == 128 else inst_id
                    # 创建乐器（Instrument）
                    # miditoolkit不强制要求不同instrument_id映射到特定音色，你可以根据实际需要调整program值。
                    instrument = miditoolkit.midi.containers.Instrument(
                        program=program,
                        is_drum=(inst_id == 128),  # 若为打击乐
                        name=f"Instrument_{inst_id}"
                    )
                    instrument_map[inst_id] = instrument
                    midi_obj.instruments.append(instrument)
                else:
                    instrument = instrument_map[inst_id]

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

    def get_all_notes(self) -> List[Note]:
        '''
        Get all notes in the MultiTrack.
        '''
        all_notes = []
        for bar in self.bars:
            all_notes.extend(bar.get_all_notes())
        return all_notes
    
    def get_content_seq(self, include_drum=False, of_inst=None, return_str=False):
        '''
        Convert the MultiTrack object to a content sequence.
        Including information about all notes being played
        Without instrument information.
        '''
        content_seq = []
        for bar in self.bars:
            content_seq.extend(bar.get_content_seq(include_drum=include_drum, of_inst=of_inst))

        if return_str:
            return ' '.join(content_seq)
        else:
            return content_seq
    
