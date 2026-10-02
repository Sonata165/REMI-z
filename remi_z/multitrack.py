import bisect
import os

import miditoolkit
import pretty_midi
import numpy as np
from typing import List, Tuple

from .note import Note, NoteSeq
from .bar import Bar, deduplicate_notes
from .track import Track
from .utils import read_yaml
from .midi_encoding import (
    MidiEncoder,
    load_midi,
    fill_pos_ts_and_tempo_,
    adjust_pos_info_offset_overlap_,
    convert_tempo_to_id,
    convert_id_to_tempo,
)
from .time_signature_utils import TimeSignatureUtil
from .keys_normalization import detect_key


class MultiTrack:
    """
    A complete multi-instrument music piece represented as a sequence of bars.

    This is the top-level container in the REMI-z hierarchy:
    ``MultiTrack → Bar → Track → Note``.

    Attributes
    ----------
    bars : List[Bar]
        Ordered list of Bar objects that make up the piece.
    time_signatures : List[Tuple[int, int]]
        Deduplicated set of time signatures present across all bars.
    tempos : List[float]
        Deduplicated set of tempos (BPM) present across all bars.
    ts_dict : dict
        Dictionary mapping time-signature tokens to ``(numerator, denominator)``
        tuples, loaded from ``dict_time_signature.yaml``.
    """

    def __init__(self, bars: List[Bar]):
        """
        Parameters
        ----------
        bars : List[Bar]
            Ordered list of Bar objects that make up the piece.
        """
        # Parameters check
        assert isinstance(bars, list), "bars must be a list"

        self.bars = bars

        self.update_ts_and_tempo()

        # Load the time signature dictionary
        ts_fp = os.path.join(os.path.dirname(__file__), "dict_time_signature.yaml")
        self.ts_dict = read_yaml(ts_fp)

        self.fn = '<unknown>'

    def update_ts_and_tempo(self):
        """
        Recompute ``self.time_signatures`` and ``self.tempos`` from the current bars.

        Call this after any in-place modification of bar tempo or time-signature
        values to keep the summary attributes in sync.
        """
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
        return f"MultiTrack: {len(self.bars)} bars"

    def __repr__(self) -> str:
        return self.__str__()

    def set_tempo(self, tempo: float):
        """
        Set the tempo for all bars in the MultiTrack object.
        """
        assert isinstance(tempo, (int, float)), "tempo must be an integer or float"
        tempo = round(tempo, 2)
        for bar in self.bars:
            bar.tempo = tempo

        # Update the tempo in the tempo set
        self.update_ts_and_tempo()

    def set_velocity(self, velocity: int, track_id: int = None):
        """
        Set the velocity for all notes in the MultiTrack object.
        """
        assert isinstance(velocity, int), "velocity must be an integer"
        assert 0 <= velocity <= 127, "velocity must be in the range of [0, 127]"

        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                if track_id is not None and inst_id != track_id:
                    continue
                else:
                    for note in track.notes:
                        note.velocity = velocity

    def set_instrument(self, track_name:str, new_inst_id: int):
        '''
        When in the multi-instance for same program ID setting,
        Assign new instrument id according to the track name. 
        '''
        for bar in self.bars:
            bar.assign_instrument(track_name, new_inst_id)

    def key_norm(self):
        """
        Normalize the pitch of all notes in the MultiTrack object.
        """
        pitch_shift = self.normalize_pitch()
        return pitch_shift

    def normalize_pitch(self):
        """
        Normalize the pitch of all notes in the MultiTrack object.
        """

        """ Detect major/minor key and pitch shift needed for the key normalization """
        is_major, pitch_shift = self.detect_key()

        """ Apply the pitch shift to the notes """
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

        return pitch_shift

    def shift_pitch(self, pitch_shift: int, track_id: int = None):
        """
        Transpose all non-drum notes by a fixed number of semitones.

        Pitches that drop below 0 are wrapped up by one octave (+12).
        Drum tracks are always skipped.

        Parameters
        ----------
        pitch_shift : int
            Semitones to add (positive = up, negative = down).
        track_id : int, optional
            If given, only the track with this instrument ID is transposed.
            ``None`` (default) transposes all non-drum tracks.
        """
        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                if track.is_drum:
                    continue
                if track_id is not None and inst_id != track_id:
                    continue
                else:
                    for note in track.notes:
                        note.pitch += pitch_shift

                        # If the pitch is out of range, adjust it
                        if note.pitch < 0:
                            note.pitch += 12

    def quantize_to_16th(self):
        """
        Quantize the MultiTrack object to 16th notes.
        On both onset and duration.
        """
        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                for note in track.notes:
                    note.onset = round(note.onset / 3) * 3
                    note.duration = max(1, round(note.duration / 3)) * 3

                    if note.duration > 20:
                        a = 3

    def detect_key(self):
        """
        Determine the major/minor key and pitch shift needed for key normalization.

        Returns:
        - is_major: True if major, False if minor
        - pitch_shift: The pitch shift needed for key normalization
        """
        note_list = self.get_note_list(with_drum=False)
        is_major, pitch_shift = detect_key(note_list)
        # detect_key computes these with numpy; cast so key_norm() leaves Note.pitch a plain int
        return bool(is_major), int(pitch_shift)

    def get_unique_insts(self):
        """
        Get all unique instruments in the MultiTrack object.
        """
        all_insts = set()
        for bar in self.bars:
            for inst_id in bar.tracks.keys():
                all_insts.add(inst_id)
        return all_insts

    def get_inst_layout(self):
        """
        Get the instrumentation layout of the MultiTrack object.
        A list of list of tokens:
        [[i-0, i-24, i-32, b-1], [i-0, i-24, i-40, b-1], ...]
        Representing instrument and voice of each bar.
        """
        inst_conf = []
        for bar in self.bars:
            bar_inst_conf = []

            # Get instruments in this bar
            inst_of_the_bar = [f"i-{i}" for i in bar.get_unique_insts()]
            bar_inst_conf.extend(inst_of_the_bar)
            bar_inst_conf.append("b-1")
            inst_conf.append(bar_inst_conf)

        return inst_conf

    @classmethod
    def concat(cls, mts: List["MultiTrack"]):
        """
        Concatenate multiple MultiTrack objects into a single MultiTrack object.
        """
        assert isinstance(mts, list), "mts must be a list"
        assert len(mts) > 0, "mts must not be empty"
        for mt in mts:
            assert isinstance(
                mt, MultiTrack
            ), "mts must be a list of MultiTrack objects"

        all_bars = []
        for mt in mts:
            all_bars.extend(mt.bars)

        return cls(bars=all_bars)

    @classmethod
    def from_bars(cls, bars: List[Bar]):
        """
        Create a MultiTrack object from a list of Bar objects.
        """
        assert isinstance(bars, list), "bars must be a list"
        return cls(bars=bars)

    @classmethod
    def from_note_seqs(cls, note_seq: List[NoteSeq], program_id: int = 0):
        """
        Create a MultiTrack object from a list of NoteSeq objects.
        Each NoteSeq will be converted to a Bar with a single instrument.
        """
        assert isinstance(note_seq, list), "note_seq must be a list"
        assert len(note_seq) > 0, "note_seq must not be empty"
        assert isinstance(program_id, int), "program_id must be an integer"
        assert 0 <= program_id <= 127, "program_id must be in the range of [0, 127]"

        # Create a list of Bar objects
        bars = []
        for bar_id, seq in enumerate(note_seq):
            assert isinstance(
                seq, NoteSeq
            ), "note_seq must be a list of NoteSeq objects"
            notes_of_insts = {
                program_id: {
                    note.onset: [[note.pitch, note.duration, note.velocity]]
                    for note in seq.notes
                }
            }
            bar = Bar(id=bar_id, notes_of_insts=notes_of_insts)
            bars.append(bar)

        return cls(bars=bars)

    @classmethod
    def from_midi(
        cls,
        midi_fp: str,
        support_same_program_multi_instance=False,
        adjust_offset_overlap: bool = True,
    ):
        """
        Create a MultiTrack object from a MIDI file.

        MIDI tracks with the same program are merged into one track, unless
        support_same_program_multi_instance (then one track per (program, track name)).

        Args:
            adjust_offset_overlap: If True (default), within each track, same-pitch notes
                sharing an onset are deduplicated (longest kept) and offsets are shortened so
                that no note overlaps a later note of the same pitch, also across bar lines
                (see midi_encoding.adjust_pos_info_offset_overlap_). Merging tracks and
                quantizing produce such overlaps, which MIDI cannot represent faithfully.
                Drum tracks are left unchanged.
        """
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
            multi_instance_same_program_support=support_same_program_multi_instance,  # support multiple instances of the same program ID
        )
        # bar: every pos
        # ts: only at pos where it changes, otherwise None
        # in-bar position
        # tempo: only at pos where it changes, otherwise None
        # insts_notes: only at pos where the note starts, otherwise None

        if adjust_offset_overlap:
            pos_info = adjust_pos_info_offset_overlap_(pos_info)

        # Fill time signature and tempo info to the first pos of each bar
        pos_info = fill_pos_ts_and_tempo_(pos_info)

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
                next_bar_id, _, next_pos, _, _ = pos_info[i + 1]
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

            # Save filename
            fn = os.path.basename(midi_fp)

        ret = cls(bars=bar_seqs)
        ret.fn = fn

        return ret

    @classmethod
    def from_remiz_seq(cls, remiz_seq: List[str]):
        """
        Create a MultiTrack from a REMI-z token list.

        Parameters
        ----------
        remiz_seq : List[str]
            Tokenized REMI-z sequence.  Bar boundaries are marked by ``'b-1'``
            tokens.

        Returns
        -------
        MultiTrack
        """
        assert isinstance(remiz_seq, list), "remiz_seq must be a list"

        remiz_str = " ".join(remiz_seq)
        return cls.from_remiz_str(remiz_str)

    @classmethod
    def from_remiz_str(
        cls, remiz_str: str, verbose: bool = True, remove_repeated_eob: bool = False
    ):
        """
        Create a MultiTrack object from a remiz string.
        """
        assert isinstance(remiz_str, str), "remiz_str must be a string"
        if "b-1" not in remiz_str:
            if verbose:
                print(
                    f'WARNING: remiz_str "{remiz_str}" does not contain any bar information. Adding a end-of-bar at the end.'
                )
            remiz_str += " b-1"
        if "v" in remiz_str:
            with_velocity = True
        else:
            with_velocity = False

        # Remove repeated 'b-1' token
        if remove_repeated_eob:
            while "b-1 b-1" in remiz_str:
                remiz_str = remiz_str.replace("b-1 b-1", "b-1")

        bar_seqs = []

        # Split to bars
        bar_strs = remiz_str.split("b-1")
        bar_strs.pop()
        for bar_id, bar_str in enumerate(bar_strs):
            bar_seq = bar_str.strip().split()

            inst_id = None
            time_sig = None
            tempo = None
            need_create_note = False
            notes_of_instruments = {}
            pitch = None
            duration = None
            velocity = None
            pos = None
            for tok in bar_seq:
                if tok.startswith("s-"):
                    time_sig = TimeSignatureUtil.convert_time_signature_token_to_tuple(
                        tok
                    )
                elif tok.startswith("t-"):
                    tempo_id = int(tok[2:])
                    tempo = convert_id_to_tempo(tempo_id)
                elif tok.startswith("i-"):
                    inst_id = int(tok[2:])
                elif tok.startswith("o-"):
                    pos = int(tok[2:])
                elif tok.startswith("p-"):
                    pitch = int(tok[2:])
                    if pitch >= 128:
                        pitch -= 128
                elif tok.startswith("d-"):
                    duration = int(tok[2:])
                    if not with_velocity:
                        velocity = 96  # 64
                        need_create_note = True
                elif tok.startswith("v-"):
                    velocity = int(tok[2:])
                    need_create_note = True

                if need_create_note:
                    # Create the note for the current instrument and position,
                    # If pitch, duration, velocity are all not None
                    if (
                        pos is None
                        or pitch is None
                        or duration is None
                        or velocity is None
                    ):
                        pass
                    else:
                        if inst_id is None:
                            inst_id = 0
                        if inst_id not in notes_of_instruments:
                            notes_of_instruments[inst_id] = {}
                        if pos not in notes_of_instruments[inst_id]:
                            notes_of_instruments[inst_id][pos] = []
                        notes_of_instruments[inst_id][pos].append(
                            [pitch, duration, velocity]
                        )

                    need_create_note = False

                    pitch = None
                    duration = None
                    velocity = None

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
        """
        Convert the MultiTrack object to a REMI-z sequence of tokens.
        """
        ret = []
        for bar in self.bars:
            bar_seq = []

            bar_seq = bar.to_remiz_seq(
                with_ts=with_ts, with_tempo=with_tempo, with_velocity=with_velocity
            )
            ret.extend(bar_seq)
        return ret

    def to_remiz_str(self, with_ts=False, with_tempo=False, with_velocity=False):
        """
        Convert the MultiTrack object to a REMI-z string.
        """
        ret = self.to_remiz_seq(
            with_ts=with_ts, with_tempo=with_tempo, with_velocity=with_velocity
        )
        return " ".join(ret)

    def to_midi(self, midi_fp: str, tempo: float = None, verbose: bool = True):
        """
        Create a MIDI file from the MultiTrack object using miditoolkit.

        Args:
            midi_fp: The file path to save the MIDI file.
            tempo: If specified, override all tempos in the MultiTrack with this value.
        """
        assert isinstance(midi_fp, str), "midi_fp must be a string"

        if tempo is not None:
            assert isinstance(tempo, (int, float)), "tempo must be an integer or float"
            tempo = round(tempo, 2)
            self.set_tempo(tempo)

        # 创建一个空的 MidiFile 对象
        # 默认 ticks_per_beat 是480，你可以根据需要修改
        midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=480)
        ticks_per_beat = midi_obj.ticks_per_beat

        # 如果有小节，则获取初始速度，否则用默认值120 BPM
        if len(self.bars) > 0:
            initial_tempo = self.bars[0].tempo
        else:
            initial_tempo = 120.0

        cumulative_bar_ticks = 0

        if len(self.bars) > 0:
            # 初始拍号
            numerator, denominator = self.bars[0].time_signature
            # TimeSignature 与 TempoChange 用 ticks 来定位事件位置
            midi_obj.time_signature_changes.append(
                miditoolkit.midi.containers.TimeSignature(
                    numerator=numerator,
                    denominator=denominator,
                    time=0,  # 第一个小节拍号从0 tick开始
                )
            )

            # 初始速度
            midi_obj.tempo_changes.append(
                miditoolkit.midi.containers.TempoChange(
                    tempo=initial_tempo, time=0  # 初始速度从0 tick开始生效
                )
            )

        # 乐器映射表：inst_id -> Instrument对象
        instrument_map = {}

        last_time_signature = (
            self.bars[0].time_signature if len(self.bars) > 0 else None
        )
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
                        time=cumulative_bar_ticks,
                    )
                )
                last_time_signature = bar.time_signature

            # 如果速度变了，插入新的 TempoChange 事件
            if bar.tempo != last_tempo:
                midi_obj.tempo_changes.append(
                    miditoolkit.midi.containers.TempoChange(
                        tempo=bar.tempo, time=cumulative_bar_ticks
                    )
                )
                last_tempo = bar.tempo

            # 计算本小节的长度（以拍为单位）
            # 拍数 = 分子 * (4 / 分母)
            # 与之前一样的计算方法
            beats_per_bar = bar.time_signature[0] * (4.0 / bar.time_signature[1])
            bar_length_ticks = int(beats_per_bar * ticks_per_beat)

            for track_id, track in bar.tracks.items():
                if track_id not in instrument_map:
                    prog_id = track.inst_id
                    program = 0 if prog_id == 128 else prog_id
                    # 创建乐器（Instrument）
                    # miditoolkit不强制要求不同instrument_id映射到特定音色，你可以根据实际需要调整program值。
                    instrument = miditoolkit.midi.containers.Instrument(
                        program=program,
                        is_drum=(prog_id == 128),  # 若为打击乐
                        name=f"Instrument_{prog_id}",  # Set Track name to Instrument_{inst_id} # Note CA v2 need {inst_id}
                        # name=f"{inst_id}" # Use with Composer's Assistant
                        # name='Pixel EP' # See if work with garageband  result: no
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
                    note_start = cumulative_bar_ticks + int(
                        onset_time_beats * ticks_per_beat
                    )
                    note_end = note_start + int(duration_beats * ticks_per_beat)

                    midi_note = miditoolkit.midi.containers.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note_start,
                        end=note_end,
                    )
                    instrument.notes.append(midi_note)

            # 更新累计的ticks数，以便下一个小节从正确的时间点开始
            cumulative_bar_ticks += bar_length_ticks

        # 写入MIDI文件
        midi_obj.dump(midi_fp)

        if verbose:
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

        last_time_signature = (
            self.bars[0].time_signature if len(self.bars) > 0 else None
        )

        tempo_change_times = []
        tempi = []

        if len(self.bars) > 0:
            initial_tempo = self.bars[0].tempo
        else:
            initial_tempo = 120.0

        # Set the initial tempo event at time zero
        tempo_change_times.append(0.0)
        tempi.append(60_000_000 / initial_tempo)
        last_tempo = initial_tempo

        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)

        for bar in self.bars:
            # Handle time signature changes
            if bar.time_signature != last_time_signature:
                numerator, denominator = bar.time_signature
                midi.time_signature_changes.append(
                    pretty_midi.TimeSignature(
                        numerator, denominator, cumulative_bar_time
                    )
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
                    instrument = pretty_midi.Instrument(
                        program=program, is_drum=(inst_id == 128)
                    )
                    instrument_map[inst_id] = instrument
                    midi.instruments.append(instrument)
                else:
                    instrument = instrument_map[inst_id]

                for note in track.notes:
                    onset_time_beats = note.onset / 12.0
                    onset_time_seconds = cumulative_bar_time + (
                        onset_time_beats * (60.0 / bar.tempo)
                    )
                    duration_beats = note.duration / 12.0
                    duration_seconds = duration_beats * (60.0 / bar.tempo)

                    midi_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=onset_time_seconds,
                        end=onset_time_seconds + duration_seconds,
                    )
                    instrument.notes.append(midi_note)

            # Calculate beats per bar using the current bar's time signature
            beats_per_bar = bar.time_signature[0] * (
                4 / bar.time_signature[1]
            )  # Numerator * (4 / Denominator)
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
            if v == "({}, {})".format(numerator, denominator):
                valid = True
                return k
        if not valid:
            raise ValueError(
                "Invalid time signature: {}/{}".format(numerator, denominator)
            )

    def get_note_list(self, with_drum=False) -> List[Tuple[int, int, int, int]]:
        """
        Get all notes in the MultiTrack.

        Returns:
            List of tuples (onset, pitch, duration, velocity)
        """
        all_notes = []
        for bar in self.bars:
            for inst_id, track in bar.tracks.items():
                if not with_drum and track.is_drum:
                    continue
                all_notes.extend(track.get_note_list())
        return all_notes

    def flatten(self, adjust_offset_overlap: bool = True) -> "MultiTrack":
        """
        Flatten the content of MultiTrack object to a single track, but still save to a MultiTrack object.
        Keep all info the same, such as bars, time signature, tempo, etc.
        Remove drum tracks, if any.

        This will merge all tracks into a single track. Notes sharing the same onset and
        pitch within a bar are deduplicated (longest duration kept), see :meth:`Bar.flatten`.

        Args:
            adjust_offset_overlap: If True (default), afterwards shorten offsets so that no
                note overlaps a later note of the same pitch, also across bar lines (see
                :meth:`adjust_offset_overlap`). Merging tracks routinely produces such
                overlaps, which MIDI cannot represent faithfully.
        """
        assert len(self.bars) > 0, "MultiTrack must have at least one bar"

        new_bars = []
        for bar in self.bars:
            t = bar.flatten()
            new_bars.append(t)

        ret = MultiTrack(bars=new_bars)
        if adjust_offset_overlap:
            ret = ret.adjust_offset_overlap()
        return ret

    def _bar_starts(self) -> List[int]:
        """Song-level position of each bar's start (REMI-z: 12 positions per quarter
        note, so a bar of time signature num/den is num * 48 // den positions long)."""
        bar_starts, pos = [], 0
        for bar in self.bars:
            bar_starts.append(pos)
            num, den = bar.time_signature
            pos += num * 48 // den
        return bar_starts

    def adjust_offset_overlap(self, include_drum: bool = False) -> "MultiTrack":
        """
        Return a copy of the MultiTrack with same-pitch offset overlaps removed.

        MIDI Note-On/Note-Off events carry only ``(channel, pitch)``, so two notes of the
        same pitch in one track whose intervals overlap (in particular one nested inside
        another) cannot be round-tripped through a MIDI file. The quantized counterpart of
        ``note_abs._adjust_offset_overlap``: per track and pitch, on song-level positions
        (so a note held into a later bar is handled too), each note's duration is capped
        at the distance to the onset of the next note of the same pitch. A note may end
        exactly where the next one starts.

        Only durations are shortened, never lengthened; onsets, pitches, velocities, bars,
        tracks, time signatures and tempos are kept. Notes of one track sharing a
        song-level onset and pitch are deduplicated first (longest duration, then highest
        velocity kept), so every duration stays >= 1 and no other note is dropped.
        ``self`` is not modified (all Note objects are new).

        Args:
            include_drum: If False (default), drum tracks are copied unchanged.
        """
        bar_starts = self._bar_starts()

        # (track key, pitch) -> [(song_onset, -duration, -velocity, bar_idx, note)]
        groups = {}
        for i, bar in enumerate(self.bars):
            for tid, track in bar.tracks.items():
                if track.is_drum and not include_drum:
                    continue
                for n in track.notes:
                    groups.setdefault((tid, n.pitch), []).append(
                        (bar_starts[i] + n.onset, -n.duration, -n.velocity, i, n)
                    )

        # bar_idx -> track key -> adjusted notes
        adjusted = {}
        for (tid, _pitch), group in groups.items():
            group.sort(key=lambda g: g[:3])  # by onset, then longest, then loudest
            kept = [g for k, g in enumerate(group) if k == 0 or g[0] != group[k - 1][0]]
            for k, (on, _d, _v, i, n) in enumerate(kept):
                dur = n.duration
                if k + 1 < len(kept):
                    dur = min(dur, kept[k + 1][0] - on)
                adjusted.setdefault(i, {}).setdefault(tid, []).append(
                    Note(onset=n.onset, duration=dur, pitch=n.pitch, velocity=n.velocity)
                )

        new_bars = []
        for i, bar in enumerate(self.bars):
            new_bar = Bar(
                id=bar.bar_id,
                notes_of_insts={},
                time_signature=bar.time_signature,
                tempo=bar.tempo,
            )
            for tid, track in bar.tracks.items():  # keeps the track order
                if track.is_drum and not include_drum:
                    notes = [
                        Note(onset=n.onset, duration=n.duration, pitch=n.pitch, velocity=n.velocity)
                        for n in track.notes
                    ]
                else:
                    notes = adjusted.get(i, {}).get(tid, [])
                new_track = Track.from_note_list(inst_id=track.inst_id, note_list=notes)
                new_track.track_id = track.track_id
                new_bar.tracks[tid] = new_track
            new_bars.append(new_bar)

        return MultiTrack(bars=new_bars)

    def get_all_notes(
        self, include_drum=True, of_insts: List[int] = None
    ) -> List[Note]:
        """
        Deprecated. Get all notes in the MultiTrack. 
        Please use get_all_notes_by_bar() instead, which will return a list of list of notes, with each inner list representing the notes in a bar. This will be more useful for most cases, as it will keep the temporal structure of the music.
        """
        all_notes = []
        for bar in self.bars:
            all_notes.extend(
                bar.get_all_notes(include_drum=include_drum, of_insts=of_insts)
            )
        return all_notes

    def get_all_notes_by_bar(
        self, include_drum=True, of_insts: List[int] = None
    ) -> List[List[Note]]:
        """
        Get all notes in the MultiTrack.
        """
        all_notes = []
        for bar in self.bars:
            all_notes.append(
                bar.get_all_notes(include_drum=include_drum, of_insts=of_insts)
            )
        return all_notes

    def get_content_seq(
        self, include_drum=False, of_insts=None, with_dur=True, return_str=False
    ):
        """
        Convert the MultiTrack object to a content sequence.
        Including information about all notes being played
        Without instrument information.
        """
        content_seq = []
        for bar in self.bars:
            content_seq.extend(
                bar.get_content_seq(
                    include_drum=include_drum, of_insts=of_insts, with_dur=with_dur
                )
            )

        if return_str:
            return " ".join(content_seq)
        else:
            return content_seq

    def get_pitch_range(self, return_range=False):
        """
        Calculate the range of the notes in the Bar.
        Will return max_pitch - min_pitch + 1
        If no notes found, return -1.
        """
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

    def get_melody_of_song(
        self, mel_def: str = "hi_note_dur_plus", inst_id: int = 0, trim_overlap: bool = False
    ) -> "MultiTrack":
        """
        Extract the melody of the whole song as a new MultiTrack in which each
        bar holds a single Track of melody notes.

        Unlike ``get_melody``, this works on one timeline for the whole song, not
        bar by bar: a note held across a bar line still counts as sounding in the
        next bar.

        Parameters
        ----------
        mel_def : str
            Melody policy:

            - ``'hi_track'`` — all notes of the track with the highest average
              pitch over the song.
            - ``'hi_note_dur_plus'`` (default) — the highest non-drum note at each
              onset; a note that starts while melody notes are still sounding is
              kept unless it is lower than one of them (a higher line entering
              over a held note, or a same-pitch re-attack, is melody; a lower
              note is accompaniment). Every still-sounding melody note is
              compared, including ones held from earlier bars. Onsets and
              durations are compared exactly: quantized MIDI sits on a grid, so
              no timing tolerance is applied (see ``NoteStream.get_melody`` for
              performance MIDI).
        inst_id : int, optional
            GM program number of the melody Track in every bar (default ``0``).
        trim_overlap : bool, optional
            If True, a melody note still sounding at the next (later) melody
            onset is cut to end there, across bar lines too, so no two melody
            notes sound at once (notes sharing an onset, e.g. a chord in
            ``'hi_track'``, are left as they are). Default False.

        Returns
        -------
        MultiTrack
            Same number of bars as ``self``, each with the source bar's id, time
            signature and tempo. Each melody note stays in the bar where it
            starts. Bars with no melody notes have no tracks. The notes are new
            Note objects, so editing the result leaves ``self`` untouched.
        """
        assert mel_def in ["hi_track", "hi_note_dur_plus"], "mel_def must be 'hi_track' or 'hi_note_dur_plus'"
        assert isinstance(inst_id, int), "inst_id must be an integer"
        assert 0 <= inst_id <= 127, "inst_id must be in the range of [0, 127]"

        bar_starts = self._bar_starts()

        # Candidate notes as (song_onset, bar_idx, note)
        if mel_def == "hi_track":
            # the track with the highest average pitch over all bars it appears in
            track_avg_pitches = {}
            for bar in self.bars:
                for tid, track in bar.tracks.items():
                    if not track.is_drum:
                        track_avg_pitches.setdefault(tid, []).append(track.get_avg_pitch())
            if not track_avg_pitches:
                return self._melody_to_multitrack([[] for _ in self.bars], inst_id)
            highest_track_id = max(track_avg_pitches, key=lambda t: sum(track_avg_pitches[t]) / len(track_avg_pitches[t]))
            melody = [
                (bar_starts[i] + note.onset, i, note)
                for i, bar in enumerate(self.bars) if highest_track_id in bar.tracks
                for note in bar.tracks[highest_track_id].notes
            ]
        else:
            # highest note at each song-level onset
            tops = {}
            for i, bar in enumerate(self.bars):
                for track in bar.tracks.values():
                    if track.is_drum:
                        continue
                    for note in track.notes:
                        on = bar_starts[i] + note.onset
                        if on not in tops or note.pitch > tops[on][2].pitch:
                            tops[on] = (on, i, note)
            melody, sounding = [], []  # sounding: (song_end, pitch) of melody notes still ringing
            for on in sorted(tops):
                sounding = [(end, p) for end, p in sounding if end > on]
                cand = tops[on][2]
                if sounding and cand.pitch < max(p for _end, p in sounding):  # lower than a held melody note
                    continue
                melody.append(tops[on])
                sounding.append((on + cand.duration, cand.pitch))

        # (song_onset, bar_idx, onset_in_bar, pitch, duration, velocity), in time order
        melody = sorted(
            [(on, i, n.onset, n.pitch, n.duration, n.velocity) for on, i, n in melody],
            key=lambda m: (m[0], -m[3]),
        )
        if trim_overlap:
            onsets = sorted({m[0] for m in melody})
            for k, (on, i, onset, pitch, dur, vel) in enumerate(melody):
                j = bisect.bisect_right(onsets, on)  # next strictly later melody onset
                if j < len(onsets) and onsets[j] < on + dur:
                    melody[k] = (on, i, onset, pitch, onsets[j] - on, vel)

        notes_per_bar = [[] for _ in self.bars]
        for _on, i, onset, pitch, dur, vel in melody:
            notes_per_bar[i].append((onset, pitch, dur, vel))
        return self._melody_to_multitrack(notes_per_bar, inst_id)

    def _melody_to_multitrack(self, notes_per_bar, inst_id: int) -> "MultiTrack":
        """Builds a MultiTrack with self's bar ids / time signatures / tempos and,
        per bar, one ``inst_id`` Track from ``(onset, pitch, duration, velocity)``
        tuples (no track for a bar with no notes). Notes sharing an onset are all
        kept."""
        bars = []
        for bar, notes in zip(self.bars, notes_per_bar):
            notes_by_onset = {}
            for onset, pitch, duration, velocity in notes:
                notes_by_onset.setdefault(onset, []).append((pitch, duration, velocity))
            bars.append(
                Bar(
                    id=bar.bar_id,
                    notes_of_insts={inst_id: notes_by_onset} if notes_by_onset else {},
                    time_signature=bar.time_signature,
                    tempo=bar.tempo,
                )
            )
        return MultiTrack(bars=bars)

    def get_melody(self, mel_def: str, inst_id: int = 0) -> "MultiTrack":
        """
        Extract the melody of every bar as a new MultiTrack in which each bar
        holds a single Track of melody notes.

        NOTE: This algorithm calculates the melody for each bar independently, so
        a note held across a bar line is not seen by the next bar. Use
        ``get_melody_of_song`` to extract on one timeline for the whole song.

        Parameters
        ----------
        mel_def : str
            Melody definition, passed to ``Bar.get_melody`` for every bar:
            ``'hi_track'``, ``'hi_note'`` or ``'hi_note_dur'``.
        inst_id : int, optional
            GM program number of the melody Track in every bar (default ``0``).
            The melody can come from different instruments in different bars
            (or, for ``'hi_note'``, within one bar), so one fixed program is used.

        Returns
        -------
        MultiTrack
            Same number of bars as ``self``, each with the source bar's id,
            time signature and tempo. Bars with no melody notes (empty, or only
            drums) have no tracks, like empty bars elsewhere in REMI-z. The notes
            are new Note objects, so editing the result leaves ``self`` untouched.
        """
        assert isinstance(inst_id, int), "inst_id must be an integer"
        assert 0 <= inst_id <= 127, "inst_id must be in the range of [0, 127]"

        notes_per_bar = []
        for bar in self.bars:
            has_pitched_track = any(not track.is_drum for track in bar.tracks.values())
            mel_notes = bar.get_melody(mel_def) if has_pitched_track else []
            notes_per_bar.append([(n.onset, n.pitch, n.duration, n.velocity) for n in mel_notes])
        return self._melody_to_multitrack(notes_per_bar, inst_id)

    def insert_empty_bars_at_front(self, num_bars):
        """
        Insert empty bars at the front of the MultiTrack object.
        """
        assert isinstance(num_bars, int), "num_bars must be an integer"
        assert num_bars >= 0, "num_bars must be a non-negative integer"

        ts = self.bars[0].time_signature
        tempo = self.bars[0].tempo

        empty_bars = []
        for i in range(1, num_bars + 1):
            empty_bars.insert(
                0, Bar(id=-i, notes_of_insts={}, time_signature=ts, tempo=tempo)
            )
        self.bars = empty_bars + self.bars

    def merge_with(self, other: "MultiTrack", other_prog_id) -> "MultiTrack":
        """
        Merge two MultiTrack objects.
        Both MultiTrack objects must have the same number of bars, time signature, and tempo.
        """
        assert isinstance(other, MultiTrack), "other must be a MultiTrack object"
        assert len(self.bars) == len(
            other.bars
        ), "Both MultiTrack objects must have the same number of bars"

        new_bars = []
        for bar1, bar2 in zip(self.bars, other.bars):
            # Merge the two bars
            merged_bar = Bar(
                id=bar1.bar_id,
                notes_of_insts={},
                time_signature=bar1.time_signature,
                tempo=bar1.tempo,
            )
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

    def permute_phrase(self, new_phrase_list: List[Tuple[int, int]]) -> "MultiTrack":
        """
        Permute the phrases in the MultiTrack object.
        Args:
            new_phrase_list: List of tuples (start_bar, end_bar) for each phrase.
        Returns:
            New MultiTrack object with permuted phrases.
        """
        new_bars = []
        for start_bar, end_bar in new_phrase_list:
            new_bars.extend(self.bars[start_bar:end_bar])

        return MultiTrack(bars=new_bars)

    def filter_tracks(self, insts: List[int]):
        """
        Filter the tracks in the MultiTrack object. Only keep the tracks in the insts list.
        """
        for bar in self.bars:
            bar.filter_tracks(insts=insts)

    def remove_tracks(self, insts: List[int]):
        """
        Remove the tracks in the MultiTrack object. Remove the tracks in the insts list.
        """
        for bar in self.bars:
            insts_to_keep = [
                inst_id for inst_id in bar.tracks.keys() if inst_id not in insts
            ]
            bar.filter_tracks(insts=insts_to_keep)

    def change_instrument(self, old_inst_id: int, new_inst_id: int):
        """
        Change the instrument ID of the tracks in the MultiTrack object.
        """
        for bar in self.bars:
            bar.change_instrument(old_inst_id=old_inst_id, new_inst_id=new_inst_id)


def save_remiz_str_to_midi(remiz_str: str, midi_fp: str):
    """
    Save a REMI-z string to a MIDI file.
    """
    mt = MultiTrack.from_remiz_str(remiz_str, verbose=False)
    mt.to_midi(midi_fp)
