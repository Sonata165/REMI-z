import pytest
from remi_z.note import Note
from remi_z.track import Track
from remi_z.bar import Bar
from remi_z.multitrack import MultiTrack


def make_note(onset=0, duration=12, pitch=60, velocity=64):
    return Note(onset=onset, duration=duration, pitch=pitch, velocity=velocity)


def make_bar(bar_id=0, notes_of_insts=None, time_signature=(4, 4), tempo=120.0):
    if notes_of_insts is None:
        notes_of_insts = {0: {0: [(60, 12, 64)], 12: [(64, 12, 64)]}}
    return Bar(id=bar_id, notes_of_insts=notes_of_insts,
               time_signature=time_signature, tempo=tempo)


def make_multitrack(n_bars=2, inst_id=0, pitch=60):
    bars = [make_bar(bar_id=i, notes_of_insts={inst_id: {0: [(pitch, 12, 64)]}})
            for i in range(n_bars)]
    return MultiTrack(bars=bars)


# ============================================================
# Construction
# ============================================================

class TestMultiTrackInit:

    def test_bars_stored(self):
        mt = make_multitrack(n_bars=3)
        assert len(mt.bars) == 3

    def test_requires_list(self):
        with pytest.raises(AssertionError):
            MultiTrack(bars="not a list")

    def test_time_signatures_collected(self):
        mt = make_multitrack()
        assert (4, 4) in mt.time_signatures

    def test_tempos_collected(self):
        mt = make_multitrack()
        assert 120.0 in mt.tempos

    def test_multiple_time_signatures(self):
        bars = [
            make_bar(bar_id=0, time_signature=(4, 4)),
            make_bar(bar_id=1, time_signature=(3, 4)),
        ]
        mt = MultiTrack(bars=bars)
        assert len(mt.time_signatures) == 2


# ============================================================
# __len__ / __getitem__
# ============================================================

class TestMultiTrackGetItem:

    def test_len(self):
        assert len(make_multitrack(n_bars=4)) == 4

    def test_getitem_int_returns_bar(self):
        mt = make_multitrack(n_bars=3)
        assert isinstance(mt[0], Bar)

    def test_getitem_int_correct_bar(self):
        bars = [make_bar(bar_id=i) for i in range(3)]
        mt = MultiTrack(bars=bars)
        assert mt[1].bar_id == 1

    def test_getitem_slice_returns_multitrack(self):
        mt = make_multitrack(n_bars=4)
        sliced = mt[1:3]
        assert isinstance(sliced, MultiTrack)
        assert len(sliced) == 2

    def test_str(self):
        mt = make_multitrack(n_bars=2)
        assert "MultiTrack" in str(mt)
        assert "2" in str(mt)

    def test_repr_equals_str(self):
        mt = make_multitrack()
        assert repr(mt) == str(mt)


# ============================================================
# set_tempo / set_velocity
# ============================================================

class TestMultiTrackSetters:

    def test_set_tempo_all_bars(self):
        mt = make_multitrack(n_bars=3)
        mt.set_tempo(90.0)
        assert all(bar.tempo == pytest.approx(90.0) for bar in mt.bars)

    def test_set_tempo_updates_tempos(self):
        mt = make_multitrack()
        mt.set_tempo(90.0)
        assert 90.0 in mt.tempos

    def test_set_velocity_all_notes(self):
        mt = make_multitrack(n_bars=2)
        mt.set_velocity(100)
        for bar in mt.bars:
            for track in bar.tracks.values():
                assert all(n.velocity == 100 for n in track.notes)

    def test_set_velocity_specific_track(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {0: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        mt.set_velocity(100, track_id=0)
        assert mt.bars[0].tracks[0].notes[0].velocity == 100
        assert mt.bars[0].tracks[24].notes[0].velocity == 64  # unchanged


# ============================================================
# quantize_to_16th
# ============================================================

class TestQuantize:

    def test_onset_snapped_to_16th(self):
        bars = [make_bar(notes_of_insts={0: {1: [(60, 12, 64)]}})]  # onset=1
        mt = MultiTrack(bars=bars)
        mt.quantize_to_16th()
        onset = mt.bars[0].tracks[0].notes[0].onset
        assert onset % 3 == 0

    def test_duration_snapped_to_16th(self):
        bars = [make_bar(notes_of_insts={0: {0: [(60, 1, 64)]}})]  # duration=1
        mt = MultiTrack(bars=bars)
        mt.quantize_to_16th()
        dur = mt.bars[0].tracks[0].notes[0].duration
        assert dur % 3 == 0 or dur == 3


# ============================================================
# get_unique_insts
# ============================================================

class TestMultiTrackGetUniqueInsts:

    def test_single_inst(self):
        mt = make_multitrack(inst_id=24)
        assert mt.get_unique_insts() == {24}

    def test_multiple_insts_across_bars(self):
        bars = [
            make_bar(bar_id=0, notes_of_insts={0: {0: [(60, 12, 64)]}}),
            make_bar(bar_id=1, notes_of_insts={24: {0: [(72, 12, 64)]}}),
        ]
        mt = MultiTrack(bars=bars)
        assert mt.get_unique_insts() == {0, 24}


# ============================================================
# to_remiz_str / from_remiz_str roundtrip
# ============================================================

class TestMultiTrackRemizRoundtrip:

    def test_to_remiz_str_contains_bar_end(self):
        mt = make_multitrack(n_bars=1)
        s = mt.to_remiz_str()
        assert "b-1" in s

    def test_to_remiz_seq_ends_with_bar_end(self):
        mt = make_multitrack(n_bars=1)
        seq = mt.to_remiz_seq()
        assert seq[-1] == "b-1"

    def test_roundtrip_bar_count(self):
        mt = make_multitrack(n_bars=3)
        s = mt.to_remiz_str()
        mt2 = MultiTrack.from_remiz_str(s, verbose=False)
        assert len(mt2) == 3

    def test_roundtrip_note_count(self):
        mt = make_multitrack(n_bars=1)
        s = mt.to_remiz_str()
        mt2 = MultiTrack.from_remiz_str(s, verbose=False)
        assert len(mt2.bars[0].get_all_notes()) == len(mt.bars[0].get_all_notes())

    def test_from_remiz_str_with_velocity(self):
        mt = make_multitrack(n_bars=1)
        s = mt.to_remiz_str(with_velocity=True)
        mt2 = MultiTrack.from_remiz_str(s, verbose=False)
        assert len(mt2) == 1

    def test_from_remiz_seq(self):
        mt = make_multitrack(n_bars=1)
        seq = mt.to_remiz_seq()
        mt2 = MultiTrack.from_remiz_seq(seq)
        assert len(mt2) == 1

    def test_from_remiz_str_adds_missing_bar_end(self):
        mt = MultiTrack.from_remiz_str("i-0 o-0 p-60 d-12", verbose=False)
        assert len(mt) == 1

    def test_remove_repeated_eob(self):
        mt = MultiTrack.from_remiz_str("i-0 o-0 p-60 d-12 b-1 b-1", verbose=False,
                                        remove_repeated_eob=True)
        assert len(mt) == 1


# ============================================================
# from_bars / concat
# ============================================================

class TestMultiTrackFactories:

    def test_from_bars(self):
        bars = [make_bar(bar_id=i) for i in range(2)]
        mt = MultiTrack.from_bars(bars)
        assert len(mt) == 2

    def test_concat(self):
        mt1 = make_multitrack(n_bars=2)
        mt2 = make_multitrack(n_bars=3)
        mt = MultiTrack.concat([mt1, mt2])
        assert len(mt) == 5

    def test_concat_requires_nonempty_list(self):
        with pytest.raises(AssertionError):
            MultiTrack.concat([])


# ============================================================
# get_all_notes / get_all_notes_by_bar
# ============================================================

class TestMultiTrackGetNotes:

    def test_get_all_notes_count(self):
        mt = make_multitrack(n_bars=2)
        notes = mt.get_all_notes()
        assert len(notes) == 2  # 1 note per bar

    def test_get_all_notes_exclude_drum(self):
        bars = [Bar(id=0, notes_of_insts={
            0: {0: [(60, 12, 64)]},
            128: {0: [(36, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        notes = mt.get_all_notes(include_drum=False)
        assert all(n.pitch != 36 for n in notes)

    def test_get_all_notes_by_bar_length(self):
        mt = make_multitrack(n_bars=3)
        result = mt.get_all_notes_by_bar()
        assert len(result) == 3

    def test_get_all_notes_by_bar_each_is_list(self):
        mt = make_multitrack(n_bars=2)
        for bar_notes in mt.get_all_notes_by_bar():
            assert isinstance(bar_notes, list)


# ============================================================
# flatten
# ============================================================

class TestMultiTrackFlatten:

    def test_single_track_per_bar(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {12: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        flat = mt.flatten()
        assert len(flat.bars[0].tracks) == 1

    def test_note_count_preserved(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {12: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        flat = mt.flatten()
        assert len(flat.get_all_notes()) == 2


    def test_offset_overlap_adjusted_by_default(self):
        # piano holds C4 for a whole note; guitar re-attacks C4 on beat 2
        mt = MultiTrack(bars=[make_bar(notes_of_insts={
            0: {0: [(60, 48, 64)]},
            24: {12: [(60, 12, 64)]},
        })])
        notes = mt.flatten().bars[0].tracks[0].notes
        assert [(n.onset, n.duration) for n in notes] == [(0, 12), (12, 12)]

    def test_offset_overlap_kept_when_disabled(self):
        mt = MultiTrack(bars=[make_bar(notes_of_insts={
            0: {0: [(60, 48, 64)]},
            24: {12: [(60, 12, 64)]},
        })])
        notes = mt.flatten(adjust_offset_overlap=False).bars[0].tracks[0].notes
        assert [(n.onset, n.duration) for n in notes] == [(0, 48), (12, 12)]

    def test_offset_overlap_across_bar_line(self):
        mt = MultiTrack(bars=[
            make_bar(bar_id=0, notes_of_insts={0: {36: [(60, 30, 64)]}}),
            make_bar(bar_id=1, notes_of_insts={24: {6: [(60, 12, 64)]}}),
        ])
        flat = mt.flatten()
        assert flat.bars[0].tracks[0].notes[0].duration == 18  # 36 + 18 = 48 + 6
        assert flat.bars[1].tracks[0].notes[0].duration == 12

    def test_source_not_modified(self):
        mt = MultiTrack(bars=[make_bar(notes_of_insts={
            0: {0: [(60, 48, 64)]},
            24: {12: [(60, 12, 64)]},
        })])
        mt.flatten()
        assert mt.bars[0].tracks[0].notes[0].duration == 48


# ============================================================
# adjust_offset_overlap
# ============================================================

class TestMultiTrackAdjustOffsetOverlap:

    @staticmethod
    def durations(mt):
        return [[(tid, n.onset, n.pitch, n.duration) for tid, t in bar.tracks.items() for n in t.notes]
                for bar in mt.bars]

    def test_nested_same_pitch_trimmed(self):
        mt = MultiTrack(bars=[make_bar(notes_of_insts={0: {0: [(60, 48, 64)], 12: [(60, 6, 64)]}})])
        assert self.durations(mt.adjust_offset_overlap()) == [[(0, 0, 60, 12), (0, 12, 60, 6)]]

    def test_three_four_bar_start(self):
        # bar 0 is 3/4 (36 positions): onset 30 + 24 reaches bar 1's onset 6 (song position 42) after 12
        mt = MultiTrack(bars=[
            make_bar(bar_id=0, notes_of_insts={0: {30: [(60, 24, 64)]}}, time_signature=(3, 4)),
            make_bar(bar_id=1, notes_of_insts={0: {6: [(60, 12, 64)]}}),
        ])
        assert mt._bar_starts() == [0, 36]
        assert self.durations(mt.adjust_offset_overlap())[0] == [(0, 30, 60, 12)]

    def test_untouched_cases(self):
        mt = MultiTrack(bars=[make_bar(notes_of_insts={
            0: {0: [(60, 12, 64), (64, 48, 64)], 12: [(60, 12, 64), (67, 12, 64)]},  # touch; other pitches
            24: {6: [(60, 12, 64)]},   # same pitch, other track
            128: {0: [(36, 48, 64)], 12: [(36, 12, 64)]},  # drums
        })])
        assert self.durations(mt.adjust_offset_overlap()) == self.durations(mt)

    def test_drums_included_on_request(self):
        mt = MultiTrack(bars=[make_bar(notes_of_insts={128: {0: [(36, 48, 64)], 12: [(36, 12, 64)]}})])
        notes = mt.adjust_offset_overlap(include_drum=True).bars[0].tracks[128].notes
        assert [n.duration for n in notes] == [12, 12]

    def test_same_onset_duplicates_in_track_deduplicated(self):
        mt = MultiTrack(bars=[make_bar(notes_of_insts={0: {0: [(60, 6, 64), (60, 24, 80)], 12: [(60, 12, 64)]}})])
        notes = mt.adjust_offset_overlap().bars[0].tracks[0].notes
        assert [(n.onset, n.duration, n.velocity) for n in notes] == [(0, 12, 80), (12, 12, 64)]

    def test_metadata_and_source_kept(self):
        mt = MultiTrack(bars=[
            make_bar(bar_id=0, notes_of_insts={0: {0: [(60, 96, 64)]}, 24: {0: [(72, 12, 64)]}},
                     time_signature=(3, 4), tempo=90.0),
            make_bar(bar_id=1, notes_of_insts={0: {0: [(60, 12, 64)]}}, tempo=100.0),
        ])
        adj = mt.adjust_offset_overlap()
        assert [(b.bar_id, b.time_signature, b.tempo, list(b.tracks)) for b in adj.bars] == \
               [(b.bar_id, b.time_signature, b.tempo, list(b.tracks)) for b in mt.bars]
        assert adj.bars[0].tracks[0].notes[0].duration == 36
        assert mt.bars[0].tracks[0].notes[0].duration == 96
        assert adj.bars[0].tracks[0].notes[0] is not mt.bars[0].tracks[0].notes[0]

    def test_midi_round_trip(self, tmp_path):
        mt = MultiTrack(bars=[
            make_bar(bar_id=0, notes_of_insts={
                0: {0: [(60, 48, 64), (64, 24, 64)], 24: [(64, 12, 64)]},
                24: {12: [(60, 12, 64)], 36: [(67, 24, 64)]},
            }),
            make_bar(bar_id=1, notes_of_insts={24: {0: [(67, 12, 64)], 12: [(60, 12, 64)]}}),
        ])
        flat = mt.flatten()
        fp = str(tmp_path / "flat.mid")
        flat.to_midi(fp, verbose=False)
        assert self.durations(MultiTrack.from_midi(fp)) == self.durations(flat)


# ============================================================
# from_midi: offset adjustment
# ============================================================

def write_midi(fp, instruments):
    """instruments: [(program, is_drum, name, [(start_pos, end_pos, pitch, velocity)])],
    times in REMI-z positions (12 per beat; 480 ticks per beat -> 40 ticks per position)."""
    import miditoolkit
    midi = miditoolkit.midi.parser.MidiFile(ticks_per_beat=480)
    midi.time_signature_changes.append(miditoolkit.midi.containers.TimeSignature(4, 4, 0))
    midi.tempo_changes.append(miditoolkit.midi.containers.TempoChange(120, 0))
    for program, is_drum, name, notes in instruments:
        inst = miditoolkit.midi.containers.Instrument(program=program, is_drum=is_drum, name=name)
        inst.notes = [miditoolkit.midi.containers.Note(velocity=v, pitch=p, start=s * 40, end=e * 40)
                      for s, e, p, v in notes]
        midi.instruments.append(inst)
    midi.dump(fp)
    return fp


def song_notes(mt):
    starts = mt._bar_starts()
    return sorted((tid, starts[i] + n.onset, n.pitch, n.duration, n.velocity)
                  for i, bar in enumerate(mt.bars) for tid, t in bar.tracks.items() for n in t.notes)


class TestFromMidiOffsetAdjustment:

    def test_same_program_tracks_merged_and_adjusted(self, tmp_path):
        fp = write_midi(str(tmp_path / "a.mid"), [
            (0, False, "piano 1", [(0, 48, 60, 64), (24, 36, 64, 64)]),
            (0, False, "piano 2", [(12, 18, 60, 70), (24, 30, 64, 90)]),  # nested C4; doubled E4
        ])
        assert song_notes(MultiTrack.from_midi(fp)) == [
            (0, 0, 60, 12, 64), (0, 12, 60, 6, 70), (0, 24, 64, 12, 64)]

    def test_disabled_keeps_old_behavior(self, tmp_path):
        fp = write_midi(str(tmp_path / "a.mid"), [
            (0, False, "piano 1", [(0, 48, 60, 64), (24, 36, 64, 64)]),
            (0, False, "piano 2", [(12, 18, 60, 70), (24, 30, 64, 90)]),
        ])
        assert song_notes(MultiTrack.from_midi(fp, adjust_offset_overlap=False)) == [
            (0, 0, 60, 48, 64), (0, 12, 60, 6, 70), (0, 24, 64, 6, 90), (0, 24, 64, 12, 64)]

    def test_across_bar_line(self, tmp_path):
        fp = write_midi(str(tmp_path / "a.mid"), [
            (0, False, "a", [(36, 72, 60, 64)]),
            (0, False, "b", [(54, 60, 60, 64)]),  # bar 1, onset 6
        ])
        mt = MultiTrack.from_midi(fp)
        assert song_notes(mt) == [(0, 36, 60, 18, 64), (0, 54, 60, 6, 64)]
        assert mt.bars[0].tracks[0].notes[0].duration == 18

    def test_zero_length_after_quantization(self, tmp_path):
        # a 1-tick note quantizes to duration 0 (Note makes it 1): it must not pass the next onset
        import miditoolkit
        fp = str(tmp_path / "a.mid")
        midi = miditoolkit.midi.parser.MidiFile(ticks_per_beat=480)
        inst = miditoolkit.midi.containers.Instrument(program=0, name="a")
        inst.notes = [miditoolkit.midi.containers.Note(64, 60, 400, 401),
                      miditoolkit.midi.containers.Note(64, 60, 440, 480)]
        midi.instruments.append(inst)
        midi.dump(fp)
        assert song_notes(MultiTrack.from_midi(fp)) == [(0, 10, 60, 1, 64), (0, 11, 60, 1, 64)]

    def test_untouched_cases(self, tmp_path):
        tracks = [
            (0, False, "piano", [(0, 48, 60, 64), (0, 12, 64, 64), (12, 24, 64, 64)]),  # touch
            (24, False, "guitar", [(12, 24, 60, 64)]),          # same pitch, other program
            (0, True, "drums", [(0, 48, 36, 64), (12, 24, 36, 64), (12, 18, 36, 80)]),
        ]
        fp = write_midi(str(tmp_path / "a.mid"), tracks)
        assert song_notes(MultiTrack.from_midi(fp)) == song_notes(
            MultiTrack.from_midi(fp, adjust_offset_overlap=False))

    def test_multi_instance_tracks_not_adjusted_against_each_other(self, tmp_path):
        fp = write_midi(str(tmp_path / "a.mid"), [
            (0, False, "piano 1", [(0, 48, 60, 64)]),
            (0, False, "piano 2", [(12, 24, 60, 64)]),
        ])
        mt = MultiTrack.from_midi(fp, support_same_program_multi_instance=True)
        assert sorted(n[1:4] for n in song_notes(mt)) == [(0, 60, 48), (12, 60, 12)]

    def test_round_trip(self, tmp_path):
        fp = write_midi(str(tmp_path / "a.mid"), [
            (0, False, "piano 1", [(0, 48, 60, 64), (24, 90, 67, 64), (60, 72, 67, 64)]),
            (0, False, "piano 2", [(12, 24, 60, 64), (30, 40, 67, 64)]),
            (33, False, "bass", [(0, 96, 36, 64), (48, 60, 36, 64)]),
        ])
        mt = MultiTrack.from_midi(fp)
        out = str(tmp_path / "b.mid")
        mt.to_midi(out, verbose=False)
        assert song_notes(MultiTrack.from_midi(out)) == song_notes(mt)
        assert song_notes(MultiTrack.from_midi(out, adjust_offset_overlap=False)) == song_notes(mt)


# ============================================================
# filter_tracks / remove_tracks / change_instrument
# ============================================================

class TestMultiTrackFilterAndChange:

    def test_filter_tracks(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {0: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        mt.filter_tracks([0])
        assert 24 not in mt.bars[0].tracks

    def test_remove_tracks(self):
        bars = [make_bar(notes_of_insts={
            0: {0: [(60, 12, 64)]},
            24: {0: [(72, 12, 64)]},
        })]
        mt = MultiTrack(bars=bars)
        mt.remove_tracks([24])
        assert 24 not in mt.bars[0].tracks
        assert 0 in mt.bars[0].tracks

    def test_change_instrument(self):
        mt = make_multitrack(n_bars=1, inst_id=0)
        mt.change_instrument(old_inst_id=0, new_inst_id=24)
        assert 24 in mt.bars[0].tracks
        assert 0 not in mt.bars[0].tracks


# ============================================================
# permute_phrase
# ============================================================

class TestMultiTrackPermute:

    def test_permute_reorders_bars(self):
        bars = [make_bar(bar_id=i, notes_of_insts={0: {0: [(60 + i, 12, 64)]}})
                for i in range(4)]
        mt = MultiTrack(bars=bars)
        permuted = mt.permute_phrase([(2, 4), (0, 2)])
        assert permuted.bars[0].bar_id == 2
        assert permuted.bars[2].bar_id == 0

    def test_permute_bar_count(self):
        mt = make_multitrack(n_bars=4)
        permuted = mt.permute_phrase([(0, 2), (2, 4)])
        assert len(permuted) == 4


# ============================================================
# get_content_seq
# ============================================================

class TestMultiTrackGetContentSeq:

    def test_returns_list(self):
        mt = make_multitrack(n_bars=1)
        assert isinstance(mt.get_content_seq(), list)

    def test_return_str(self):
        mt = make_multitrack(n_bars=1)
        result = mt.get_content_seq(return_str=True)
        assert isinstance(result, str)

    def test_bar_count_reflected(self):
        mt = make_multitrack(n_bars=2)
        seq = mt.get_content_seq()
        assert seq.count("b-1") == 2


# ============================================================
# get_melody
# ============================================================

class TestMultiTrackGetMelody:

    def make_two_inst_mt(self):
        # bar 0: piano chord low, strings high; bar 1: empty; bar 2: drums only
        bars = [
            make_bar(bar_id=0, notes_of_insts={
                0: {0: [(48, 12, 64), (52, 12, 64)], 24: [(50, 12, 64)]},
                48: {0: [(72, 24, 80)], 24: [(74, 12, 80)]},
            }),
            make_bar(bar_id=1, notes_of_insts={}),
            make_bar(bar_id=2, notes_of_insts={128: {0: [(36, 6, 100)]}}, time_signature=(3, 4), tempo=90.0),
        ]
        return MultiTrack(bars=bars)

    def test_returns_multitrack_same_bar_count(self):
        mt = self.make_two_inst_mt()
        mel = mt.get_melody("hi_note")
        assert isinstance(mel, MultiTrack)
        assert len(mel) == len(mt)

    def test_single_track_per_bar(self):
        mel = self.make_two_inst_mt().get_melody("hi_track")
        assert list(mel.bars[0].tracks.keys()) == [0]

    def test_inst_id_param(self):
        mel = self.make_two_inst_mt().get_melody("hi_note", inst_id=73)
        assert list(mel.bars[0].tracks.keys()) == [73]
        assert mel.bars[0].tracks[73].inst_id == 73

    def test_hi_track_notes(self):
        mel = self.make_two_inst_mt().get_melody("hi_track")
        pitches = [n.pitch for n in mel.bars[0].tracks[0].notes]
        assert sorted(pitches) == [72, 74]

    def test_hi_note_notes(self):
        mel = self.make_two_inst_mt().get_melody("hi_note")
        notes = mel.bars[0].tracks[0].notes
        assert [(n.onset, n.pitch) for n in notes] == [(0, 72), (24, 74)]

    def test_matches_bar_get_melody(self):
        mt = self.make_two_inst_mt()
        for mel_def in ["hi_track", "hi_note", "hi_note_dur"]:
            mel = mt.get_melody(mel_def)
            expected = [(n.onset, n.pitch, n.duration, n.velocity) for n in mt.bars[0].get_melody(mel_def)]
            got = [(n.onset, n.pitch, n.duration, n.velocity) for n in mel.bars[0].tracks[0].notes]
            assert sorted(got) == sorted(expected), mel_def

    def test_chord_notes_kept(self):
        # hi_track on a single chordal track: several notes share an onset and must all survive
        mt = MultiTrack(bars=[make_bar(notes_of_insts={0: {0: [(48, 12, 64), (52, 12, 64), (55, 12, 64)]}})])
        mel = mt.get_melody("hi_track")
        assert sorted(n.pitch for n in mel.bars[0].tracks[0].notes) == [48, 52, 55]

    def test_empty_and_drum_only_bars_have_no_track(self):
        mel = self.make_two_inst_mt().get_melody("hi_track")
        assert mel.bars[1].tracks == {}
        assert mel.bars[2].tracks == {}

    def test_bar_metadata_kept(self):
        mt = self.make_two_inst_mt()
        mel = mt.get_melody("hi_note")
        for src, out in zip(mt.bars, mel.bars):
            assert (out.bar_id, out.time_signature, out.tempo) == (src.bar_id, src.time_signature, src.tempo)

    def test_source_not_modified(self):
        mt = self.make_two_inst_mt()
        before = mt.to_remiz_str()
        mel = mt.get_melody("hi_note")
        mel.shift_pitch(12)
        assert mt.to_remiz_str() == before

    def test_bad_inst_id(self):
        with pytest.raises(AssertionError):
            self.make_two_inst_mt().get_melody("hi_note", inst_id=128)

    def test_after_key_norm(self):
        # key_norm used to leave numpy ints in Note.pitch, which Note() rejects when get_melody rebuilds notes
        bars = [make_bar(bar_id=i, notes_of_insts={0: {0: [(62, 12, 64)], 12: [(66, 12, 64)], 24: [(69, 12, 64)]}})
                for i in range(4)]  # D major arpeggio -> shifted to C
        mt = MultiTrack(bars=bars)
        shift = mt.key_norm()
        assert type(shift) is int
        assert all(type(n.pitch) is int for b in mt.bars for t in b.tracks.values() for n in t.notes)
        mel = mt.get_melody("hi_note_dur")
        assert len(mel) == 4


# ============================================================
# get_melody_of_song
# ============================================================

def mel_notes(mt):
    """[(bar_idx, onset_in_bar, pitch, duration)] of a melody MultiTrack."""
    return sorted((i, n.onset, n.pitch, n.duration) for i, b in enumerate(mt.bars) for t in b.tracks.values() for n in t.notes)


class TestMultiTrackGetMelodyOfSong:

    def test_note_held_across_bar_line_blocks_lower_notes(self):
        # the bug case: bar 0's 60 starts at 36 and lasts 36 -> ends at bar 1 pos 24.
        # bar 1's 52 (0-12) and 43 (12-18) lie inside it and are lower -> not melody; 64 at 30 is after it.
        mt = MultiTrack(bars=[
            make_bar(0, {0: {36: [(60, 36, 64)]}}),
            make_bar(1, {0: {0: [(52, 12, 64)], 12: [(43, 6, 64)], 30: [(64, 12, 64)]}}),
        ])
        assert mel_notes(mt.get_melody_of_song("hi_note_dur_plus")) == [(0, 36, 60, 36), (1, 30, 64, 12)]
        # bar-by-bar get_melody can't see the held note -- this is what the song-level version fixes
        assert (1, 0, 52, 12) in mel_notes(mt.get_melody("hi_note_dur"))

    def test_default_policy(self):
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(60, 96, 64)]}}), make_bar(1, {0: {12: [(55, 12, 64)]}})])
        assert mel_notes(mt.get_melody_of_song()) == mel_notes(mt.get_melody_of_song("hi_note_dur_plus"))

    def test_higher_note_over_held_note_kept(self):
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(60, 96, 64)]}}), make_bar(1, {0: {12: [(67, 12, 64)]}})])
        assert mel_notes(mt.get_melody_of_song()) == [(0, 0, 60, 96), (1, 12, 67, 12)]

    def test_same_pitch_reattack_kept(self):
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(64, 60, 64)]}}), make_bar(1, {0: {6: [(64, 12, 64)]}})])
        assert mel_notes(mt.get_melody_of_song()) == [(0, 0, 64, 60), (1, 6, 64, 12)]

    def test_exact_comparison_no_tolerance(self):
        # a note starting exactly where the held note ends is not overlapping; one position earlier is
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(64, 12, 64)], 12: [(60, 12, 64)]}})])
        assert [p for _, _, p, _ in mel_notes(mt.get_melody_of_song())] == [64, 60]
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(64, 13, 64)], 12: [(60, 12, 64)]}})])
        assert [p for _, _, p, _ in mel_notes(mt.get_melody_of_song())] == [64]

    def test_highest_note_per_onset_across_tracks(self):
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(48, 12, 64), (55, 12, 64)]}, 48: {0: [(72, 12, 80)]}})])
        assert mel_notes(mt.get_melody_of_song()) == [(0, 0, 72, 12)]

    def test_three_four_bar_length(self):
        # 3/4 bars are 36 positions: 60 at 24 lasting 24 ends at bar 1 pos 12
        mt = MultiTrack(bars=[
            make_bar(0, {0: {24: [(60, 24, 64)]}}, time_signature=(3, 4)),
            make_bar(1, {0: {6: [(55, 6, 64)], 12: [(57, 6, 64)]}}, time_signature=(3, 4)),
        ])
        assert mel_notes(mt.get_melody_of_song()) == [(0, 24, 60, 24), (1, 12, 57, 6)]

    def test_trim_overlap_across_bar_line(self):
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(60, 96, 64)]}}), make_bar(1, {0: {12: [(67, 12, 64)]}})])
        # 60 is cut at song position 48 + 12 = 60
        assert mel_notes(mt.get_melody_of_song(trim_overlap=True)) == [(0, 0, 60, 60), (1, 12, 67, 12)]
        assert mel_notes(mt.get_melody_of_song()) == [(0, 0, 60, 96), (1, 12, 67, 12)]  # default: untrimmed

    def test_trim_overlap_leaves_no_overlap(self):
        import random
        rng = random.Random(0)
        bars = []
        for i in range(8):
            notes = {}
            for _ in range(6):
                notes.setdefault(rng.randrange(0, 48, 3), []).append((rng.randint(40, 90), rng.randrange(3, 60, 3), 64))
            bars.append(make_bar(i, {0: notes}))
        mel = MultiTrack(bars=bars).get_melody_of_song(trim_overlap=True)
        song = sorted((i * 48 + n.onset, i * 48 + n.onset + n.duration) for i, b in enumerate(mel.bars)
                      for t in b.tracks.values() for n in t.notes)
        assert all(nxt[0] >= cur[1] for cur, nxt in zip(song, song[1:]))
        assert all(end > on for on, end in song)

    def test_hi_track_returns_multitrack_with_chords(self):
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(48, 12, 64)]}, 48: {0: [(72, 24, 64), (76, 24, 64)], 12: [(74, 12, 64)]}})])
        mel = mt.get_melody_of_song("hi_track")
        assert isinstance(mel, MultiTrack)
        assert mel_notes(mel) == [(0, 0, 72, 24), (0, 0, 76, 24), (0, 12, 74, 12)]
        # trimming cuts the chord at the next onset but never to zero length
        assert mel_notes(mt.get_melody_of_song("hi_track", trim_overlap=True)) == [(0, 0, 72, 12), (0, 0, 76, 12), (0, 12, 74, 12)]

    def test_empty_drum_only_and_metadata(self):
        mt = MultiTrack(bars=[
            make_bar(0, {0: {0: [(60, 12, 64)]}}, tempo=90.0),
            make_bar(1, {}),
            make_bar(2, {128: {0: [(36, 6, 100)]}}, time_signature=(3, 4)),
        ])
        for policy in ["hi_note_dur_plus", "hi_track"]:
            mel = mt.get_melody_of_song(policy, inst_id=73)
            assert len(mel) == 3
            assert list(mel.bars[0].tracks) == [73] and mel.bars[1].tracks == {} and mel.bars[2].tracks == {}
            assert [(b.bar_id, b.time_signature, b.tempo) for b in mel.bars] == [(b.bar_id, b.time_signature, b.tempo) for b in mt.bars]
        drums = MultiTrack(bars=[make_bar(0, {128: {0: [(36, 6, 100)]}})])
        assert drums.get_melody_of_song("hi_track").bars[0].tracks == {}

    def test_source_not_modified(self):
        mt = MultiTrack(bars=[make_bar(0, {0: {0: [(60, 96, 64)]}}), make_bar(1, {0: {12: [(67, 12, 64)]}})])
        before = mt.to_remiz_str()
        mel = mt.get_melody_of_song(trim_overlap=True)
        mel.shift_pitch(12)
        assert mt.to_remiz_str() == before

    def test_bad_args(self):
        mt = make_multitrack()
        with pytest.raises(AssertionError):
            mt.get_melody_of_song("hi_note")
        with pytest.raises(AssertionError):
            mt.get_melody_of_song(inst_id=128)
