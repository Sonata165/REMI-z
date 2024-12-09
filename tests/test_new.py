import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(__file__)))

from remi_z.core import MultiTrack
from remi_z.legacy_tokenizer import RemiTokenizer

def main():
    test_key_norm_new()

def test_key_norm_new():
    multitrack = MultiTrack.from_midi('/Users/sonata/Code/REMI-z/_misc/4bros.mid')
    multitrack.normalize_pitch()
    multitrack.to_midi('/Users/sonata/Code/REMI-z/_misc/4bros_norm_new.mid')

def test_from_remi():
    multitrack = MultiTrack.from_midi('/Users/sonata/Code/REMI-z/_misc/4bros.mid')
    remiz_str = multitrack.to_remiz_str(
        with_ts=False,
        with_tempo=False,
        with_velocity=True,
    )
    multitrack2 = MultiTrack.from_remiz_str(remiz_str)
    remiz_str2 = multitrack2.to_remiz_str(
        with_ts=False,
        with_tempo=False,
        with_velocity=True,
    )
    a = 3

def test_drum_notes():
    multitrack = MultiTrack.from_midi('/Users/sonata/Code/REMI-z/_misc/4bros.mid')
    a = 2

def test_key_norm():
    multitrack = MultiTrack.from_midi('/Users/sonata/Code/REMI-z/_misc/4bros.mid')
    multitrack.normalize_pitch()
    multitrack.to_midi('/Users/sonata/Code/REMI-z/_misc/4bros_norm.mid')

def test_decoding():
    midi_fp = '/Users/sonata/Code/REMI-z/_misc/alaylm.mid'
    multitrack = MultiTrack.from_midi(midi_fp)
    multitrack.normalize_pitch()
    multitrack.to_midi('/Users/sonata/Code/REMI-z/_misc/alaylm_norm.mid')

def result_comparison():
    # Generate REMI-z for a MIDI file
    multitrack = MultiTrack.from_midi('/Users/sonata/Code/REMI-z/_misc/butterfly_ours.mid')
    remiz_seq = multitrack.to_remiz_seq(with_ts=True, with_tempo=True, key_norm=True)
    multitrack.to_midi('/Users/sonata/Code/REMI-z/_misc/butterfly_ours_dec.mid')
    multitrack[1:4]

    # Generate by the legacy tokenizer (deprecated)
    tokenizer = RemiTokenizer()
    remi_old = tokenizer.midi_to_remi(
        '/Users/sonata/Code/REMI-z/_misc/butterfly_ours.mid', 
        normalize_pitch=False, 
        reorder_by_inst=False,
        include_ts=True,
        include_tempo=True,
    )
    tokenizer.remi_to_midi(remi_old, '/Users/sonata/Code/REMI-z/_misc/butterfly_ours_old.mid')

    same_cnt = 0
    for i in range(len(remiz_seq)):
        if remiz_seq[i] == remi_old[i]:
            same_cnt += 1
    percent = same_cnt / len(remiz_seq) * 100
    print(f"Same count: {same_cnt}/{len(remiz_seq)}")
    print(f"Same percent: {percent}%")


if __name__ == '__main__':
    main()