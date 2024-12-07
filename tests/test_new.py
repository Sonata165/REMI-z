import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(__file__)))

from multitrack.core import MultiTrack
from multitrack.legacy_tokenizer import RemiTokenizer

def main():
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