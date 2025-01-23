from remi_z import MultiTrack, Bar

midi_fp = '/Users/sonata/Code/GuitarArranger/misc/caihong-4bar.midi'
mt = MultiTrack.from_midi(midi_fp)
bar = mt[0]
res = bar.get_chord()
print(res)