# REMI-z tokenizer and MultiTrack data structure

This is the official implementation of the REMI-z tokenizer in the paper [*Unlocking Potential in Pre-Trained Music Language Models for Versatile Multi-Track Music Arrangement*](https://arxiv.org/abs/2408.15176).

This tool helps to convert your music between MIDI and REMI-z representation, meanwhile facilitate manipulate the music at bar level.

The core of this tokenizer is the MultiTrack class as the data structure for multitrack music, which is a hierachical format. Here are the structural details:
- The music is represented by an MultiTrack object, which is list of bars.
    - Each Bar object represents all notes being played within one bar, grouped by Track object, together with time signature and tempo info of this bar.
        - Each Track object represents one instrument, contatining notes of that instrument in this bar.
            - Each Note object represent one note, including onset, offset, pitch, velocity information.

This Multitrack object can be create from various formats (supporting MIDI for now), and convert into various formats (e.g., MIDI, and REMI-z representation).

## Install
Install from pip

    pip install REMI-z

Install from source

    git clone https://github.com/Sonata165/REMI-z.git
    cd REMI-z
    pip install -r requirements.txt
    pip install .
