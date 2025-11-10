# *REMI-z* tokenizer and *MultiTrack* music data structure

This is the official implementation of the REMI-z tokenizer in the paper [*Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization*](https://arxiv.org/abs/2408.15176). 

[ [Paper](https://arxiv.org/abs/2408.15176) | [Demo](https://www.oulongshen.xyz/automatic_arrangement) | [Github](https://github.com/Sonata165/REMI-z) | [PyPI](https://pypi.org/project/REMI-z/) | [Author](https://www.oulongshen.xyz/)]

This tool helps to convert your music between MIDI and REMI-z representation, which is an efficient sequence representation of multitrack music, meanwhile facilitate manipulate the music at bar level.

The core of this tokenizer is the MultiTrack class as the data structure for multitrack music, which is a hierachical format. Here are the structural details:
- The music is represented by an MultiTrack object, which is list of bars.
    - Each Bar object represents all notes being played within one bar, grouped by Track object, together with time signature and tempo info of this bar.
        - Each Track object represents one instrument, contatining notes of that instrument in this bar.
            - Each Note object represent one note, including onset, offset, pitch, velocity information.

This Multitrack object can be create from various formats (e.g., MIDI or REMI-z), and convert into various formats (e.g., MIDI, and REMI-z representation).

## Install
Install from pip

    pip install REMI-z

Install from source

    git clone https://github.com/Sonata165/REMI-z.git
    cd REMI-z
    pip install -r Requirements.txt
    pip install -e .

## Usage
Please refer to the `demo.ipynb`.

## Updates
- [2025-09-09] Support reading and writing MIDIs containing multiple tracks of same program ID.

## Known issues
- Create MultiTrack object from REMI-z sequence does not yet support multiple tracks of same program ID. 

## Citation

    @inproceedings{ou2025unifying,
        title     = {Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization},
        author    = {Ou, Longshen and Zhao, Jingwei and Wang, Ziyu and Xia, Gus and Liang, Qihao and Hopkins, Torin and Wang, Ye},
        booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
        year      = {2025}
    }