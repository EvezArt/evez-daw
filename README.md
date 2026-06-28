# EVEZ DAW v2

Pure-code music production engine. Zero samples. Zero paid APIs. Pure NumPy/SciPy synthesis.

## Features
- 6 beat presets: breakcore 170, dubstep 140, phonk 130, amen break, 404 architecture 200, industrial 160
- 7 bass engines: sub, reese, wobble, phonk 808, scream, FM growl, square sub
- 10 FX: distortion, bitcrush, reverb, delay, lowpass, highpass, formant, comb, chorus, waveshaper
- Sidechain ducking
- Master bus: 3-band EQ + stereo widening + soft clip limiter
- Multi-section arrangement (intro/verse/drop/break/drop/outro)
- Swing/groove
- Drumkit generator (WAV export)
- Voice chopping & rearranging
- WAV + MP3 export
- CLI interface — no server needed

## Usage
```bash
python3 evez_daw_v2.py presets
python3 evez_daw_v2.py render --preset 404_architecture --bars 8 --bass scream --sidechain 0.6 --mp3
python3 evez_daw_v2.py render --arrange --bass wobble --sidechain 0.6 --mp3
python3 evez_daw_v2.py drumkit --style breakcore --name my_kit
python3 evez_daw_v2.py chop --input sample.wav --slices 16 --pattern 0,2,1,3
```

## Requirements
```bash
pip install numpy scipy soundfile
# MP3 export requires ffmpeg
```

Author: EVEZ // Steven Crawford-Maggard
License: MIT
