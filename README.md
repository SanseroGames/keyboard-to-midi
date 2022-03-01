# Audio Keyboard to MIDI Device

I have a keyboard that only has a audio output, but way to use midi with my computer. But I wanted to use a keyboard to work with in my DAW so I made this script.

This script tries to recognize which keys are pressed based on the audio in the microphone input. It is based on the project [audio-to-midi](https://github.com/NFJones/audio-to-midi]) but it tries to perform the analysis in real time and emit midi signals based on the recognized keys.

The script has a visualisation graph to help adjust the microphone input volume such that only the keys are recognized that should be, but you should disable it if you don't need it, as it adds latency to the process. Also the script maps a Xbox controller to midi controller, for example for pitch bending or velocity control.

This script is quite rough and "works on my machine:tm:"

## Limitations

The script is by no means accurate. The accuracy highly depends on the chosen instrument of the keyboard. The more "noisy" an instrument is (i.e. has a lot of frequency for one note) the more notes might be recognized. Also be aware that the accuracy of the script drops quite enormous the lower the note gets in pitch. For me it is fairly accurate from the middle C and above, and quite terrible below it.

The script does not try to assume velocity for the notes. It just uses a simple threshold to recognize if a frequency is considered a note. You need to be able to adjust the volume of your microphone input in order to use this script.

Also the script is not optimized, so you notice a slight latency between the keypress and the not playing in the DAW. If you record on eights notes you sometimes might miss your note by an eight, just be aware of that.

## Requirements

This script requires the following python packages
- `mido`
- `rtmidi`
- `pyaudio`
- `numpy`
- `matplotlib` For visualisation
- `inputs` For controller to midi mapping

Additionaly, on Windows you require [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) to be installed and running. Windows by itself doea not support software midi signals routed to a DAW, so this is what you need this for.

## Usage

This script is only tested on Windows.

To just run the script simply start it with

```
python3 main.py
```

If you want to use the visualisation run

```
python3 main.py --visual
```