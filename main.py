import time
import mido
import pyaudio
import numpy
from functools import lru_cache
import os
import pathlib
import sys
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=sys.maxsize)
path = pathlib.Path().resolve()

os.add_dll_directory(path)
#outport = mido.open_output('Microsoft GS Wavetable Synth 0')
#mido.set_backend('mido.backends.pygame')
#outport = mido.open_output('Microsoft GS Wavetable Synth')
mido.set_backend('mido.backends.rtmidi_python')
outport = mido.open_output(b'loopMIDI Port')
a_pressed = False

TIME_WINDOW = 1 # in milliseconds
WIDTH = 4
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
CHUNK = 1024 # frames

def generate_notes():
    """
    Generates a dict of midi note codes with their corresponding
        frequency ranges.
    """

    # C0
    base = [7.946362749, 8.1757989155, 8.4188780665]
    
    # 12th root of 2
    multiplier = numpy.float_power(2.0, 1.0 / 12)

    notes = {0: base}
    for i in range(1, 128):
        mid = multiplier * notes[i - 1][1]
        low = (mid + notes[i - 1][1]) / 2.0
        high = (mid + (multiplier * mid)) / 2.0
        notes.update({i: [low, mid, high]})

    return notes

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_data(x_vec, y1_data)
    # adjust limits if new data goes beyond bounds
    if numpy.min(y1_data)<=line1.axes.get_ylim()[0] or numpy.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([numpy.min(y1_data)-numpy.std(y1_data),numpy.max(y1_data)+numpy.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1

class Converter:
    def __init__(
        self,
        activation_level=0,
        condense=None,
        condense_max=False,
        max_note_length=0,
        transpose=0,
        pitch_set=None,
        pitch_range=None,
        note_count=0,
        progress=None,
        samplerate=44100,
        channels=2,
        format=2,
        frames=1024,
        bpm=60,
    ):
        self.condense = condense
        self.condense_max = condense_max
        self.max_note_length = max_note_length
        self.transpose = transpose
        self.pitch_set = pitch_set
        self.pitch_range = pitch_range or [0, 127]
        self.note_count = note_count
        self.progress = progress
        self.samplerate = samplerate
        self.channels = channels
        self.frames=frames
        self.bpm = bpm
        if format != 2 and format != 4:
            raise ValueError("Did not figure out how numpy can use something else than 2 or 4")
        self.format = format

        self.activation_level = int(127 * activation_level) or 1
        self.block_size = self.frames

        self._determine_ranges()

    def _determine_ranges(self):
        self.notes = generate_notes()
        self.max_freq = min(self.notes[127][-1], self.samplerate / 2)
        self.min_freq = self.notes[0][-1]
        self.bins = self.block_size//2
        self.frequencies = numpy.fft.fftfreq(self.bins, 1/self.samplerate)[
            : self.bins // 2
        ]
        for i, f in enumerate(self.frequencies):
            if f >= self.min_freq:
                self.min_bin = i
                break
        else:
            self.min_bin = 0
        for i, f in enumerate(self.frequencies):
            if f >= self.max_freq:
                self.max_bin = i
                break
        else:
            self.max_bin = len(self.frequencies)
           
            
    @staticmethod
    def _time_window_to_block_size(time_window, rate):
        """
        time_window is the time in ms over which to compute fft's.
        rate is the audio sampling rate in samples/sec.
        Transforms the time window into an index step size and
            returns the result.
        """

        # rate/1000(samples/ms) * time_window(ms) = block_size(samples)
        rate_per_ms = rate / 1000
        block_size = rate_per_ms * time_window

        return int(block_size)
        
    def _freqs_to_midi(self, freqs):
        """
        freq_list is a list of frequencies with normalized amplitudes.
        Takes a list of notes and transforms the amplitude to a
            midi volume as well as adding track and channel info.
        """

        notes = [None for _ in range(128)]
        for pitch, velocity in freqs:
            if not (self.pitch_range[0] <= pitch <= self.pitch_range[1]):
                continue
            #velocity = min(int(127 * (velocity / self.bins)), 127)

            if velocity > 1: #self.activation_level:
                if not notes[pitch]:
                    notes[pitch] = Note(pitch, velocity)
                else:
                    notes[pitch].velocity = int(
                        ((notes[pitch].velocity * notes[pitch].count) + velocity)
                        / (notes[pitch].count + 1)
                    )
                    notes[pitch].count += 1

        notes = [note for note in notes if note]

        if self.note_count > 0:
            max_count = min(len(notes), self.note_count)
            notes = sorted(notes, key=attrgetter("velocity"))[::-1][:max_count]

        return notes

    def _snap_to_key(self, pitch):
        if self.pitch_set:
            mod = pitch % 12
            pitch = (12 * (pitch // 12)) + min(
                self.pitch_set, key=lambda x: abs(x - mod)
            )
        return pitch
    
    @lru_cache(None)
    def _freq_to_pitch(self, freq):
        for pitch, freq_range in self.notes.items():
            # Find the freq's equivalence class, adding the amplitudes.
            if freq_range[0] <= freq <= freq_range[2]:
                return self._snap_to_key(pitch) + self.transpose
        raise RuntimeError("Unmappable frequency: {}".format(freq))
        
    def _reduce_freqs(self, freqs):
        """
        freqs is a list of amplitudes produced by _fft_to_frequencies().
        Reduces the list of frequencies to a list of notes and their
            respective volumes by determining what note each frequency
            is closest to. It then reduces the list of amplitudes for each
            note to a single amplitude by summing them together.
        """

        reduced_freqs = {}
        for freq in freqs:
            reduced_freqs.append((self._freq_to_pitch(freq[0]), freq[1]))
        return reduced_freqs
        
    def _samples_to_freqs(self, samples):
        global line1
        amplitudes = numpy.fft.fft(samples)
        freqs = []

        for index in range(self.min_bin, self.max_bin):
            # frequency, amplitude
            freqs.append(
                [
                    self.frequencies[index],
                    numpy.sqrt(
                        numpy.float_power(amplitudes[index].real, 2)
                        + numpy.float_power(amplitudes[index].imag, 2)
                    ),index
                ]
            )
        xdata = []
        ydata = []
        #for i in freqs[:80]:
        #    xdata.append(i[0])
        #    ydata.append(i[1])
        #line1 = live_plotter(xdata,ydata,line1)
        # Transform the frequency info into midi compatible data.
        return self._reduce_freqs(freqs)
        
    def _block_to_notes(self, block):
        global line1
        channels = [[] for _ in range(self.channels)]
        notes = [None for _ in range(self.channels)]

        for sample in block:
            for channel in range(self.channels):
                channels[channel].append(sample[channel])

        for channel, samples in enumerate(channels):
            freqs = self._samples_to_freqs(samples)
            notes[channel] = self._freqs_to_midi(freqs)

        return notes
        
    def _fast_freqs_to_midi(self, freqs):
        """
        freq_list is a list of frequencies with normalized amplitudes.
        Takes a list of notes and transforms the amplitude to a
            midi volume as well as adding track and channel info.
        """

        notes = []
        for pitch, velocity in enumerate(freqs):
            if not (self.pitch_range[0] <= pitch <= self.pitch_range[1]):
                continue
            #velocity = min(int(127 * (velocity / self.bins)), 127)

            if velocity > 1: #self.activation_level:
                notes.append(Note(pitch, velocity))
        return notes
        
    def _fast_reduce_freqs(self, freqs):
        global line1
        """
        freqs is a list of amplitudes produced by _fft_to_frequencies().
        Reduces the list of frequencies to a list of notes and their
            respective volumes by determining what note each frequency
            is closest to. It then reduces the list of amplitudes for each
            note to a single amplitude by summing them together.
        """

        reduced_freqs = [0] * 128
        for i in range(self.min_bin, self.max_bin):
            reduced_freqs[self._freq_to_pitch(self.frequencies[i])] += freqs[i]
        #line1 = live_plotter(range(128),reduced_freqs,line1)
        return reduced_freqs
     
    def _fast_samples_to_freqs(self, samples):
        global line1
        amplitudes = numpy.fft.fft(samples)
        freqs = numpy.sqrt(numpy.float_power(amplitudes.real, 2)+numpy.float_power(amplitudes.imag, 2))

        #line1 = live_plotter(self.frequencies[self.min_bin: self.max_bin],freqs[self.min_bin: self.max_bin],line1)
        # Transform the frequency info into midi compatible data.
        return self._fast_reduce_freqs(freqs)
        
    def _block_to_notes_single_channel(self, block):
        global line1
        notes = [None for _ in range(self.channels)]
        
        freqs = self._fast_samples_to_freqs(block.reshape(-1))
        notes[0] = self._fast_freqs_to_midi(freqs)

        return notes
        
    def convert(self, in_data, frame_count):
        block = numpy.frombuffer(in_data, dtype=numpy.dtype("f4")).reshape(-1,self.channels)
        notes = self._block_to_notes_single_channel(block)
        return notes

class Note:
    __slots__ = ["pitch", "velocity", "count"]

    def __init__(self, pitch, velocity, count=0):
        self.pitch = pitch
        self.velocity = velocity
        self.count = count
        
    def __repr__(self):
        return f"Note(pitch={self.pitch},velocity={self.velocity},count={self.count})"
   
line1=[]
c = Converter(samplerate=RATE, channels=CHANNELS, frames=CHUNK, pitch_range=[24,108], activation_level=0.0000001/127, transpose=-24)

oldNotes = set()

def callback(in_data, frame_count, time_info, status):
    global oldNotes
    notes = c.convert(in_data, frame_count)
    newNotes = set()
    for i in notes[0]:
        msg = mido.Message("note_on", note=i.pitch, velocity=63)
        outport.send(msg)
        newNotes.add(i.pitch)
    for i in oldNotes:
        if i not in newNotes:
            msg = mido.Message("note_off", note=i)
        outport.send(msg)
    oldNotes=newNotes
    return (None, pyaudio.paContinue)


# This somehow silences my webbrowser?... (reload of the page fixes it though)
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paFloat32,#p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback =callback)

stream.start_stream()

print("Running...")
try:
    while stream.is_active():
        time.sleep(0.1)
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

print("* terminated")
