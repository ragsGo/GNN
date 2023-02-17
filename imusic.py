import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import IPython.display as ipd
import librosa
from midiutil import MIDIFile
import random
from pedalboard import Pedalboard, Chorus, Reverb
from pedalboard.io import AudioFile
from scipy.io import wavfile
song = np.array([])
octaves = np.array([0.5,1,2])
sr = 22050 # sample rate
T = 0.1    # 0.1 second duration
t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
#Make a song with numpy array :]
#nPixels = int(len(frequencies))#All pixels in image
nPixels = 60
scale_freqs = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 415.30]


def hue2freq(h, scale_freqs):
    thresholds = [26, 52, 78, 104, 128, 154, 180]
    note = scale_freqs[0]
    if (h <= thresholds[0]):
        note = scale_freqs[0]
    elif (h > thresholds[0]) & (h <= thresholds[1]):
        note = scale_freqs[1]
    elif (h > thresholds[1]) & (h <= thresholds[2]):
        note = scale_freqs[2]
    elif (h > thresholds[2]) & (h <= thresholds[3]):
        note = scale_freqs[3]
    elif (h > thresholds[3]) & (h <= thresholds[4]):
        note = scale_freqs[4]
    elif (h > thresholds[4]) & (h <= thresholds[5]):
        note = scale_freqs[5]
    elif (h > thresholds[5]) & (h <= thresholds[6]):
        note = scale_freqs[6]
    else:
        note = scale_freqs[0]

    return note

def img2music(img, scale=[220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 415.30],
              sr=22050, T=0.1, nPixels=60, useOctaves=True, randomPixels=False,
              harmonize='U0'):
    """
    Args:
        img    :     (array) image to process
        scale  :     (array) array containing frequencies to map H values to
        sr     :     (int) sample rate to use for resulting song
        T      :     (int) time in seconds for dutation of each note in song
        nPixels:     (int) how many pixels to use to make song
    Returns:
        song   :     (array) Numpy array of frequencies. Can be played by ipd.Audio(song, rate = sr)
    """
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get shape of image
    height, width, depth = img.shape

    i = 0;
    j = 0;
    k = 0
    # Initialize array the will contain Hues for every pixel in image
    hues = []
    for i in range(height):
        for j in range(width):
            hue = hsv[i][j][0]  # This is the hue value at pixel coordinate (i,j)
            hues.append(hue)

    # Make dataframe containing hues and frequencies
    pixels_df = pd.DataFrame(hues, columns=['hues'])
    pixels_df['frequencies'] = pixels_df.apply(lambda row: hue2freq(row['hues'], scale), axis=1)
    frequencies = pixels_df['frequencies'].to_numpy()

    # Make harmony dictionary (i.e. fundamental, perfect fifth, major third, octave)
    # unison           = U0 ; semitone         = ST ; major second     = M2
    # minor third      = m3 ; major third      = M3 ; perfect fourth   = P4
    # diatonic tritone = DT ; perfect fifth    = P5 ; minor sixth      = m6
    # major sixth      = M6 ; minor seventh    = m7 ; major seventh    = M7
    # octave           = O8
    harmony_select = {'U0': 1,
                      'ST': 16 / 15,
                      'M2': 9 / 8,
                      'm3': 6 / 5,
                      'M3': 5 / 4,
                      'P4': 4 / 3,
                      'DT': 45 / 32,
                      'P5': 3 / 2,
                      'm6': 8 / 5,
                      'M6': 5 / 3,
                      'm7': 9 / 5,
                      'M7': 15 / 8,
                      'O8': 2
                      }
    harmony = np.array([])  # This array will contain the song harmony
    harmony_val = harmony_select[harmonize]  # This will select the ratio for the desired harmony

    song_freqs = np.array([])  # This array will contain the chosen frequencies used in our song :]
    song = np.array([])  # This array will contain the song signal
    octaves = np.array([0.5, 1, 2])  # Go an octave below, same note, or go an octave above
    t = np.linspace(0, T, int(T * sr), endpoint=False)  # time variable
    # Make a song with numpy array :]
    # nPixels = int(len(frequencies))#All pixels in image
    for k in range(nPixels):
        if useOctaves:
            octave = random.choice(octaves)
        else:
            octave = 1

        if randomPixels == False:
            val = octave * frequencies[k]
        else:
            val = octave * random.choice(frequencies)

        # Make note and harmony note
        note = 0.5 * np.sin(2 * np.pi * val * t)
        h_note = 0.5 * np.sin(2 * np.pi * harmony_val * val * t)

        # Place notes into corresponfing arrays
        song = np.concatenate([song, note])
        harmony = np.concatenate([harmony, h_note])
        # song_freqs = np.concatenate([song_freqs, val])

    return song, pixels_df, harmony


def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    base_freq = 440  # Frequency of Note A4
    keys = np.array([x + str(y) for y in range(0, 9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end + 1]

    note_freqs = dict(zip(keys, [2 ** ((n + 1 - 49) / 12) * base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0  # stop
    return note_freqs


def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    t = np.linspace(0, duration, int(sample_rate * duration))  # Time axis
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def makeScale(whichOctave, whichKey, whichScale, makeHarmony='U0'):
    # Load note dictionary
    note_freqs = get_piano_notes()

    # Define tones. Upper case are white keys in piano. Lower case are black keys
    scale_intervals = ['A', 'a', 'B', 'C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g']

    # Find index of desired key
    index = scale_intervals.index(whichKey)

    # Redefine scale interval so that scale intervals begins with whichKey
    new_scale = scale_intervals[index:12] + scale_intervals[:index]

    # Choose scale
    if whichScale == 'AEOLIAN':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'BLUES':
        scale = [0, 2, 3, 4, 5, 7, 9, 10, 11]
    elif whichScale == 'PHYRIGIAN':
        scale = [0, 1, 3, 5, 7, 8, 10]
    elif whichScale == 'CHROMATIC':
        scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif whichScale == 'DIATONIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'DORIAN':
        scale = [0, 2, 3, 5, 7, 9, 10]
    elif whichScale == 'HARMONIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 11]
    elif whichScale == 'LYDIAN':
        scale = [0, 2, 4, 6, 7, 9, 11]
    elif whichScale == 'MAJOR':
        scale = [0, 2, 4, 5, 7, 9, 11]
    elif whichScale == 'MELODIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 9, 10, 11]
    elif whichScale == 'MINOR':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'MIXOLYDIAN':
        scale = [0, 2, 4, 5, 7, 9, 10]
    elif whichScale == 'NATURAL_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'PENTATONIC':
        scale = [0, 2, 4, 7, 9]
    else:
        print('Invalid scale name')
    harmony_select = {'U0': 1,
                      'ST': 16 / 15,
                      'M2': 9 / 8,
                      'm3': 6 / 5,
                      'M3': 5 / 4,
                      'P4': 4 / 3,
                      'DT': 45 / 32,
                      'P5': 3 / 2,
                      'm6': 8 / 5,
                      'M6': 5 / 3,
                      'm7': 9 / 5,
                      'M7': 15 / 8,
                      'O8': 2
                      }

    # Get length of scale (i.e., how many notes in scale)
    nNotes = len(scale)

    # Initialize arrays
    freqs = []
    # harmony = []
    # harmony_val = harmony_select[makeHarmony]
    for i in range(nNotes):
        note = new_scale[scale[i]] + str(whichOctave)
        freqToAdd = note_freqs[note]
        freqs.append(freqToAdd)
        # harmony.append(harmony_val*freqToAdd)
    return freqs  # ,harmony

waterfall = cv2.imread('Noah.jpg')
waterfall2 = cv2.cvtColor(waterfall, cv2.COLOR_BGR2RGB)

waterfall_scale = makeScale(1, 'd', 'MAJOR')
waterfall_song, waterfall_df,waterfall_song_harmony  = img2music(waterfall,
                                                                waterfall_scale,
                                                                T = 0.3,
                                                                randomPixels = True,
                                                                useOctaves = True)

wavfile.write('Noah.wav'    , rate = 22050, data = waterfall_song.astype(np.float32))
ipd.Audio(waterfall_song, rate = sr)