import os
import tensorflow as tf
from tensorflow import audio
import tensorflow_io as tfio


def get_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".WAV" or ".wav"):
            files.append(file)
    return files


def load_wav(path):
    raw = tf.io.read_file(path)
    clip, sampleRate = audio.decode_wav(raw)
    clip = tf.squeeze(clip, axis=-1)
    sampleRate = tf.cast(sampleRate, tf.int64)
    clip = tfio.audio.resample(clip, sampleRate, 16000)
    return clip


if __name__ == "__main__":
    stimDir = "./stimuli/"
    files = get_files(stimDir)
    load_wav(os.path.join(stimDir, files[0]))
