import os
import tensorflow as tf
from tensorflow import audio
import tensorflow_io as tfio
import pydub


def get_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".WAV") or file.endswith(".wav"):
            files.append(file)

    # Sort files
    files.sort()
    return files


def load_wav(path):
    raw = tf.io.read_file(path)
    clip, sampleRate = audio.decode_wav(raw)
    # If stereo, take the mean
    if clip.shape[-1] == 2:
        clip = tf.reduce_mean(clip, axis=-1)
    else:
        clip = tf.squeeze(clip, axis=-1)
    sampleRate = tf.cast(sampleRate, tf.int64)
    clip = tfio.audio.resample(clip, sampleRate, 16000)
    return clip


def convert_mp3(path):
    # Search files for mp3
    mps3 = os.listdir(path)

    for file in mps3:
        if file.endswith(".mp3"):
            # Convert to wav
            sound = pydub.AudioSegment.from_mp3(os.path.join(path, file))
            sound.export(os.path.join(path, file[:-4] + ".wav"), format="wav")

            # Delete mp3
            os.remove(os.path.join(path, file))


if __name__ == "__main__":
    convert_mp3("./stimuli/")
