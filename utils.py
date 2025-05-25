from pathlib import Path

import yaml
from pydub import AudioSegment


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def make_patches(wav_file, start, end):
    start_time, end_time = int(start * 1000), int(end * 1000)  # to milliseconds
    trimmed_audio = AudioSegment.from_wav(wav_file)[start_time:end_time].set_channels(1)
    output_path = Path(wav_file).parent / f"{Path(wav_file).stem}_{start}_to_{end}.wav"
    trimmed_audio.export(output_path, format="wav")
    return output_path
