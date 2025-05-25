import argparse
import os
import logging
from datetime import timedelta

import srt
import yaml
from pydub import AudioSegment, silence
import nemo.collections.asr as nemo_asr
import ffmpeg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mp4_to_wav(input_file, output_file):
    (
        ffmpeg
        .input(input_file)
        .output(output_file, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(quiet=False)
    )
    logger.info(f"✅ Successfully created: {output_file}")


def generate_subtitles(wav_file: str, srt_file: str):
    with open('config.yml', 'r') as file:
        CONFIG = yaml.safe_load(file)

    logger.info("Loading audio...")
    audio = AudioSegment.from_wav(wav_file)
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=CONFIG['audio']['silence_min'],
        silence_thresh=CONFIG['audio']['silence_thresh'],
        keep_silence=True
    )

    logger.info(f"Detected {len(chunks)} segments.")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=CONFIG['model'])

    subtitles = []
    start_ms = 0
    index = 1

    for chunk in chunks:
        chunk = chunk.set_channels(1)
        if len(chunk) < CONFIG['audio']['segment_min']:
            start_ms += len(chunk)
            continue

        chunk_path = f"chunk_{index}.wav"
        chunk.export(chunk_path, format="wav")

        logger.info(f"Transcribing segment {index}...")
        output = model.transcribe([chunk_path])
        text = output[0].text.strip()

        if text:
            end_ms = start_ms + len(chunk)
            subtitle = srt.Subtitle(
                index=index,
                start=timedelta(milliseconds=start_ms),
                end=timedelta(milliseconds=end_ms),
                content=text
            )
            subtitles.append(subtitle)
            index += 1

        start_ms += len(chunk)
        os.remove(chunk_path)

    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

    logger.info(f"Saved subtitles to {srt_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Audio transcription and subtitle generation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input WAV file')
    parser.add_argument('--output_file', type=str, default='output.srt', help='Path to the input WAV file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_file.endswith('.mp4'):
        wav_file = args.input_file.replace('.mp4', '.wav')
        mp4_to_wav(args.input_file, wav_file)
        args.input_file = wav_file

    generate_subtitles(args.input_file, args.output_file)
