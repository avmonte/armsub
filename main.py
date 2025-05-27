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

    processed_subs = postprocess_subtitles(subtitles)

    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(processed_subs))

    logger.info(f"✅ Saved subtitles to {srt_file}")


def postprocess_subtitles(subtitles, min_duration=timedelta(seconds=1.5), max_duration=timedelta(seconds=6), max_chars=80):
    processed = []
    buffer = None

    for i, sub in enumerate(subtitles):
        duration = sub.end - sub.start
        text_len = len(sub.content.strip())

        if duration < min_duration or text_len < 20:
            if buffer is None:
                buffer = sub
            else:
                buffer.content += ' ' + sub.content
                buffer.end = sub.end
        else:
            if buffer:
                processed.append(buffer)
                buffer = None

            if duration > max_duration or text_len > max_chars:
                parts = split_subtitle(sub, max_chars)
                processed.extend(parts)
            else:
                processed.append(sub)

    if buffer:
        processed.append(buffer)

    for i, sub in enumerate(processed, 1):
        sub.index = i

    return processed


def split_subtitle(sub, max_chars=80):
    text = sub.content.strip()
    total_duration = sub.end - sub.start
    words = text.split()
    seconds_per_word = total_duration.total_seconds() / max(len(words), 1)

    segments = [seg.strip() for seg in text.split(",") if seg.strip()]

    parts = []
    current_text = ""
    current_len = 0
    current_words = 0
    start_time = sub.start

    def flush_segment(text, words_count):
        nonlocal start_time
        end_time = start_time + timedelta(seconds=seconds_per_word * words_count)
        part = srt.Subtitle(index=0, start=start_time, end=end_time, content=text.strip())
        start_time = end_time
        return part

    for seg in segments:
        seg_len = len(seg)
        seg_words = len(seg.split())

        if current_len + seg_len + 2 <= max_chars:
            current_text += (", " if current_text else "") + seg
            current_len += seg_len + 2
            current_words += seg_words
        else:
            if current_text:
                parts.append(flush_segment(current_text, current_words))
            current_text = seg
            current_len = seg_len
            current_words = seg_words

    if current_text:
        parts.append(flush_segment(current_text, current_words))

    final_parts = []
    for part in parts:
        if len(part.content) > max_chars:
            final_parts.extend(split_by_words(part, max_chars, seconds_per_word))
        else:
            final_parts.append(part)

    return final_parts


def split_by_words(sub, max_chars, seconds_per_word):
    words = sub.content.split()
    parts = []
    temp = []
    current_len = 0
    start_time = sub.start

    for word in words:
        if current_len + len(word) + 1 > max_chars:
            end_time = start_time + timedelta(seconds=seconds_per_word * len(temp))
            parts.append(srt.Subtitle(index=0, start=start_time, end=end_time, content=' '.join(temp)))
            start_time = end_time
            temp = [word]
            current_len = len(word)
        else:
            temp.append(word)
            current_len += len(word) + 1

    if temp:
        end_time = sub.end
        parts.append(srt.Subtitle(index=0, start=start_time, end=end_time, content=' '.join(temp)))

    return parts


def parse_args():
    parser = argparse.ArgumentParser(description="Audio transcription and subtitle generation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input WAV file')
    parser.add_argument('--output_file', type=str, default='subtitles.srt', help='Path to the input WAV file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_file.endswith('.mp4'):
        wav_file = args.input_file.replace('.mp4', '.wav')
        mp4_to_wav(args.input_file, wav_file)
        args.input_file = wav_file

    generate_subtitles(args.input_file, args.output_file)
