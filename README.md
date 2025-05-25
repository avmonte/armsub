## About

This script converts MP4 or WAV audio/video files into time-aligned subtitle files (`.srt`) using an Armenian ASR model.

## Usage

```bash
python main.py --input_file path/to/input.mp4 --output_file path/to/output.srt 
```
```--input_file``` can be an MP4 or WAV file. 

```--output_file``` is optional.

## License and Attribution

This project uses the [Armenian ASR model](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) by **NVIDIA**, licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). We have modified the model for subtitle generation.
