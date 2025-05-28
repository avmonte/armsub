## About

This script converts MP4 or WAV audio/video files into time-aligned subtitle files (`.srt`) using an Armenian ASR model.

## Setup

> **Note**: This script is only supported on Unix-based systems (Linux/macOS). NVIDIA NeMo is not officially supported on Windows. For more information, refer to the [NVIDIA NeMo GitHub repository](https://github.com/NVIDIA/NeMo).

1. **Install Conda**  
   Download and install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create and activate a new environment**  
   ```bash
   conda create -n armsub python=3.10 -y
   conda activate armsub
   ```

3. **Upgrade pip (required before installing NeMo)**
    ```bash
    python -m pip install --upgrade pip   
    ```

4. **Install required packages**  
   ```bash
    pip install srt pyyaml pydub ffmpeg-python 'nemo_toolkit[asr]'
    ```


## Usage

```bash
python main.py --input_file path/to/input.mp4 --output_file path/to/output.srt 
```
```--input_file``` can be an MP4 or WAV file. 

```--output_file``` is optional.

## License and Attribution

This project uses the [Armenian ASR model](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc) by **NVIDIA**, licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). We have modified the model for subtitle generation.
