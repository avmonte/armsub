import nemo.collections.asr as nemo_asr

from utils import *

config = read_yaml('config.yml')


def main(wav_files):
    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=config['model'])

    output = asr_model.transcribe(wav_files)
    print(output[0].text)


if __name__ == "__main__":
    make_patches('data/sample0.wav', 0, 10)
    main(['/Users/gevorg/PycharmProjects/armsub/data/sample0_0_to_10.wav'])

