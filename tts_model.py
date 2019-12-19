import sys
sys.path.append("espnet/egs/ljspeech/tts1/local")
sys.path.append("espnet")

from argparse import Namespace

import nltk
import torch
import yaml
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import
from parallel_wavegan.models import ParallelWaveGANGenerator
from text.cleaners import custom_english_cleaners

nltk.download('punkt')


class TTSModel:
    def __init__(self, device='cpu'):
        dict_path = "downloads/data/lang_1char/train_no_dev_units.txt"
        model_path = "downloads/exp/train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best"
        vocoder_path = "downloads/ljspeech.parallel_wavegan.v1/checkpoint-400000steps.pkl"
        vocoder_conf = "downloads/ljspeech.parallel_wavegan.v1/config.yml"

        device = torch.device(device)

        idim, odim, train_args = get_model_conf(model_path)
        model_class = dynamic_import(train_args.model_module)
        model = model_class(idim, odim, train_args)
        torch_load(model_path, model)
        model = model.eval().to(device)
        inference_args = Namespace(**{"threshold": 0.5, "minlenratio": 0.0, "maxlenratio": 10.0})

        with open(vocoder_conf) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        vocoder = ParallelWaveGANGenerator(**config["generator_params"])
        vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"]["generator"])
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)

        with open(dict_path) as f:
            lines = f.readlines()
        lines = [line.replace("\n", "").split(" ") for line in lines]
        char_to_id = {c: int(i) for c, i in lines}

        self.device = device
        self.char_to_id = char_to_id
        self.idim = idim
        self.model = model
        self.inference_args = inference_args
        self.config = config
        self.vocoder = vocoder

    def frontend(self, text):
        """Clean text and then convert to id sequence."""
        text = custom_english_cleaners(text)

        charseq = list(text)
        idseq = []
        for c in charseq:
            if c.isspace():
                idseq += [self.char_to_id["<space>"]]
            elif c not in self.char_to_id.keys():
                idseq += [self.char_to_id["<unk>"]]
            else:
                idseq += [self.char_to_id[c]]
        idseq += [self.idim - 1]  # <eos>
        return torch.LongTensor(idseq).view(-1).to(self.device)

    def generate(self, input_text):
        with torch.no_grad():
            x = self.frontend(input_text)
            c, _, _ = self.model.inference(x, self.inference_args)
            z = torch.randn(1, 1, c.size(0) * self.config["hop_size"]).to(self.device)
            c = torch.nn.ReplicationPad1d(
                self.config["generator_params"]["aux_context_window"])(c.unsqueeze(0).transpose(2, 1))
            y = self.vocoder(z, c).view(-1)

            y = y.view(-1).cpu().numpy()
            return y


if __name__ == '__main__':
    import librosa

    tts = TTSModel()
    y = tts.generate('This is really awesome!')
    sr = tts.config['sampling_rate']
    librosa.output.write_wav('file.wav', y, sr=sr)
