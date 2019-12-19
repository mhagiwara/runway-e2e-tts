import base64
import io

import runway
import soundfile as sf
import torch
from runway.data_types import text

from tts_model import TTSModel


@runway.setup(options={})
def setup(opts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TTSModel(device=device)
    return model


inputs = {'input_text': text}
outputs = {'audio': text}


@runway.command('generate', inputs=inputs, outputs=outputs, description='Generate an audio file.')
def generate(model, input_args):
    y = model.generate(input_args['input_text'])
    buffer = io.BytesIO()
    sf.write(buffer, y, model.config['sampling_rate'], format='WAV', closefd=False)
    return base64.b64encode(bytes(buffer.getbuffer()))


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=9000)
