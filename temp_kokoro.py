# !pip install -q kokoro>=0.9.2 soundfile
# !apt-get -qq -y install espeak-ng > /dev/null 2>&1
from kokoro import KPipeline, KModel
from IPython.display import display, Audio
import soundfile as sf
import torch
import torchaudio
model = KModel(
    config=r"C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\models\Kroko-82M\config.json",
    model=r"C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\models\Kroko-82M\kokoro-v1_0.pth",
)

# exit()

pipeline = KPipeline(
    lang_code="a",
    model=model,
)
import time

start = time.time()

text = """
TED was established in February 1984, but became an annual conference from 1990.
An average TED talk is 18 minutes or under 18 minutes long – which is backed by strategy and neuroscience.
The conference covers a broad spectrum of topics – from tech, business and innovation. To culture, feminism and spirituality.

It produces content in more than 100 languages.

TED.com currently hosts over 2,400 talks, with new additions daily.
There are 3,400 Youtube TED talks on the official TED channel.
The TedX Youtube channel hosts over 90,000 videos, with new additions daily.
"""
generator = pipeline(
    text,
    voice="af_heart",
    #  split_pattern=r"(?:\r?\n)|(?<=\.)\s+(?=[A-Z])"
    split_pattern=None,
)
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i == 0))
    sf.write(f"{i}.wav", audio, 24000)
    end = time.time()
    print('Duration:',end - start)
    print(audio)
