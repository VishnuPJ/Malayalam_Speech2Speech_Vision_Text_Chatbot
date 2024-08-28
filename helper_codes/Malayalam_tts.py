# pip install -q torchaudio omegaconf
# pip install aksharamukha
import torch
from pprint import pprint
from omegaconf import OmegaConf
from IPython.display import Audio, display

torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)
models = OmegaConf.load('latest_silero_models.yml')

"""## List models"""

# see latest avaiable models
available_languages = list(models.tts_models.keys())
print(f'Available languages {available_languages}')

for lang in available_languages:
    _models = list(models.tts_models.get(lang).keys())
    print(f'Available models for {lang}: {_models}')

"""## V4"""

import torch

language = 'indic'
model_id = 'v4_indic'
device = torch.device('cuda')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)  # gpu or cpu

"""### Speakers"""

print(model.speakers)

"""### Text"""

from aksharamukha import transliterate
orig_text = "ചിത്രം വിശകലനം ചെയ്ത് പ്രധാന ഘടകങ്ങളെ ഉൾക്കൊള്ളുന്ന ഒരു വിവരണം നൽകുക. പ്രധാന വിഷയം, പശ്ചാത്തല വിശദാംശങ്ങൾ, വർണ്ണങ്ങൾ, പ്രവർത്തനങ്ങൾ, വികാരങ്ങൾ എന്നിവയിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കുക. വിവരണം സരളവും, സമഗ്രവും , ദൃശ്യത്തിന്റെ കൃത്യമായ വിശദീകരണം നൽകുന്നതുമായിരിക്കണം."
roman_text = transliterate.process('Malayalam', 'ISO', orig_text)
print(roman_text)

# roman_text2 = transliterate.process('Malayalam', 'ISO', "൧,൨,൩,൪,൫,൬,൭,൮,൯")

sample_rate = 48000
speaker = 'malayalam_female'
put_accent=True
put_yo=True

audio = model.apply_tts(text=roman_text,
                        speaker=speaker,
                        sample_rate=sample_rate,
                        put_accent=put_accent,
                        put_yo=put_yo)
display(Audio(audio, rate=sample_rate))

"""### SSML"""

ssml_sample = """
              <speak>
              <p>
                  ഞാൻ ഉണർന്നപ്പോൾ, <prosody rate="x-slow">ഞാൻ വളരെ പതുക്കെ സംസാരിക്കുന്നു</prosody>.
                  അതിന് ശേഷം ഞാൻ സാധാരണ ശബ്ദത്തിൽ സംസാരിക്കുന്നു,
                  <prosody pitch="x-high"> അല്ലെങ്കിൽ ഞാൻ ഉയർന്ന ശബ്ദത്തിൽ സംസാരിക്കാം </prosody>,
                  അല്ലെങ്കിൽ <prosody pitch="x-low">താഴത്തെ ശബ്ദത്തിൽ സംസാരിക്കാം</prosody>.
                  അതിന് ശേഷം, ഭാഗ്യം ഉണ്ടെങ്കിൽ – <prosody rate="fast">ഞാൻ വളരെ വേഗത്തിൽ സംസാരിക്കാം.</prosody>
                  എനിക്ക് ഏതെങ്കിലും ദൈർഘ്യമുള്ള ഇടവേളകൾ ചെയ്യാനും കഴിയും, ഉദാഹരണത്തിന്, രണ്ട് സെക്കന്റ് <break time="2000ms"/>.
                  <p>
                    എന്റെ ഇടവേളകൾ ഓരോ പാരഗ്രാഫുകൾക്കിടയിലും ചെയ്യാൻ കഴിയും.
                  </p>
                  <p>
                    <s>എനിക്ക് വാചകങ്ങൾക്കിടയിലും ഇടവേളകൾ ചെയ്യാൻ കഴിയും</s>
                    <s>ഇപ്പോൾ ഞാൻ എങ്ങനെ ചെയ്യുന്നുവെന്ന് കാണൂ</s>
                  </p>
              </p>
              </speak>
              """


sample_rate = 48000
speaker = 'malayalam_female'
audio = model.apply_tts(ssml_text=transliterate.process('Malayalam', 'ISO', ssml_sample),
                        speaker=speaker,
                        sample_rate=sample_rate)
display(Audio(audio, rate=sample_rate))