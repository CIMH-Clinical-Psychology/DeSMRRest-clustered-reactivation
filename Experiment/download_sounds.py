1# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:20:31 2021

This file downloads text-to-speech from google servers

@author: Simon Kern
"""

import os
from google.cloud import texttospeech
from scipy.io import wavfile

# file needs to be present
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../google-cloud-creds.json'



def save_tts_file(word, filename, language_code="en-US"):
    # Instantiates a client
    tts_client = texttospeech.TextToSpeechClient()
    # translate_client = translate_v2.Client()

    # Set the text input to be synthesized

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    synthesis_input = texttospeech.SynthesisInput(text=word)

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    # The response's audio_content is binary.
    with open(f"{filename}", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)

#%%


words = ospath.list_files('stimuli', exts=['png', 'jpg'], subfolders=True)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
lengths = []

for word in words:
    word = ospath.splitext(ospath.basename(word))[0]
    if 'example' in word: continue
    secs = save_tts_file(word, f"de_{word}.wav")
    lengths.append(secs)


save_tts_file("Fehler, Warnung", "error.wav")

print(f'Lengths DE: {", ".join([f"{s:.1f}" for s in lengths])}')
print(f"lengths DE: {min(lengths):.2f}-{max(lengths):.2f} seconds")

#%% Now do the same but translated
# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
lengths = []

for word in words:
    word = ospath.splitext(ospath.basename(word))[0]
    word_en = translate_client.translate(word, source_language='de')['translatedText']
    if word_en=='writing desk': word_en="desk"
    if word_en=='automobile': word_en="car"

    if 'example' in word: continue
    secs = save_tts_file(word_en, f"en_{word}.wav", language_code="en-EN-D")
    lengths.append(secs)


save_tts_file("Error, warning", "error.wav")

print(f'Lengths EN: {", ".join([f"{s:.1f}" for s in lengths])}')
print(f"lengths EN: {min(lengths):.2f}-{max(lengths):.2f} seconds")


#%%
lengths = []

for word in words:
    word = ospath.splitext(ospath.basename(word))[0]
    word_en = translate_client.translate(word, source_language='de', target_language='tr')['translatedText']
    if 'example' in word: continue
    secs = save_tts_file(word_en, f"tr_{word}.wav", language_code="tr-TR-D")
    lengths.append(secs)


save_tts_file("Error, warning", "error.wav")

print(f'Lengths EN: {", ".join([f"{s:.1f}" for s in lengths])}')
print(f"lengths EN: {min(lengths):.2f}-{max(lengths):.2f} seconds")
