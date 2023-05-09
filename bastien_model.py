from spleeter.separator import Separator
import gradio as gr
import os
import random2

# Initiate a file separator with 2 stems (instruments and vocals)
separator = Separator('spleeter:2stems')



# Gradio function to split audio stems and return their filepaths
def extract_stems(audio):
    
    # initiate a folder for splitted files
    foldername = str(random2.randrange(100000000))

    # Separate audio input. Synchronous is true to wait for the end of split before going further
    separator.separate_to_file(audio, "output/", filename_format= foldername + "/{instrument}.wav", synchronous=True)
    
    # To get full filepath, need to clean the input audio filepath (ex : /tmp/audioexample.mp3)
    filepath = audio[5:] # remove first five chars aka /tmp/
    filepath = filepath[:max([idx for idx, x in enumerate(filepath) if x == '.'])] # remove extension
    
    vocals = f"./output/"+ foldername +"/vocals.wav"
    accompaniment = f"./output/"+ foldername +"/accompaniment.wav"
    
    return vocals, accompaniment

# Launch a Gradio interface
# Input is an audio file, returning his filepath
# Output is an also audio files

title = "Demo: Deezer Spleeter / Audio separation library"
description = "This demo is a basic interface for <a href='https://research.deezer.com/projects/spleeter.html' target='_blank'>Deezer Spleeter</a>. It uses the Spleeter library for separate audio file in two stems : accompaniments and vocals."
#examples = [["examples/" + mp3] for mp3 in os.listdir("examples/")]

demo = gr.Interface(
    fn=extract_stems, 
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=[gr.Audio(label="Vocals stem", source="upload", type="filepath"), gr.Audio(label="Accompaniment stem", source="upload", type="filepath")],
    title=title,
    description=description,
    #examples=examples,
    allow_flagging="never"
    )

demo.launch(server_name="0.0.0.0")
