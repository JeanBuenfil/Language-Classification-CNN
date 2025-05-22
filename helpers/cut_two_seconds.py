import os
from pydub import AudioSegment
import librosa

def remove_first_two_seconds_from_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"): 
            input_audio = os.path.join(input_folder, file)
            output_audio = os.path.join(output_folder, file)

            audio = AudioSegment.from_file(input_audio)
            trimmed_audio = audio[2000:len(audio)-2000]  # Trim first and last 2 seconds (2000 ms)

            trimmed_audio.export(output_audio, format="wav")
            print(f"Procesado: {file}")

remove_first_two_seconds_from_folder("audiosMaya/medico_maya_prueba", "audiosMaya/medico_maya_sin_silencio")

