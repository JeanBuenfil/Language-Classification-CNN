import librosa
import numpy as np
import soundfile as sf
import os

def trim_all_silences(y, top_db=30):
    """Elimina todos los segmentos silenciosos usando librosa.effects.split"""
    intervals = librosa.effects.split(y, top_db=top_db)
    y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
    return y_trimmed

def process_folder(input_dir, output_dir, top_db):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.wav', '.flac', '.mp3')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # Carga el audio sin cambiar la frecuencia de muestreo original
                y, sr = librosa.load(input_path, sr=None)
                y_trimmed = trim_all_silences(y, top_db=top_db)
                sf.write(output_path, y_trimmed, sr)
                print(f"Procesado: {filename}")
            except Exception as e:
                print(f"Error con {filename}")

input_folder = "audiosMaya/medico_maya_sin_silencio"
output_folder = "audiosMaya/medico_maya_sin_silencio10"
process_folder(input_folder, output_folder, top_db=44)

def remove_silence_automatically(audio):
    y, sr = librosa.load(audio, sr=16000)
    yt, index = librosa.effects.trim(y, top_db=30)
    print(librosa.get_duration(y=y, sr=sr), librosa.get_duration(y=yt, sr=sr))
    

#remove_silence_automatically("audiosMaya/medico_maya_con_silencio/000101_1555.wav")