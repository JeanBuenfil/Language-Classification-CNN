import librosa
import numpy as np
import soundfile as sf
import os

def trim_all_silences(y, top_db=30):
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
                y, sr = librosa.load(input_path, sr=None)
                y_trimmed = trim_all_silences(y, top_db=top_db)
                sf.write(output_path, y_trimmed, sr)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error with {filename}")

input_folder = ""
output_folder = ""
process_folder(input_folder, output_folder, top_db=44)

