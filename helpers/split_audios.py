import os
from pydub import AudioSegment

def split_audio(input_audio, output_folder, clip_length):
    audio = AudioSegment.from_wav(input_audio)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Calcule amount of clips
    num_clips = len(audio) // clip_length
    
    filename = os.path.splitext(os.path.basename(input_audio))[0]
    for i in range(num_clips + 1):
        start_time = i * clip_length
        end_time = start_time + clip_length
        
        
        clip = audio[start_time:end_time]
        
        
        output_path = os.path.join(output_folder, f'{filename}_clip_{i + 1}.wav')
        
        # Save clip
        if len(clip) == 10000: #milliseconds
            clip.export(output_path, format='wav')
            print(f'Clip guardado: {output_path}')


input_folder = ''
output_folder = ''
clip_length = 10000
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)

    if file.lower().endswith('.wav'):
        input_audio = file_path
        split_audio(input_audio, output_folder, clip_length)

