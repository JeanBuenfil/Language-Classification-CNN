# Language-Classification-CNN

This repository contains the methodology employed to train a image classification model, with the goal of language classification (specifically mexican spanish and mayan).  
This methodology includes deciding the use of image classification, getting the audios and relevant code for audio management and training

# Folder Structure
## audios
Contains the audio files used in the different experiments separated into the different languages  
NOTE: Due to github's file size restrains, nahuatl audios before clipping couldn't be included in the repository, the link to the original corpus and the processing done to it can be found in a following section

### mayan
original: Contains the original audio files before any kind of processing besides from clipping  
audios_maasab_with_silence: Contains the audios from the Maasab corpus  
audios_maasab_no_silence: Contains the audios from the Maasab corpus after using the audacity macro "trim_silence_audacity.txt" to remove the silence  
mayan_one_second: Contains all the one second clips from the original audios  
mayan_one_second_without_maasab_corpus: Contains all the one second clips from the original audios excluding the ones from the maasab corpus  

### nahuatl
nahuatl_ten_seconds: Contains all the ten seconds clips from the selected audios  
nahuatl_one_second: Contain all the one second clips from the ten seconds clips  
chosen_one_speaker_audios.txt: Contains the info about the audios selected from the original corpus after separating the speakers into different channels that were after clipped and stored in the folder nahuatl_ten_seconds  

### spanish
original: Contains the original audio files before any kind of processing  
spanish_no_silence: Contains the audios after using the audacity macro "trim_silence_audacity.txt" to remove the silence  
spanish_no_silence_without_maasab: Contains the audios after using the audacity macro "trim_silence_audacity.txt" to remove the silence excluding the ones from the maasab corpus  

## spectrograms
Contains the spectrograms used in the different experiments separated into the different languages

Every audio folder has its corresponding spectrogram folder

## helpers
Contains a variety of scripts/macros to help modify audio

remove_silence.py: Removes the silence from the audios using the Librosa library  
split_audios.py: Splits the audios given a specific length  
trim_silence_audacity.txt: Macro than can be imported into the software "Audacity" to remove the silence from the audios   

NOTE: Is recommended to use trim_silence_audacity.txt over remove_silence.py to remove silences as it was more effective during the experiments.

## graphs
Contains the confusion matrix and plot of the training of most of the experiments  
Graphs are separated into folders with the name of the languages the training was done for  
Graphs are identified as [ID_Exp]_plot and [ID_Exp]_matrix, where [ID_Exp] is the unique identifier of the experiment according to the experiment log (log.csv)  

## Pueble-Nahuatl-Manifest
Contains all the official documentation for the Puebla-Nahuatl corpus

## log.csv
Log containing relevant details of the experiments, the fields are the following:  
ID_Exp: Unique auto incremental identifier for every experiment  
Hash: Unique hash generated from the different parameters used in the experiment  
Languages: List of languages included in the experiment  
Model: Model used in the training  
Audio_Length: Audio lenght of every audio (if audio length wasn't standarized the field is left blank)  
Image_Amount: Number of image for every language included in the experiment  
Parameters: Variety of parameters that can be ajusted for every experiment (audio frecuency, learning rate, batch size, etc.)  
Split_Seed: Seed used at the moment of splitting the images into training and testing  
hasSilence: "True" if audios weren't processed to remove the silence, "False" if the silence was removed from the audios  
Results: Acuraccy, Recall, Precision and F1 values of the experiment  
Audio_Folders_Used: List of paths to the audio folders used in the experiment   
Notes: Additional observations  

All fields except for Notes are recorded automatically when following the training process as stated in "Audio Classification (CNN).ipynb"

# Image classification

The training was done using the spectograms images extracted from the audios, the reason why spectrograms were used is because working with them it means there is no need for annotations or transcriptions, and previous works using this characteristic exist [1].

# Corpora
The public domain audios can be found in the following sources:

## Mexican Spanish
Ciempiess-Balance (https://mega.nz/folder/k0QTyA6C#iRwTHwEqjYdlOlAVbMSFKw) [2].  
The CIEMPIESS BALANCE is a Radio Corpus designed to create acoustic models for automatic speech recognition and it is made up by recordings of spontaneous conversations in Mexican Spanish between a radio moderator and his guests.   
The CB is made up by 8555 audio files with transcripts. 2447 of those files (28.6%) come from male speakers and 6108 files (71.39%) come from female speakers.  
1317 audio files were selected at random from the original 8555 while trying to keep a balance between male and female speakers, resulting in 778 audio files with female speakers and 539 audio files from male speakers.   

## Mayan
T'aantsil (https://taantsil.com.mx/) [3].  
The T’aanTsil is an audio corpus in the Yucatec Maya language, created with the aim of supporting the development of speech technologies for indigenous languages. It is part of a broader effort to document and preserve native languages through digital resources.


Yucatec Maya DoReCo (https://sharedocs.huma-num.fr/wl/?id=OEm80dNUe88cfpejRnhmFE5IIFeGdCp9) [4].   
The Yucatec Maya DoReCo dataset was compiled by Stavros Skopeteas in 2015 and further processed for DoReCo by Alejandra Camelo Cruz, Ludger Paschen, and Matthew Stave between 2019 and 2022. The files that the Yucatec Maya DoReCo dataset are based on are part of a larger collection of Stavros Skopeteas's Yucatec Maya data that is archived at TLA (https://hdl.handle.net/1839/00-0000-0000-0021-E91B-F). This dataset is made up by 10 audios files, with a length between 6 and 13 minutes, splitting the audios resulted in a total of 792 clips used in the training.

## Nahuatl
Puebla-Nahuatl (https://www.openslr.org/92) [6]  
Documentation about this corpus can be found in the folder Pueble-Nahuatl-Manifest  
The audios used in the training were all taken from the "Medicina_79" subfolder from the original corpus, resulting in 79 audios, most of these audios have two speakers, each one recorded in a different channel, so the audios were separated into two (one for each channel) using Audacity
Audio length was between one minute to more than one hour, so using "Pueble-Nahuatl-Manifest\Metadata\Metadata_Cuetzalan-954-Recordings.xml" some audios were selected based on gender and length to be clipped into 10 seconds clips. This resulted in a total of 1747 clips, by checking "chosen_one_speaker_audios.txt" from the audios/nahuatl subfolder is possible to know which audios were chosen to clip into 10 seconds. 


# Model training

The file "Audio Classification (CNN).ipynb" found in the repository contains all the necessary code to perform the training once the audio data has been collected. The contents of the notebook include an adaptation and some additional functions based on the one found at https://github.com/jeffprosise/Deep-Learning/blob/master/Audio%20Classification%20(CNN).ipynb [5].

Although the "Audio Classification (CNN).ipynb" file describes the procedure for a single model, is possible to use different models for the training process. The current base model can be identified by examining the following code snippet:


```
from tensorflow.keras.applications import ResNet152V2  
from tensorflow.keras.applications.resnet import preprocess_input

base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

To perform training with other base models, it's necessary to change the imports  
"ResNet152V2"  
where it can be replaced with any of the official keras models list found in : https://keras.io/api/applications/ [7]  

and "tensorflow.keras.applications.resnet" where "resnet" has to be replaced according to the model used and the offical keras applications list: https://www.tensorflow.org/api_docs/python/tf/keras/applications [8]


# References
[1] Mukherjee, H., Ghosh, S., Sen, S., Sk Md, O., Santosh, K. C., Phadikar, S., & Roy, K. (2019). Deep learning for spoken language identification: Can we visualize speech signal patterns?. Neural Computing and Applications, 31, 8483-8501.   
[2] Ciempiess-unam. Retrieved from https://ciempiess.org/downloads   
[3] T’aantsil. Retrieved from https://taantsil.com.mx/   
[4] Skopeteas, Stavros, Amedee Colli Colli, Daniela Schellenbach, Carolin Brokmann, Florian Fischer and Maya Gálvez Wimmelmann. 2024. Yucatec Maya DoReCo dataset. In Seifart, Frank, Ludger Paschen and Matthew Stave (eds.). Language Documentation Reference Corpus (DoReCo) 2.0. Lyon: Laboratoire Dynamique Du Langage (UMR5596, CNRS & Université Lyon 2). (https://doi.org/10.34847/nkl.9cbb3619).   
[5] Jeff Prosise (2021). Deep-Learning [Software]. GitHub. https://github.com/jeffprosise/Deep-Learning/blob/master/Audio%20Classification%20(CNN).ipynb   
[6]  Highland Puebla Nahuatl speech translation corpus for endangered language documentation. Shi, Jiatong and Amith, Jonathan D and Chang, Xuankai and Dalmia, Siddharth and Yan, Brian and Watanabe, Shinji. Proceedings of the First Workshop on Natural Language Processing for Indigenous Languages of the Americas. 53--63, 2021  
[7] Keras. Retrieved from https://keras.io/api/applications/  
[8] Tensorflow. Retrieved from https://www.tensorflow.org/api_docs/python/tf/keras/applications  