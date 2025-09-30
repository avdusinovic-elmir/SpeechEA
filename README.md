# Speech-EA: Evolutionary Algorithm-based Attack on Automatic Speech Recognition Systems
This repository contains the code for the paper "Speech-EA: Evolutionary Algorithm-based Attack on Automatic Speech Recognition Systems."

## Repository Structure
- original_audio - Contains the 10 command folders and their respective 100 audio samples per command.
- adversarial_audios - Contains the folders for the 10 runs and the results resulting from the attack (adversarial audio, waveform graph, Mel-spectrogram difference graph and the log files).
- runSamples.py - Python source code file to start the Speech-EA attack and save the achieved result.
- EA_untargeted.py - Python source code file containing the Speech-EA attack and its evaluation functions.
- All_Runs.xlsx - Consists of an Excel file containing all the results achieved through out the 10 runs.

## Reproducibility Notice

Please note that **Speech-EA includes inherent randomness**. This means that running the code multiple times may lead to results that are different then the ones achieved in the paper.

To improve reproducibility, consider setting a fixed seed for all of the functions that use randomness. 

For any questions or contributions, feel free to open an issue or a pull request.
