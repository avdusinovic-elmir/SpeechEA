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

## Library Information

All algorithms and experiments were implemented using Python 3.8.20 [1] with NumPy 1.24.3 [2], Keras 2.13.1 [3], Torch 2.4.1 [4], and Scikit-learn 1.3.2 [5] libraries.


[1] Van Rossum, G., Drake, F.L.: Python 3 Reference Manual. CreateSpace, Scotts
Valley, CA (2009)

[2] Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cour-
napeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N.J., Kern, R., Picus, M., Hoyer,
S., van Kerkwijk, M.H., Brett, M., Haldane, A., del Río, J.F., Wiebe, M., Peter-
son, P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H.,
Gohlke, C., Oliphant, T.E.: Array programming with numpy. Nature 585(7825),
357–362 (Sep 2020). https://doi.org/10.1038/s41586-020-2649-2

[3] Chollet, F., et al.: Keras documentation. keras.io 33 (2015), https://github.com/
fchollet/keras

[4] Contributors, P.: Pytorch documentation, https://docs.pytorch.org/docs/
stable/index.html

[5] Kramer, O., Kramer, O.: Scikit-learn. Machine learning for evolution strategies pp.
45–53 (2016)