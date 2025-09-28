import os

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torchaudio.functional as F

import tensorflow as tf
print("TensorFlow version:", tf.version)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Using GPU:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

from EA_untargeted import EA

print("Yes run")

command = "yes"
folder = "../original_audio/"+command
results_folder = "audio_untargeted/"+command+"/"
dir_list = os.listdir(folder)
for file in dir_list:
    audio_file = folder+"/"+file
    if not os.path.isfile(audio_file) or file==".DS_Store":
        continue

    file_name = command+"_"+file.split(".")[0]
    population = 50
    elits = int(population*0.3)
    epochs = 1000
    mutatation_range = 0.7
    epsilon = 0.5

    speech_array, sampling_rate = torchaudio.load(audio_file)
    speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
    speech_array = speech_array.squeeze().numpy()
    print("speech_array", speech_array.shape)
    print("speech_array", speech_array)

    speech_array_tensor = torch.tensor(speech_array, dtype=torch.float32).unsqueeze(0)
    print("speech_array", speech_array_tensor.shape)
    print("speech_array", speech_array_tensor)

    sample_name = file_name
    attackEA = EA(pop=population, elits=elits, mutatation_range=mutatation_range,
                    epsilon=epsilon, folder_path=results_folder, sample_name=sample_name)

    result, noise, similarity, final_epoch, perceptual_loss = attackEA.attack_speech(org=speech_array_tensor, epochs=epochs)
    
    mel_adv = result
    result = result.squeeze().numpy()
    result_array = result
    result = processor(result, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
    with torch.no_grad():
        logits = model(result).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    result_text = processor.batch_decode(predicted_ids)[0]

    print("Transcription:", result_text)
    print("Result:", result_text)
    print("Similarity:", similarity)

    file_nbr = file_name.split("_")[1]

    plt.figure()
    plt.plot(result_array, label='Adversarial Audio')
    plt.plot(speech_array, label='Clean Audio')
    # plt.plot(noise, label='Adversarial Noise')
    plt.title(f"{result_text}_{perceptual_loss}")
    plt.legend()
    plt.savefig(f"{results_folder}{result_text}_{file_nbr}.png")
    plt.show()
    print("Result Type:", type(result))
    print("Result Shape:", result.shape)
    result = result.detach().cpu().numpy()
    result = np.asarray(result, dtype=np.float32)
    result = result.squeeze()
    print("Type of result:", type(result))

    if isinstance(result, np.ndarray):
        print("Shape of result:", result.shape)
        print("Data type of result:", result.dtype)
    sf.write(f"{results_folder}{result_text}_{file_nbr}.wav", result_array, sampling_rate)

    # Mel-Scale Representation
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,  
        win_length=400,  
        hop_length=160,  
        n_mels=80  
    )

    print(mel_adv.shape)
    original_mel = mel_transform(speech_array_tensor)
    adversarial_mel = mel_transform(mel_adv)
    log_adversarial = torch.log(adversarial_mel + 1e-6)
    log_original = torch.log(original_mel + 1e-6)
    adv_mel = log_adversarial.squeeze().numpy()
    orig_mel = log_original.squeeze().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(adv_mel, aspect='auto', origin='lower')
    plt.title(f"{result_text}_{perceptual_loss}")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(f"{results_folder}{result_text}_{file_nbr}_MEL.png")
    plt.close()

    diff = adv_mel - orig_mel  

    plt.figure(figsize=(10, 4))

    plt.imshow(orig_mel, aspect='auto', origin='lower', cmap='Greys')
    plt.imshow(diff, aspect='auto', origin='lower', cmap='bwr', alpha=0.6)

    plt.title("Original Mel with Differences (red=adversarial > original, blue=less)")
    plt.colorbar(label="Difference (adv - orig, log scale)")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bands")

    plt.tight_layout()
    plt.savefig(f"{results_folder}{result_text}_{file_nbr}_MEL_DIFF.png")
    plt.show()
