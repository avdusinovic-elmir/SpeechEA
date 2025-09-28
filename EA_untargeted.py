import torch
import numpy as np
import random
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio.transforms as T
import torchaudio.functional as F
import time
import soundfile as sf
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/wav2vec2-large-960h"  # Pretrained ASR model
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

class Individual:
    def __init__(self, solution, check_similarity, perceptual_fitness):
        self.solution = solution
        self.check_similarity = check_similarity
        self.perceptual_fitness = perceptual_fitness


class EA:
    def __init__(self, pop, elits, mutatation_range, epsilon, folder_path, sample_name="unknown"):
        self.pop = pop
        self.elits = elits
        self.mutation_range = mutatation_range
        self.epsilon = epsilon
        self.sample_name = sample_name
        self.log_buffer = []
        self.folder_path = folder_path
        
        os.makedirs(self.folder_path+"logs2", exist_ok=True)
        
        self.log_path = f"{self.folder_path}logs2/log_{self.sample_name}.txt"
        with open(self.log_path,"w") as f:
            f.write(f"Starting untargeted EA for sample: {self.sample_name}\n")
            
    
    def log(self, message):
        self.log_buffer.append(message)
    
    def clip_audio(self, combination, original):
        clipped_combination = torch.clamp(combination, -1, 1)

        adv_noise = clipped_combination - original
        return adv_noise.squeeze().numpy()
        

    def preprocess_audio(self, org_audio):
        org_audio = org_audio.squeeze().numpy()

        preprocessed_audio = processor(org_audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)

        return preprocessed_audio

    def transcript_audio(self, audio_file):
        audio_file = self.preprocess_audio(audio_file)
        with torch.no_grad():
            logits = model(audio_file).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription

    def perceptual_loss_combined(self, original_audio, adversarial_noise):
        mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,  
            win_length=400,  
            hop_length=160,  
            n_mels=80  
        )

        original_mel = mel_transform(original_audio)
        adversarial_mel = mel_transform(adversarial_noise)

        log_original = torch.log(original_mel + 1e-6)
        log_adversarial = torch.log(adversarial_mel + 1e-6)
        mel_loss = torch.nn.functional.mse_loss(log_original, log_adversarial)
        self.log(f"Mel loss: {mel_loss.squeeze().numpy()}")

        waveform_loss = torch.nn.functional.mse_loss(original_audio, adversarial_noise)

        total_loss = 0.7 * mel_loss + 0.3 * waveform_loss

        return total_loss.item()

    def generate_population(self, original, pop):
        population = []
        size = len(original[0])

        for _ in range(pop):
            new_solution = np.zeros(size, dtype=np.float32)

            new_solution[0:16000] = np.random.uniform(
                -self.epsilon, self.epsilon, 16000
            ).astype(np.float32)

            envelope = torch.abs(torch.tensor(original.clone().detach(), dtype=torch.float32))  
            new_solution *= envelope.squeeze().numpy()  

            noise_tensor = torch.tensor(new_solution, dtype=torch.float32)
            filtered_noise = F.bandpass_biquad(noise_tensor, 16000, central_freq=6000, Q=0.707)
            
            population.append(Individual(solution=filtered_noise, check_similarity=True, perceptual_fitness=None))

        return population

    def sort_population(self, original, population, epoch):
        words = set()
        original_transcript = self.transcript_audio(original)
        for indv in population:
            combination = original + indv.solution
            combination = torch.clamp(combination, -1, 1)
            # print(indv.solution)
            adversarial_text = self.transcript_audio(combination)
            words.add(adversarial_text)

            indv.check_similarity = original_transcript.upper()==adversarial_text.upper()
            indv.perceptual_fitness = self.perceptual_loss_combined(original, combination)

        
        population.sort(key=lambda x: (x.check_similarity, x.perceptual_fitness))
        
        self.log(f"Epoch {epoch}: Similarity check ={population[0].check_similarity}, Percep={population[0].perceptual_fitness:.4f}")
        self.log(f"Unique transactions: "+", ".join(sorted(words)))

        return population

    def selection(self, population):
        for i in range(self.elits):
            population.pop()
        return population

    def crossover(self, population):
        for i in range(self.elits):
            parent1 = population[i].solution
            parent2 = population[(i + 1) % self.elits].solution

            mask = np.random.randint(0, 2, size=parent1.shape).astype(bool)
            child = np.where(mask, parent1, parent2)
            new_ind = Individual(solution=child, check_similarity=True, perceptual_fitness=None)
            population.append(new_ind)
        return population

    def mutation(self, population, original):
        for indv in population[-self.elits:]:
            random_array = np.zeros(len(indv.solution), dtype=np.float32)
            for i in range(0, 16000):
                if random.random() > 0.3:
                    random_array[i] += np.random.uniform(-self.mutation_range, self.mutation_range)

            envelope = np.abs(indv.solution)
            random_array *= envelope
            noise_tensor = torch.tensor(random_array, dtype=torch.float32)
            filtered_noise = F.bandpass_biquad(noise_tensor, 16000, central_freq=6000, Q=0.707)

            indv.solution += filtered_noise.numpy()
            
        return population

    def attack_speech(self, org, epochs):
        population = self.generate_population(org, self.pop)
        final_epoch = 0

        start_attack = time.time()
        for _ in range(epochs):

            print("Epoch:" + str(_))
            final_epoch = _

            b0 = time.time()
            population = self.sort_population(org, population, _)
            e0 = time.time()
            self.log(f"Sort Duration (in seconds): {e0-b0}")
            
            # print("SORT_POPULATION: ", e0 - b0)

            if not population[0].check_similarity:
                self.log(f"WORD ACHIEVED AT EPOCH {_}")
                print("We reached our destination! OLLAAAAAA")
                break

            # Stop if fitness of one individual is 0
            b1 = time.time()
            population = self.selection(population)
            # print("POPULATION SIZE after selection: ", len(population))
            e1 = time.time()
            self.log(f"Selection Duration (in seconds): {e1-b1}")

            b2 = time.time()
            population = self.crossover(population)
            # print("POPULATION SIZE after crossover: ", len(population))
            e2 = time.time()
            self.log(f"Crossover Duration (in seconds): {e2-b2}")
            # print("POPU aft Cross: ", population)
            b3 = time.time()
            population = self.mutation(population, org)
            # print("POPULATION SIZE after mutation: ", len(population))
            e3 = time.time()
            self.log(f"Mutation Duration (in seconds): {e3-b3}")
        
        end_attack = time.time()
        total_duration = end_attack - start_attack
        self.log(f"Similarity check: {population[0].check_similarity}")
        self.log(f"Perceptual loss: {population[0].perceptual_fitness}")
        self.log(f"Attack Duration (in seconds): {total_duration}")
        result = org + population[0].solution
        result = torch.clamp(result, -1, 1)

        with open(self.log_path, "w") as f:
            f.write("\n".join(self.log_buffer))
        return (result, population[0].solution, population[0].check_similarity,
                final_epoch, population[0].perceptual_fitness)
