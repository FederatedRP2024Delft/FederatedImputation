import random

import numpy as np
import torch.utils.data

from vae.mnist_vae import VaeAutoencoderClassifier, ConditionalVae


def impute_naive(k, trained_vae:VaeAutoencoderClassifier, initial_dataset):
    # Return a dataset imputed k images from trained_vae
    generated_dataset = trained_vae.generate_data(n_samples=k)
    to_be_zipped = []
    for image_ind in range(k):
        to_be_zipped.append(
            (generated_dataset[0][image_ind], np.argmax(generated_dataset[1][image_ind].detach().numpy())))
    return torch.utils.data.ConcatDataset([initial_dataset, to_be_zipped])

def impute_cvae_naive(k, trained_cvae:ConditionalVae, initial_dataset):
    # Return a dataset imputed k images from trained_vae
    generated_dataset = []
    uniform_digits = [random.randint(0, 9) for _ in range(k)]
    for i in uniform_digits:
        generated_dataset.append((trained_cvae.generate_data(n_samples=1, target_label=i).squeeze(1), i))

    to_be_zipped = []
    for image_ind in range(k):
        to_be_zipped.append(
            (generated_dataset[image_ind][0], generated_dataset[image_ind][1]))
    return torch.utils.data.ConcatDataset([initial_dataset, to_be_zipped])