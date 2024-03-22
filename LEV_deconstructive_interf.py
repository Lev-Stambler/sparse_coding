#!/usr/bin/env python
# coding: utf-8

# In[1]:


from huggingface_hub import hf_hub_download
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


from transformer_lens import HookedTransformer
model_name = "EleutherAI/pythia-70m-deduped"

model = HookedTransformer.from_pretrained(model_name, device=device)


# In[4]:


# Downnload dataset
from datasets import Dataset, load_dataset
dataset_name = "JeanKaddour/minipile"
token_amount= 40
#TODO: change train[:1000] to train if you want whole dataset
# 100_000 datasets
# I think that we want to use the full 100_000 at some point...
# dataset = load_dataset(dataset_name, split="train[:100000]").map(
dataset = load_dataset(dataset_name, split="train[:10000]").map( # 1_000 to get started
    lambda x: model.tokenizer(x['text']),
    batched=True,
).filter(
    lambda x: len(x['input_ids']) > token_amount
).map(
    lambda x: {'input_ids': x['input_ids'][:token_amount]}
)
# TODO: we can maybe make this faster for the larger dataset?


# In[5]:


setting = "mlp_out"

def get_cache_name_neurons(layer: int):
    if setting == "residual":
        cache_name = f"blocks.{layer}.hook_resid_post"
        neurons = model.cfg.d_model
    elif setting == "mlp":
        cache_name = f"blocks.{layer}.mlp.hook_post"
        neurons = model.cfg.d_mlp
    elif setting == "attention":
        cache_name = f"blocks.{layer}.hook_attn_out"
        neurons = model.cfg.d_model
    elif setting == "mlp_out":
        cache_name = f"blocks.{layer}.hook_mlp_out"
        neurons = model.cfg.d_model
    else:
        raise NotImplementedError
    return cache_name, neurons


# In[6]:


n_layers = model.cfg.n_layers
model.cfg.d_model, n_layers


# # Get Dictionary Activations

# In[7]:


# TODO: in chunks...
# Now we can use the model to get the activations
from torch.utils.data import DataLoader
from datasets import DatasetDict
from tqdm.auto import tqdm
from einops import rearrange
import math

# MAX_CHUNK_SIZE = 1_000

# TODO: move to a separate file or something
def get_activations(layer: int):
    datapoints = dataset.num_rows
    embedding_size = model.cfg.d_model
    activations_final = np.memmap(f'layer-{layer}.mymemmap', dtype='float32', mode='w+', shape=(datapoints, token_amount, embedding_size))
    batch_size = 32

    with torch.no_grad(), dataset.formatted_as("pt"):
        dl = DataLoader(dataset["input_ids"], batch_size=batch_size)
        cache_name = get_cache_name_neurons(layer)[0]
        for i, batch in enumerate(tqdm(dl)):
            # print(batch)
            _, cache = model.run_with_cache(batch.to(device))
            # print("AA", cache[cache_name].shape)
            # batched_neuron_activations = rearrange(cache[cache_name], "b s n -> (b s) n" )

            real_batch_size = batch.shape[0]
            activations_final[i*batch_size:i*batch_size + real_batch_size, :, :] = cache[cache_name].cpu().numpy()
    return activations_final

dict_activations = [get_activations(layer) for layer in range(n_layers)]


# ## Get activations for a specific feature and visualize them

# In[8]:


layer = 0
dict_activations[0].shape, dict_activations[layer].reshape(-1, dict_activations[layer].shape[-1]).shape


# In[9]:


from interp_utils import *
import torch
import numpy as np

# Get the activations for the best dict features
def get_feature_datapoints_with_idx(feature_index, dictionary_activations, tokenizer, token_amount, dataset, k=10, setting="max"):
    if len(dictionary_activations.shape) == 3:
        best_feature_activations = dictionary_activations[:, :, feature_index].flatten()
    else:
        best_feature_activations = dictionary_activations
    # Sort the features by activation, get the indices
    if setting=="max":
        found_indices = np.argsort(best_feature_activations)[:k]
        # found_indices = np.argsort(best_feature_activations, descending=True)[:k]
    elif setting=="uniform":
        # min_value = torch.min(best_feature_activations)
        min_value = np.min(best_feature_activations)
        max_value = np.max(best_feature_activations)

        # Define the number of bins
        num_bins = k

        # Calculate the bin boundaries as linear interpolation between min and max
        bin_boundaries = np.linspace(min_value, max_value, num_bins + 1)

        # Assign each activation to its respective bin
        # TODO: hmm
        # np bucketize?
        # bins = torch.bucketize(best_feature_activations, bin_boundaries)
        bins = np.digitize(best_feature_activations, bin_boundaries)

        # Initialize a list to store the sampled indices
        sampled_indices = []

        # Sample from each bin
        for bin_idx in np.unique(bins):
            if(bin_idx==0): # Skip the first one. This is below the median
                continue
            # Get the indices corresponding to the current bin
            bin_indices = np.array(np.nonzero(bins == bin_idx)).squeeze(axis=0)
            # print(bin_indices.shape)
            
            # Randomly sample from the current bin
            sampled_indices.extend(np.random.choice(bin_indices, size=1, replace=False))

        # Convert the sampled indices to a PyTorch tensor & reverse order
        found_indices = np.flip(np.array(sampled_indices), axis=[0])
    else: # random
        # get nonzero indices
        nonzero_indices = torch.nonzero(best_feature_activations)[:, 0]
        # shuffle
        shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0])]
        found_indices = shuffled_indices[:k]
    num_datapoints = int(dictionary_activations.shape[0])
    datapoint_indices =[np.unravel_index(i, (num_datapoints, token_amount)) for i in found_indices]
    text_list = []
    full_text = []
    token_list = []
    full_token_list = []
    for md, s_ind in datapoint_indices:
        md = int(md)
        s_ind = int(s_ind)
        full_tok = torch.tensor(dataset[md]["input_ids"])
        full_text.append(tokenizer.decode(full_tok))
        tok = dataset[md]["input_ids"][:s_ind+1]
        text = tokenizer.decode(tok)
        text_list.append(text)
        token_list.append(tok)
        full_token_list.append(full_tok)
    return text_list, full_text, token_list, full_token_list, found_indices


# ## Baseline before looking at "deconstructive interference"

# In[10]:


import interp_utils
import importlib
importlib.reload(interp_utils)

feature = 10
layer = 0

text_list, full_text, token_list, full_token_list, indices = get_feature_datapoints_with_idx(feature, dict_activations[layer], model.tokenizer, token_amount, dataset, setting="uniform")
interp_utils.visualize_text(text_list, feature, model, None, layer=layer, setting="model")


# ## Looking at constructive interference

# In[11]:


other_layers = list(range(1, n_layers))
concat_other_layers = np.concatenate([dict_activations[i] for i in other_layers], axis=-1)
concat_other_layers = concat_other_layers.reshape(-1, concat_other_layers.shape[-1])
concat_other_layers.shape, len(other_layers), dict_activations[0].shape


# In[12]:


import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

def apply_incremental_pca(X, pca_dims):
    # TODO: look at this...
    n_batches = 10  # Adjust based on your memory capacity
    ipca = IncrementalPCA(n_components=pca_dims)
    for X_batch in np.array_split(X, n_batches):
        ipca.partial_fit(X_batch)
    return ipca.transform(X)

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# KMeans with cosine similarity
def kmeans_cosine(X, k, pca_dims=None, iterations=100):
    # Normalize input data
    X_normalized = normalize(X, axis=1)

     # Apply PCA if specified
    if pca_dims is not None:
        X_normalized = apply_incremental_pca(X_normalized, pca_dims)
        print("DONE PCA-ing!")

    # Randomly initialize centroids
    n_samples, n_features = X_normalized.shape
    centroids = X_normalized[np.random.choice(n_samples, k, replace=False)]

    for iter in range(iterations):
        # Cluster assignment step
        clusters = [[] for _ in range(k)]
        for idx, x in enumerate(X_normalized):
            similarities = [cosine_similarity(x, centroid) for centroid in centroids]
            closest = np.argmax(similarities)
            clusters[closest].append(idx)

        # Update centroids
        # TODO: we maybe able to just **not use** PCA at all here.... slow it may be
        new_centroids = []
        for cluster in clusters:
            if cluster:  # Check if cluster is not empty
                new_centroid = np.mean(X_normalized[cluster], axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(np.random.rand(n_features))  # Reinitialize empty clusters

        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        print("Done with iteration", iter)

    return centroids, clusters

# TODO: can we speed this up??? Maybe we use PCA
_, cluster_by_idx = kmeans_cosine(concat_other_layers, iterations=50, k=7, pca_dims=None)


# In[ ]:


# In theory, the larger clusters should be the ones that are less important to care about??? Idk...
[len(c) for c in cluster_by_idx]


# In[ ]:


feature = 10
layer = 0

label_idx = 7
cluster = cluster_by_idx[label_idx]

activations = np.expand_dims(dict_activations[layer].reshape(-1, dict_activations[layer].shape[-1])[cluster, :], 1)
print(activations.shape)
text_list, full_text, token_list, full_token_list, indices = get_feature_datapoints_with_idx(feature, activations, model.tokenizer, token_amount, dataset, setting="uniform")
interp_utils.visualize_text(text_list, feature, model, None, layer=layer, setting="model")

