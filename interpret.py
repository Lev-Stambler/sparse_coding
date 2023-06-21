import asyncio
import importlib
import json
import os
import pickle
import requests
from typing import Any, Dict, Union

from baukit import Trace
from datasets import load_dataset
from einops import rearrange
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer

from argparser import parse_args
from utils import dotdict, make_tensor_name
from nanoGPT_model import GPT
from run import AutoEncoder

# set OPENAI_API_KEY environment variable from secrets.json['openai_key']
# needs to be done before importing openai interp bits
with open("secrets.json") as f:
    secrets = json.load(f)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, ActivationRecord, NeuronRecord, NeuronId
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.fast_dataclasses import loads

EXPLAINER_MODEL_NAME = "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003"

# Replaces the load_neuron function in neuron_explainer.activations.activations because couldn't get blobfile to work
def load_neuron(
    layer_index: Union[str, int], neuron_index: Union[str, int],
    dataset_path: str = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/collated-activations",
) -> NeuronRecord:
    """Load the NeuronRecord for the specified neuron."""
    url = os.path.join(dataset_path, str(layer_index), f"{neuron_index}.json")
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Neuron record not found at {url}.")
    neuron_record = loads(response.content)

    if not isinstance(neuron_record, NeuronRecord):
        raise ValueError(
            f"Stored data incompatible with current version of NeuronRecord dataclass."
        )
    return neuron_record


def make_feature_activation_dataset(cfg, model: HookedTransformer, feature_dict: torch.Tensor, use_baukit: bool = False, max_sentences: int = 10000):
    """
    Takes a dict of features and returns the top k activations for each feature in pile10k
    """
    sentence_dataset = load_dataset(cfg.dataset_name)
    sentence_dataset = sentence_dataset["train"]
    if max_sentences is not None and max_sentences < len(sentence_dataset):
        sentence_dataset = sentence_dataset.select(range(max_sentences))

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tensor_name = make_tensor_name(cfg)
    # make list of sentence, tokenization pairs

    print(f"Computing internals for all {len(sentence_dataset)} sentences")

    # Make dataframe with columns for each feature, and rows for each sentence fragment
    # each row should also have the full sentence, the current tokens and the previous tokens

    sentence_fragment_dicts = []
    for sentence_id, sentence in tqdm(enumerate(sentence_dataset)):
        tokens = tokenizer(sentence["text"], return_tensors="pt")["input_ids"].to(cfg.device)[:, : cfg.max_length]
        if use_baukit:
            with Trace(model, tensor_name) as ret:
                _ = model(tokens)
                mlp_activation_data = ret.output
                mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n").to(cfg.device)
                mlp_activation_data = nn.functional.gelu(mlp_activation_data)
        else:
            _, cache = model.run_with_cache(tokens)
            mlp_activation_data = cache[tensor_name].to(cfg.device)  # NOTE: could do all layers at once, but currently just doing 1 layer
            mlp_activation_data = rearrange(mlp_activation_data, "b s n -> (b s) n")

        # Project the activations into the feature space
        feature_activation_data = torch.matmul(mlp_activation_data, feature_dict.to(torch.float32))

        full_sentence = tokenizer.decode(tokens[0, 1:])
        for i in range(1, tokens.shape[1]):
            partial_str_dict: Dict[str, Any] = {}
            partial_str_dict["pre_tokens"] = tokens[0, 1 : i + 1].tolist()
            partial_str_dict["token_id"] = i
            partial_str_dict["sentence_id"] = sentence_id
            partial_str_dict["full_tokens"] = tokens[0, 1:].tolist()
            partial_str_dict["sentence"] = full_sentence
            partial_sentence = tokenizer.decode(tokens[0, 1 : i + 1])
            partial_str_dict["context"] = partial_sentence
            partial_str_dict["current_token"] = tokenizer.decode(tokens[0, i])
            if i < tokens.shape[1] - 1:
                partial_str_dict["next_token"] = tokenizer.decode(tokens[0, i + 1])
            else:
                partial_str_dict["next_token"] = ""

            if i != 1:
                partial_str_dict["prev_token"] = tokenizer.decode(tokens[0, i - 1])
            else:
                partial_str_dict["prev_token"] = ""

            for j in range(feature_dict.shape[1]):
                partial_str_dict[f"feature_{j}"] = feature_activation_data[i, j].item()
            # add a row into the dataframe for each sentence fragment
            sentence_fragment_dicts.append(partial_str_dict)

    df = pd.DataFrame(sentence_fragment_dicts)
    return df


async def interpret_neuron(neuron_record, n_examples_per_split: int= 5):
    slice_params = ActivationRecordSliceParams(n_examples_per_split=n_examples_per_split)
    train_activation_records = neuron_record.train_activation_records(slice_params)
    valid_activation_records = neuron_record.valid_activation_records(slice_params)

    explainer = TokenActivationPairExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        prompt_format=PromptFormat.HARMONY_V4,
        max_concurrent=1,
    )

    explanations = await explainer.generate_explanations(
    all_activation_records=train_activation_records,
    max_activation=calculate_max_activation(train_activation_records),
    num_samples=1,
)
    assert len(explanations) == 1
    explanation = explanations[0]
    print(f"{explanation=}")

    # Simulate and score the explanation.
    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=1,
            prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        )
    )
    scored_simulation = await simulate_and_score(simulator, valid_activation_records)
    return scored_simulation


async def main(cfg: dotdict):
    # Load model
    if cfg.model_name in ["gpt2", "EleutherAI/pythia-70m-deduped"]:
        model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        use_baukit = False
    elif cfg.model_name == "nanoGPT":
        model_dict = torch.load(open(cfg.model_path, "rb"), map_location="cpu")["model"]
        model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
        cfg_loc = cfg.model_path[:-3] + "cfg"  # cfg loc is same as model_loc but with .pt replaced with cfg.py
        cfg_loc = cfg_loc.replace("/", ".")
        model_cfg = importlib.import_module(cfg_loc).model_cfg
        model = GPT(model_cfg).to(cfg.device)
        model.load_state_dict(model_dict)
        use_baukit = True
    else:
        raise ValueError("Model name not recognised")
    
    # Load feature dict
    assert cfg.load_autoencoders is not None
    loaded_autoencoders = pickle.load(open(cfg.load_autoencoders, "rb"))
    autoencoder = loaded_autoencoders[0][0]
    feature_dict = autoencoder.encoder[0].weight.detach().t().to(cfg.device)

    if cfg.load_activation_dataset and os.path.exists(cfg.load_activation_dataset):
        base_df = pd.read_csv(cfg.load_activation_dataset)
    else:
        base_df = make_feature_activation_dataset(cfg, model, feature_dict, use_baukit=use_baukit, max_sentences=cfg.activation_sentences)
        # save the dataset
        base_df.to_csv(cfg.save_activation_dataset, index=False)

    df = base_df.copy()

    random_sentences = 10
    for feature_num in range(1, 6):
        # print previous, current, and next token the top scoring tokens
        sorted_df = df.sort_values(by=f"feature_{feature_num}", ascending=False)
        print(sorted_df.head(10)[["prev_token", "current_token", "next_token"]])

        random_activationrecord_list = []
        for sentence_id, sentence_df in df.groupby("sentence_id"):
            token_list = []
            activation_list = []
            if sentence_id > random_sentences: # this eventually gives > not supported between instances of str and int becau
                break
            for token_id, token_df in sentence_df.groupby("token_id"):
                token_list.append(token_df["current_token"].iloc[0])
                activation_list.append(token_df[f"feature_{feature_num}"].iloc[0])
            
            random_activationrecord_list.append(ActivationRecord(token_list, activation_list))

        # Now we want to get sentences with the highest activation, so we group by sentence_id and then get average activation
        # then we sort by activation and take the top 10

        n_top_sentences = 8
        sentence_averages = []
        for sentence_id, sentence_df in df.groupby("sentence_id"):
            sentence_averages.append(
                (sentence_id, sentence_df[f"feature_{feature_num}"].mean())
            )
        
        sentence_averages.sort(key=lambda x: x[1], reverse=True)
        top_sentence_ids = [x[0] for x in sentence_averages[:n_top_sentences]]

        most_positive_activationrecord_list = []
        for sentence_id in top_sentence_ids:
            sentence_df = df[df["sentence_id"] == sentence_id]
            token_list = []
            activation_list = []
            for token_id, token_df in sentence_df.groupby("token_id"):
                token_list.append(token_df["current_token"].iloc[0])
                activation_list.append(token_df[f"feature_{feature_num}"].iloc[0])
            
            most_positive_activationrecord_list.append(ActivationRecord(token_list, activation_list))
        
        neuron_id = NeuronId(layer_index=2, neuron_index=feature_num)

        neuron_record = NeuronRecord(neuron_id=neuron_id, random_sample=random_activationrecord_list, most_positive_activation_records=most_positive_activationrecord_list)
        n_examples_per_split = 2
        # note that we need n_sentences >= n_examples_per_split * 4 (numsplits)


        scored_simulation = await interpret_neuron(neuron_record, n_examples_per_split)
        print(f"score={scored_simulation.get_preferred_score():.2f}")

async def run_openai_example():
    neuron_record = load_neuron(9, 10)

    # Grab the activation records we'll need.
    slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
    train_activation_records = neuron_record.train_activation_records(
        activation_record_slice_params=slice_params
    )
    valid_activation_records = neuron_record.valid_activation_records(
        activation_record_slice_params=slice_params
    )

    # Generate an explanation for the neuron.
    explainer = TokenActivationPairExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        prompt_format=PromptFormat.HARMONY_V4,
        max_concurrent=1,
    )
    explanations = await explainer.generate_explanations(
        all_activation_records=train_activation_records,
        max_activation=calculate_max_activation(train_activation_records),
        num_samples=1,
    )
    assert len(explanations) == 1
    explanation = explanations[0]
    print(f"{explanation=}")

    # Simulate and score the explanation.
    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=1,
            prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        )
    )
    scored_simulation = await simulate_and_score(simulator, valid_activation_records)
    print(f"score={scored_simulation.get_preferred_score():.2f}")


if __name__ == "__main__":
    asyncio.run(run_openai_example())
    # cfg = parse_args()
    # asyncio.run(main(cfg))