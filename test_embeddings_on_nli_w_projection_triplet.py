from collections import defaultdict
from itertools import product
from math import ceil
import os
import sys
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


torch.manual_seed(42)
n_experiments = int(sys.argv[3])


class ProjectionMatrix(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=32):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim, bias=False)
        if torch.cuda.is_available():
            self.projection.cuda()
            self.using_cuda = True
        else:
            self.using_cuda = False

    def forward(self, x):
        return self.projection(x)
    

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1
                                                   ).expand(token_embeddings.size()).float()
    return torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def train_epoch(epoch_n, batch_size, data, layer, model, optimiser):
    data_size = data['premises'][layer].size(0)
    n_steps = ceil(data_size / batch_size)
    epoch_train_losses = torch.zeros(n_steps)
    for step_n in tqdm(range(n_steps), desc=f'Epoch {epoch_n+1}, training', leave=False):
        optimiser.zero_grad()
        lo = step_n * batch_size
        hi = lo + batch_size
        premises = data['premises'][layer][lo:hi]
        entailments = data['entailments'][layer][lo:hi]
        contradictions = data['contradictions'][layer][lo:hi]
        if model.using_cuda:
            premises = premises.cuda()
            entailments = entailments.cuda()
            contradictions = contradictions.cuda()
        
        # Squared Euclidean distances between premises and entailments
        # should be smaller than those between entailments and contradictions.
        projected_premises = model(premises)
        projected_entailments = model(entailments)
        projected_contradictions = model(contradictions)
        
        distances_to_entailments = (
            projected_premises - projected_entailments
        ).square().sum(axis=1)
        distances_to_contradictions = (
            projected_premises - projected_contradictions
        ).square().sum(axis=1)
        
        loss = torch.relu(distances_to_entailments - distances_to_contradictions).mean() 
        loss.backward()
        optimiser.step()
        epoch_train_losses[step_n] = loss.item()
    return epoch_train_losses.mean().item()


def validate_epoch(epoch_n, batch_size, data, layer, model):
    data_size = data['premises'][layer].size(0)
    n_steps = ceil(data_size / batch_size)
    all_differences = []
    for step_n in tqdm(range(n_steps), desc=f'Epoch {epoch_n+1}, validation', leave=False):
        lo = step_n * batch_size
        hi = lo + batch_size
        premises = data['premises'][layer][lo:hi]
        entailments = data['entailments'][layer][lo:hi]
        contradictions = data['contradictions'][layer][lo:hi]
        if model.using_cuda:
            premises = premises.cuda()
            entailments = entailments.cuda()
            contradictions = contradictions.cuda()
        
        # Squared Euclidean distances between premises and entailments
        # should be smaller than those between entailments and contradictions.
        with torch.no_grad():
            projected_premises = model(premises)
            projected_entailments = model(entailments)
            projected_contradictions = model(contradictions)
        
        distances_to_entailments = (
            projected_premises - projected_entailments
        ).square().sum(axis=1)
        distances_to_contradictions = (
            projected_premises - projected_contradictions
        ).square().sum(axis=1)
        all_differences.extend(
            (distances_to_contradictions - distances_to_entailments).flatten().tolist())
    return (torch.tensor(all_differences) > 0.0).double().mean().item()


def encode_batch_w_mean_pooling(batch, tokeniser, model):
    """
    Return mean-pooled representation of all layers.
    """
    result = defaultdict(list)
    tokenisation = tokeniser(batch, return_tensors='pt', max_length=512,
                            padding='longest', truncation=True)
    model_inputs = {k: v.cuda() for k, v in tokenisation.items()}
    hidden_states = model(
        **model_inputs, output_hidden_states=True).hidden_states
    for layer in range(len(hidden_states)):
        result[layer] = mean_pooling(
            hidden_states[layer], model_inputs['attention_mask']
        ).detach().cpu()
    return result


def encode_sentences_w_mean_pooling(job_name, sentences, batch_size, tokeniser, model):
    n_steps = ceil(len(sentences) / batch_size)
    result = defaultdict(list)
    for step_n in tqdm(range(n_steps), desc=job_name, leave=False):
        lo = step_n * batch_size
        hi = lo + batch_size
        batch = sentences[lo:hi]
        by_layer_encodings = encode_batch_w_mean_pooling(
            batch, tokeniser, model)
        for k, v in by_layer_encodings.items():
            result[k].append(v)
    return {
        k: torch.concat(v, dim=0)
        for k, v in result.items()
    }


def encode_sentences_w_gpt(job_name, sentences, tokeniser, model):
    results = defaultdict(list)
    for sentence in tqdm(sentences, desc=job_name, leave=False):
        tokenisation = tokeniser(
            sentence,
            truncation=True,
            # To avoid OOM errors with larger models. The number of exx longer than 200 tokens
            # is very limited.
            max_length=256,
            return_tensors='pt')
        inputs = {k: v.cuda() for k, v in tokenisation.items()}
        outputs = model(**inputs, output_hidden_states=True).hidden_states
        for layer_n in range(len(outputs)):
            results[layer_n].append(
                outputs[layer_n][0, -1].reshape(1, -1).detach().cpu())
    return {
        k: torch.concat(v, dim=0)
        for k, v in results.items()
    }


def encode_sentences_w_gpt_w_mean_pooling(job_name, sentences, tokeniser, model):
    results = defaultdict(list)
    for sentence in tqdm(sentences, desc=job_name, leave=False):
        tokenisation = tokeniser(
            sentence,
            truncation=True,
            return_tensors='pt')
        inputs = {k: v.cuda() for k, v in tokenisation.items()}
        outputs = model(**inputs, output_hidden_states=True).hidden_states
        for layer_n in range(len(outputs)):
            results[layer_n].append(
                mean_pooling(outputs[layer_n].cpu(), inputs['attention_mask'])
            )
    return {
        k: torch.concat(v, dim=0)
        for k, v in results.items()
    }


def main():
    task = sys.argv[1]  # snli / anli / mnli
    model_name = sys.argv[2]
    model_name_path = model_name.replace('/', '_')
    # Encode everything
    data = {
        'train': {'premises': [], 'entailments': [], 'contradictions': []},
        'dev': {'premises': [], 'entailments': [], 'contradictions': []},
        'test': {'premises': [], 'entailments': [], 'contradictions': []}
    }
    for split in ['train', 'dev', 'test']:
        df = pd.read_csv(f'../data/{task}_{split}.tsv', sep='\t').dropna()
        # Testing
        # df = df.iloc[:1000,]
        print(f'{df.shape[0]} {split} instances')
        for row in df.itertuples():
            data[split]['premises'].append(row.Premise)
            data[split]['entailments'].append(row.Entailment)
            data[split]['contradictions'].append(row.Contradiction)

    used_device_map = False
    if 'llama' in model_name:
        from transformers import LlamaTokenizer, LlamaForCausalLM
        tokeniser = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(model_name, device_map='auto')
        used_device_map = True
    elif 't5' in model_name.lower():
        # Extract the encoder from the model; use it in the same way
        # as BERT-like models
        tokeniser = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).encoder
    elif 'opt' in model_name:
        tokeniser = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModel.from_pretrained(model_name, device_map='auto')
        used_device_map = True
    else:
        tokeniser = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    model.eval()
    if not used_device_map:
        model.cuda()


    embeddings = defaultdict(dict)
    embedding_batch_size = 32
    # Encoder-type models
    if 'bert' in model_name or 'roberta' in model_name or 'electra' in model_name or 't5' in model_name.lower():
        with torch.no_grad():
            for split in data:
                print(f'Encoding {split}...')
                for category in ['premises', 'entailments', 'contradictions']:
                    embeddings[split][category] = encode_sentences_w_mean_pooling(
                        f'{split}_{category}', data[split][category], embedding_batch_size, tokeniser, model
                    )
    # Causal LM models
    else:
        with torch.no_grad():
            for split in data:
                print(f'Encoding {split}...')
                for category in ['premises', 'entailments', 'contradictions']:
                    embeddings[split][category] = encode_sentences_w_gpt(
                        f'{split}_{category}', data[split][category], tokeniser, model
                    )
    # Release GPU memory
    del model
    torch.cuda.empty_cache()

    # Run the experiments
    layer_min = 1
    layer_max = max(embeddings['train']['premises'])
    layers = list(range(layer_min, layer_max+1))
    embedding_dimension = embeddings['train']['premises'][1].size(1)
    # We assume that all models have embedding_dimension > 512.
    # The last element in the index is a hack to encode the fact
    # that for this row we are using the original dimensionalithy
    # and do not use projection.
    dimensionalities = [2, 4, 8, 16, 32, 64, 128, 256, 512,
                        embedding_dimension, embedding_dimension+1]
    batch_size = 32
    n_epochs = 5
    for experiment_n in range(1, n_experiments+1):
        out_table_path = f'../tables/{task}_{model_name_path}_{experiment_n}.csv'
        result = pd.DataFrame(0.0, index=dimensionalities, columns=layers)
        
        if os.path.exists(out_table_path):
            # Index and colnames of previous_results should be subsets
            # of the current set.
            previous_results = pd.read_csv(out_table_path, index_col=0)
            for idx, colname in product(previous_results.index, previous_results.columns):
                result.loc[idx, int(colname)] = previous_results.loc[idx, colname]

        for layer, dimensionality in product(layers, dimensionalities[:-1]):
            print(f'{dimensionality=}, {layer=}')
            if result.loc[dimensionality, layer] > 0.0001:
                continue
            # Reinitialise the projection matrix
            projection_matrix = ProjectionMatrix(input_dim=embedding_dimension,
                                                 output_dim=dimensionality)
            optimiser = torch.optim.AdamW(projection_matrix.parameters(), lr=1e-5)
            current_best = 0.0
            test_tmp = 0.0
            best_epoch = 0
            for epoch_n in tqdm(range(n_epochs), leave=False):
                train_epoch(epoch_n, batch_size, embeddings['train'], layer,
                            projection_matrix, optimiser)
                epoch_dev_accuracy = round(validate_epoch(
                    epoch_n, batch_size, embeddings['dev'], layer, projection_matrix), 3)
                if epoch_dev_accuracy > current_best:
                    current_best = epoch_dev_accuracy
                    best_epoch = epoch_n + 1
                    test_tmp = validate_epoch(
                        epoch_n, batch_size, embeddings['test'], layer, projection_matrix)

            result.loc[dimensionality, layer] = test_tmp
            print(f'Test accuracy: {round(test_tmp, 2)} (epoch {best_epoch})')

        # Now repeat the analysis with the original embedding dimensionality 
        # without doing the projection.
        for layer in layers:
            # The index hack
            dimensionality = dimensionalities[-2]
            dimensionality_idx = dimensionalities[-1]
            dummy_projection_matrix = ProjectionMatrix(
                dimensionality, dimensionality)
            dummy_projection_matrix.projection.weight.data.copy_(
                torch.eye(dimensionality, dimensionality))
            with torch.no_grad():
                vanilla_test_accuracy = validate_epoch(-1, batch_size, embeddings['test'], layer, dummy_projection_matrix)
            result.loc[dimensionality_idx, layer] = vanilla_test_accuracy
            print(
                f'Vanilla test accuracy for layer {layer}: {round(vanilla_test_accuracy, 2)}')
        result.to_csv(out_table_path)


if __name__ == '__main__':
    main()
