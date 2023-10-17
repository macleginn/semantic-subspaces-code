from collections import defaultdict
from itertools import product
from math import ceil
import csv
import os
import sys
import pickle
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from scipy.stats import spearmanr
from test_models_on_sts import mean_pooling

random_seed = sys.argv[2]
torch.manual_seed(random_seed)


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


def train_epoch(epoch_n, batch_size, data, layer, model, optimiser):
    data_size = data['embeddings1'][layer].size(0)
    n_steps = ceil(data_size / batch_size)
    # print(f'{data_size} training examples; {n_steps} training steps')
    epoch_train_losses = torch.zeros(n_steps)
    for step_n in tqdm(range(n_steps), desc=f'Epoch {epoch_n+1}, training', leave=False):
        optimiser.zero_grad()
        lo = step_n * batch_size
        hi = lo + batch_size
        embeddings1 = data['embeddings1'][layer][lo:hi]
        embeddings2 = data['embeddings2'][layer][lo:hi]
        scores = torch.tensor(data['scores'][lo:hi])
        if model.using_cuda:
            embeddings1 = embeddings1.cuda()
            embeddings2 = embeddings2.cuda()
            scores = scores.cuda()
        # Squared Euclidean distances between projected Embeddings
        # should mimic the semantic-dissimilarity scores.
        projected_embeddings1 = model(embeddings1)
        projected_embeddings2 = model(embeddings2)
        distances = (
            projected_embeddings1 - projected_embeddings2
        ).square().sum(axis=1).sqrt()

        # param = model.projection.weight
        # sym = torch.mm(param, torch.t(param))
        # if model.using_cuda:
        #     sym -= torch.eye(param.shape[0]).cuda()
        # else:
        #     sym -= torch.eye(param.shape[0])
        # ls_ort = sym.pow(2.0).sum()

        # Beware of zero loss on identical sentences.
        # UPD: we remove those in preprocessing.
        loss = ((distances - scores).square()).mean()  # + ls_ort
        # if torch.isclose(loss, torch.tensor(0.0)):
        #    continue

        loss.backward()
        optimiser.step()
        # print(f"proj matrix grad norm: {model.projection.weight.grad.detach().data.norm(2).item()}\n")
        # print(f"loss: {loss.item()}")

        # if torch.isnan(model.projection.weight.grad.detach().data.norm(2)):
        #     print('Grad is nan!')

        #     print(distances)
        #     print(sentence_data['sentences1'][lo:hi])
        #     print(sentence_data['sentences2'][lo:hi])
        #     print(sentence_data['scores'][lo:hi])
        #     print(f"emb 1 norm: {embeddings1.norm(2).item()}\n"
        #           f"emb 2 norm: {embeddings2.norm(2).item()}\n"
        #           f"proj matrix norm: {model.projection.weight.detach().norm(2).item()}\n"
        #           f"proj emb 1 norm: {projected_embeddings1.detach().norm(2).item()}\n"
        #           f"proj emb 2 norm: {projected_embeddings2.detach().norm(2).item()}\n"
        #           f"scores norm: {scores.detach().norm(2).item()}\n"
        #           f"distances norm: {distances.detach().norm(2).item()}")

        #     sys.exit(1)
        epoch_train_losses[step_n] = loss.item()
    return epoch_train_losses.mean().item()


def validate_epoch(epoch_n, batch_size, data, layer, model):
    data_size = data['embeddings1'][layer].size(0)
    n_steps = ceil(data_size / batch_size)
    all_distances = []
    for step_n in tqdm(
            range(n_steps), desc=f'Epoch {epoch_n+1}, validation', leave=False):
        lo = step_n * batch_size
        hi = lo + batch_size
        embeddings1 = data['embeddings1'][layer][lo:hi]
        embeddings2 = data['embeddings2'][layer][lo:hi]
        if model.using_cuda:
            embeddings1 = embeddings1.cuda()
            embeddings2 = embeddings2.cuda()
        with torch.no_grad():
            projected_embeddings1 = model(embeddings1)
            projected_embeddings2 = model(embeddings2)
        distances = (
            projected_embeddings1 - projected_embeddings2).square().sum(axis=1).sqrt()
        all_distances.extend(distances.flatten().tolist())
    return spearmanr(all_distances, data['scores']).statistic


def encode_batch_w_mean_pooling(batch, tokeniser, model):
    """
    Return mean-pooled representation of all layers.
    """
    result = defaultdict(list)
    tokenisation = tokeniser(batch, return_tensors='pt',
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
            max_length=512,
            return_tensors='pt')
        if 'opt-66b' not in model.config._name_or_path:
            inputs = {k: v.cuda() for k, v in tokenisation.items()}
        else:
            inputs = tokenisation
        outputs = model(**inputs, output_hidden_states=True).hidden_states
        for layer_n in range(len(outputs)):
            results[layer_n].append(
                outputs[layer_n][0, -1].reshape(1, -1).cpu())
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
    model_name = sys.argv[1]
    model_name_path = model_name.replace('/', '_')
    embedding_path = f'../embeddings/{model_name_path}.pickle'
    if not os.path.exists(embedding_path):
        # Encode everything
        data = {
            'train': {
                'sentences1': [],
                'sentences2': [],
                'scores': []
            },
            'dev': {
                'sentences1': [],
                'sentences2': [],
                'scores': []
            },
            'test': {
                'sentences1': [],
                'sentences2': [],
                'scores': []
            },
        }
        with open('../data/stsbenchmark.tsv', 'r', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                # Some models fail on identical sentences
                if row['sentence1'].lower() == row['sentence2'].lower():
                    continue
                split = row['split']
                # Convert the scores to dissimilarity rankings in the [0.0, 1.0] range.
                data[split]['scores'].append(1.0 - float(row['score']) / 5.0)
                data[split]['sentences1'].append(row['sentence1'])
                data[split]['sentences2'].append(row['sentence2'])

        used_device_map = False
        if 'llama' in model_name:
            from transformers import LlamaTokenizer, LlamaForCausalLM
            tokeniser = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(model_name, device_map='sequential')
            used_device_map = True
        elif 't5' in model_name.lower():
            # Extract the encoder from the model; use it in the same way
            # as BERT-like models
            tokeniser = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).encoder
        # OPT models needs to be downloaded to be loaded on multiple GPUs.
        # Also fast tokenisers do not work for at least some of them.
        elif 'opt-30b' in model_name:
            tokeniser = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModel.from_pretrained('/mount/arbeitsdaten33/projekte/tcl/Users/nikolady/opt-30b', device_map='balanced_low_0')
            used_device_map = True
        elif 'opt' in model_name:
            tokeniser = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModel.from_pretrained(model_name, device_map='sequential')
            used_device_map = True
        else:
            tokeniser = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        model.eval()
        # Do not use GPUs for the biggest models
        if not used_device_map:
            model.cuda()

    
        embeddings = defaultdict(dict)
        # Encoder-type models
        if 'bert' in model_name or 'roberta' in model_name or 'electra' in model_name or 't5' in model_name.lower():
            with torch.no_grad():
                for split in data:
                    print(f'Encoding {split}...')
                    embeddings[split]['embeddings1'] = encode_sentences_w_mean_pooling(
                        f'{split}_1', data[split]['sentences1'], 128, tokeniser, model
                    )
                    embeddings[split]['embeddings2'] = encode_sentences_w_mean_pooling(
                        f'{split}_2', data[split]['sentences2'], 128, tokeniser, model
                    )
                    embeddings[split]['scores'] = data[split]['scores']
            print('Saving embeddings...')
            with open(embedding_path, 'wb') as out:
                pickle.dump(embeddings, out)
        # Causal LM models
        else:
            with torch.no_grad():
                for split in data:
                    print(f'Encoding {split}...')
                    embeddings[split]['embeddings1'] = encode_sentences_w_gpt(
                        f'{split}_1', data[split]['sentences1'], tokeniser, model
                    )
                    embeddings[split]['embeddings2'] = encode_sentences_w_gpt(
                        f'{split}_2', data[split]['sentences2'], tokeniser, model
                    )
                    embeddings[split]['scores'] = data[split]['scores']
            print('Saving embeddings...')
            with open(embedding_path, 'wb') as out:
                pickle.dump(embeddings, out)

        # Release GPU memory
        del model
        torch.cuda.empty_cache()
    else:
        print('Loading embeddings...')
        with open(embedding_path, 'rb') as inp:
            embeddings = pickle.load(inp)

    # Train the projection matrix on the test/dev set
    layer_min = 1
    layer_max = max(embeddings['train']['embeddings1'])
    layers = list(range(layer_min, layer_max+1))
    embedding_dimension = embeddings['train']['embeddings1'][1].size(1)
    # We assume that all models have embedding_dimension > 512.
    # The last element in the index is a hack to encode the fact
    # that for this row we are using the original dimensionalithy
    # and do not use projection.
    dimensionalities = [2, 4, 8, 16, 32, 64, 128, 256, 512,
                        embedding_dimension, embedding_dimension+1]

    out_table_path = f'../tables/{model_name_path}_{random_seed}.csv'
    result = pd.DataFrame(0.0, index=dimensionalities, columns=layers)
    if os.path.exists(out_table_path):
        # Index and colnames of previous_results should be subsets
        # of the current set.
        previous_results = pd.read_csv(out_table_path, index_col=0)
        for idx, colname in product(previous_results.index, previous_results.columns):
            result.loc[idx, int(colname)] = previous_results.loc[idx, colname]

    batch_size = 32
    n_epochs = 500

    for layer, dimensionality in product(layers, dimensionalities[:-1]):
        print(f'{dimensionality=}, {layer=}')
        if result.loc[dimensionality, layer] > 0.0001:
            continue
        projection_matrix = ProjectionMatrix(input_dim=embedding_dimension,
                                             output_dim=dimensionality)
        optimiser = torch.optim.AdamW(projection_matrix.parameters(), lr=1e-5)
        current_best = 0.0
        epochs_without_improvement = 0
        early_stopping_threshold = 5
        epochs = 0
        for epoch_n in tqdm(range(n_epochs), leave=False):
            train_epoch(epoch_n, batch_size, embeddings['train'], layer,
                        projection_matrix, optimiser)
            epoch_dev_correlation = round(validate_epoch(
                epoch_n, batch_size, embeddings['dev'], layer, projection_matrix), 3)
            epochs += 1
            if epoch_dev_correlation > current_best:
                current_best = epoch_dev_correlation
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == early_stopping_threshold:
                    break

        test_correlation = validate_epoch(
            epoch_n, batch_size, embeddings['test'], layer, projection_matrix)
        result.loc[dimensionality, layer] = test_correlation
        print(
            f'Test correlation after {epochs} epochs: {round(test_correlation, 2)}')

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
            vanilla_test_correlation = validate_epoch(-1, batch_size, embeddings['test'], layer, dummy_projection_matrix)
        result.loc[dimensionality_idx, layer] = vanilla_test_correlation
        print(
            f'Vanilla test correlation for layer {layer}: {round(vanilla_test_correlation, 2)}')
    
    result.to_csv(out_table_path)


if __name__ == '__main__':
    main()
