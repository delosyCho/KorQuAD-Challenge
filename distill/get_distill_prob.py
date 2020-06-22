import numpy
import json
import codecs

import numpy as np
import tokenization as tokenization
from modeling_bert import QuestionAnswering, Config
import torch
from tqdm import tqdm
from evaluate_korquad import evaluate as korquad_eval

import collections
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


config_name = 'data/large_config.json'
checkpoint = 'pretrain_ckpt/korquad_5e-05_4_3_4.bin'
#device = torch.device("cuda" if torch.cuda.is_available() and not True else "cpu")
device = torch.device("cuda")

print("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, 1, 0))

config = Config.from_json_file(config_name)
config.dropout_prob = 0.0

model = QuestionAnswering(config)
#model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(checkpoint))
model.to(device)

model = torch.nn.DataParallel(model)

max_length = 512
n_best_size = 10
max_answer_length = 12

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])


count = 0

mini_batch_sz = 96

input_ids = np.load('input_ids.npy')
input_mask = np.load('input_mask.npy')
input_segments = np.load('input_segments.npy')

start_prob = np.zeros(shape=input_ids.shape, dtype=np.float32)
end_prob = np.zeros(shape=input_ids.shape, dtype=np.float32)

paragraph = torch.tensor(input_ids.astype(np.int64)).type(dtype=torch.long).cuda()
paragraph_mask = torch.tensor(input_mask.astype(np.int64)).type(dtype=torch.long).cuda()
paragraph_segments = torch.tensor(input_segments.astype(np.int64)).type(dtype=torch.long).cuda()
all_example_index = torch.arange(paragraph.size(0), dtype=torch.long).cuda()

eval_data = TensorDataset(paragraph, paragraph_mask, paragraph_segments, all_example_index)

eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=128)

for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    with torch.no_grad():
        print('batch...')
        batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        batch_start_logits = batch_start_logits.cpu()
        batch_end_logits = batch_end_logits.cpu()

        input_ids = np.array(input_ids.cpu(), dtype=np.int32)

    start_prob_logits = np.array(batch_start_logits, dtype=np.float32)
    stop_prob_logits = np.array(batch_end_logits, dtype=np.float32)

    for m, example_index in enumerate(example_indices):
        start_prob[example_index] = start_prob_logits[m]
        end_prob[example_index] = stop_prob_logits[m]

np.save('start_prob', start_prob)
np.save('end_prob', end_prob)

