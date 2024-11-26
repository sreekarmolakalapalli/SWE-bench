import torch
import re
import numpy as np
import os
import subprocess

from pathlib import Path

from typing import Optional, Union, List

from torch import nn
from transformers import AutoModel, PreTrainedTokenizer, BertModel, AutoTokenizer
from swebench.harness.utils import load_swebench_dataset

class CrossMatchLoss(nn.Module):
    def __init__(
        self,
        semantic_match_factor=0.,
    ):
        super().__init__()
        self.semantic_match_factor = semantic_match_factor
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(
        self,
        input_vector: torch.Tensor,  # (B * H)
        positive_vectors: torch.Tensor = None,  # (B * P * H)
        negative_vectors: torch.Tensor = None,  # (B * N * H)
        positive_semantic_match_scores: torch.Tensor = None,  # (B * P)
        negative_semantic_match_scores: torch.Tensor = None,  # (B * N)
    ):
        if positive_vectors is None and negative_vectors is None:
            raise ValueError(
                "CrossMatchLoss does not know how to calculate the loss if" +
                "Both the positive vectors and negative vectors are None." +
                "Please provide at least one non-None vectors"
            )
        if positive_semantic_match_scores is None \
                and negative_semantic_match_scores is None:
            semantic_match_factor = 0.
        else:
            semantic_match_factor = self.semantic_match_factor

        input_norm = torch.norm(
            input_vector, dim=-1, keepdim=True, p=2
        ).unsqueeze(1)
        modified_input_vector = input_vector.unsqueeze(1)
        if positive_vectors is not None:
            positive_norm = torch.norm(
                positive_vectors, dim=-1, keepdim=True, p=2
            )
            positive_products = torch.matmul(
                input_norm, positive_norm.transpose(1, 2)
            ).squeeze(1)
            modified_pv = positive_vectors.transpose(1, 2)
            positive_scores = torch.abs(
                torch.matmul(modified_input_vector, modified_pv)
            ).squeeze(1)
            positive_scores = positive_scores / positive_products
            positive_labels = torch.ones_like(positive_scores)
        else:
            positive_scores = torch.zeros(
                size=(input_vector.shape[0], 0), dtype=input_vector.dtype,
                device=input_vector.device
            )
            positive_labels = torch.zeros(
                size=(input_vector.shape[0], 0), dtype=input_vector.dtype,
                device=input_vector.device
            )
        if positive_vectors is None or positive_semantic_match_scores is None:
            positive_semantic_match_scores = torch.zeros_like(positive_scores)

        if negative_vectors is not None:
            negative_norm = torch.norm(
                negative_vectors, dim=-1, keepdim=True, p=2
            )
            negative_products = torch.matmul(
                input_norm, negative_norm.transpose(1, 2)
            ).squeeze(1)
            modified_nv = negative_vectors.transpose(1, 2)
            negative_scores = torch.abs(
                torch.matmul(modified_input_vector, modified_nv)
            ).squeeze(1)
            negative_scores = negative_scores / negative_products
            negative_labels = torch.zeros_like(negative_scores)
        else:
            negative_scores = torch.zeros(
                size=(input_vector.shape[0], 0), dtype=input_vector.dtype,
                device=input_vector.device
            )
            negative_labels = torch.zeros(
                size=(input_vector.shape[0], 0), dtype=input_vector.dtype,
                device=input_vector.device
            )
        if negative_vectors is None or negative_semantic_match_scores is None:
            negative_semantic_match_scores = torch.zeros_like(negative_scores)

        labels = torch.cat([positive_labels, negative_labels], dim=-1)
        scores = torch.cat([positive_scores, negative_scores], dim=-1)
        semantic_match_scores = torch.cat(
            [positive_semantic_match_scores, negative_semantic_match_scores],
            dim=-1
        )
        labels = semantic_match_factor * semantic_match_scores + \
            (1 - semantic_match_factor) * labels
        loss = self.loss_fn(scores, labels)
        return {
            "loss": loss,
            "scores": {
                "positive": positive_scores,
                "negative": negative_scores
            },
            "input_vector": input_vector,
            "positive_vectors": positive_vectors,
            "negative_vectors": negative_vectors,
            "semantic_match_factor": semantic_match_factor,
        }

class CodeBERTBasedModel(nn.Module):
    def __init__(
        self,
        model_name: str = 'codebert',
        semantic_match_factor: float = 0.1,
    ):
        super().__init__()
        assert model_name in [
            'codebert', 'graphcodebert', 'roberta'
        ], "Only codebert, graphcodebert, and roberta are supported"
        if model_name == 'codebert':
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        elif model_name == 'graphcodebert':
            self.model = AutoModel.from_pretrained(
                "microsoft/graphcodebert-base")
        else:
            self.model = AutoModel.from_pretrained("roberta-base")
        self.semantic_match_factor = semantic_match_factor
        self.loss_fn = CrossMatchLoss(
            semantic_match_factor=semantic_match_factor)

    def get_vector(
        self,
        input_ids: torch.Tensor,  # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
    ):
        batched = True
        if input_ids.ndim == 1:
            batched = False
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        assert input_ids.ndim == 2 and attention_mask.ndim == 2
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        _vector = output.pooler_output
        if not batched:
            _vector = _vector.squeeze(0)
        return _vector.detach()

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
        pos_input_ids: Optional[torch.Tensor] = None,  # (B, P, L)
        pos_attn_mask: Optional[torch.Tensor] = None,  # (B, P, L)
        pos_semantic_scores: Optional[torch.Tensor] = None,  # (B, P)
        neg_input_ids: Optional[torch.Tensor] = None,  # (B, N, L)
        neg_attn_mask: Optional[torch.Tensor] = None,  # (B, N, L)
        neg_semantic_scores: Optional[torch.Tensor] = None,  # (B, N)
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        input_vector = output.pooler_output
        if pos_input_ids is not None and pos_input_ids.shape[1] > 0:
            output = self.model(
                input_ids=pos_input_ids.reshape(-1, pos_input_ids.shape[-1]),
                attention_mask=None if pos_attn_mask is None
                else pos_attn_mask.reshape(-1, pos_attn_mask.shape[-1])
            )
            positive_vectors = output.pooler_output
            positive_vectors = positive_vectors.reshape(
                pos_input_ids.shape[0], pos_input_ids.shape[1], -1
            )
        else:
            positive_vectors = None

        if neg_input_ids is not None and neg_input_ids.shape[1] > 0:
            output = self.model(
                input_ids=neg_input_ids.reshape(-1, neg_input_ids.shape[-1]),
                attention_mask=None if neg_attn_mask is None
                else neg_attn_mask.reshape(-1, neg_attn_mask.shape[-1])
            )
            negative_vectors = output.pooler_output
            negative_vectors = negative_vectors.reshape(
                neg_input_ids.shape[0], neg_input_ids.shape[1], -1
            )
        else:
            negative_vectors = None
        return self.loss_fn(
            input_vector=input_vector,
            positive_vectors=positive_vectors,
            negative_vectors=negative_vectors,
            positive_semantic_match_scores=pos_semantic_scores,
            negative_semantic_match_scores=neg_semantic_scores
        )
    
def calculate_scores(vector, other_vectors):
    scores = []
    for o in other_vectors:
        scores.append(
            np.dot(vector, o) / (
                np.abs(np.linalg.norm(o, ord=2)) * \
                np.abs(np.linalg.norm(vector, ord=2))
            )
        )
    return np.array(scores)

def get_vector(
        cls,
        tokenizer: PreTrainedTokenizer,
        model: CodeBERTBasedModel,
        texts: Union[str, List[str]],
        no_train_rank: bool = False
    ):
        assert isinstance(model, CodeBERTBasedModel)
        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]
        tokenizer_output = tokenizer(
            texts, max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors='pt'
        )
        input_ids, attention_mask = tokenizer_output.input_ids, \
            tokenizer_output.attention_mask
        assert isinstance(input_ids, torch.LongTensor) \
            and isinstance(attention_mask, torch.LongTensor)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        vector = model.get_vector(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        if not batched:
            vector = vector.squeeze(0)
        return vector.cpu().numpy().tolist()

def parse_test_command(command):

    pattern = r'(?P<file_path>[\w/\\.-]+)(?:[:.]{2,}|\.)(?P<test_name>[\w.]+)'

    match = re.search(pattern, command)
    if not match:
        raise ValueError("Command format not recognized. Make sure it's a pytest or unittest command.")
    
    file_path = match.group('file_path')
    test_name = match.group('test_name').replace('.', '::')
    if ".py" not in file_path:
        file_path = file_path + ".py"

    return file_path, test_name

def normalize_lists(list1, list2):
    min_length = min(len(list1), len(list2))
    
    normalized_list1 = list1[:min_length]
    normalized_list2 = list2[:min_length]

    zipped_list = list(zip(normalized_list1, normalized_list2))
    
    return zipped_list


def extract_test_source(test_name, filepath:str=None, file_contents:str=None):
    if filepath:
        with open(f"{filepath}", 'r') as file:
            lines = file.readlines()
    if file_contents:
        lines = file_contents.splitlines()
    test_code = []
    in_test = False
    
    function_pattern = re.compile(rf"def {re.escape(test_name)}\b")

    for line in lines:
        if function_pattern.search(line):
            in_test = True
        if in_test:
            test_code.append(line)
            if line.strip() == "" or line.startswith("def "):
                in_test = False
                break

    return "".join(test_code)

def create_playground(repo: str, commit_sha: str):
    playground_path = Path("./playground")
    if not os.path.exists(playground_path):
        os.mkdir(playground_path)

    os.chdir(playground_path)

    rel_repo_path = repo.split("/")[-1]
    if not os.path.exists(rel_repo_path):
        print(f"repo {repo} does not exist. Cloning now...")
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git"], check=True)
    else:
        print(f"repo {repo} already exists. Skipping cloning.")
    
    os.chdir(rel_repo_path)
    
    subprocess.run(["git", "reset", "--hard"])
    subprocess.run(["git", "fetch", "--all"], check=True)
    subprocess.run(["git", "checkout", commit_sha], check=True)

    print(f"Checked out commit {commit_sha} in {repo} and changed directory to {rel_repo_path}")

def load_model(model, ckpt_under_check, dont_load=False):
    if not dont_load:
        ckpt_file = os.path.join(ckpt_under_check, 'pytorch_model.bin')
        if not os.path.exists(ckpt_file):
            print('Model file does not exist. Please train first!')
            exit()
        with open(ckpt_file, 'rb') as fp:
            model.load_state_dict(torch.load(fp, map_location="mps"))

def run_neural_net_scores(submitted_test_patch, gold_test_patch):
    model = CodeBERTBasedModel()
    load_model(model, "/Users/ethansin/Capstone/SWE-bench/swebench/harness/models/atcoder/semantic_data/python/with_score/codebert_0.2-0.2/checkpoint-best-eval_rank_gap")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    v = get_vector(None, tokenizer=tokenizer, model=model, texts=[submitted_test_patch, gold_test_patch])    
    
    score = float(calculate_scores(v[0], [v[1]])[0])

    if score > 1:
        score = 1

    score = (0.01 - (1 - score)) / 0.01 if score > 0.99 else 0

    return score


# def test():
#     dataset = load_swebench_dataset(instance_ids=["astropy__astropy-12907", "astropy__astropy-14182"])
#     sample = dataset[0]
#     ref_patch = sample["test_patch"]
#     repo = sample["repo"]
#     create_playground(repo, sample["base_commit"])
#     directives = get_test_directives(repo, ref_patch)
#     file_path, test_name = parse_test_command("pytest astropy/modeling/tests/test_separable.py::test_custom_model_separable")
#     ref_func = extract_test_source(file_path, test_name)

#     gold_path, gold_name = parse_test_command("pytest astropy/io/ascii/tests/test_rst.py::test_read_normal")
#     # gold_func = extract_test_source(gold_path, gold_name)
#     gold_func = "def function():\nreturn None"

#     model = CodeBERTBasedModel()
#     load_model(model, "/Users/ethansin/Capstone/SWE-bench/swebench/harness/models/atcoder/semantic_data/python/with_score/codebert_0.2-0.2/checkpoint-best-eval_rank_gap")
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

#     v = get_vector(None, tokenizer=tokenizer, model=model, texts=[ref_func, gold_func])

#     scores = calculate_scores(v[0], [v[1]])

#     print(scores)

if __name__ == "__main__":
    # test()
    pass