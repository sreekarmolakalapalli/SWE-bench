import json
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

from swebench.harness.reinforest_utils import (
    CodeBERTBasedModel,
    run_neural_net_scores,
    load_model,
)
from swebench.harness.run_test_metric import (
    calculate_crystalbleu
)

def main(logs_path: Path = Path("logs/run_evaluation/agentless_base_metric/agentless")):
    results = {
        "total_base_count": 0,
        "total_instances": 500,
        "total_weighted_ngram_count": 0,
        "total_weighted_nnet_count": 0,
        "base_score": 0,
        "weighted_ngram_score": 0,
        "weighted_nnet_score": 0,
    }
    model = CodeBERTBasedModel()
    load_model(model, "/Users/ethansin/Capstone/SWE-bench/swebench/harness/models/atcoder/semantic_data/python/with_score/codebert_0.2-0.2/checkpoint-best-eval_rank_gap")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    for instance in tqdm(logs_path.iterdir()):
        with open(Path(instance / "report.json"), "r") as f:
            instance_results = json.load(f)

        bleu_weighted = 0.5 + (0.5 * calculate_crystalbleu(instance_results["model_patch"], instance_results["gold_patch"])) if instance_results["base"] == 1 else 0
        nnet_weighted = 0.5 + (0.5 * run_neural_net_scores(instance_results["model_patch"], instance_results["gold_patch"], model, tokenizer)) if instance_results["base"] == 1 else 0

        results["total_base_count"] += instance_results["base"]
        results["total_weighted_ngram_count"] += bleu_weighted
        results["total_weighted_nnet_count"] += nnet_weighted

    
    results["base_score"] = results["total_base_count"] / results["total_instances"]
    results["weighted_ngram_score"] = results["total_weighted_ngram_count"] / results["total_instances"]
    results["weighted_nnet_score"] = results["total_weighted_nnet_count"] / results["total_instances"]

    with open(Path(logs_path / "total_score.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()