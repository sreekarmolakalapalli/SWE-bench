from __future__ import annotations

import docker
import json
import resource
import traceback
import logging
import re
import os
import subprocess
import pickle

import difflib

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from collections import Counter
from nltk.util import ngrams
from crystalbleu import sentence_bleu

# from crystalbleu import sentence_bleu

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
    MAP_REPO_VERSION_TO_SPECS,
    NON_TEST_EXTS,
)
from swebench.harness.docker_utils import (
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    cleanup_container,
    list_images,
    should_remove,
    clean_images,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.reinforest_utils import (
    parse_test_command,
    # run_neural_net_scores,
    get_modified_or_added_functions_for_file,
)
from swebench.harness.grading import get_logs_eval, test_passed, test_failed
from swebench.harness.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import load_swebench_dataset, str2bool

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"

class EvaluationError(Exception):
    def __init__(self, instance_id, message):
        super().__init__(message)
        self.super_str = super().__str__()
        self.instance_id = instance_id

    def __str__(self):
        return (
            f"Evaluation error for {self.instance_id}: {self.super_str}\n"
        )
    
def get_test_directives(repo: str, test_patch: str) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # For seq2seq code repos, testing command is fixed
    if repo == "swe-bench/humaneval":
        return ["test.py"]

    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if repo == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    return directives

def make_eval_script_list(repo, version, specs, env_name, repo_directory, base_commit, test_patch, test_command=None):
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )
    if not test_command:
        test_command = " ".join(
            [
                MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"],
                *get_test_directives(repo, test_patch),
            ]
        )
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        test_command,
    ]
    return eval_commands

def run_instance(
        test_spec: TestSpec,
        prediction: dict,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        timeout: int | None = None,
    ):

    instance_id = test_spec.instance_id
    repo = test_spec.repo
    version = test_spec.version
    specs = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]
    env_name = "testbed"
    repo_directory = f"/{env_name}"
    base_commit = test_spec.base_commit

    dummy_logger = logging.Logger("instance")

    container = build_container(test_spec, client, run_id, dummy_logger, rm_image, force_rebuild)
    container.start()

    def apply_patch(patch: str):
            patch_file = Path(f"{instance_id}/patch.diff")
            patch_file.write_text(patch or "")
            copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

            val = container.exec_run(
                "git apply --allow-empty -v /tmp/patch.diff",
                workdir="/testbed",
                user="root",
            )
            if val.exit_code != 0:
                print(f"Failed to apply patch to container, trying again...")
                
                # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
                val = container.exec_run(
                    "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
                    workdir="/testbed",
                    user="root",
                )
                if val.exit_code != 0:
                    print(f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}")
                    raise EvaluationError(
                        instance_id,
                        f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}",
                    )
                else:
                    print(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")
            else:
                print(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")
            val = container.exec_run(
                    'git add .&&git commit -m "add changes"',
                    workdir="/testbed",
                    user="root",
                )
            
    try:
        model_name_or_path = prediction.get("model_name_or_path", "None").replace("/", "__")
        # pred_patch = test_spec.test_patch
        ## COMMENTED OUT CODE BELOW IS FOR THE REAL RUN, ABOVE IS A PLACEHOLDER FOR DEVELOPMENT
        pred_patch = prediction['test_patch']

        if model_name_or_path != "agentless":
            apply_patch(pred_patch)
            fail_to_pass = []

            test_commands = [*get_test_directives(repo, pred_patch)]
            for test_case in test_commands:
                test_filepath, _ = parse_test_command(test_case)
                test_func_names = get_modified_or_added_functions_for_file(pred_patch, test_filepath)
                for test_func_name in test_func_names:
                    fail_to_pass.append(f"{test_func_name}")

            fail_to_pass = list(set(fail_to_pass))

            eval_script_list = make_eval_script_list(repo, version, specs, env_name, repo_directory, base_commit, pred_patch)
            eval_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"
            log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
            log_dir.mkdir(parents=True, exist_ok=True)
            eval_file = Path(log_dir / "eval.sh")
            eval_file.write_text(eval_script)
            copy_to_container(container, eval_file, Path("/eval.sh"))

            result1, timed_out1, total_runtime1 = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
            result1_output_path = log_dir / "result_output1.txt"
            with open(result1_output_path, "w") as f:
                f.write(result1)
                if timed_out1:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    raise EvaluationError(
                        instance_id,
                        f"Test timed out after {timeout} seconds.",
                    )
            eval_sm1, _ = get_logs_eval(result1_output_path)

            f2p_success_1 = []
            f2p_failure_1 = []

            for test_case in fail_to_pass:
                if test_passed(test_case, eval_sm1):
                    f2p_success_1.append(test_case)
                elif test_failed(test_case, eval_sm1):
                    f2p_failure_1.append(test_case)

            apply_patch(test_spec.gold_patch)
            result2, timed_out2, total_runtime2 = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)


            result2_output_path = log_dir / "result_output2.txt"
            with open(result2_output_path, "w") as f:
                f.write(result2)
                if timed_out2:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    raise EvaluationError(
                        instance_id,
                        f"Test timed out after {timeout} seconds.",
                    )

            eval_sm2, _ = get_logs_eval(result2_output_path)
            
            f2p_success_2 = []
            f2p_failure_2 = []
            for test_case in fail_to_pass:
                if test_passed(test_case, eval_sm2):
                    f2p_success_2.append(test_case)
                elif test_failed(test_case, eval_sm2):
                    f2p_failure_2.append(test_case)

            if f2p_failure_1 and f2p_success_2 and not f2p_success_1 and not f2p_failure_2:
                score = 1
            else:
                score = 0

            # smell_weighted_score = run_test_smells(pred_patch) * score
            # ngram_weighted_score = run_similarity_score(pred_patch) * score
            # neural_net_score = run_neural_net_scores(pred_patch, test_spec.test_patch)
            n_gram_score = calculate_crystalbleu(pred_patch, test_spec.test_patch)

            scores = {
                "base": score,
                "weighted_n-gram": 0.5 + (0.5 * n_gram_score) if score == 1 else 0,
                # "weighted_neural-net": 0.5 + (0.5 * neural_net_score) if score == 1 else 0,
                "model_patch": pred_patch,
                "gold_patch": test_spec.test_patch,
                "fail_to_pass_detected": fail_to_pass,
                "gold_fail_to_pass": test_spec.FAIL_TO_PASS,
            }
        else:
            apply_patch(pred_patch)
            test_command = "python reproduce_bug.py"
            
            eval_script_list = make_eval_script_list(repo, version, specs, env_name, repo_directory, base_commit, pred_patch, test_command)
            eval_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"
            log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
            log_dir.mkdir(parents=True, exist_ok=True)
            eval_file = Path(log_dir / "eval.sh")
            eval_file.write_text(eval_script)
            copy_to_container(container, eval_file, Path("/eval.sh"))

            result1, timed_out1, total_runtime1 = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
            result1_output_path = log_dir / "result_output1.txt"
            with open(result1_output_path, "w") as f:
                f.write(result1)
                if timed_out1:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    raise EvaluationError(
                        instance_id,
                        f"Test timed out after {timeout} seconds.",
                    )
            f2p_success_1 = True if "Issue resolved" in result1 else False

            apply_patch(test_spec.gold_patch)
            result2, timed_out2, total_runtime2 = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)


            result2_output_path = log_dir / "result_output2.txt"
            with open(result2_output_path, "w") as f:
                f.write(result2)
                if timed_out2:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    raise EvaluationError(
                        instance_id,
                        f"Test timed out after {timeout} seconds.",
                    )
            f2p_success_2 = True if "Issue resolved" in result2 else False

            if f2p_success_2 and not f2p_success_1:
                score = 1
            else:
                score = 0

            n_gram_score = calculate_crystalbleu(pred_patch, test_spec.test_patch)
            # neural_net_score = run_neural_net_scores(pred_patch, test_spec.test_patch)

            scores = {
                "base": score,
                "weighted_n-gram": 0.5 + (0.5 * n_gram_score) if score == 1 else 0,
                # "weighted_neural-net": 0.5 + (0.5 * neural_net_score) if score == 1 else 0,
                "model_patch": pred_patch,
                "gold_patch": test_spec.test_patch,
            }

        with open(log_dir / "report.json", "w") as f:
            json.dump(scores, f, indent=4)

        print(f"scores written to {log_dir}/results.json")
    except RuntimeError as e:
        print('blah blah blah')
    finally:
        cleanup_container(client, container, dummy_logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, dummy_logger)
        close_logger(dummy_logger)
        print(f"{instance_id} container removed")

def run_test_smells(test_patch: str) -> float:
    # Starting with a perfect score of 1.0.
    smell_score = 1.0

    # 1. open files without try/catch
    if "open(" in test_patch and "try:" not in test_patch:
        smell_score -= 0.2
    
    # 2. Dupilicate Setup Code 
    setup_matches = re.findall(r"def setUp\(\):", test_patch)
    if len(setup_matches) > 1:
        smell_score -= 0.2 
    
    # 3. Check for overly generic mock usage 
    if "mock" in test_patch and "return_value" not in test_patch:
        smell_score -= 0.2 
    
    # 4. Verbose Setup 
    setup_code = re.search(r"def setUp\(.*?\):([\s\S]+?)def ", test_patch)
    if setup_code and len(setup_code.group(1).splitlines()) > 10:
        smell_score -= 0.2
    
    return max(smell_score, 0.5)

def run_instances(
        predictions: dict,
        instances: list,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    client = docker.from_env()
    test_specs = list(map(make_test_spec, instances))

    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    predictions[test_spec.instance_id],
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
                ): None
                for test_spec in test_specs
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if instance ran successfully
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    continue
    print("All instances run.")

def get_dataset_from_preds(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions: dict,
        run_id: str,
        exclude_completed: bool = True,
        log_to_check: str = "report.json",
    ):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # load dataset
    dataset = load_swebench_dataset(dataset_name, split)
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        # check that all instance IDs have predictions
        missing_preds = set(instance_ids) - set(predictions.keys())
        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs.")
    
    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )
    if instance_ids:
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

    # check which instance IDs have already been run
    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            # skip instances without predictions
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / log_to_check
        )
        if report_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {k for k, v in predictions.items() if v["test_patch"] == "" or v["test_patch"] is None}

    # filter dataset to only instances with predictions
    dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in prediction_ids and i[KEY_INSTANCE_ID] not in empty_patch_ids]
    return dataset

def get_gold_predictions(dataset_name: str, split: str):
    """
    Get gold predictions for the given dataset and split.
    """
    dataset = load_swebench_dataset(dataset_name, split)
    return [
        {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            "model_patch": datum["patch"],
            "model_name_or_path": "gold",
        } for datum in dataset
    ]

def calculate_crystalbleu(model_patch, gold_patch):
    trivial_ngrams_file = "trivial_ngrams.pkl"  
    with open(trivial_ngrams_file, "rb") as file:
        trivial_ngrams = pickle.load(file)

    def tokenize_code(code):
        token_pattern = r"[A-Za-z_][A-Za-z0-9_]*|[\{\}\[\]\(\)\.;,]|[+\-*/=%<>!]+"
        tokens = re.findall(token_pattern, code)
        return tokens

    model_patch_tokens = tokenize_code(model_patch)
    gold_patch_tokens = tokenize_code(gold_patch)
    
    score = sentence_bleu([model_patch_tokens], gold_patch_tokens, ignoring = trivial_ngrams)
    return score

def main(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions_path: str,
        max_workers: int,
        force_rebuild: bool,
        cache_level: str,
        clean: bool,
        open_file_limit: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

# load predictions as map of instance_id to prediction
    if predictions_path == 'gold':
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, split)
    else:
        if predictions_path.endswith(".json"):
            with open(predictions_path, "r") as f:
                predictions = json.load(f)
        elif predictions_path.endswith(".jsonl"):
            with open(predictions_path, "r") as f:
                predictions = [json.loads(line) for line in f]
        else:
            raise ValueError("Predictions path must be \"gold\", .json, or .jsonl")
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    dataset = get_dataset_from_preds(dataset_name, split, instance_ids, predictions, run_id)
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)
    existing_images = list_images(client)
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    else:
        # build environment images + run instances
        build_env_images(client, dataset, force_rebuild, max_workers)
        run_instances(predictions, dataset, cache_level, clean, force_rebuild, max_workers, run_id, timeout)

    # Run evaluation with test_metrics.py
    #run_test_metrics_evaluation(predictions, dataset)

    # clean images + make final report
    clean_images(client, existing_images, cache_level, clean)

def run_test_metrics_evaluation(predictions):
    # Write predictions to a temporary JSON file for test_metrics.py to read
    predictions_file = "temp_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(predictions, f)

    # Call test_metrics.py with the path to the temporary predictions file
    result = subprocess.run(
        ["python3", "test_metrics.py", "--predictions", predictions_file],
        capture_output=True,
        text=True
    )

    # Output results from test_metrics.py
    print("Evaluation Results:", result.stdout)

    # Clean up the temporary predictions file
    os.remove(predictions_file)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument(
        "--timeout", type=int, default=1_800, help="Timeout (in seconds) for running tests for each instance"
        )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images"
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))
