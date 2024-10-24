from __future__ import annotations

import docker
import json
import resource
import traceback

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
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
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import load_swebench_dataset, str2bool, extract_new_test_command

class EvaluationError(Exception):
    def __init__(self, instance_id, message):
        super().__init__(message)
        self.super_str = super().__str__()
        self.instance_id = instance_id

    def __str__(self):
        return (
            f"Evaluation error for {self.instance_id}: {self.super_str}\n"
        )

def run_instance(
        test_spec: TestSpec,
        pred: str,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        timeout: int | None = None,
    ):

    instance_id = test_spec.instance_id

    container = build_container(test_spec, client, run_id, rm_image, force_rebuild)
    container.start()

    pred = test_spec.test_patch
    
    def apply_patch(patch: str):
        patch_file = Path("patch.diff")
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

    apply_patch(pred)
    cmd = extract_new_test_command(pred)
    result1 = container.exec_run(cmd)
    apply_patch(test_spec.gold_patch)
    result2 = container.exec_run(cmd)

    if result2 and not result1:
        score = 1
    else:
        score = 0

    smell_weighted_score = run_test_smells(pred) * score
    similarity_weighted_score = run_similarity_score(pred) * score

    return (smell_weighted_score, similarity_weighted_score, score)


def run_test_smells(patch) -> float:
    return 0.5
    
def run_similarity_score(patch) -> float:
    return 0.6
        