"""Paths and global settings shared by every pipeline stage."""
import os

# student_resource/ root (this file lives in student_resource/code/business_entity_resolution/src/)
ROOT = os.environ.get(
    "BER_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
DATA = os.path.join(ROOT, "dataset")
WORK = os.environ.get("BER_WORK", os.path.join(ROOT, "work"))
OUTPUT = os.path.join(ROOT, "output")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

N_JOBS = int(os.environ.get("BER_JOBS", "14"))
SEED = 42


def src_path(split, k):
    """Raw TSV path for split in {train,test} and source k in {1,2,3}."""
    return os.path.join(DATA, split, f"{split}_source{k}.tsv")


GT_PATH = os.path.join(DATA, "train", "train_ground_truth.tsv")


def work_path(name):
    return os.path.join(WORK, name)


# BER_SIM=1: train on the "withheld S1" simulation (make_drop.py) so train looks like test
SIM = os.environ.get("BER_SIM", "0") == "1"
TAG = "_sim" if SIM else ""
TRAIN_FEAT = "train_drop_feat" if SIM else "train_feat"
