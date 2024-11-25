import os
import torch

import __main__
from dataclasses import dataclass
from colbert.utils.utils import timestamp

from .core_config import DefaultVal


@dataclass
class RunSettings:
    """
    The defaults here have a special status in Run(), which initially calls assign_defaults(),
    so these aren't soft defaults in that specific context.
    """

    overwrite: bool = DefaultVal(False)

    root: str = DefaultVal(os.path.join(os.getcwd(), "experiments"))
    experiment: str = DefaultVal("default")

    index_root: str = DefaultVal(None)
    name: str = DefaultVal(timestamp(daydir=True))

    rank: int = DefaultVal(0)
    nranks: int = DefaultVal(1)
    amp: bool = DefaultVal(True)

    total_visible_gpus = torch.cuda.device_count()
    gpus: int = DefaultVal(total_visible_gpus)

    avoid_fork_if_possible: bool = DefaultVal(False)

    @property
    def gpus_(self):
        value = self.gpus

        if isinstance(value, int):
            value = list(range(value))

        if isinstance(value, str):
            value = value.split(",")

        value = list(map(int, value))
        value = sorted(list(set(value)))

        assert all(
            device_idx in range(0, self.total_visible_gpus) for device_idx in value
        ), value

        return value

    @property
    def index_root_(self):
        return self.index_root or os.path.join(self.root, self.experiment, "indexes/")

    @property
    def script_name_(self):
        if "__file__" in dir(__main__):
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                script_path = script_path[len(cwd) :]

            else:
                try:
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath) :]
                except:
                    pass

            assert script_path.endswith(".py")
            script_name = script_path.replace("/", ".").strip(".")[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)

            return script_name

        return "none"

    @property
    def path_(self):
        return os.path.join(self.root, self.experiment, self.script_name_, self.name)

    @property
    def device_(self):
        return self.gpus_[self.rank % self.nranks]


@dataclass
class TokenizerSettings:
    query_token_id: str = DefaultVal("[unused0]")
    doc_token_id: str = DefaultVal("[unused1]")
    query_token: str = DefaultVal("[Q]")
    doc_token: str = DefaultVal("[D]")


@dataclass
class ResourceSettings:
    checkpoint: str = DefaultVal(None)
    triples: str = DefaultVal(None)
    collection: str = DefaultVal(None)
    queries: str = DefaultVal(None)
    index_name: str = DefaultVal(None)


@dataclass
class DocSettings:
    dim: int = DefaultVal(128)
    doc_maxlen: int = DefaultVal(220)
    mask_punctuation: bool = DefaultVal(True)


@dataclass
class QuerySettings:
    query_maxlen: int = DefaultVal(32)
    attend_to_mask_tokens: bool = DefaultVal(False)
    interaction: str = DefaultVal("colbert")


@dataclass
class TrainingSettings:
    """
    TrainingSettings is a dataclass that holds configuration settings for training a model.

    Attributes:
        similarity (str): The similarity metric to use. Default is "cosine".
        bsize (int): The batch size for training. Default is 32.
        accumsteps (int): The number of accumulation steps. Default is 1.
        lr (float): The learning rate for training. Default is 3e-06.
        maxsteps (int): The maximum number of training steps. Default is 500,000.
        save_every (int): The interval at which to save the model. Default is None.
        resume (bool): Whether to resume training from a checkpoint. Default is False.
        warmup (int): The number of warmup steps. Default is None.
        warmup_bert (int): The number of warmup steps specifically for BERT. Default is None.
        relu (bool): Whether to use ReLU activation. Default is False.
        nway (int): The number of ways for N-way classification. Default is 2.
        use_ib_negatives (bool): Whether to use in-batch negatives. Default is False.
        reranker (bool): Whether to use a reranker. Default is False.
        distillation_alpha (float): The alpha value for distillation. Default is 1.0.
        ignore_scores (bool): Whether to ignore scores. Default is False.
        model_name (str): The name of the model to use. Default is None.
    """

    similarity: str = DefaultVal("cosine")

    bsize: int = DefaultVal(32)

    accumsteps: int = DefaultVal(1)

    lr: float = DefaultVal(3e-06)

    maxsteps: int = DefaultVal(500_000)

    save_every: int = DefaultVal(None)

    resume: bool = DefaultVal(False)

    ## NEW:
    warmup: int = DefaultVal(None)

    warmup_bert: int = DefaultVal(None)

    relu: bool = DefaultVal(False)

    nway: int = DefaultVal(2)

    use_ib_negatives: bool = DefaultVal(False)

    reranker: bool = DefaultVal(False)

    distillation_alpha: float = DefaultVal(1.0)

    ignore_scores: bool = DefaultVal(False)

    model_name: str = DefaultVal(None)  # DefaultVal('bert-base-uncased')

    batch_idx: int = DefaultVal(0)

    optimizer_state_dict: dict = DefaultVal(None)

    model_state_dict: dict = DefaultVal(None)

    arguments: dict = DefaultVal(None)


@dataclass
class IndexingSettings:
    """
    IndexingSettings is a dataclass that holds configuration settings for indexing.

    Attributes:
        index_path (str): Path to the index. Default is None.
        index_bsize (int): Batch size for indexing. Default is 64.
        nbits (int): Number of bits for quantization. Default is 1.
        kmeans_niters (int): Number of iterations for k-means clustering. Default is 4.
        resume (bool): Flag to indicate whether to resume indexing. Default is False.
        pool_factor (int): Factor for pooling. Default is 1.
        clustering_mode (str): Mode of clustering to use. Default is "hierarchical".
        protected_tokens (int): Number of protected tokens. Default is 0.

    Properties:
        index_path_ (str): Returns the index path or constructs it from index_root_ and index_name.
    """

    index_path: str = DefaultVal(None)

    index_bsize: int = DefaultVal(64)

    nbits: int = DefaultVal(1)

    kmeans_niters: int = DefaultVal(4)

    resume: bool = DefaultVal(False)

    pool_factor: int = DefaultVal(1)

    clustering_mode: str = DefaultVal("hierarchical")

    protected_tokens: int = DefaultVal(0)

    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)


@dataclass
class SearchSettings:
    """
    SearchSettings is a dataclass that holds configuration settings for search operations.

    Attributes:
        ncells (int): Number of cells to use in the search. Default is None.
        centroid_score_threshold (float): Threshold for the centroid score. Default is None.
        ndocs (int): Number of documents to retrieve. Default is None.
        load_index_with_mmap (bool): Flag to determine if the index should be loaded with memory mapping. Default is False.
    """

    ncells: int = DefaultVal(None)
    centroid_score_threshold: float = DefaultVal(None)
    ndocs: int = DefaultVal(None)
    load_index_with_mmap: bool = DefaultVal(False)
