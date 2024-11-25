import os
import torch

# from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS
from colbert.infra.run import Run
import mlflow
from colbert.modeling.colbert import ColBERT
from colbert.infra import ColBERTConfig
from typing import cast


def print_progress(scores):
    positive_avg, negative_avg = (
        round(scores[:, 0].mean().item(), 2),
        round(scores[:, 1].mean().item(), 2),
    )
    print(
        "#>>>   ", positive_avg, negative_avg, "\t\t|\t\t", positive_avg - negative_avg
    )
    # mlflow.log_metrics(
    #    {
    #       "positive_avg": positive_avg,
    ##       "negative_avg": negative_avg,
    #     "positive_minus_negative_avg": positive_avg - negative_avg,
    # },
    # synchronous=False,
    # )


def manage_checkpoints(
    args: ColBERTConfig,
    colbert: torch.nn.parallel.DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    batch_idx: int,
    savepath=None,
    consumed_all_triples=False,
):
    """
    Manages the saving of checkpoints during training.

    Parameters:
    args (Namespace): The arguments for the training process.
    colbert (DistributedDataParallel): The ColBERT model wrapped in DistributedDataParallel.
    optimizer (Optimizer): The optimizer used for training.
    batch_idx (int): The current batch index.
    savepath (str, optional): The path where checkpoints will be saved. Defaults to None.
    consumed_all_triples (bool, optional): Flag indicating if all triples have been consumed. Defaults to False.

    Returns:
    str: The path where the checkpoint was saved, or None if no checkpoint was saved.
    """
    # arguments = dict(args)

    # TODO: Call provenance() on the values that support it??

    checkpoints_path = savepath or os.path.join(Run().path_, "checkpoints")
    name = None

    try:
        save = colbert.save
    except:
        save = colbert.module.save

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    path_save = None

    if consumed_all_triples or (batch_idx % 2000 == 0):
        # name = os.path.join(path, "colbert.dnn")
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, "colbert")

    if batch_idx in SAVED_CHECKPOINTS:
        # name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, f"colbert-{batch_idx}")

    if path_save:
        print(f"#> Saving a checkpoint to {path_save} ..")

        # checkpoint = {}
        # checkpoint["batch"] = batch_idx
        # checkpoint['epoch'] = 0
        # checkpoint["model_state_dict"] = colbert.state_dict()
        # checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        # checkpoint["arguments"] = args.export()
        # colbert.colbert_config.set_new_key("model_state_dict", colbert.state_dict())
        # colbert.colbert_config.set_new_key("optimizer_state_dict", optimizer.state_dict())
        # c#olbert.colbert_config.set_new_key("arguments", args.export())
        # colbert.colbert_config.set_new_key("batch", batch_idx)
        colbert_module = cast(ColBERT, colbert.module)
        colbert_module.colbert_config.set("batch_idx", batch_idx)
        # colbert.colbert_config.set("model_state_dict", colbert.state_dict())
        colbert_module.colbert_config.set(
            "optimizer_state_dict", optimizer.state_dict()
        )
        #colbert_module.colbert_config.set("arguments", args.export().tolist())

        save(path_save)

    return path_save
