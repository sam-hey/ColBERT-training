import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
import mlflow


def train(config: ColBERTConfig, triples, queries=None, collection=None):
    mlflow.active_run()
    config.checkpoint = config.checkpoint or "bert-base-uncased"

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print(
        "Using config.bsize =",
        config.bsize,
        "(per process) and config.accumsteps =",
        config.accumsteps,
    )
    mlflow.log_params({"bsize": config.bsize, "accumsteps": config.accumsteps})

    if collection is not None:
        # config.reranker = False => LazyBatcher
        if config.reranker:
            reader = RerankBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
        else:
            reader = LazyBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert: ColBERT = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    if config.resume:
        print("#> Resuming training from checkpoint:", config.checkpoint)
        assert config.checkpoint is not None
        checkpoint = torch.load(config.checkpoint + "/optimizer.pt")
        colbert.load_state_dict(checkpoint["model"])

    colbert: torch.nn.parallel.DistributedDataParallel = (
        torch.nn.parallel.DistributedDataParallel(
            colbert,
            device_ids=[config.rank],
            output_device=config.rank,
            find_unused_parameters=True,
        )
    )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8
    )
    if not config.resume:
        # Reset the gradients of all optimized torch.Tensor s.
        optimizer.zero_grad()
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler = None
    if config.warmup is not None:
        print(
            f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps."
        )
        mlflow.log_params(
            {"warmup": config.warmup, "maxsteps": config.maxsteps}, synchronous=False
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup,
            num_training_steps=config.maxsteps,
        )

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)
    # config.amp = True
    amp = MixedPrecisionManager(config.amp)

    # initalize the labels tensor with 0s
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    if config.resume:
        start_batch_idx = checkpoint["batch_idx"]
        reader.skip_to_batch(start_batch_idx, config.bsize)
        train_loss = checkpoint["train_loss"]

        # reader.skip_to_batch(start_batch_idx, config.checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        """
        We use simple defaults with limited manual exploration on the official development 
        set for the learning rate (10−5), batch size (32 examples), and warm up (for 20,000 steps) with linear decay.
        """
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = (
                        torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    )
                    # config.distillation_alpha = 1.0
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(
                        target_scores, dim=-1
                    )

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(
                        log_scores, target_scores
                    )

                else:
                    """
                    We also employ in-batch negatives per GPU, where a cross-entropy
                    loss is applied to the positive score of each query against all passages
                    corresponding to other queries in the same batch.
                    """
                    loss = nn.CrossEntropyLoss()(scores, labels[: scores.size(0)])

                if config.use_ib_negatives:
                    if config.rank < 1:
                        print(f"\t\t\t\tLoss: {loss.item()}, IB Loss: {ib_loss.item()}")
                        mlflow.log_metrics(
                            {"ib_loss": ib_loss.item(), "loss": loss.item()},
                            step=batch_idx,
                            synchronous=False,
                        )

                    loss += ib_loss

                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        # exponential moving average (EMA)
        # momentum factor = 0.999 controlling how much weight is given to the previous train_loss
        # the new values do not have a big impact
        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        mlflow.log_metric("train_loss", train_loss, step=batch_idx, synchronous=False)

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            print_message(batch_idx, train_loss)
            manage_checkpoints(
                config,
                colbert,
                optimizer,
                batch_idx + 1,
                savepath=None,
                train_loss=train_loss,
            )

    if "batch_idx" not in locals():
        print("#> Checkpoint is the end! Exiting.")
        return config.checkpoint

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(
            config,
            colbert,
            optimizer,
            batch_idx + 1,
            savepath=None,
            consumed_all_triples=True,
            train_loss=train_loss,
        )

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
