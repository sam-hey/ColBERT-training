from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT

import torch
import string

import os
import pathlib
from torch.utils.cpp_extension import load
from typing import Union, Tuple


class ColBERT(BaseColBERT):
    """
    This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name="bert-base-uncased", colbert_config: ColBERTConfig = None):
        super().__init__(name, colbert_config)
        self.use_gpu = colbert_config.total_visible_gpus > 0

        ColBERT.try_load_torch_extensions(self.use_gpu)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [
                    symbol,
                    self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0],
                ]
            }
        self.pad_token = self.raw_tokenizer.pad_token_id

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(
            "Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def forward(self, Q, D) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the ColBERT model.

        Args:
            Q (tuple): A tuple containing the query tensor and any additional arguments required by the `query` method.
            D (tuple): A tuple containing the document tensor and any additional arguments required by the `doc` method.

        Returns:
            torch.Tensor: The similarity scores between the query and document encodings.
            torch.Tensor (optional): The information bottleneck loss if `use_ib_negatives` is enabled in the configuration. CrossEntropyLoss is used for the IB loss.
        """
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims="return_mask")

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            # ib_loss is CrossEntropyLoss
            #
            return scores, ib_loss

        return scores

    def compute_ib_loss(
        self, Q: torch.Tensor, D: torch.Tensor, D_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the In-Batch (IB) loss for the given query and document tensors.

        In-Batch (IB) loss is a CrossEntropyLoss that is used to train the model with negative samples from the same batch.
        All negative samples are generated from the same batch, excluding the query itself.

        Args:
            Q (torch.Tensor): The query tensor of shape (batch_size, query_length, embedding_dim).
            D (torch.Tensor): The document tensor of shape (batch_size, doc_length, embedding_dim).
            D_mask (torch.Tensor): The mask tensor for documents of shape (batch_size, doc_length).

        Returns:
            torch.Tensor: The computed IB loss as a scalar tensor.

        Notes:
            - The function first computes the interaction scores between queries and documents.
            - The scores are reshaped to the appropriate dimensions.
            - Labels are generated for the CrossEntropyLoss computation.
            - Finally, the CrossEntropyLoss is computed between the scores and the labels.
        """
        # Compute the scores between queries and documents
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)

        # Reduce the scores using the colbert_score_reduce function
        scores = colbert_score_reduce(
            scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config
        )

        # Get the number of negative samples per query = 2
        nway = self.colbert_config.nway

        # Generate indices for all negative samples, excluding the query itself
        all_except_self_negatives = [
            list(range(qidx * D.size(0), qidx * D.size(0) + nway * qidx + 1))
            + list(
                range(
                    qidx * D.size(0) + nway * (qidx + 1), qidx * D.size(0) + D.size(0)
                )
            )
            for qidx in range(Q.size(0))
        ]

        # Flatten the scores for all negative samples
        scores = scores[flatten(all_except_self_negatives)]

        # Reshape the scores to have the correct dimensions
        scores = scores.view(Q.size(0), -1)

        # Generate labels for the CrossEntropyLoss
        # creates a tensor of size (Q.size(0),) with values from 0 to Q.size(0) - 1
        # nway = 2
        # output = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, ....])
        labels = torch.arange(0, Q.size(0), device=scores.device) * (
            self.colbert_config.nway
        )

        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = (
            input_ids.to(self.device),
            attention_mask.to(self.device),
        )
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = (
            torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device)
            .unsqueeze(2)
            .float()
        )
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, "return_mask"]

        input_ids, attention_mask = (
            input_ids.to(self.device),
            attention_mask.to(self.device),
        )
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = (
            torch.tensor(
                self.mask(input_ids, skiplist=self.skiplist), device=self.device
            )
            .unsqueeze(2)
            .float()
        )
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == "return_mask":
            return D, mask.bool()

        return D

    def score(
        self, Q: torch.Tensor, D_padded: torch.Tensor, D_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the similarity score between query embeddings (Q) and document embeddings (D_padded).

        Args:
            Q (torch.Tensor): Query embeddings of shape (batch_size, query_len, embedding_dim).
            D_padded (torch.Tensor): Padded document embeddings of shape (batch_size, doc_len, embedding_dim).
            D_mask (torch.Tensor): Mask for the document embeddings of shape (batch_size, doc_len).

        Returns:
            torch.Tensor: The similarity score between the query and document embeddings.
        """
        # assert self.colbert_config.similarity == 'cosine'
        if self.colbert_config.similarity == "l2":
            assert self.colbert_config.interaction == "colbert"
            return (
                (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1)) ** 2).sum(-1))
                .max(-1)
                .values.sum(-1)
            )
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [
            [(x not in skiplist) and (x != self.pad_token) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer


# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(
    scores_padded: torch.Tensor, D_mask: torch.Tensor, config: ColBERTConfig
) -> torch.Tensor:
    """
    Reduces the ColBERT scores by applying a mask and performing max-pooling or top-k pooling based on the interaction type.

    Args:
        scores_padded (torch.Tensor): The padded scores tensor of shape (batch_size, sequence_length).
        D_mask (torch.Tensor): The mask tensor indicating valid positions in the scores tensor.
        config (ColBERTConfig): Configuration object containing interaction type and query_maxlen.

    Returns:
        torch.Tensor: The reduced scores tensor of shape (batch_size,).

    Raises:
        AssertionError: If the interaction type in the config is not "colbert" or "flipr".
        AssertionError: If the interaction type is "flipr" and query_maxlen is not 64.
    """
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction in ["colbert", "flipr"], config.interaction

    if config.interaction == "flipr":
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, : config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen :].topk(K2, dim=-1).values.sum(1)

        return A + B

    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
    Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
    If Q.size(0) is 1, the matrix will be compared with all passages.
    Otherwise, each query matrix will be compared against the *aligned* passage.

    EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
    Works with a single query only.
    """

    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    if use_gpu or config.interaction == "flipr":
        scores_padded, scores_mask = StridedTensor(
            scores, D_lengths, use_gpu=use_gpu
        ).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        return ColBERT.segmented_maxsim(scores, D_lengths)
