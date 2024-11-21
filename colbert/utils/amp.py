import torch

from contextlib import contextmanager
from colbert.utils.utils import NullContextManager


class MixedPrecisionManager:
    def __init__(self, activated):
        """
        Initializes the AMP (Automatic Mixed Precision) utility.

        Args:
            activated (bool): A flag indicating whether AMP is activated. If True,
                              a GradScaler instance is created for scaling gradients.
        """
        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        """
        Returns a context manager for automatic mixed precision (AMP) if AMP is activated.

        If AMP is activated, this method returns `torch.cuda.amp.autocast()`, which enables
        automatic mixed precision for the operations within the context. If AMP is not activated,
        it returns a `NullContextManager`, which does nothing.

        Returns:
            contextlib.AbstractContextManager: A context manager for AMP if activated, otherwise a no-op context manager.
        """
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        """
        Performs the backward pass on the given loss.

        If automatic mixed precision (AMP) is activated, scales the loss before
        performing the backward pass to prevent underflow. Otherwise, performs
        the standard backward pass.

        Args:
            loss (torch.Tensor): The loss tensor to backpropagate.
        """
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None):
        """
        Performs a single optimization step for the given model.

        Args:
            colbert (torch.nn.Module): The model to be optimized.
            optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.

        Returns:
            None
        """
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                colbert.parameters(), 2.0, error_if_nonfinite=False
            )

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
