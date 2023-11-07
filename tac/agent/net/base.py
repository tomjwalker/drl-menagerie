from abc import ABC, abstractmethod


class Net(ABC):
    """Abstract Net class to define the API methods for all net classes"""

    def __init__(self, net_spec, in_dim, out_dim):

        self.net_spec = net_spec
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_norms = None
        # TODO: add functionality for GPU below
        self.device = "cpu"

    @abstractmethod
    def forward(self):
        """The forward step for a specific net class"""
        raise NotImplementedError

    def train_step(self, loss, optim, lr_scheduler=None, clock=None, global_net=None):
        """Performs a single training step. This is a backward pass, followed by a gradient update.

        Arguments:
        ----------
        loss: torch.Tensor
            The loss to be minimised
        optim: torch.optim.Optimiser
            The optimiser to use for the gradient update
        lr_scheduler: torch.optim.lr_scheduler
            The learning rate scheduler to use
        clock: Clock
            The clock object to use for timing
        global_net: Net
            The global net to use for the gradient clipping
        """
        # TODO: lr scheduler

        # optimiser.zero_grad() clears the gradients of all optimisable variables, from the previous time step
        optim.zero_grad()
        # loss.backward() computes the gradient of the loss w.r.t. all optimisable variables, del(loss)/del(theta)
        loss.backward()
        # optimiser.step() updates the value of the optimisable variables, using the gradients computed in the previous
        # step
        optim.step()

        # TODO: clip grad and global net

        if clock is not None:
            clock.tick("opt_step")

        return loss

    def store_grad_norms(self):
        """Stores the gradient norms for each layer of the net"""
        norms = [param.grad.norm().item() for param in self.parameters()]
        self.grad_norms = norms
