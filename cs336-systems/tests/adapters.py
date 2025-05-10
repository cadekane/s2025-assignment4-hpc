#!/usr/bin/env python3
from __future__ import annotations

from typing import Type

import torch

from typing import Type
import torch
import triton
import triton.language as tl

import torch.distributed as dist

def get_rmsnorm_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm.
    The expectation is that this class will implement RMSNorm
    using standard PyTorch operations.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyRMSNormAutogradFunctionClass
    class RMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps=1e-8):
            # ctx.save_for_backward(x, weight)
            # ctx.eps = eps

            # rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
            # normed_x = x / rms
            # return normed_x * weight

            rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)  # shape (*, 1)
            normed_x = x / rms  # shape (*, H)
            ctx.save_for_backward(x, weight, rms, normed_x)
            ctx.eps = eps
            return normed_x * weight

        @staticmethod
        def backward(ctx, grad_output):
            x, weight, rms, normed_x = ctx.saved_tensors
            H = x.shape[-1]
            eps = ctx.eps

            # grad w.r.t. weight (g)
            grad_weight = (grad_output * normed_x).sum(dim=tuple(range(grad_output.ndim - 1)))

            # grad w.r.t. x
            gx = grad_output * weight  # shape (*, H)
            dot = (gx * normed_x).sum(dim=-1, keepdim=True)  # shape (*, 1)
            grad_x = (gx - normed_x * dot / H) / rms  # shape (*, H)

            return grad_x, grad_weight, None  # None for eps

    return RMSNormFunction



import torch
import triton
import triton.language as tl

# Define the Triton kernel for the forward pass
@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel invocation for forward pass of RMS normalization with fused operations.
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # Initialize RMS accumulator
    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _rms += a * a

    # Calculate RMS (reciprocal of standard deviation)
    rms = tl.sqrt(tl.sum(_rms) / N + eps)

    tl.store(Rstd + row, rms)

    # Fused normalization, scaling, and bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
        x_hat = x / rms  # Normalize
        y = x_hat * w + b  # Scale and add bias
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _rms_norm_bwd_fused( # backward RMSNorm kernel
    x_ptr, dy_ptr, dx_ptr,
    w_ptr, rstd_ptr,
    dg_partial_ptr,  # M x N partials
    M: tl.constexpr, N: tl.constexpr, stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x_ptrs = x_ptr + row_idx * stride + offsets
    dy_ptrs = dy_ptr + row_idx * stride + offsets
    dx_ptrs = dx_ptr + row_idx * stride + offsets
    w_ptrs = w_ptr + offsets
    rstd_val = tl.load(rstd_ptr + row_idx)

    x = tl.load(x_ptrs, mask=mask)
    dy = tl.load(dy_ptrs, mask=mask)
    w = tl.load(w_ptrs, mask=mask)

    # normalize input
    x_hat = x * rstd_val

    # ∂y/∂x part (scaled residual)
    dy_scaled = dy * w
    mean = tl.sum(x_hat * dy_scaled, axis=0) / N
    dx = rstd_val * (dy_scaled - x_hat * mean)

    tl.store(dx_ptrs, dx, mask=mask)

    # partial ∂L/∂g = dy * x_hat
    dg_partial = dy * x_hat
    dg_ptrs = dg_partial_ptr + row_idx * N + offsets
    tl.store(dg_ptrs, dg_partial, mask=mask)

# Define the RMSNorm autograd function class
class TritonRMSNormAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, eps=1e-5):
        ctx.eps = eps
    
        # Handle optional bias
        if bias is None:
            bias = torch.zeros_like(weight)
    
        # Flatten input if necessary
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])  # [M, N]
        M, N = x.shape

        # Added orig_shape to ctx
        ctx.orig_shape = orig_shape 
    
        # Compute norms for each row
        norm2 = torch.sum(x ** 2, dim=-1)
        rstd = 1.0 / torch.sqrt(norm2 / x.shape[-1] + eps)
        
        # Save rstd for backward
        ctx.save_for_backward(x, weight, bias, rstd)

    
        y = torch.empty_like(x, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        grid = (M,)
        stride = x.stride(0)
    
        _rms_norm_fwd_fused[grid](
            x, y, weight, bias, rstd, stride, N, eps, BLOCK_SIZE=128
        )
    
        return y.view(*orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, rstd = ctx.saved_tensors
        M, N = x.shape
        dy = dy.contiguous().view(M, N)
    
        dx = torch.empty_like(x)
        dg_partial = torch.empty_like(x)
    
        grid = (M,)
        stride = x.stride(0)
    
        _rms_norm_bwd_fused[grid](
            x, dy, dx,
            weight, rstd,
            dg_partial,
            M, N, stride,
            BLOCK_SIZE=128
        )
    
        dw = dg_partial.sum(dim=0)
        db = dy.sum(dim=0)
    
        dx = dx.view(ctx.orig_shape)  # restore input shape
    
        if bias is not None and bias.requires_grad:
            return dx, dw, db, None
        else:
            return dx, dw, None, None


def get_rmsnorm_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm
    using Triton kernels.
    """
    return TritonRMSNormAutogradFunction



def rmsnorm_backward_g_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to g.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to g. Shape: (H,)
    """
    eps = 1e-5  # small epsilon for numerical stability

    # Compute the norm over the last dimension
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)  # shape: (*, 1)

    # Normalize x
    x_normalized = x / rms  # shape: (*, H)

    # Gradient w.r.t. g is sum over the batch (and other leading dims if any)
    grad_g = torch.sum(x_normalized * grad_output, dim=tuple(range(x.dim() - 1)))  # shape: (H,)

    return grad_g


def rmsnorm_backward_x_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to x.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to x. Shape: (*, H)
    """
    # Get the dimension of the last axis (H)
    H = x.shape[-1]
    eps = 1e-5

    # Compute r = sqrt(1/H * sum(x^2) + eps), shape: (*, 1)
    norm_squared = x.pow(2).mean(dim=-1, keepdim=True)
    r = torch.sqrt(norm_squared + eps)

    # Normalize x: x̂ = x / r
    x_hat = x / r

    # Apply element-wise multiplication with g: y = x̂ ⊙ g
    gx = grad_output * g  # shape (*, H)

    # Compute dot(x̂, gx) across last dimension
    dot = (x_hat * gx).sum(dim=-1, keepdim=True)  # shape: (*, 1)

    # Compute gradient with respect to x
    grad_input = (gx - x_hat * dot / H) / r

    return grad_input

################################################################################################################

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union, Any
import collections


class DDPIndividualParameters(torch.nn.Module):
    """
    Distributed Data Parallel implementation that handles parameter broadcasting
    and gradient synchronization for distributed training.
    
    This implementation overlaps communication with computation during the backward
    pass by asynchronously communicating gradients as they become available.
    """
    
    def __init__(self, module: torch.nn.Module):
        """
        Initialize DDP container for a PyTorch module.
        
        Args:
            module: The PyTorch module to be parallelized
        """
        super().__init__()
        
        # Store the underlying module
        self.module = module
        
        # Check if distributed is initialized
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed package is not available or not initialized")
        
        # Get rank and world size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Get the device of the module
        self.device = next(module.parameters()).device
        
        # Handle parameter deduplication - important for tied weights
        self._build_params_for_sync()
        
        # Dictionary to store handles for asynchronous communication
        self.handles = []
        
        # Broadcast initial parameters to ensure all ranks start with the same weights
        self._broadcast_parameters()
        
        # Register backward hooks for gradient synchronization
        self._register_hooks()
    
    def _build_params_for_sync(self):
        """
        Build a deduplicated list of parameters that need synchronization.
        This is important for handling tied weights (parameters that are reused).
        """
        # Use dict to handle parameter deduplication by object ID
        self.param_to_name = {param: f"param_{i}" for i, param in enumerate([p for p in self.module.parameters() if p.requires_grad])}
        
        # Create a deduplicated list for parameter synchronization
        self.parameters_to_sync = list(self.param_to_name.keys())
        
        # For tracking params that have already been reduced (to handle tied weights)
        self.params_already_reduced = set()
    
    def _broadcast_parameters(self):
        """
        Broadcast parameters from rank 0 to all other ranks,
        preserving shared weights (tied parameters).
        """
        dist.barrier()
        
        seen_data_ptrs = set()
        for param in self.parameters_to_sync:
            data_ptr = param.data_ptr()
            if data_ptr not in seen_data_ptrs:
                dist.broadcast(param.data, src=0)
                seen_data_ptrs.add(data_ptr)
        
        dist.barrier()

    
    def _register_hooks(self):
        """
        Register backward hooks on parameters to trigger asynchronous gradient
        all-reduce operations when gradients become available.
        """
        # Track parameter object IDs to handle tied weights correctly
        self.grad_accs = []
        
        for param in self.parameters_to_sync:
            # Skip parameters that don't require gradient
            if not param.requires_grad:
                continue
            
            # Define the hook function to be called during backward pass
            def hook(grad, param_id=id(param)):
                # Skip parameters with no gradient
                if grad is None:
                    return grad
                
                # Only proceed if this parameter hasn't been reduced yet
                # This is important for tied weights where the same parameter
                # may have multiple hooks registered
                if param_id not in self.params_already_reduced:
                    # Find the parameter object from its ID
                    for p in self.parameters_to_sync:
                        if id(p) == param_id and p.grad is not None:
                            # Create a communication handle for this parameter
                            handle = self._all_reduce_grad_async(p)
                            if handle is not None:
                                self.handles.append(handle)
                            # Mark this parameter as already reduced
                            self.params_already_reduced.add(param_id)
                            break
                
                return grad
            
            # Use PyTorch's AccumulateGrad object to ensure hooks are properly chained
            # This is important for models that might have custom hooks or multiple
            # backwards passes using the same parameters
            grad_acc = param.grad_fn.next_functions[0][0] if param.grad_fn else None
            
            if grad_acc is not None and hasattr(grad_acc, 'variable'):
                # Register the hook on the AccumulateGrad object
                handle = grad_acc.register_hook(hook)
                self.grad_accs.append((grad_acc, handle))
            else:
                # Fallback to registering directly on the parameter
                param.register_hook(hook)
    
    def _all_reduce_grad_async(self, param: torch.Tensor) -> Any:
        """
        Perform asynchronous all-reduce on gradients.
        
        Args:
            param: Parameter whose gradient needs to be synchronized
            
        Returns:
            A handle to the asynchronous communication
        """
        if param.grad is None:
            # Skip parameters with no gradient
            return None
        
        # Get a reference to the gradient
        grad = param.grad.data
        
        # Start an asynchronous all-reduce operation
        # We divide by world_size to compute the average of gradients across all processes
        handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
        
        # Create a callback to divide gradients by world_size after the all-reduce completes
        def _callback():
            # Compute the average by dividing by the world size
            param.grad.data.div_(self.world_size)
        
        # If the handle has a callback method (newer PyTorch versions)
        if hasattr(handle, "add_callback"):
            handle.add_callback(_callback)
            return handle
        else:
            # For older PyTorch versions, we need to manually track the handle
            # and perform the division later in finish_gradient_synchronization()
            return handle
    
    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method with the provided arguments.
        
        Args:
            *inputs: Positional arguments to be passed to the module's forward
            **kwargs: Keyword arguments to be passed to the module's forward
            
        Returns:
            Output from the wrapped module's forward method
        """
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous communication operations to complete.
        This method should be called before the optimizer step.
        """
        # Wait for all handles to complete
        for handle in self.handles:
            if handle is not None:
                handle.wait()
        
        # For older PyTorch versions without callbacks, we need to manually 
        # divide gradients by world_size here
        if self.handles and not hasattr(self.handles[0], "add_callback"):
            for param in self.parameters_to_sync:
                if param.grad is not None:
                    param.grad.data.div_(self.world_size)
        
        # Clear the handles and reset the set of reduced parameters for the next iteration
        self.handles.clear()
        self.params_already_reduced.clear()

def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(
    params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
