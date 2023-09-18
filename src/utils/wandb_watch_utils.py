# This file is based on wandb/sdk/lib/wandb_torch.py
"""watch."""

import logging
from typing import Optional

import torch
from wandb.sdk.lib import telemetry
from wandb.wandb_torch import TorchHistory, log_track_init, log_track_update

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import wandb

logger = logging.getLogger("wandb")

_global_watch_idx = 0


def add_log_activations_hook(
    self: TorchHistory,
    module: "torch.nn.Module",
    name: str = "",
    prefix: str = "",
    log_freq: int = 0,
) -> None:
    """This instruments hooks into the pytorch module log activations after a forward pass.

    log_freq - log gradients/parameters every N batches
    """
    # if name is not None:
    prefix = prefix + name

    if not hasattr(module, "_wandb_hook_names"):
        module._wandb_hook_names = []

    def activation_log_hook(module, input_, output, log_track):
        if not log_track_update(log_track):
            return
        if isinstance(output, (list, tuple)):
            for i, out in enumerate(output):
                self.log_tensor_stats(out, "activations/" + prefix + str(i))
        elif isinstance(output, dict):
            for key, out in output.items():
                self.log_tensor_stats(out, "activations/" + prefix + key)
        else:
            self.log_tensor_stats(output, "activations/" + prefix)

    log_track_params = log_track_init(log_freq)
    try:
        hook = module.register_forward_hook(
            lambda mod, inp, outp: activation_log_hook(mod, inp, outp, log_track_params)
        )
        self._hook_handles["activations/" + prefix] = hook
        module._wandb_hook_names.append("activations/" + prefix)
    except RuntimeError as e:
        wandb.termwarn(
            f"Trying to register forward_hook failed ({e}) - skipping parameter tracking."
        )


def watch(
    modules,
    criterion=None,
    log: Optional[Literal["gradients", "parameters", "activations", "all"]] = "gradients",
    log_freq: int = 1000,
    idx: Optional[int] = None,
    log_graph: bool = False,
):
    """Hook into the torch model to collect gradients and the topology.

    Should be extended to accept arbitrary ML models.

    Args:
        modules: (torch.Module) The model to hook, can be a tuple or dict
        criterion: (torch.F) An optional loss value being optimized
        log: (str) One of "gradients", "parameters", "all", or None
        log_freq: (int) log gradients and parameters every N batches
        idx: (int) an index to be used when calling wandb.watch on multiple models
        log_graph: (boolean) log graph topology

    Returns:
        `wandb.Graph`: The graph object that will populate after the first backward pass

    Raises:
        ValueError: If called before `wandb.init` or if any of models is not a torch.nn.Module.
    """
    global _global_watch_idx

    with telemetry.context() as tel:
        tel.feature.watch = True

    logger.info("Watching")

    if wandb.run is None:
        raise ValueError("You must call `wandb.init` before calling watch")

    if log not in {"gradients", "parameters", "activations", "all", None}:
        raise ValueError("log must be one of 'gradients', 'parameters', 'all', or None")

    log_parameters = log in {"parameters", "all"}
    log_gradients = log in {"gradients", "all"}
    log_activations = log in {"activations", "all"}

    if isinstance(modules, (tuple, list)):
        modules = {str(i): model for i, model in enumerate(modules)}
    elif isinstance(modules, dict):
        pass
    else:
        modules = {"": modules}

    torch = wandb.util.get_module(
        "torch", required="wandb.watch only works with pytorch, couldn't import torch."
    )

    for model in modules.values():
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "Expected a pytorch model (torch.nn.Module). Received " + str(type(model))
            )

    graphs = []
    prefix = ""

    if idx is None:
        idx = _global_watch_idx
    for local_idx, model_id in enumerate(modules):
        model = modules[model_id]
        global_idx = idx + local_idx
        _global_watch_idx += 1
        if global_idx > 0:
            prefix = f"graph_{global_idx}/"
        if model_id != "":
            prefix += f"{model_id}/"

        if log_parameters:
            wandb.run._torch.add_log_parameters_hook(
                model,
                prefix=prefix,
                log_freq=log_freq,
            )

        if log_gradients:
            wandb.run._torch.add_log_gradients_hook(
                model,
                prefix=prefix,
                log_freq=log_freq,
            )

        if log_activations:
            add_log_activations_hook(
                wandb.run._torch,
                model,
                prefix=prefix,
                log_freq=log_freq,
            )

        if log_graph:
            graph = wandb.run._torch.hook_torch(model, criterion, graph_idx=global_idx)
            graphs.append(graph)
            # NOTE: the graph is set in run.summary by hook_torch on the backward pass
    return graphs


def unwatch(models=None):
    """Remove pytorch model topology, gradient and parameter hooks.

    Args:
        models: (list) Optional list of pytorch models that have had watch called on them
    """
    if models:
        if not isinstance(models, (tuple, list)):
            models = (models,)
        for model in models:
            if not hasattr(model, "_wandb_hook_names"):
                wandb.termwarn("%s model has not been watched" % model)
            else:
                for name in model._wandb_hook_names:
                    wandb.run._torch.unhook(name)
                delattr(model, "_wandb_hook_names")
                # TODO: we should also remove recursively model._wandb_watch_called

    else:
        wandb.run._torch.unhook_all()
