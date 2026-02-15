import copy
import sys
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

def _register_global(cls: type) -> None:
    """Make cls picklable by registering it into its module globals."""
    mod = sys.modules[cls.__module__]
    setattr(mod, cls.__name__, cls)


def moeify(num_experts: int = 4, fixed_expert: int = 0):
    """
    Class decorator: turn an nn.Module class into a MoE container class (same name),
    while also generating and registering an Expert class for internal use and pickling.

    - old .pt (pickled old structure) will unpickle into MoE class (same name)
      and be upgraded inside __setstate__.
    - legacy state_dict (no experts.*) will be loaded into expert0 and replicated.
    """
    assert num_experts >= 1
    assert fixed_expert >= 0

    def deco(OrigCls: type):
        if not issubclass(OrigCls, nn.Module):
            raise TypeError("moeify can only decorate nn.Module subclasses")

        module_name = OrigCls.__module__
        orig_name = OrigCls.__name__
        expert_name = f"{orig_name}Expert"

        # 1) Build Expert class by copying OrigCls dict but inherit from nn.Module
        #    (avoid depending on OrigCls at load time).
        expert_dict = {}
        for k, v in OrigCls.__dict__.items():
            # keep most things, but skip descriptors that would conflict
            if k in ("__dict__", "__weakref__"):
                continue
            expert_dict[k] = v

        ExpertCls = type(expert_name, (nn.Module,), expert_dict)
        ExpertCls.__module__ = module_name
        _register_global(ExpertCls)

        # 2) Build MoE container class (same name as original)
        class MoEClass(nn.Module):
            __module__ = module_name
            __name__ = orig_name  # for readability; pickling uses globals binding anyway

            def __init__(self, *args, **kwargs):
                super().__init__()
                self.num_experts = int(num_experts)
                self.register_buffer("_fixed_expert", torch.tensor(int(fixed_expert), dtype=torch.long))
                self._moe_upgraded = True  # constructed as MoE already

                # create experts
                e0 = ExpertCls(*args, **kwargs)
                experts = [e0]
                for _ in range(self.num_experts - 1):
                    experts.append(copy.deepcopy(e0))
                self.experts = nn.ModuleList(experts)

                # (optional) placeholder gate; for now fixed
                # self.gate = ...

            def _replicate_from0_(self):
                """Copy parameters & buffers from expert0 to others (in-place)."""
                with torch.no_grad():
                    src = self.experts[0].state_dict()
                    for i in range(1, self.num_experts):
                        self.experts[i].load_state_dict(src, strict=True)

            def forward(self, *args, **kwargs):
                idx = int(self._fixed_expert.item())
                if idx < 0 or idx >= len(self.experts):
                    idx = 0
                return self.experts[idx](*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                # delegate unknown attributes to expert0 for backward compatibility
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    experts = super().__getattribute__("__dict__").get("experts", None)
                    if isinstance(experts, nn.ModuleList) and len(experts) > 0:
                        return getattr(experts[0], name)
                    raise

            def __setstate__(self, state: Dict[str, Any]):
                """
                Called by pickle on torch.load(.pt).
                Old checkpoints may contain the *old* module state (no experts).
                We upgrade that legacy state into experts[0] and replicate.
                """
                # Let nn.Module restore raw dict first
                super().__setstate__(state)

                # If already MoE (new ckpt), nothing to do
                if hasattr(self, "experts") and isinstance(self.experts, nn.ModuleList):
                    return

                # Legacy: the current object holds old fields (_modules, _parameters, etc.)
                legacy = dict(self.__dict__)  # shallow copy of restored state

                # Re-init MoE container cleanly
                nn.Module.__init__(self)
                self.num_experts = int(num_experts)
                self.register_buffer("_fixed_expert", torch.tensor(int(fixed_expert), dtype=torch.long))
                self._moe_upgraded = True

                # Build expert0 by injecting legacy dict into an ExpertCls instance
                e0 = ExpertCls.__new__(ExpertCls)
                nn.Module.__init__(e0)
                e0.__dict__.update(legacy)

                experts = [e0]
                for _ in range(self.num_experts - 1):
                    experts.append(copy.deepcopy(e0))
                self.experts = nn.ModuleList(experts)

            def _load_from_state_dict(
                self,
                state_dict: Dict[str, torch.Tensor],
                prefix: str,
                local_metadata: Dict[str, Any],
                strict: bool,
                missing_keys: list,
                unexpected_keys: list,
                error_msgs: list,
            ):
                """
                Support loading legacy state_dict that contains keys for original module
                (prefix + 'conv.weight', ...) rather than MoE ('experts.0.conv.weight', ...).
                """
                # If state_dict already has experts.* keys, default behavior is fine
                has_experts_keys = any(k.startswith(prefix + "experts.") for k in state_dict.keys())
                if has_experts_keys:
                    return super()._load_from_state_dict(
                        state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs
                    )

                # Legacy: collect keys belonging to this module but not experts.*
                legacy_sub: Dict[str, torch.Tensor] = {}
                legacy_keys = []
                for k in list(state_dict.keys()):
                    if k.startswith(prefix) and not k.startswith(prefix + "experts."):
                        legacy_sub[k[len(prefix):]] = state_dict[k]
                        legacy_keys.append(k)

                # Consume them to prevent children recursion from seeing them
                for k in legacy_keys:
                    state_dict.pop(k)

                # Load into expert0
                # Use strict=False here because upstream (Ultralytics) usually loads with strict=False anyway.
                self.experts[0].load_state_dict(legacy_sub, strict=False)
                self._replicate_from0_()

                # Do not call super; we already handled this module's legacy weights.
                return

        MoEClass.__name__ = orig_name
        MoEClass.__qualname__ = OrigCls.__qualname__
        MoEClass.__module__ = module_name

        _register_global(MoEClass)

        return MoEClass

    return deco
