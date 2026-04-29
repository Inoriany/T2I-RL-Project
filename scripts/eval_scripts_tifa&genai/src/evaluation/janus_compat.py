from __future__ import annotations

import copy
import dataclasses
import functools
from typing import Any, Dict


META_TENSOR_ITEM_ERROR = "Tensor.item() cannot be called on meta tensors"


def _patch_mutable_defaults_on_config_class(cls: type[Any]) -> bool:
    patched = False
    for field_name in getattr(cls, "__annotations__", {}):
        if not hasattr(cls, field_name):
            continue
        current_value = getattr(cls, field_name)
        if isinstance(current_value, dataclasses.Field):
            continue
        if isinstance(current_value, dict):
            setattr(cls, field_name, dataclasses.field(default_factory=dict))
            patched = True
        elif isinstance(current_value, list):
            setattr(cls, field_name, dataclasses.field(default_factory=list))
            patched = True
        elif isinstance(current_value, set):
            setattr(cls, field_name, dataclasses.field(default_factory=set))
            patched = True
        else:
            try:
                hash(current_value)
            except TypeError:
                setattr(
                    cls,
                    field_name,
                    dataclasses.field(
                        default_factory=lambda value=current_value: copy.deepcopy(value)
                    ),
                )
                patched = True
            else:
                continue
    return patched


def _patch_missing_post_init_on_model_class(cls: type[Any]) -> bool:
    if getattr(cls, "__codex_post_init_patched__", False):
        return False

    post_init = getattr(cls, "post_init", None)
    original_init = getattr(cls, "__init__", None)
    if post_init is None or original_init is None:
        return False

    @functools.wraps(original_init)
    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()

    cls.__init__ = _patched_init
    cls.__codex_post_init_patched__ = True
    return True


def import_janus_vlchatprocessor() -> type[Any]:
    from transformers import configuration_utils as transformers_configuration_utils

    original_stdlib_dataclass = dataclasses.dataclass
    original_transformers_dataclass = getattr(transformers_configuration_utils, "dataclass", None)

    def _safe_dataclass(cls=None, **kwargs):
        if cls is None:
            return lambda wrapped_cls: _safe_dataclass(wrapped_cls, **kwargs)
        try:
            return original_stdlib_dataclass(cls, **kwargs)
        except ValueError as exc:
            if "mutable default" not in str(exc):
                raise
            if not _patch_mutable_defaults_on_config_class(cls):
                raise
            return original_stdlib_dataclass(cls, **kwargs)

    dataclasses.dataclass = _safe_dataclass
    if original_transformers_dataclass is not None:
        transformers_configuration_utils.dataclass = _safe_dataclass
    try:
        from janus.models import MultiModalityCausalLM, VLChatProcessor
    finally:
        dataclasses.dataclass = original_stdlib_dataclass
        if original_transformers_dataclass is not None:
            transformers_configuration_utils.dataclass = original_transformers_dataclass
    _patch_missing_post_init_on_model_class(MultiModalityCausalLM)
    return VLChatProcessor


def build_janus_model_load_kwargs(
    *,
    torch_dtype: Any,
    quantization_config: Any = None,
) -> Dict[str, Any]:
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = "auto"
        return load_kwargs

    # Janus' SigLIP tower calls `.item()` during module construction, which
    # breaks when transformers/accelerate build the model on the meta device.
    load_kwargs["low_cpu_mem_usage"] = False
    return load_kwargs


def build_janus_retry_load_kwargs(load_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    retry_kwargs = dict(load_kwargs)
    retry_kwargs["low_cpu_mem_usage"] = False
    retry_kwargs.pop("device_map", None)
    return retry_kwargs


def is_meta_tensor_item_error(exc: BaseException) -> bool:
    return META_TENSOR_ITEM_ERROR in str(exc)
