import os
from typing import Optional, Union
import infinicore

__all__ = ["AutoQwen3Model"]


class AutoQwen3Model:
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, os.PathLike]],
        device: infinicore.device,
        dtype=infinicore.dtype,
        backend="python",
    ):
        if backend == "python":
            from . import modeling_qwen3

            return modeling_qwen3.Qwen3ForCausalLM.from_pretrained(
                model_path,
                device=device,
                dtype=dtype,
            )

        elif backend == "cpp":
            raise KeyError("not support")
        raise KeyError("invalid backend")
