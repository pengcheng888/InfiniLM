import os
from typing import Optional, Union
import infinicore

__all__ = ["AutoFM9GModel"]


class AutoFM9GModel:
    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, os.PathLike]],
        device: infinicore.device,
        dtype=infinicore.dtype,
        backend="python",
    ):
        if backend == "python":
            from . import modeling_fm9g

            return modeling_fm9g.FM9GForCausalLM.from_pretrained(
                model_path,
                device=device,
                dtype=dtype,
            )

        elif backend == "cpp":
            raise NotImplementedError(
                "CPP backend is not implemented yet for FM9G model"
            )

        raise KeyError("invalid backend")
