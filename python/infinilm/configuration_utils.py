import copy
from typing import Any


class PretrainedConfig:
    def __init__(*args,
                 **kwargs):
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["transformers_version"] = "None"

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]

            output[key] = value

        # self._remove_keys_not_serialized(output)

        self.dict_dtype_to_str(output)

        return output

    def is_encoder_decoder(self):
        return False

    def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype") is not None and not isinstance(d["dtype"], str):
            d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_dtype_to_str(value)

    def get_text_config(self, decoder=None, encoder=None):
        """
        elif encoder and hasattr(config_to_return, "encoder_layers"):
                config_to_return.num_hidden_layers = config_to_return.encoder_layers

        return config_to_returnReturns the text config related to the text input (encoder) or text output (decoder) of the model. The
        `decoder` and `encoder` input arguments can be used to specify which end of the model we are interested in,
        which is useful on models that have both text input and output modalities.

        There are three possible outcomes of using this method:
        1. On most models, it returns the original config instance itself.
        2. On newer (2024+) composite models, it returns the text section of the config, which is nested under a set
            of valid names.
        3. On older (2023-) composite models, it discards decoder-only parameters when `encoder=True` and vice-versa.

        Args:
            decoder (`Optional[bool]`, *optional*):
                If set to `True`, then only search for decoder config names.
            encoder (`Optional[bool]`, *optional*):
                If set to `True`, then only search for encoder config names.
        """
        return_both = decoder == encoder  # both unset or both set -> search all possible names

        decoder_possible_text_config_names = ("decoder", "generator", "text_config")
        encoder_possible_text_config_names = ("text_encoder",)
        if return_both:
            possible_text_config_names = encoder_possible_text_config_names + decoder_possible_text_config_names
        elif decoder:
            possible_text_config_names = decoder_possible_text_config_names
        else:
            possible_text_config_names = encoder_possible_text_config_names

        valid_text_config_names = []
        for text_config_name in possible_text_config_names:
            if hasattr(self, text_config_name):
                text_config = getattr(self, text_config_name, None)
                if text_config is not None:
                    valid_text_config_names += [text_config_name]

        if len(valid_text_config_names) > 1:
            raise ValueError(
                f"Multiple valid text configs were found in the model config: {valid_text_config_names}. In this "
                "case, using `get_text_config()` would be ambiguous. Please specify the desired text config directly, "
                "e.g. `text_config = config.sub_config_name`"
            )
        elif len(valid_text_config_names) == 1:
            config_to_return = getattr(self, valid_text_config_names[0])
        else:
            config_to_return = self

        # handle legacy models with flat config structure, when we only want one of the configs
        if not return_both and len(valid_text_config_names) == 0 and config_to_return.is_encoder_decoder:
            config_to_return = copy.deepcopy(config_to_return)
            prefix_to_discard = "encoder" if decoder else "decoder"
            for key in config_to_return.to_dict():
                if key.startswith(prefix_to_discard):
                    delattr(config_to_return, key)
            # old encoder/decoder models may use "encoder_layers"/"decoder_layers" instead of "num_hidden_layers"
            if decoder and hasattr(config_to_return, "decoder_layers"):
                config_to_return.num_hidden_layers = config_to_return.decoder_layers
            elif encoder and hasattr(config_to_return, "encoder_layers"):
                config_to_return.num_hidden_layers = config_to_return.encoder_layers

        return config_to_return
