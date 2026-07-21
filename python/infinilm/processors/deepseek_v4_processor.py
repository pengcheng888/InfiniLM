from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("deepseek_v4")
class DeepSeekV4Processor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        self._fix_missing_chat_template()

    def _fix_missing_chat_template(self):
        if getattr(self.tokenizer, "chat_template", None):
            return

        user_token = self.tokenizer.convert_ids_to_tokens(128803)
        assistant_token = self.tokenizer.convert_ids_to_tokens(128804)
        thinking_end_token = "</think>"

        self.tokenizer.chat_template = (
            "{{ bos_token }}"
            "{%- for message in messages -%}"
            "{%- if message['role'] == 'system' -%}"
            "{{ message['content'] }}"
            "{%- elif message['role'] == 'developer' -%}"
            f"{user_token}{{{{ message['content'] }}}}"
            "{%- if loop.last and add_generation_prompt -%}"
            f"{assistant_token}{thinking_end_token}"
            "{%- endif -%}"
            "{%- elif message['role'] == 'user' -%}"
            f"{user_token}{{{{ message['content'] }}}}"
            "{%- if (not loop.last and messages[loop.index0 + 1]['role'] == 'assistant') or (loop.last and add_generation_prompt) -%}"
            f"{assistant_token}{thinking_end_token}"
            "{%- endif -%}"
            "{%- elif message['role'] == 'assistant' -%}"
            "{{ message['content'] }}{{ eos_token }}"
            "{%- endif -%}"
            "{%- endfor -%}"
        )
