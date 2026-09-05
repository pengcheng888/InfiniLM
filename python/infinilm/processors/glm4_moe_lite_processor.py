from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("glm4_moe_lite")
class Glm4MoeLiteProcessor(BasicLLMProcessor):
    pass
