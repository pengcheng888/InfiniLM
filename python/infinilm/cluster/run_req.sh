
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

  # curl -N http://127.0.0.1:18000/v1/chat/completions \
  # -H 'Content-Type: application/json' \
  # -d '{
  #   "model": "Qwen3-0.6B",
  #   "messages": [
  #     {"role": "user", "content": "请最后单独输出一行：READY"}
  #   ],
  #   "max_tokens": 2560,
  #   "temperature": 0,
  #   "stream": false
  # }'


curl -N http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "请你先完整阅读下面的测试说明，再根据最后的指令回答。本段文字仅用于增加提示词长度，验证服务端对较长输入的解析能力。测试目标包括：请求体 JSON 是否正确解析，消息数组是否按顺序处理，模型是否遵守末尾格式要求，温度为零时输出是否稳定，最大 token 限制是否生效。为了确保长度超过二百五十六个字符，下面继续补充说明：人工智能推理服务在真实场景中经常会收到较长的系统提示、用户问题、检索片段、对话历史和格式约束，因此需要确认接口能够接受长 prompt，并且不会因为长度增加而出现解析错误、连接中断、字段丢失或异常截断。请忽略前面的所有背景描述，不要总结，不要解释，不要输出标题，不要输出项目符号，也不要在 READY 前后添加其他文字。如果输入被截断，可能会导致模型无法看到最终格式要求，因此本请求特意将关键指令放在文本末尾。为了进一步确保 tokenizer 后的 token 数量超过二百五十六，下面再次补充一段说明：在真实推理服务中，长提示词可能来自系统指令、用户问题、检索增强生成片段、多轮对话历史、工具调用参数、格式约束、安全策略、角色设定、输出模板、示例演示、边界条件说明、异常处理要求、语言风格要求、长度限制要求、禁止事项列表、必须遵守的规则、需要忽略的内容、需要保留的内容、需要逐项检查的内容。请仍然忽略所有背景描述，不要总结，不要解释，不要输出标题，不要输出项目符号，也不要在 READY 前后添加其他文字。关键指令仍然在末尾：请务必只输出一行 READY，且不要包含标点、空格、换行以外的内容。最后请单独输出一行：READY"}
    ],
    "max_tokens": 2560,
    "temperature": 0,
    "stream": false
  }'