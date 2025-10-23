"""
SegLLM主模块初始化文件
提供核心模型的导入接口

输入:
- 无直接输入，提供模块级导入

输出:
- LlavaLlamaForCausalLM: LLaVA-Llama因果语言模型类

功能:
- 从model模块导入LlavaLlamaForCausalLM类
- 为外部使用提供统一的模型访问接口
- 简化模型导入过程
"""

from .model import LlavaLlamaForCausalLM