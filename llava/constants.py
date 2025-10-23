"""
SegLLM框架常量定义文件
定义了模型训练、推理和评估过程中使用的各种常量

输入:
- 无直接输入，定义常量供其他模块使用

输出:
- 各种常量定义，包括token索引、特殊token字符串、心跳超时等
- REPLACEMENT_TYPE类：定义输入替换类型枚举

功能:
- 定义模型训练相关的常量（IGNORE_INDEX、IMAGE_TOKEN_INDEX等）
- 定义各种特殊token字符串（图像、音频、视频、分割等）
- 定义纵向推理相关的特殊token（T0、T1、变化等）
- 提供输入替换类型的枚举定义
- 为整个SegLLM框架提供统一的常量引用
"""

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_AUDIO_TOKEN = "<audio>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_IM_GEN_START_TOKEN = "<im_gen_start>"
DEFAULT_AUDIO_GEN_START_TOKEN = "<au_gen_start>"
DEFAULT_VIDEO_GEN_START_TOKEN = "<vd_gen_start>"
DEFAULT_IM_GEN_END_TOKEN = "<im_gen_end>"
DEFAULT_IM_GEN_TOKEN = "<im_gen>"
DEFAULT_AUDIO_GEN_TOKEN = "<audio_gen>"
DEFAULT_AUDIO_GEN_START_TOKEN = "<audio_gen_start>"
DEFAULT_MSK_TOKEN = '<mask_gen>'
DEFAULT_BASE_TOKEN = '<base>'
DEFAULT_BASE_NULL_TOKEN = '<base_null>'
DEFAULT_SEGMENTATION_TOKEN = "<seg>"
DEFAULT_SEGMENTATION_INPUT_TOKEN = "<seg_input>"
FAKE_SEGM_TOKEN = '<FAKE_MASK_OUT>'

# 纵向推理相关token
DEFAULT_LONGITUDINAL_TOKEN = "<longitudinal>"
DEFAULT_IMAGE_T0_TOKEN = "<image_t0>"
DEFAULT_IMAGE_T1_TOKEN = "<image_t1>"
DEFAULT_CHANGE_TOKEN = "<change>"

class REPLACEMENT_TYPE:
    INPUT = 0
    BASE = 1
    GEN = 2
    SEG = 3