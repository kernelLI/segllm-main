"""
SegLLM多模态编码器构建器
根据配置构建视觉塔编码器

输入:
- vision_tower_cfg: 视觉塔配置对象，包含mm_vision_tower等参数
- **kwargs: 额外关键字参数

输出:
- CLIPVisionTower: CLIP视觉塔实例
- LanguageBindVisionTower: LanguageBind视觉塔实例（未实现）

功能:
- 根据视觉塔名称构建对应的视觉编码器
- 支持CLIP和LanguageBind等视觉模型
- 处理绝对路径和模型名称两种情况
- 提供统一的视觉塔构建接口
"""

import os
from .clip_encoder import CLIPVisionTower
# from .image_bind_encoder import LanguageBindVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower == 'languagebind':
        raise NotImplemented
        #return LanguageBindVisionTower(vision_tower,args=vision_tower_cfg, **kwargs)
    raise ValueError(f'Unknown vision tower: {vision_tower}')
