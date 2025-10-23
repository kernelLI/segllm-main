"""
SegLLM训练日志回调模块
提供WandB日志记录和图像可视化功能

输入:
- imgs: 图像列表，可以是张量或numpy数组
- name: 日志名称，默认为"vis"
- keys: 图像标题列表，可选
- **kwargs: 额外关键字参数传递给wandb.log

输出:
- wandb_dump_images(): 无返回值，直接将图像记录到WandB

功能:
- 将多个图像拼接成一行网格图
- 支持张量和numpy数组格式的图像
- 自动创建matplotlib图形并记录到WandB
- 提供图像标题设置功能
- 自动清理图形内存
"""

import wandb
from matplotlib import pyplot as plt
import torch
def wandb_dump_images(imgs, name="vis", keys=None, **kwargs):
    """
    x: H X W X C
    y: H X W X C
    """
    if wandb.run is not None:
        n_imgs = len(imgs)
        fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
        for idx, img in enumerate(imgs):
            if torch.is_tensor(img):
                img = img.detach().cpu().float().numpy()
            axes[idx].imshow(img)
            if keys:
                axes[idx].title.set_text(keys[idx])
        fig.tight_layout()
        wandb.log({name: wandb.Image(fig), **kwargs})
        plt.close(fig)