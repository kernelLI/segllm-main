"""
SegLLM语言模型模块 (LLaVA-Llama)
实现基于Llama架构的多模态因果语言模型

输入:
- config: LlavaConfig配置对象，包含模型超参数
- input_ids: 输入token ID序列
- attention_mask: 注意力掩码
- images: 图像字典，包含图像数据
- labels: 标签序列用于训练
- generation_target: 生成目标字典
- extra_replacement: 额外替换数据用于多模态处理

输出:
- LlavaLlamaModel: LLaVA-Llama模型实例
- LlavaLlamaForCausalLM: LLaVA-Llama因果语言模型实例
- forward(): 返回CausalLMOutputWithPast包含logits和损失
- encode_images(): 返回图像特征张量
- process_extra_replacement_data(): 返回处理后的替换张量和剩余任务

功能:
- 实现多模态输入处理（图像+文本）
- 支持图像编码和特征投影
- 处理额外替换数据（分割、生成等）
- 实现因果语言建模和前向传播
- 支持训练模式下的多任务损失计算
"""

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union,Dict
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast,ModelOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX
from transformers.modeling_outputs import BaseModelOutputWithPast
import logging
import numpy as np
from transformers.models.llama.modeling_llama import LlamaAttention
logger = logging.getLogger(__name__)

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
    def _prepare_decoder_attention_mask():
        return None
        
from llava.constants import REPLACEMENT_TYPE

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    base_model_prefix = "model"
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #self.lm_head_img = nn.Linear(3, config.vocab_size, bias=False) # FIXME: Add config
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def process_extra_replacement_data(self,data,ref):
        '''
        List[Union[torch.Tensor,Query]]
        '''
        REMAIN_LIST = ['mask-decode']
        final_tensors = []
        all_job_list = {}
        all_job_result = {}
        remaining_jobs = {}
        for sample_idx,row in enumerate(data):
            if isinstance(row,torch.Tensor):
                final_tensors.append(row)
            elif isinstance(row,list):  
                for b in  row:
                    query,args = b
                    if query in REMAIN_LIST:
                        if query not in remaining_jobs:
                            remaining_jobs[query] = []
                        if query == 'mask-decode':
                            args[0]['sample_idx'] = sample_idx
                        remaining_jobs[query].append((query,args))
                        final_tensors.append(torch.zeros(1,self.config.hidden_size).to(ref))
                        continue
                    if query not in all_job_list:
                        all_job_list[query] = []
                    idx = len(all_job_list[query])
                    all_job_list[query].append(args)
                    final_tensors.append((query,idx))
            else:
                raise NotImplemented
        ## process
        for q,v in all_job_list.items():
            with torch.autocast(device_type = ref.device.type):
                if q == 'image-encode':
                    all_job_result[q] = self.encode_images(torch.stack(v).to(ref)) # N L D
                elif q == 'mask-encode':
                    v = [x[0] for x in v]
                    all_job_result[q] = self.encode_images(torch.stack(v).to(ref),features='cls') # N L D
                    if self.training:
                        dropout = torch.rand(all_job_result[q].shape[0],1,1) > 0.5
                        dropout = dropout.bool().int().to(all_job_result[q])
                        all_job_result[q] =  all_job_result[q] * dropout
                    else:
                        all_job_result[q] =  all_job_result[q] * 0.5
                    

                elif q == 'bbox-encode':
                    v = [x[0] for x in v]
                    v_enc =  self.get_model().mask_enc_head(torch.stack(v).to(ref))     # project (N, 4) --> (N, D)
                    if self.training:
                        dropout = torch.rand(v_enc.shape[0],1) > 0.5
                        dropout = dropout.bool().int().to(v_enc)
                        v_enc = v_enc * dropout
                    else:
                        v_enc = v_enc * 0.5
                    all_job_result[q] = v_enc.unsqueeze(1)                              # (N, D) --> (N, 1, D)       N L D
                else:
                    raise NotImplemented
        for i in range(len(final_tensors)):
            if isinstance(final_tensors[i],tuple):
                query,idx = final_tensors[i]
                final_tensors[i] = all_job_result[query][idx]
                if len(final_tensors[i].shape) == 1:
                    final_tensors[i] = final_tensors[i][None]
        return torch.cat(final_tensors),remaining_jobs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Dict] = None,
        return_dict: Optional[bool] = None,
        generation_target: Optional[Dict] = None,
        return_generations=True,
        extra_inputs=None,
        extra_replacement=None,
        dataset_index=None, # hack, do not delete
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.training:
            encodings = {} 
        else:
            encodings = {}
        # encode vae for 
        replacement_mask = torch.zeros_like(input_ids,dtype=bool).to(input_ids.device)
        raw_input_ids = input_ids
        if labels is not None:
            raw_labels = labels # N L
        else:
            raw_labels = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images,novision=True)
        remaining_jobs = {} # list of additional targets
        if extra_replacement is not None:
            if not isinstance(extra_replacement['data'],torch.Tensor):
                lazy_encode = False
                extra_replacement['data'],remaining_jobs= self.process_extra_replacement_data(extra_replacement['data'],ref=torch.zeros(1).to(inputs_embeds))
            else:
                lazy_encode = True
            if self.training or labels is not None:
                extra_replacement_mask = (raw_input_ids == self.DEFAULT_VIDEO_TOKEN_IDX ) | ((raw_input_ids ==self.DEFAULT_SEGMENTATION_TOKEN_IDX )& (raw_labels == self.DEFAULT_SEGMENTATION_TOKEN_IDX) ) # | (
                if extra_replacement['mask'].shape[0] != inputs_embeds[extra_replacement_mask].shape[0]:
                    print("SKIPPED",extra_replacement['mask'].shape[0], inputs_embeds[extra_replacement_mask].shape[0])
                    # commnet this line for image generation where num tokens is expected to be larger
                    raise ValueError("Mismatch replacement shape, this should not happen for segmentation data. Plesae double check number of embeddings matches number of tokens to replace.")
                    extra_replacement['mask'] = torch.zeros(inputs_embeds[extra_replacement_mask].shape[0]).to(extra_replacement['mask'])     
                z = torch.zeros_like(inputs_embeds)
                if lazy_encode:
                    z2 = self.get_model().mm_projector(
                        extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])
                else:
                    z2 = extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]
                a,b = torch.where(extra_replacement_mask)
                z[a[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT],b[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]] += z2
                inputs_embeds[extra_replacement_mask][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = 0.0
                z = z + inputs_embeds
                inputs_embeds = z
                # print("Replaced:",len(extra_replacement['mask']==REPLACEMENT_TYPE.INPUT),len(extra_replacement['mask']==REPLACEMENT_TYPE.GEN))
                extra_tgt_mask = (extra_replacement['mask']==REPLACEMENT_TYPE.BASE )| (extra_replacement['mask']==REPLACEMENT_TYPE.GEN)
                extra_replacement_gt = extra_replacement['data'][extra_tgt_mask]
                loss_fn_extra = nn.L1Loss()
                if extra_replacement_gt.shape[0]==0:
                    loss_fn_extra = None
            else:
                assert labels is None
                extra_replacement_mask = (raw_input_ids == self.DEFAULT_VIDEO_TOKEN_IDX )
                print(len(extra_replacement['mask']==REPLACEMENT_TYPE.INPUT))

                z = torch.zeros_like(inputs_embeds)
                if lazy_encode:
                    z2 = self.get_model().mm_projector(
                        extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])
                else:
                    z2 = extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]
                print("z2",z2)
                a,b = torch.where(extra_replacement_mask)
                a = a[:extra_replacement['mask'].shape[0]]
                b = b[:extra_replacement['mask'].shape[0]]
                z[a[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT],b[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]] += z2
                inputs_embeds[extra_replacement_mask][:extra_replacement['mask'].shape[0]][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = 0.0
                z = z + inputs_embeds
                inputs_embeds = z
                print("HERE")
                #print(inputs_embeds[extra_replacement_mask][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])
                
                # inputs_embeds[extra_replacement_mask][:extra_replacement['mask'].shape[0]][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = self.get_model().vae_projector_image(
                #     extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT].to(inputs_embeds))
            # z.sum().backward()


        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
           # replacement_mask=replacement_mask, # allow looking into future for images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        img_decode = None
        aud_decode=None
        individual_losses = {}
        extra_gen = None
        extra_gen_idx = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            hidden_states.shape

            loss_fct = CrossEntropyLoss()

                #prediction = logits_img.argmax(-1)
     
            # Flatten the tokens
            
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss_lang = loss.detach().item()
            individual_losses['loss_lang'] = loss_lang
            if remaining_jobs:
                for job,job_tgt in remaining_jobs.items():
                    if job == 'mask-decode':
                        seg_mask_idx =  ((raw_input_ids ==self.DEFAULT_SEGMENTATION_TOKEN_IDX )& (raw_labels == self.DEFAULT_SEGMENTATION_TOKEN_IDX))
                        mask_preds = self.get_model().segmentator_predictor(hidden_states[:,:-1][seg_mask_idx[:,1:]].contiguous()) # L X D (project [SEG] hidden states transformer dim --> SAM prompt dim)
                        job_tgt = list([x[1][0] for x in job_tgt])
                        all_seg_images = np.unique([x['image_path'] for x in job_tgt]) # unique_image_path
                        all_seg_images_lookup = {v:idx for idx,v in enumerate(all_seg_images)}
                        image_indices = np.array([all_seg_images_lookup[x['image_path']] for x in job_tgt])
                        sample_indices = np.array([x['sample_idx'] for x in job_tgt])
                        job_tgt = np.array(job_tgt)
                        mask_inputs = dict(
                            images=[],          # input images          B x 3 x 1024 x 1024 (B = num convs = num images)
                            prompts=[],         # [SEG] hidden states   B x num_masks x 512 (prompt_dim * num_tokens)
                            targets=[],         # GT masks              B x num_masks x 1 x 1024 x 1024
                            prev_masks=[],       # GT REF masks          B x num_masks x 1 x 1024 x 1024,
                            sample_indices=sample_indices,
                        )
                        for idx,v in enumerate(all_seg_images):           # interate through each conv in batch, correspond to different image
                            mask_prompts = mask_preds[image_indices==idx] # L X D (for the current image)
                            mask_data = job_tgt[image_indices==idx]
                            mask_targets = torch.stack([x['mask'] for x in mask_data]) # L x H X W
                            mask_targets_aux = torch.stack([x['aux_mask'] for x in mask_data]) # L x H X W
                            mask_inputs['images'].append(mask_data[0]['image'])     # 3 x 1024 x 1024
                            mask_inputs['prompts'].append(mask_prompts)             # L x 512 (or 256)      (each image/conv has L masks)
                            mask_inputs['targets'].append(mask_targets)             # L x 1 x 1024 x 1024   GT tgt masks
                            mask_inputs['prev_masks'].append(mask_targets_aux)      # L x 1 x 1024 x 1024   GT ref masks
                        with torch.autocast(device_type=hidden_states.device.type):
                            _, mask_loss = self.get_segmentator()(**mask_inputs)
                            if self.training:
                                for k,v in mask_loss.items():
                                    individual_losses[f'segm_loss_{k}']=v.item()
                            else:
                                for k,v in mask_loss.items():                   # eval, there are additional metrics
                                    if k in [
                                        'mask', 
                                        'iou_per_mask', 
                                        'inter_per_mask', 
                                        'union_per_mask'                        # these are tensors, cannot take .item()
                                    ]:
                                        individual_losses[f'segm_loss_{k}']=v
                                    else:
                                        try:
                                            individual_losses[f'segm_loss_{k}']=v.item()
                                        except:
                                            individual_losses[f'segm_loss_{k}']=v
                                individual_losses['mask_data']=mask_data        # mask_data, not from SAM output (maybe GT masks)
                        loss += mask_loss['total_loss']
                    else:
                        raise ValueError
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            extra_gen=extra_gen,
            extra_gen_idx=extra_gen_idx,
            attentions=outputs.attentions,
            img_decode=img_decode,
            aud_decode=aud_decode,
            individual_losses=individual_losses,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "extra_replacement": kwargs.get("extra_replacement", None),
            }
        )
        return model_inputs

# AutoConfig.register("llava", LlavaConfig)
# AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
