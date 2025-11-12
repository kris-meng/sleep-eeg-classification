# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BART model configuration"""
import warnings
from collections import OrderedDict
from typing import Any, Mapping, Optional

from transformers import PretrainedConfig, PreTrainedTokenizer, TensorType, is_torch_available
from transformers.onnx import OnnxSeq2SeqConfigWithPast, OnnxConfigWithPast
from transformers.onnx.utils import compute_effective_axis_dimension
from transformers.utils import logging

logger = logging.get_logger(__name__)

BART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/config.json",
    # See all BART models at https://huggingface.co/models?filter=bart
}


class Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BartModel`]. It is used to instantiate a BART
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BART
    [facebook/bart-large](https://huggingface.co/facebook/bart-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BartModel`] or [`TFBartModel`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        max_source_positions('int',*optional*, defaults to 7500):
            the maximum sequence length that this model might ever be used with.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        num_labels (`int`, *optional*, defaults to 3):
            The number of labels to use in [`BartForSequenceClassification`].
        forced_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.

    Example:

    ```python

    ```"""

    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        encoder_layers=6,
        encoder_attention_heads=6,
        decoder_layers=2,
        decoder_attention_heads=6,
        decoder_ffn_dim=1536,
        encoder_ffn_dim=1536,
        num_labels=5,
        encoder_layerdrop=0.1,
        decoder_layerdrop=0.1,
        use_cache=False,
        activation_function="gelu",
        d_model=384,
        dropout=0.1,
        init_std=0.02,
        scale_embedding=False,
        max_source_positions=316,
        max_target_positions=448,
        suppress_tokens=None,
        begin_suppress_tokens=[220, 50256],
        median_filter_width=7,
        vocab_size=64,
        hidden_size=768,
        intermediate_size=3072,
        input_dim=1,
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.0,
        feat_quantizer_dropout=0.0,
        final_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="layer",
        feat_extract_activation="gelu",
        conv_dim=(64, 64, 64),
        conv_stride=(3,2,2),
        conv_kernel=(1250,625,100),
        conv_bias=True,
        deconv_dim=(64, 64, 1),
        deconv_stride=(2, 2, 2),  
        deconv_kernel=(40,113,100),
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=8,
        do_stable_layer_norm=True,
        apply_spec_augment=True,
        mask_time_prob=0.65,
        mask_time_length=50,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        num_codevectors_per_group=64,
        num_codevector_groups=2,
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        codevector_dim=128,
        proj_codevector_dim=128,
        diversity_loss_weight=0.1,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=32,
        tdnn_dim=(512, 512, 512, 512, 1500),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        xvector_output_dim=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        add_adapter=False,
        adapter_kernel_size=3,
        adapter_stride=2,
        num_adapter_layers=3,
        output_hidden_size=768,
        output_hidden_states=True,
        adapter_attn_dim=None,
        **kwargs,
    ):
        """

        :type max_source_positions: object
        """
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.deconv_dim = list(deconv_dim)
        self.deconv_stride = list(deconv_stride)
        self.deconv_kernel = list(deconv_kernel)
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim)
        self.intermediate_size = intermediate_size
        self.input_dim = input_dim
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.output_hidden_states = output_hidden_states
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.use_weighted_layer_sum = use_weighted_layer_sum

        if (
                (len(self.conv_stride) != self.num_feat_extract_layers)
                or (len(self.conv_kernel) != self.num_feat_extract_layers)
                or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        # parameters for pretraining with codevector quantized representations
        self.num_codevectors_per_group = num_codevectors_per_group
        self.num_codevector_groups = num_codevector_groups
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.feat_quantizer_dropout = feat_quantizer_dropout
        self.num_negatives = num_negatives
        self.codevector_dim = codevector_dim
        self.max_source_positions = max_source_positions
        self.proj_codevector_dim = proj_codevector_dim
        self.diversity_loss_weight = diversity_loss_weight

        # ctc loss
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # adapter
        self.add_adapter = add_adapter
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_stride = adapter_stride
        self.num_adapter_layers = num_adapter_layers
        self.output_hidden_size = output_hidden_size or hidden_size
        self.adapter_attn_dim = adapter_attn_dim

        # SequenceClassification-specific parameter. Feel free to ignore for other classes.
        self.classifier_proj_size = classifier_proj_size

        # XVector-specific parameters. Feel free to ignore for other classes.
        self.tdnn_dim = list(tdnn_dim)
        self.tdnn_kernel = list(tdnn_kernel)
        self.tdnn_dilation = list(tdnn_dilation)
        self.xvector_output_dim = xvector_output_dim

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        # Audio Classification-specific parameters. Feel free to ignore for other classes.
        self.classifier_proj_size = classifier_proj_size
        self.use_weighted_layer_sum = use_weighted_layer_sum

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        self.median_filter_width = median_filter_width

        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            **kwargs,
        )
