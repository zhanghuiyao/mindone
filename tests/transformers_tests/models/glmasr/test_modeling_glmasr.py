# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the MindSpore glmasr model."""

import unittest

import numpy as np
import pytest

import mindspore
from mindone.transformers import (
    GlmAsrConfig,
    GlmAsrForConditionalGeneration,
)


class GlmAsrModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        audio_token_id=0,
        seq_length=35,
        feat_seq_length=64,
        text_config={
            "model_type": "llama",
            "intermediate_size": 64,
            "initializer_range": 0.02,
            "hidden_size": 16,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "use_labels": True,
            "use_mrope": False,
            "vocab_size": 99,
            "head_dim": 8,
            "pad_token_id": 1,  # can't be the same as the audio token id
        },
        is_training=True,
        audio_config={
            "model_type": "glmasr_encoder",
            "hidden_size": 128,
            "num_attention_heads": 2,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_mel_bins": 128,
            "max_source_positions": 32,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.audio_token_id = audio_token_id
        self.text_config = text_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        return GlmAsrConfig(
            text_config=self.text_config,
            encoder_config=self.audio_config,
            ignore_index=self.ignore_index,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_features = mindspore.Tensor(
            np.random.randn(self.batch_size, self.feat_seq_length, self.audio_config["num_mel_bins"]),
            dtype=mindspore.float32,
        )
        decoder_input_ids = mindspore.Tensor(
            np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_length)),
            dtype=mindspore.int64,
        )

        return config, input_features, decoder_input_ids

    def prepare_config_and_inputs_for_common(self):
        config, input_features, decoder_input_ids = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
        }
        return config, inputs_dict

    def create_and_check_model(self, config, input_features, decoder_input_ids):
        model = GlmAsrForConditionalGeneration(config)
        model.set_train(False)
        result = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
        self.parent.assertEqual(
            result.logits.shape,
            (self.batch_size, self.seq_length + self.feat_seq_length // 10, self.vocab_size),
        )


class GlmAsrModelTest(unittest.TestCase):
    all_model_classes = (GlmAsrForConditionalGeneration,)
    test_headmasking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = GlmAsrModelTester(self)

    def test_model(self):
        config, input_features, decoder_input_ids = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_features, decoder_input_ids)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = set(model.construct.__code__.co_varnames)
            # Check that the model has expected inputs
            expected_inputs = {"input_features", "decoder_input_ids", "attention_mask", "labels"}
            self.assertTrue(expected_inputs.issubset(signature))

    def test_model_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = GlmAsrForConditionalGeneration(config)
        # Check that model can be initialized
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.language_model)
        self.assertIsNotNone(model.multi_modal_projector)

    def test_config_save_load(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        # Test that config can be saved and loaded
        config_dict = config.to_dict()
        loaded_config = GlmAsrConfig.from_dict(config_dict)
        self.assertEqual(config.to_dict(), loaded_config.to_dict())

    @pytest.mark.skip(reason="GlmAsr does not support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @pytest.mark.skip(reason="GlmAsr does not have a standalone forward for the base model")
    def test_model_common_attributes(self):
        pass

    @pytest.mark.skip(reason="Not applicable for GlmAsr")
    def test_retain_grad_hidden_states_attentions(self):
        pass


if __name__ == "__main__":
    unittest.main()
