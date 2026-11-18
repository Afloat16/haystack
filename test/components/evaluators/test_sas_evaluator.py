# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.models import BoW, Pooling, StaticEmbedding, Transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.nn import Identity, Sigmoid
from transformers import BertConfig, BertForSequenceClassification, BertModel, PreTrainedTokenizerFast

from haystack import Pipeline
from haystack.components.evaluators.sas_evaluator import SASEvaluator
from haystack.utils import expit
from haystack.utils.device import ComponentDevice


class TestSASEvaluator:
    @pytest.fixture(params=["static", "bow"])
    def static_sentence_transformer(self, tmp_path, request):
        module: StaticEmbedding | BoW
        if request.param == "static":
            tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "cat": 1, "dog": 2, "vehicle": 3}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            module = StaticEmbedding(
                tokenizer=tokenizer, embedding_weights=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
            )
            expected_embeddings = [[1.0, 0.0], [0.8, 0.4], [0.0, 1.0]]
            expected_scores = [1.0, 2 / np.sqrt(5), 0.0]
        else:
            module = BoW(vocab=["cat", "dog", "vehicle"])
            expected_embeddings = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            expected_scores = [1.0, 1 / np.sqrt(2), 0.0]
        path = tmp_path / "similarity_model"
        SentenceTransformer(modules=[module], device="cpu").save_pretrained(str(path), create_model_card=False)
        assert (path / "modules.json").exists()
        assert not (path / "config.json").exists()
        loaded = SentenceTransformer(str(path), device="cpu")
        np.testing.assert_allclose(loaded.encode(["cat", "cat dog", "vehicle"]), expected_embeddings, atol=1e-7)
        return str(path), expected_scores

    def test_run_with_static_sentence_transformer(self, static_sentence_transformer, del_hf_env_vars):
        path, expected_scores = static_sentence_transformer
        evaluator = SASEvaluator.from_dict(SASEvaluator(model=path, device=ComponentDevice.from_str("cpu")).to_dict())
        result = evaluator.run(ground_truth_answers=["cat"] * 3, predicted_answers=["cat", "cat dog", "vehicle"])

        assert result["individual_scores"] == pytest.approx(expected_scores)
        assert result["score"] == pytest.approx(sum(expected_scores) / 3)
        loaded = evaluator._similarity_model
        assert isinstance(loaded, SentenceTransformer)
        evaluator.warm_up()
        assert evaluator._similarity_model is loaded

    @pytest.mark.parametrize("run_async", [False, True])
    @pytest.mark.asyncio
    async def test_pipeline_with_static_sentence_transformer(
        self, static_sentence_transformer, run_async, del_hf_env_vars
    ):
        path, expected_scores = static_sentence_transformer
        pipeline = Pipeline()
        pipeline.add_component("sas", SASEvaluator(model=path, device=ComponentDevice.from_str("cpu")))
        data = {"sas": {"ground_truth_answers": ["cat"] * 3, "predicted_answers": ["cat", "cat dog", "vehicle"]}}
        result = await pipeline.run_async(data) if run_async else pipeline.run(data)

        assert result["sas"]["individual_scores"] == pytest.approx(expected_scores)
        assert result["sas"]["score"] == pytest.approx(sum(expected_scores) / 3)

    @pytest.mark.parametrize("config_error", [OSError("missing configuration"), ValueError("unrecognized model")])
    def test_warm_up_static_model_passes_token(self, config_error, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")
        with (
            patch("haystack.components.evaluators.sas_evaluator.AutoConfig.from_pretrained", side_effect=config_error),
            patch(
                "haystack.components.evaluators.sas_evaluator.util.is_sentence_transformer_model", return_value=True
            ) as probe,
            patch("haystack.components.evaluators.sas_evaluator.SentenceTransformer") as model,
        ):
            SASEvaluator(model="private/static-model", device=ComponentDevice.from_str("cpu")).warm_up()

        probe.assert_called_once_with("private/static-model", token="fake-token")
        model.assert_called_once_with("private/static-model", device="cpu", token="fake-token")

    @pytest.mark.parametrize("config_error", [OSError("missing configuration"), ValueError("unrecognized model")])
    def test_warm_up_preserves_configuration_error_for_other_models(self, config_error):
        with (
            patch("haystack.components.evaluators.sas_evaluator.AutoConfig.from_pretrained", side_effect=config_error),
            patch(
                "haystack.components.evaluators.sas_evaluator.util.is_sentence_transformer_model", return_value=False
            ),
            patch("haystack.components.evaluators.sas_evaluator.SentenceTransformer") as model,
            pytest.raises(type(config_error), match=re.escape(str(config_error))) as exc_info,
        ):
            SASEvaluator(model="missing/model").warm_up()

        assert exc_info.value is config_error
        model.assert_not_called()

    @pytest.mark.parametrize("cross_encoder", [False, True])
    def test_warm_up_keeps_transformer_model_type(self, tmp_path, cross_encoder, del_hf_env_vars):
        base_path = tmp_path / "backbone"
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "cat": 2}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
        tokenizer_fast.save_pretrained(base_path)
        config = BertConfig(
            vocab_size=3, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, num_labels=1
        )
        model: CrossEncoder | SentenceTransformer
        if cross_encoder:
            backbone = BertForSequenceClassification(config)
            with torch.no_grad():
                for parameter in backbone.parameters():
                    parameter.zero_()
                backbone.classifier.bias.fill_(0.75)
            backbone.save_pretrained(base_path)
            model = CrossEncoder(str(base_path), device="cpu")
            expected = expit(0.75)
        else:
            BertModel(config).save_pretrained(base_path)
            model = SentenceTransformer(modules=[Transformer(str(base_path)), Pooling(8)], device="cpu")
            expected = 1.0
        model_path = tmp_path / "similarity_model"
        model.save_pretrained(str(model_path), create_model_card=False)
        evaluator = SASEvaluator(model=str(model_path), device=ComponentDevice.from_str("cpu"))

        result = evaluator.run(ground_truth_answers=["cat"], predicted_answers=["cat"])

        assert isinstance(evaluator._similarity_model, CrossEncoder if cross_encoder else SentenceTransformer)
        assert result["individual_scores"] == pytest.approx([expected])
        assert result["score"] == pytest.approx(expected)

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")
        evaluator = SASEvaluator()

        assert evaluator._model == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert evaluator._batch_size == 32
        assert evaluator._device is None
        assert evaluator._token.resolve_value() == "fake-token"

    @pytest.mark.parametrize("batch_size", [0, -1, 0.5, 32.0, "32", None])
    def test_init_invalid_batch_size(self, batch_size):
        with pytest.raises(
            ValueError, match=re.escape(f"batch_size must be a positive integer, but got {batch_size!r}")
        ):
            SASEvaluator(batch_size=batch_size)

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")

        evaluator = SASEvaluator(device=ComponentDevice.from_str("cuda:0"))

        expected_dict = {
            "type": "haystack.components.evaluators.sas_evaluator.SASEvaluator",
            "init_parameters": {
                "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "batch_size": 32,
                "device": {"type": "single", "device": "cuda:0"},
                "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
            },
        }
        assert evaluator.to_dict() == expected_dict

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")
        evaluator = SASEvaluator.from_dict(
            {
                "type": "haystack.components.evaluators.sas_evaluator.SASEvaluator",
                "init_parameters": {
                    "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "batch_size": 32,
                    "device": {"type": "single", "device": "cuda:0"},
                    "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                },
            }
        )

        assert evaluator._model == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert evaluator._batch_size == 32
        assert evaluator._device is not None
        assert evaluator._device.to_torch_str() == "cuda:0"
        assert evaluator._token.resolve_value() == "fake-token"

    @patch("haystack.components.evaluators.sas_evaluator.SentenceTransformer")
    @patch("haystack.components.evaluators.sas_evaluator.CrossEncoder")
    @patch("haystack.components.evaluators.sas_evaluator.AutoConfig")
    def test_warm_up_passes_token_to_hf(
        self, mock_auto_config, mock_cross_encoder, mock_sentence_transformer, monkeypatch
    ):
        monkeypatch.setenv("HF_API_TOKEN", "fake-token")
        mock_auto_config.from_pretrained.return_value = MagicMock(architectures=["BertModel"])
        evaluator = SASEvaluator(model="private/model", device=ComponentDevice.from_str("cpu"))
        evaluator.warm_up()

        mock_auto_config.from_pretrained.assert_called_once_with("private/model", token="fake-token")
        mock_sentence_transformer.assert_called_once_with("private/model", device="cpu", token="fake-token")
        mock_cross_encoder.assert_not_called()

    def test_run_with_empty_inputs(self):
        evaluator = SASEvaluator()
        result = evaluator.run(ground_truth_answers=[], predicted_answers=[])
        assert len(result) == 2
        assert result["score"] == 0.0
        assert result["individual_scores"] == [0.0]

    def test_run_with_different_lengths(self):
        evaluator = SASEvaluator()
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)

    def test_run_with_none_in_predictions(self):
        evaluator = SASEvaluator()
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            None,
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        with pytest.raises(ValueError):
            evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("activation_fn", "model_scores", "expected_scores"),
        [
            # Raw logits are normalized with a sigmoid
            (Identity(), {"Lyon": -9.4, "Paris": 4.9}, {"Lyon": expit(-9.4), "Paris": expit(4.9)}),
            # Scores that the model already normalized are returned as they are
            (Sigmoid(), {"Lyon": 0.2, "Paris": 0.9}, {"Lyon": 0.2, "Paris": 0.9}),
        ],
    )
    def test_run_cross_encoder_score_does_not_depend_on_batch(
        self, activation_fn, model_scores, expected_scores, monkeypatch
    ):
        # Use the mocked model below instead of loading a real one
        monkeypatch.setattr(SASEvaluator, "warm_up", lambda self: None)
        model = MagicMock(spec=CrossEncoder)
        model.num_labels = 1
        model.activation_fn = activation_fn
        model.predict.side_effect = lambda pairs, **kwargs: np.array([model_scores[pred] for pred, _ in pairs])
        evaluator = SASEvaluator(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        evaluator._similarity_model = model

        wrong_only = evaluator.run(ground_truth_answers=["Paris"], predicted_answers=["Lyon"])
        mixed = evaluator.run(ground_truth_answers=["Paris", "Paris"], predicted_answers=["Lyon", "Paris"])

        assert wrong_only["individual_scores"] == pytest.approx([expected_scores["Lyon"]])
        assert mixed["individual_scores"] == pytest.approx([expected_scores["Lyon"], expected_scores["Paris"]])

    @pytest.mark.parametrize("num_labels", [2, 3])
    @patch("haystack.components.evaluators.sas_evaluator.CrossEncoder")
    @patch("haystack.components.evaluators.sas_evaluator.AutoConfig")
    def test_warm_up_with_multi_label_cross_encoder_raises(self, mock_auto_config, mock_cross_encoder, num_labels):
        mock_auto_config.from_pretrained.return_value = MagicMock(architectures=["BertForSequenceClassification"])
        mock_cross_encoder.return_value.num_labels = num_labels
        evaluator = SASEvaluator(model="cross-encoder/nli-deberta-v3-xsmall")

        with pytest.raises(ValueError, match=f"has {num_labels} labels"):
            evaluator.warm_up()

        assert evaluator._similarity_model is None
        mock_cross_encoder.return_value.predict.assert_not_called()

    @patch("haystack.components.evaluators.sas_evaluator.CrossEncoder")
    @patch("haystack.components.evaluators.sas_evaluator.AutoConfig")
    def test_warm_up_with_single_label_cross_encoder_is_idempotent(self, mock_auto_config, mock_cross_encoder):
        mock_auto_config.from_pretrained.return_value = MagicMock(architectures=["BertForSequenceClassification"])
        mock_cross_encoder.return_value.num_labels = 1
        evaluator = SASEvaluator(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

        evaluator.warm_up()
        evaluator.warm_up()

        assert evaluator._similarity_model is mock_cross_encoder.return_value
        mock_auto_config.from_pretrained.assert_called_once()
        mock_cross_encoder.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_with_bi_encoder_model(self, del_hf_env_vars):
        evaluator = SASEvaluator("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        ground_truths = [
            "US $2.3 billion",
            "Paris's cultural magnificence is symbolized by the Eiffel Tower",
            "Japan was transformed into a modernized world power after the Meiji Restoration.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)
        assert len(result) == 2
        assert result["score"] == pytest.approx(0.912335)
        assert result["individual_scores"] == pytest.approx([0.855047, 0.907907, 0.974050], abs=1e-5)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_with_cross_encoder_model(self, del_hf_env_vars):
        evaluator = SASEvaluator(model="cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
        ground_truths = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        predictions = [
            "A construction budget of US $2.3 billion",
            "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
            "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
        ]
        result = evaluator.run(ground_truth_answers=ground_truths, predicted_answers=predictions)
        assert len(result) == 2
        assert result["score"] == pytest.approx(0.938108, abs=1e-5)
        assert result["individual_scores"] == pytest.approx([0.930112, 0.9431504, 0.9410622], abs=1e-5)
