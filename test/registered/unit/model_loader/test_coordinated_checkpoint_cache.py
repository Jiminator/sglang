"""Guards for opt-in single-node TP checkpoint-cache release."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    coordinated_safetensors_weights_iterator,
)


class TestCoordinatedLoaderGuards(unittest.TestCase):
    def test_requires_buffered_mmap_and_no_prefetch(self):
        for disable_mmap, configured, started in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ):
            with self.subTest(
                disable_mmap=disable_mmap, configured=configured, active=started
            ):
                loader = DefaultModelLoader(
                    LoadConfig(
                        model_loader_extra_config={
                            "enable_multithread_load": True,
                            "drop_checkpoint_cache_after_model_load": True,
                            "checkpoint_cache_batch_shards": 3,
                        }
                    )
                )
                flags = SimpleNamespace(
                    weight_loader_disable_mmap=disable_mmap,
                    weight_loader_prefetch_checkpoints=configured,
                    weight_loader_prefetch_num_threads=2,
                    weight_loader_drop_cache_after_load=False,
                )
                with patch.object(
                    loader, "_prepare_weights", return_value=("fixture", [], True)
                ), patch(
                    "sglang.srt.model_loader.loader.get_model", return_value=flags
                ), self.assertRaisesRegex(
                    ValueError, "without checkpoint prefetch"
                ):
                    loader._get_weights_iterator(
                        DefaultModelLoader.Source("fixture", None),
                        startup_prefetch_active=started,
                    )

    def test_disallows_multinode(self):
        with self.assertRaisesRegex(ValueError, "one node"):
            list(coordinated_safetensors_weights_iterator([], 2, 3, nnodes=2))

    def test_disallows_pipeline_or_data_parallel_world(self):
        parallel = SimpleNamespace(tp_size=2, world_group=SimpleNamespace(world_size=4))
        with patch("torch.distributed.is_initialized", return_value=True), patch(
            "sglang.srt.model_loader.weight_utils.get_parallel", return_value=parallel
        ), self.assertRaisesRegex(ValueError, "TP-only"):
            list(coordinated_safetensors_weights_iterator([], 2, 3))

    def test_disallows_other_cpu_backends(self):
        parallel = SimpleNamespace(
            tp_size=2,
            world_group=SimpleNamespace(
                world_size=2, rank_in_group=0, cpu_group=object()
            ),
        )
        with patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.get_backend", return_value="mooncake-cpu"
        ), patch(
            "sglang.srt.model_loader.weight_utils.get_parallel", return_value=parallel
        ), self.assertRaisesRegex(
            ValueError, "Gloo"
        ):
            list(coordinated_safetensors_weights_iterator([], 2, 3))

    def test_invalid_batch_values(self):
        for value in (True, False, -1, "3", 1.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DefaultModelLoader(
                    LoadConfig(
                        model_loader_extra_config={
                            "checkpoint_cache_batch_shards": value
                        }
                    )
                )

    def test_requires_model_end_release(self):
        with self.assertRaisesRegex(ValueError, "model-end"):
            DefaultModelLoader(
                LoadConfig(
                    model_loader_extra_config={"checkpoint_cache_batch_shards": 3}
                )
            )

    def test_disabled_mode_preserves_previous_options(self):
        for options in (
            {},
            {"checkpoint_cache_batch_shards": 0},
            {"enable_multithread_load": False, "checkpoint_cache_batch_shards": 0},
        ):
            DefaultModelLoader(LoadConfig(model_loader_extra_config=options))


if __name__ == "__main__":
    unittest.main()
