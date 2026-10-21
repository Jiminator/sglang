"""CPU integration and failure probes through the actual DefaultModelLoader."""

import ctypes
import datetime
import gc
import json
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import safetensors.torch
import torch
import torch.distributed as dist

SHARDS, NUMEL = 8, 1 << 18
SUCCESS = {"balanced", "skew", "stash"}


def resident(path):
    with open(path, "rb") as f:
        mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    view = (ctypes.c_char * 1).from_buffer(mapping)
    vector = (ctypes.c_ubyte * ((len(mapping) + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
    fn = ctypes.CDLL(None, use_errno=True).mincore
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    assert fn(ctypes.addressof(view), len(mapping), vector) == 0
    result = sum(v & 1 for v in vector) / len(vector)
    del view
    mapping.close()
    return result


def worker(rank, root_text, mode):
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    root = Path(root_text)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(root / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader import loader as loader_module
    from sglang.srt.model_loader import weight_utils as weights

    group = SimpleNamespace(
        cpu_group=dist.group.WORLD,
        rank=rank,
        rank_in_group=rank,
        world_size=2,
        barrier=lambda: dist.barrier(),
    )
    parallel = SimpleNamespace(nnodes=1, tp_size=2, world_group=group)
    weights.get_parallel = loader_module.get_parallel = lambda: parallel
    original_iterator = weights.coordinated_safetensors_weights_iterator
    loader_module.coordinated_safetensors_weights_iterator = (
        lambda *a, **kw: original_iterator(*a, **kw, timeout_seconds=2)
    )
    flags = SimpleNamespace(
        weight_loader_disable_mmap=False,
        weight_loader_prefetch_checkpoints=False,
        weight_loader_prefetch_num_threads=2,
        weight_loader_drop_cache_after_load=False,
    )
    loader_module.get_model = lambda: flags
    paths = [str(root / f"model-{i}.safetensors") for i in range(SHARDS)]
    calls, snapshots, copied, errors = [], [], {}, []
    original_drop = weights._drop_file_cache_after_load
    state = {"checking": True}

    def checked_drop(path):
        index = int(Path(path).stem.split("-")[1])
        if state["checking"]:
            assert all((root / f"copied-{r}-{index}").exists() for r in (0, 1)), (
                "early release",
                rank,
                index,
            )
            if mode == "stash" and index == 0:
                assert (
                    root / "stash-released"
                ).exists(), "released a retained tensor file"
        calls.append(index)
        original_drop(path)

    weights._drop_file_cache_after_load = checked_drop
    actual_paths = list(reversed(paths)) if mode == "mismatch" and rank == 1 else paths
    if mode == "missing_file" and rank == 1:
        actual_paths = [str(root / "missing.safetensors"), *paths[1:]]
    config = LoadConfig(
        model_loader_extra_config={
            "enable_multithread_load": True,
            "num_threads": 2,
            "drop_checkpoint_cache_after_model_load": True,
            "checkpoint_cache_batch_shards": (
                2 if mode == "batch_mismatch" and rank == 1 else 3
            ),
        }
    )
    loader = loader_module.DefaultModelLoader(config)
    loader._prepare_weights = lambda *a, **kw: (str(root), actual_paths, True)
    source = loader_module.DefaultModelLoader.Source(str(root), None)
    iterator = None
    started = time.monotonic()
    try:
        if mode == "absent_rank" and rank == 1:
            time.sleep(4)
            errors.append("simulated_nonparticipant")
        else:
            iterator = loader._get_weights_iterator(source)
            stash = None
            if mode == "maps_error" and rank == 1:
                builtin_open = open

                def guarded_open(path, *a, **kw):
                    if str(path) == "/proc/self/maps":
                        raise OSError("injected maps failure")
                    return builtin_open(path, *a, **kw)

                weights.open = guarded_open
            for name, tensor in iterator:
                index = int(name[1:])
                if mode == "consumer_abort" and rank == 1:
                    raise RuntimeError("injected consumer failure")
                if mode == "changed_file" and rank == 1 and index == 0:
                    with open(paths[-1], "ab") as stream:
                        stream.write(b"x")
                if mode == "skew" and rank == 1:
                    deadline = time.monotonic() + 1.5
                    while not (root / f"copied-0-{index}").exists():
                        assert (
                            time.monotonic() < deadline
                        ), "fast reader did not advance"
                        time.sleep(0.005)
                    assert (
                        resident(paths[index]) > 0.99
                    ), "file evicted before late consumer"
                if index % 3 == 0:
                    snapshots.append(
                        {"at_shard": index, "resident": [resident(p) for p in paths]}
                    )
                if mode == "stash" and rank == 1 and index == 0:
                    stash = tensor
                if mode == "stash" and rank == 1 and index == 6:
                    assert torch.equal(stash, torch.full((NUMEL,), 0.5))
                    stash = None
                    (root / "stash-released").touch()
                copied[name] = tensor.clone()
                (root / f"copied-{rank}-{index}").touch()
            del tensor, stash
            gc.collect()
            assert mode in SUCCESS, "failure case unexpectedly succeeded"
            for i in range(SHARDS):
                assert torch.equal(copied[f"s{i}"], torch.full((NUMEL,), i + 0.5))
            # Save mid-load advice before the existing model-end cleanup.
            before_model_release = list(calls)
            state["checking"] = False
            weights.drop_checkpoint_cache_after_model_load(paths, nnodes=1)
            final = [resident(p) for p in paths]
            assert all(v == 0 for v in final), final
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc)[:400])
        if mode in SUCCESS:
            raise
    finally:
        if iterator is not None:
            iterator.close()
        elapsed = time.monotonic() - started
        result = {
            "mode": mode,
            "rank": rank,
            "elapsed_seconds": elapsed,
            "errors": errors,
            "copied_count": len(copied),
            "advised_shards_before_model_release": locals().get(
                "before_model_release", calls
            ),
            "residency_snapshots": snapshots,
            "final_resident": locals().get("final"),
            "values_equal": mode in SUCCESS and not errors,
        }
        (root / f"rank{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


def run(mode):
    with tempfile.TemporaryDirectory(
        prefix="coordinated-cache-",
        dir=os.environ.get("COORDINATED_CACHE_TEST_DIR", tempfile.gettempdir()),
    ) as name:
        root = Path(name)
        for i in range(SHARDS):
            path = root / f"model-{i}.safetensors"
            safetensors.torch.save_file(
                {f"s{i}": torch.full((NUMEL,), i + 0.5)}, str(path)
            )
            with path.open("rb") as f:
                os.fsync(f.fileno())
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if mode == "corrupt_shard":
            with (root / "model-1.safetensors").open("r+b") as stream:
                stream.write(b"\xff" * 8)
        context = mp.get_context("spawn")
        workers = [context.Process(target=worker, args=(r, name, mode)) for r in (0, 1)]
        for p in workers:
            p.start()
        for p in workers:
            p.join(70)
        codes = [p.exitcode for p in workers]
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(5)
        assert codes == [0, 0], (mode, codes)
        rows = [json.loads((root / f"rank{r}.json").read_text()) for r in (0, 1)]
        assert all(r["elapsed_seconds"] < 12 for r in rows), rows
        if mode in SUCCESS:
            assert all(r["values_equal"] for r in rows), rows
            calls = [i for r in rows for i in r["advised_shards_before_model_release"]]
            assert sorted(calls) == list(range(7)), (mode, calls)
        else:
            assert all(r["errors"] for r in rows), rows
            assert not any(r["advised_shards_before_model_release"] for r in rows), rows
        return {"mode": mode, "passed": True, "ranks": rows}


def config_checks():
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader.loader import DefaultModelLoader

    count = 0
    for value in (True, -1, 1.5, "3", None):
        try:
            DefaultModelLoader(
                LoadConfig(
                    model_loader_extra_config={"checkpoint_cache_batch_shards": value}
                )
            )
        except ValueError:
            count += 1
        else:
            raise AssertionError(value)
    for extra in (
        {"checkpoint_cache_batch_shards": 3},
        {
            "checkpoint_cache_batch_shards": 3,
            "drop_checkpoint_cache_after_model_load": True,
            "enable_multithread_load": False,
        },
    ):
        try:
            DefaultModelLoader(LoadConfig(model_loader_extra_config=extra))
        except ValueError:
            count += 1
        else:
            raise AssertionError(extra)
    for extra in ({}, {"checkpoint_cache_batch_shards": 0}):
        DefaultModelLoader(LoadConfig(model_loader_extra_config=extra))
        count += 1
    return {"cases": count, "passed": True}


if __name__ == "__main__":
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    out = {
        "config": config_checks(),
        "cases": [],
        "limits": "Real CPU/Gloo and DefaultModelLoader integration; no CUDA copy, TP8, large-model peak or production-speed evidence.",
    }
    modes = os.environ.get(
        "PROBE_CASES",
        "balanced,skew,stash,mismatch,missing_file,consumer_abort,maps_error,absent_rank",
    ).split(",")
    assert set(modes) <= {
        "balanced",
        "skew",
        "stash",
        "mismatch",
        "missing_file",
        "consumer_abort",
        "maps_error",
        "absent_rank",
        "corrupt_shard",
        "changed_file",
        "batch_mismatch",
    }
    for mode in modes:
        out["cases"].append(run(mode))
        (
            Path(os.environ.get("COORDINATED_CACHE_TEST_DIR", tempfile.gettempdir()))
            / "coordinated-cache-result.json"
        ).write_text(json.dumps(out, indent=2))
        print(json.dumps({"case": mode, "passed": True}), flush=True)
    out["passed"] = True
    (
        Path(os.environ.get("COORDINATED_CACHE_TEST_DIR", tempfile.gettempdir()))
        / "coordinated-cache-result.json"
    ).write_text(json.dumps(out, indent=2))
    print("COORDINATED_CACHE_RESULT=" + json.dumps(out), flush=True)
