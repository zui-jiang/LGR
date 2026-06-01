import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from slime.utils.memory_utils import available_memory, clear_memory, print_memory

logger = logging.getLogger(__name__)

# Flag to enable/disable enhanced NCCL cleanup
# Set via environment variable: ENABLE_NCCL_CLEANUP=1
_ENABLE_ENHANCED_CLEANUP = os.environ.get("ENABLE_NCCL_CLEANUP", "0") == "1"

old_new_group_dict = {}


def monkey_patch_torch_dist():
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))

        if len(ranks) == 1:
            return group

        group = ReloadableProcessGroup(group, ranks)
        return group

    dist.new_group = new_group

    def get_new_function(func):
        def new_function(*args, **kwargs):
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            with _wrap_low_level_call():
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_function(dist.get_rank)
    dist.get_world_size = get_new_function(dist.get_world_size)
    dist.get_backend = get_new_function(dist.get_backend)
    dist.get_global_rank = get_new_function(dist.get_global_rank)
    dist.get_group_rank = get_new_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_function(dist.all_reduce)
    dist.all_gather = get_new_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_function(dist.all_gather_object)
    dist.all_to_all = get_new_function(dist.all_to_all)
    dist.all_to_all_single = get_new_function(dist.all_to_all_single)
    dist.broadcast = get_new_function(dist.broadcast)
    dist.reduce = get_new_function(dist.reduce)
    dist.reduce_scatter = get_new_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_function(dist.scatter)
    dist.gather = get_new_function(dist.gather)
    dist.barrier = get_new_function(dist.barrier)
    dist.send = get_new_function(dist.send)
    dist.recv = get_new_function(dist.recv)
    dist._coalescing_manager = get_new_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_function(dist.isend)
    dist.irecv = get_new_function(dist.irecv)

    def get_new_p2pop_function(func):
        def new_function(*args, **kwargs):
            def convert(arg):
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    GROUPS = {}

    def __init__(self, group, ranks):
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.group_info = {
            "ranks": ranks,
        }
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        """Destroy all reloadable process groups with optional enhanced cleanup."""
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])

        # Enhanced cleanup: synchronize before destroy
        if _ENABLE_ENHANCED_CLEANUP and dist.is_initialized():
            try:
                from slime.utils.nccl_cleanup_helper import synchronize_before_destroy
                logger.info(f"[Rank {dist.get_rank()}] Synchronizing before destroying {len(reloadable_groups)} process groups")
                synchronize_before_destroy(use_gloo=True)
            except Exception as e:
                logger.warning(f"[Rank {dist.get_rank()}] Enhanced cleanup synchronization failed: {e}")

        # Destroy each process group
        for reloadable_group in reloadable_groups:
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

        # Enhanced cleanup: force cleanup of NCCL state
        if _ENABLE_ENHANCED_CLEANUP:
            try:
                from slime.utils.nccl_cleanup_helper import force_cleanup_nccl_state
                cleanup_delay = float(os.environ.get("NCCL_CLEANUP_DELAY", "2.0"))
                logger.info(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Force cleaning NCCL state with {cleanup_delay}s delay")
                force_cleanup_nccl_state(delay_seconds=cleanup_delay)
            except Exception as e:
                logger.warning(f"Enhanced NCCL cleanup failed: {e}")

    @staticmethod
    def reload_process_groups():
        """Reload all reloadable process groups with optional retry mechanism."""
        import time

        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        old_new_group = old_new_group_dict.get(pid)

        # Determine retry settings
        if _ENABLE_ENHANCED_CLEANUP:
            max_retries = int(os.environ.get("NCCL_RELOAD_RETRIES", "3"))
            retry_delay = float(os.environ.get("NCCL_RELOAD_RETRY_DELAY", "5.0"))
        else:
            max_retries = 1
            retry_delay = 0.0

        # Reload each group with retry logic
        for idx, reloadable_group in enumerate(reloadable_groups):
            if reloadable_group.group is not None:
                continue

            for attempt in range(max_retries):
                try:
                    group = old_new_group(ranks=reloadable_group.group_info["ranks"], backend="nccl")
                    reloadable_group.group = group
                    break  # Success, exit retry loop

                except Exception as e:
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"[Rank {rank}] Failed to reload process group {idx}/{len(reloadable_groups)} "
                            f"(attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)

                        # Force cleanup before retry
                        if _ENABLE_ENHANCED_CLEANUP:
                            try:
                                from slime.utils.nccl_cleanup_helper import force_cleanup_nccl_state
                                force_cleanup_nccl_state(delay_seconds=1.0)
                            except Exception:
                                pass
                    else:
                        logger.error(
                            f"[Rank {rank}] Failed to reload process group {idx}/{len(reloadable_groups)} "
                            f"after {max_retries} attempts: {e}"
                        )
                        raise

        # Enhanced cleanup: synchronize after reload to verify all ranks succeeded
        if _ENABLE_ENHANCED_CLEANUP and dist.is_initialized():
            try:
                from slime.utils.nccl_cleanup_helper import synchronize_after_reload
                logger.info(f"[Rank {dist.get_rank()}] Synchronizing after reloading {len(reloadable_groups)} process groups")
                synchronize_after_reload(use_gloo=True, retry_on_failure=True)
            except Exception as e:
                logger.warning(f"[Rank {dist.get_rank()}] Enhanced cleanup post-reload sync failed: {e}")

    def rank(self) -> int:
        return self.group.rank()

    def size(self) -> int:
        return self.group.size()

    def name(self) -> str:
        return self.group.name()

    def shutdown(self) -> None:
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call():
            return getattr(inner, method)(*args, **kwargs)

    def barrier(self, *a, **kw):
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        return self._fwd("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        return self._fwd("_get_backend_name")

    def _get_backend(self, *a, **kw):
        return self._fwd("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        return self._fwd("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        self.group.bound_device_id = dev


def destroy_process_groups():
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call():
    try:
        mem_info = available_memory()
        if mem_info["free_GB"] < 5:
            clear_memory()
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        e.add_note(f"{mem_info=}")
        raise
