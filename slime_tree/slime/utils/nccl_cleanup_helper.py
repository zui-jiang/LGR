"""
NCCL cleanup helper to address NCCL state pollution after checkpoint save.

This module provides utilities to ensure complete cleanup of NCCL state when
destroy_process_groups() is called, especially after heavy operations like
checkpoint saving that involve extensive collective communication.

The issue: After saving checkpoints with optimizer state across 64 GPUs,
NCCL's internal state (TCPStore keys, connection buffers) may not be completely
cleaned up by destroy_process_groups(), causing subsequent reload_process_groups()
to timeout when initializing NCCL communicators.
"""

import gc
import logging
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def synchronize_before_destroy(use_gloo: bool = True) -> None:
    """
    Synchronize all ranks before destroying process groups.

    This ensures:
    1. All pending NCCL operations are completed
    2. All ranks reach the same state before cleanup
    3. No lingering async operations when destroy is called

    Args:
        use_gloo: If True, use gloo backend for barrier (safer).
                  If False, use default process group.
    """
    if not dist.is_initialized():
        return

    try:
        # First, synchronize CUDA to ensure all GPU operations are done
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Then, barrier to ensure all ranks are at the same point
        if use_gloo:
            try:
                from slime.utils.distributed_utils import get_gloo_group
                gloo_group = get_gloo_group()
                dist.barrier(group=gloo_group)
                logger.info(f"[Rank {dist.get_rank()}] Gloo barrier completed before destroy")
            except Exception as e:
                logger.warning(f"[Rank {dist.get_rank()}] Gloo barrier failed: {e}, trying default group")
                dist.barrier()
        else:
            dist.barrier()

    except Exception as e:
        logger.warning(f"[Rank {dist.get_rank()}] synchronize_before_destroy failed: {e}")


def force_cleanup_nccl_state(delay_seconds: float = 2.0) -> None:
    """
    Force cleanup of NCCL-related state.

    This includes:
    1. CUDA cache cleanup
    2. Python garbage collection
    3. Delay to allow NCCL backend to finish cleanup

    Args:
        delay_seconds: How long to wait for NCCL cleanup (default: 2.0s)
    """
    try:
        # Clear CUDA cache to release any lingering NCCL buffers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection to cleanup Python objects
        gc.collect()

        # Give NCCL time to complete internal cleanup
        # NCCL cleanup is async and may take time to finish
        if delay_seconds > 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info(f"[Rank {rank}] Waiting {delay_seconds}s for NCCL cleanup...")
            time.sleep(delay_seconds)

    except Exception as e:
        logger.warning(f"force_cleanup_nccl_state failed: {e}")


def synchronize_after_reload(use_gloo: bool = True, retry_on_failure: bool = True) -> None:
    """
    Synchronize all ranks after reloading process groups.

    This ensures all ranks successfully initialized NCCL communicators
    before proceeding to actual computation.

    Args:
        use_gloo: If True, use gloo backend for barrier.
        retry_on_failure: If True, retry once on failure.
    """
    if not dist.is_initialized():
        return

    max_retries = 2 if retry_on_failure else 1

    for attempt in range(max_retries):
        try:
            if use_gloo:
                from slime.utils.distributed_utils import get_gloo_group
                gloo_group = get_gloo_group()
                dist.barrier(group=gloo_group)
            else:
                dist.barrier()

            rank = dist.get_rank()
            logger.info(f"[Rank {rank}] Barrier after reload completed (attempt {attempt + 1})")
            return

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"[Rank {dist.get_rank()}] Barrier after reload failed (attempt {attempt + 1}): {e}, retrying...")
                time.sleep(1)
            else:
                logger.error(f"[Rank {dist.get_rank()}] Barrier after reload failed after {max_retries} attempts: {e}")
                raise


def safe_destroy_with_cleanup(destroy_func, cleanup_delay: float = 2.0) -> None:
    """
    Wrapper for destroy_process_groups with comprehensive cleanup.

    Args:
        destroy_func: The original destroy_process_groups function
        cleanup_delay: Delay after destroy for NCCL cleanup (default: 2.0s)
    """
    # Step 1: Synchronize before destroy
    synchronize_before_destroy(use_gloo=True)

    # Step 2: Call original destroy
    destroy_func()

    # Step 3: Force cleanup
    force_cleanup_nccl_state(delay_seconds=cleanup_delay)


def safe_reload_with_retry(reload_func, max_retries: int = 3, retry_delay: float = 5.0) -> None:
    """
    Wrapper for reload_process_groups with retry mechanism.

    Args:
        reload_func: The original reload_process_groups function
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries (default: 5.0s)
    """
    for attempt in range(max_retries):
        try:
            # Call original reload
            reload_func()

            # Synchronize after reload to ensure all ranks succeeded
            synchronize_after_reload(use_gloo=True, retry_on_failure=False)

            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info(f"[Rank {rank}] reload_process_groups succeeded (attempt {attempt + 1})")
            return

        except Exception as e:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if attempt < max_retries - 1:
                logger.warning(
                    f"[Rank {rank}] reload_process_groups failed (attempt {attempt + 1}/{max_retries}): {e}"
                    f"\nRetrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                # Force cleanup before retry
                force_cleanup_nccl_state(delay_seconds=1.0)
            else:
                logger.error(
                    f"[Rank {rank}] reload_process_groups failed after {max_retries} attempts: {e}"
                )
                raise
