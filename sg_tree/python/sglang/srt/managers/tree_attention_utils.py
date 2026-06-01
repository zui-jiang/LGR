"""Utilities for building tree attention masks and positions from parent_ids."""

from typing import List, Optional, Tuple

import torch


def compute_tree_positions(parent_ids: List[int]) -> List[int]:
    """Compute position IDs (tree depth) from parent_ids.

    Args:
        parent_ids: parent_ids[i] is the index of token i's parent. -1 for root.

    Returns:
        List of position IDs where positions[i] = depth of token i in the tree.
    """
    n = len(parent_ids)
    positions = [0] * n
    for i in range(1, n):
        positions[i] = positions[parent_ids[i]] + 1
    return positions


def build_tree_mask_single(parent_ids: List[int]) -> torch.Tensor:
    """Build a tree attention mask for a single request.

    Uses DP: mask[i] = copy of mask[parent[i]], then set mask[i][i] = True.
    This is O(N^2) time and space.

    Args:
        parent_ids: parent_ids[i] is the index of token i's parent. -1 for root.

    Returns:
        1D bool tensor of shape (N*N,), row-major flattened NxN mask.
    """
    n = len(parent_ids)
    # Use a 2D tensor for construction, then flatten
    mask = torch.zeros(n, n, dtype=torch.bool)

    # Root token attends to itself
    mask[0, 0] = True

    # DP: each token inherits its parent's mask row, then adds itself
    for i in range(1, n):
        p = parent_ids[i]
        mask[i] = mask[p].clone()
        mask[i, i] = True

    return mask.flatten()


def build_tree_mask(
    parent_ids_list: List[List[int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build tree attention masks for a batch of requests.

    Args:
        parent_ids_list: List of parent_ids for each request in the batch.
        device: Target device for the tensors.

    Returns:
        custom_mask: 1D bool tensor, concatenation of all per-request NxN masks.
        mask_indptr: int64 tensor of shape (bs+1,), cumulative mask offsets.
    """
    masks = []
    indptr = [0]

    for parent_ids in parent_ids_list:
        mask = build_tree_mask_single(parent_ids)
        masks.append(mask)
        indptr.append(indptr[-1] + len(mask))

    custom_mask = torch.cat(masks).to(device, non_blocking=True)
    mask_indptr = torch.tensor(indptr, dtype=torch.int64, device=device)
    return custom_mask, mask_indptr


def validate_parent_ids(parent_ids: List[int], input_len: int) -> None:
    """Validate parent_ids for correctness.

    Args:
        parent_ids: The parent_ids to validate.
        input_len: Expected length (must match len(parent_ids)).

    Raises:
        ValueError: If parent_ids are invalid.
    """
    if len(parent_ids) != input_len:
        raise ValueError(
            f"parent_ids length ({len(parent_ids)}) must match "
            f"input_ids length ({input_len})"
        )
    if parent_ids[0] != -1:
        raise ValueError("parent_ids[0] must be -1 (root node)")
    for i in range(1, len(parent_ids)):
        if parent_ids[i] < 0 or parent_ids[i] >= i:
            raise ValueError(
                f"parent_ids[{i}]={parent_ids[i]} is invalid. "
                f"Must be in [0, {i-1}] (topological order)."
            )
