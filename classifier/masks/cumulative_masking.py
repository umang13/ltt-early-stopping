import numpy as np
import torch
from typing import Optional


class CumulativeMaskState:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.masks = [None] * num_samples
        self.cursor = 0

    def reset_cursor(self) -> None:
        self.cursor = 0


class TrackingMask:
    def __init__(self, mask_fn):
        self.mask_fn = mask_fn

    def _get_candidate_mask(self, inputs, num_slices=None):
        if num_slices is not None:
            try:
                _, mask = self.mask_fn(inputs, num_slices=num_slices, return_mask=True)
            except TypeError:
                masked = self.mask_fn(inputs, num_slices=num_slices)
                mask = (masked != 0).to(inputs.dtype)
        else:
            try:
                _, mask = self.mask_fn(inputs, return_mask=True)
            except TypeError:
                masked = self.mask_fn(inputs)
                mask = (masked != 0).to(inputs.dtype)
        return mask

    @staticmethod
    def _select_mask_dim(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() >= 1 and mask.shape[-1] == 2:
            return mask[..., 0]
        return mask

    @staticmethod
    def _expand_mask(base_mask: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        if ref_mask.dim() >= 1 and ref_mask.shape[-1] == 2 and base_mask.shape[-1] != 2:
            return base_mask.unsqueeze(-1).expand_as(ref_mask)
        return base_mask

    @staticmethod
    def _column_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.is_complex():
            mask = mask.abs()
        if mask.dim() < 2:
            return mask > 0
        column_axis = mask.dim() - 2
        reduce_axes = tuple(i for i in range(mask.dim()) if i != column_axis)
        return mask.any(dim=reduce_axes)

    def _is_column_mask(self, mask: torch.Tensor) -> bool:
        base = self._select_mask_dim(mask)
        if base.dim() == 1:
            return True
        if base.dim() < 2:
            return False
        if base.is_complex():
            base = base.abs()
        column_mask = self._column_mask(base)
        shape = [1] * base.dim()
        shape[-2] = column_mask.shape[0]
        expanded = column_mask.reshape(*shape).expand_as(base)
        return torch.equal((base > 0), (expanded > 0))

    def _sample_indices(self, indices: np.ndarray, weights: Optional[np.ndarray], needed: int) -> np.ndarray:
        if weights is None:
            return np.random.choice(indices, size=needed, replace=False)
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()
        return np.random.choice(indices, size=needed, replace=False, p=weights)

    def sample_with_existing(self, candidate_mask: torch.Tensor, existing_mask=None):
        # print("Enters sample_with_existing with candidate_mask.shape", candidate_mask.shape, "and existing_mask.shape", existing_mask.shape if existing_mask is not None else None)
        candidate_mask = candidate_mask.detach()
        if candidate_mask.is_complex():
            candidate_mask = candidate_mask.abs()
        if existing_mask is None:
            return candidate_mask

        existing_mask = existing_mask.to(candidate_mask.device)
        if existing_mask.is_complex():
            existing_mask = existing_mask.abs()
        cand_sel = self._select_mask_dim(candidate_mask)
        exist_sel = self._select_mask_dim(existing_mask)
        if cand_sel.is_complex():
            cand_sel = cand_sel.abs()
        if exist_sel.is_complex():
            exist_sel = exist_sel.abs()
        cand_sel_real = cand_sel
        exist_sel_real = exist_sel

        patch_size = getattr(self.mask_fn, "patch_size", None)
        if patch_size and cand_sel.dim() >= 2:
            base = cand_sel_real
            exist_base = exist_sel_real
            if base.dim() == 3 and base.shape[0] == 1:
                base = base.squeeze(0)
                exist_base = exist_base.squeeze(0)
            if base.dim() == 2:
                height, width = base.shape[-2], base.shape[-1]
                if height % patch_size == 0 and width % patch_size == 0:
                    ph = height // patch_size
                    pw = width // patch_size
                    patch_mask = base.reshape(ph, patch_size, pw, patch_size).mean(dim=(1, 3))
                    exist_patch_mask = exist_base.reshape(ph, patch_size, pw, patch_size).mean(dim=(1, 3))
                    target_count = int((patch_mask > 0).sum().item())
                    existing_count = int((exist_patch_mask > 0).sum().item())
                    if existing_count >= target_count:
                        return existing_mask
                    available = (patch_mask > 0) & (exist_patch_mask == 0)
                    available_indices = torch.nonzero(available, as_tuple=False)
                    needed = target_count - existing_count
                    if available_indices.shape[0] <= needed:
                        add_patch = patch_mask > 0
                    else:
                        chosen = self._sample_indices(
                            np.arange(available_indices.shape[0]),
                            None,
                            needed,
                        )
                        add_patch = torch.zeros_like(patch_mask, dtype=patch_mask.dtype)
                        chosen_idx = available_indices[torch.tensor(chosen, device=available_indices.device)]
                        add_patch[tuple(chosen_idx.t())] = 1.0
                    add_mask = add_patch.repeat_interleave(patch_size, dim=0).repeat_interleave(
                        patch_size, dim=1
                    )
                    if cand_sel.dim() == 3:
                        add_mask = add_mask.unsqueeze(0)
                    add_mask = self._expand_mask(add_mask, candidate_mask)
                    new_mask = torch.maximum(existing_mask, add_mask)
                    return new_mask

        if self._is_column_mask(cand_sel_real):
            cand_cols = self._column_mask(cand_sel_real)
            exist_cols = self._column_mask(exist_sel_real)
            target_count = int(cand_cols.sum().item())
            existing_count = int(exist_cols.sum().item())
            # print("target_count", target_count, "existing_count", existing_count)
            if existing_count >= target_count:
                return existing_mask

            available = (cand_cols > 0) & (exist_cols == 0)
            available_indices = torch.nonzero(available, as_tuple=False).squeeze(-1).cpu().numpy()
            needed = target_count - existing_count

            weights = None
            if hasattr(self.mask_fn, "get_prior"):
                sorted_indices = np.array(sorted(available_indices.tolist()))
                if sorted_indices.size > 0:
                    prior = self.mask_fn.get_prior(sorted_indices.tolist())
                    weight_map = {idx: prior[i] for i, idx in enumerate(sorted_indices)}
                    weights = np.array([weight_map[idx] for idx in available_indices])

            if available_indices.size <= needed:
                chosen = available_indices
            else:
                chosen = self._sample_indices(available_indices, weights, needed)

            add_cols = torch.zeros_like(cand_cols, dtype=cand_cols.dtype)
            add_cols[torch.tensor(chosen, device=cand_cols.device)] = 1.0
            add_cols = self._expand_mask(add_cols, cand_sel)
            new_mask = torch.maximum(existing_mask, self._expand_mask(add_cols, candidate_mask))
            # print(existing_mask.sum()/existing_mask.numel(), new_mask.sum()/new_mask.numel(), add_mask.sum()/add_mask.numel())
            return new_mask

        available = (cand_sel_real > 0) & (exist_sel_real == 0)
        available_indices = torch.nonzero(available, as_tuple=False)
        target_count = int((cand_sel_real > 0).sum().item())
        existing_count = int((exist_sel_real > 0).sum().item())
        # print("target_count", target_count, "existing_count", existing_count)
        needed = target_count - existing_count
        if needed <= 0:
            return existing_mask
        if available_indices.shape[0] <= needed:
            add_mask = cand_sel_real
        else:
            weights = cand_sel_real[available].float().cpu().numpy()
            chosen = self._sample_indices(
                np.arange(available_indices.shape[0]), weights, needed
            )
            add_mask = torch.zeros_like(cand_sel_real)
            chosen_idx = available_indices[torch.tensor(chosen, device=available_indices.device)]
            add_mask[tuple(chosen_idx.t())] = 1.0
        add_mask = self._expand_mask(add_mask, candidate_mask)
        new_mask = torch.maximum(existing_mask, add_mask)
        # print(existing_mask.sum()/existing_mask.numel(), new_mask.sum()/new_mask.numel(), add_mask.sum()/add_mask.numel())
        return new_mask


def build_cumulative_batch_mask(mask_fn, inputs, state: CumulativeMaskState, num_slices=None):
    batch_size = inputs.shape[0]
    tracking = TrackingMask(mask_fn)
    candidate_mask = tracking._get_candidate_mask(inputs, num_slices=num_slices).detach()
    if candidate_mask.dim() == inputs.dim() - 1:
        candidate_mask = candidate_mask.unsqueeze(0).expand(batch_size, *candidate_mask.shape)
    if candidate_mask.shape[0] != batch_size:
        candidate_mask = candidate_mask[:batch_size]

    batch_masks = []
    for i in range(batch_size):
        idx = state.cursor + i
        prev = state.masks[idx]
        cand = candidate_mask[i]
        new_mask = tracking.sample_with_existing(cand, prev)
        # print(f"{prev.sum()/prev.numel() if prev is not None else 0}, {cand.sum()/cand.numel()}, {new_mask.sum()/new_mask.numel()}, {(new_mask.cpu() - prev.cpu()).sum().item()/new_mask.cpu().numel() if prev is not None else None}")
        if new_mask.dim() == cand.dim() + 1:
            new_mask = new_mask.squeeze(0)
        state.masks[idx] = new_mask.detach().cpu()
        batch_masks.append(new_mask)
    state.cursor += batch_size
    return torch.stack(batch_masks, dim=0)


def apply_mask_to_inputs(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != inputs.shape:
        return inputs * mask
    return inputs * mask
