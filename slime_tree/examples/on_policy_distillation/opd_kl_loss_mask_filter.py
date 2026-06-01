"""
使用简化的KL散度 (student_lp - teacher_lp) 进行loss mask过滤
适用于On-Policy Distillation场景

用法:
    --rollout-data-postprocess-path examples.on_policy_distillation.opd_kl_loss_mask_filter:filter_loss_mask_by_kl_simple

    # Threshold模式（固定阈值）
    --opd-kl-filter-mode threshold
    --opd-kl-threshold-high 0.5   # KL上限
    --opd-kl-threshold-low -0.5   # KL下限

    # 默认行为（--opd-kl-filter-invert false 或不设置）:
    # 保留阈值范围内的token: -0.5 <= KL <= 0.5（过滤两端极端值）

    # 反转模式（--opd-kl-filter-invert true）:
    # 保留阈值范围外的token: KL < -0.5 或 KL > 0.5（过滤中间值）
    --opd-kl-filter-invert true

    # Quantile模式（分位数）- 保留指定分位数范围内的token
    --opd-kl-filter-mode quantile
    --opd-kl-quantile-high 0.8    # 上分位数: 过滤 top 20% 高KL的token
    --opd-kl-quantile-low 0.2     # 下分位数: 过滤 bottom 20% 低KL的token
    # 同时设置时: 保留中间 60% 的token（invert=false）或两端40%的token（invert=true）
"""
import logging
import torch

logger = logging.getLogger(__name__)


def filter_loss_mask_by_kl_simple(args, rollout_data):
    if 'teacher_log_probs' not in rollout_data or rollout_data['teacher_log_probs'] is None:
        logger.warning("[OPD KL Filter] No teacher_log_probs found, skipping")
        return

    log_probs = rollout_data['log_probs']
    teacher_log_probs = rollout_data['teacher_log_probs']
    loss_masks = rollout_data['loss_masks']

    student_lp_all = torch.cat([_to_tensor(lp) for lp in log_probs])
    teacher_lp_all = torch.cat([_to_tensor(lp) for lp in teacher_log_probs])
    loss_masks_all = torch.cat([_to_tensor(m) for m in loss_masks])


    mask = (loss_masks_all == 1)
    original_stats = {
        'student_lp': student_lp_all[mask].mean().item() if mask.any() else 0.0,
        'teacher_lp': teacher_lp_all[mask].mean().item() if mask.any() else 0.0,
        'reverse_kl': (student_lp_all[mask] - teacher_lp_all[mask]).mean().item() if mask.any() else 0.0,
    }

    logger.info(f"[OPD KL Filter] Original stats: "
                f"student_lp={original_stats['student_lp']:.4f}, "
                f"teacher_lp={original_stats['teacher_lp']:.4f}, "
                f"reverse_kl={original_stats['reverse_kl']:.4f}")


    filter_mode = getattr(args, 'opd_kl_filter_mode', 'threshold')
    filter_invert = getattr(args, 'opd_kl_filter_invert', False)
    filter_transform = getattr(args, 'opd_filter_transform', 'identity')  # 'identity' or 'exp'
    use_wandb = getattr(args, 'use_wandb', False)

    threshold_high = getattr(args, 'opd_kl_threshold_high', None)
    threshold_low = getattr(args, 'opd_kl_threshold_low', None)
    quantile_high = getattr(args, 'opd_kl_quantile_high', None)
    quantile_low = getattr(args, 'opd_kl_quantile_low', None)

    if filter_mode == 'threshold':
        if threshold_high is None and threshold_low is None:
            logger.warning("[OPD KL Filter] No thresholds set, skipping filter")
            return
    elif filter_mode == 'quantile':
        if quantile_high is None and quantile_low is None:
            logger.warning("[OPD KL Filter] No quantiles set, skipping filter")
            return

    stats = {
        'total_original': 0,
        'total_kept': 0,
        'kept_student_lp': [],
        'kept_teacher_lp': [],
        'kept_kl': [],
        'filtered_student_lp': [],
        'filtered_teacher_lp': [],
        'filtered_kl': [],
    }

    all_samples_kl = []

    if filter_mode == 'quantile':
        for i in range(len(log_probs)):
            student_lp = _to_tensor(log_probs[i])
            teacher_lp = _to_tensor(teacher_log_probs[i])
            loss_mask = _to_tensor(loss_masks[i])

            kl = _compute_diff(student_lp, teacher_lp, filter_transform)

            original_mask = (loss_mask == 1)
            if original_mask.any():
                kl_values = kl[original_mask]
                all_samples_kl.append(kl_values)

        if all_samples_kl:
            all_kl_tensor = torch.cat(all_samples_kl)
            if quantile_high is not None:
                threshold_high = torch.quantile(all_kl_tensor, quantile_high).item()
                logger.info(f"[OPD KL Filter] Quantile high {quantile_high:.2f} → threshold = {threshold_high:.4f}")
            if quantile_low is not None:
                threshold_low = torch.quantile(all_kl_tensor, quantile_low).item()
                logger.info(f"[OPD KL Filter] Quantile low {quantile_low:.2f} → threshold = {threshold_low:.4f}")

    for i in range(len(log_probs)):
        student_lp = _to_tensor(log_probs[i])
        teacher_lp = _to_tensor(teacher_log_probs[i])
        loss_mask = _to_tensor(loss_masks[i])

        kl = _compute_diff(student_lp, teacher_lp, filter_transform)
        original_mask = (loss_mask == 1)
        original_count = original_mask.sum().item()

        keep_mask = torch.ones_like(original_mask, dtype=torch.bool) if not filter_invert else torch.zeros_like(original_mask, dtype=torch.bool)
        if not filter_invert:
            if threshold_low is not None:
                keep_mask &= (kl >= threshold_low)
            if threshold_high is not None:
                keep_mask &= (kl <= threshold_high)
        else:
            if threshold_low is not None:
                keep_mask |= (kl < threshold_low)
            if threshold_high is not None:
                keep_mask |= (kl > threshold_high)

        new_mask = original_mask & keep_mask
        kept_count = new_mask.sum().item()

        stats['total_original'] += original_count
        stats['total_kept'] += kept_count

        if new_mask.any():
            stats['kept_student_lp'].extend(student_lp[new_mask].cpu().tolist())
            stats['kept_teacher_lp'].extend(teacher_lp[new_mask].cpu().tolist())
            stats['kept_kl'].extend(kl[new_mask].cpu().tolist())

        filtered_mask = original_mask & ~new_mask
        if filtered_mask.any():
            stats['filtered_student_lp'].extend(student_lp[filtered_mask].cpu().tolist())
            stats['filtered_teacher_lp'].extend(teacher_lp[filtered_mask].cpu().tolist())
            stats['filtered_kl'].extend(kl[filtered_mask].cpu().tolist())

        if isinstance(rollout_data['loss_masks'][i], list):
            rollout_data['loss_masks'][i] = new_mask.to(torch.int).tolist()
        else:
            rollout_data['loss_masks'][i] = new_mask


    if use_wandb:
        _log_to_wandb(stats, threshold_high, threshold_low, original_stats)


def _to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.tensor(data, dtype=torch.float32)


def _compute_diff(student_lp, teacher_lp, transform='identity'):
    if transform == 'exp':
        return torch.exp(student_lp) - torch.exp(teacher_lp)
    else:
        return student_lp - teacher_lp




def _log_to_wandb(stats, threshold_high, threshold_low, original_stats):
    try:
        import wandb
        if wandb.run is None:
            return

        def safe_stats(values):
            if not values:
                return {'mean': 0.0, 'max': 0.0, 'min': 0.0}
            t = torch.tensor(values)
            return {
                'mean': t.mean().item(),
                'max': t.max().item(),
                'min': t.min().item(),
            }

        kept_student_lp = safe_stats(stats['kept_student_lp'])
        kept_teacher_lp = safe_stats(stats['kept_teacher_lp'])
        kept_kl = safe_stats(stats['kept_kl'])

        filtered_student_lp = safe_stats(stats['filtered_student_lp'])
        filtered_teacher_lp = safe_stats(stats['filtered_teacher_lp'])
        filtered_kl = safe_stats(stats['filtered_kl'])

        # Calculate sign distribution for kept and filtered tokens
        kept_kl_tensor = torch.tensor(stats['kept_kl']) if stats['kept_kl'] else torch.tensor([])
        filtered_kl_tensor = torch.tensor(stats['filtered_kl']) if stats['filtered_kl'] else torch.tensor([])

        log_dict = {
            'opd_kl_filter/original_student_lp_mean': original_stats['student_lp'],
            'opd_kl_filter/original_teacher_lp_mean': original_stats['teacher_lp'],
            'opd_kl_filter/original_reverse_kl_mean': original_stats['reverse_kl'],

            'opd_kl_filter/kept_student_lp_mean': kept_student_lp['mean'],
            'opd_kl_filter/kept_student_lp_max': kept_student_lp['max'],
            'opd_kl_filter/kept_student_lp_min': kept_student_lp['min'],
            'opd_kl_filter/kept_teacher_lp_mean': kept_teacher_lp['mean'],
            'opd_kl_filter/kept_teacher_lp_max': kept_teacher_lp['max'],
            'opd_kl_filter/kept_teacher_lp_min': kept_teacher_lp['min'],
            'opd_kl_filter/kept_reverse_kl_mean': kept_kl['mean'],
            'opd_kl_filter/kept_reverse_kl_max': kept_kl['max'],
            'opd_kl_filter/kept_reverse_kl_min': kept_kl['min'],

            # Sign distribution for kept tokens
            'opd_kl_filter/kept_positive_kl_count': (kept_kl_tensor > 0).sum().item() if len(kept_kl_tensor) > 0 else 0,
            'opd_kl_filter/kept_negative_kl_count': (kept_kl_tensor < 0).sum().item() if len(kept_kl_tensor) > 0 else 0,

            'opd_kl_filter/filtered_student_lp_mean': filtered_student_lp['mean'],
            'opd_kl_filter/filtered_student_lp_max': filtered_student_lp['max'],
            'opd_kl_filter/filtered_student_lp_min': filtered_student_lp['min'],
            'opd_kl_filter/filtered_teacher_lp_mean': filtered_teacher_lp['mean'],
            'opd_kl_filter/filtered_teacher_lp_max': filtered_teacher_lp['max'],
            'opd_kl_filter/filtered_teacher_lp_min': filtered_teacher_lp['min'],
            'opd_kl_filter/filtered_reverse_kl_mean': filtered_kl['mean'],
            'opd_kl_filter/filtered_reverse_kl_max': filtered_kl['max'],
            'opd_kl_filter/filtered_reverse_kl_min': filtered_kl['min'],

            # Sign distribution for filtered tokens
            'opd_kl_filter/filtered_positive_kl_count': (filtered_kl_tensor > 0).sum().item() if len(filtered_kl_tensor) > 0 else 0,
            'opd_kl_filter/filtered_negative_kl_count': (filtered_kl_tensor < 0).sum().item() if len(filtered_kl_tensor) > 0 else 0,

            'opd_kl_filter/threshold_high': threshold_high if threshold_high is not None else 0,
            'opd_kl_filter/threshold_low': threshold_low if threshold_low is not None else 0,
        }

        wandb.log(log_dict)

        logger.info(f"[OPD KL Filter] Logged to wandb: "
                    f"original_teacher_lp={original_stats['teacher_lp']:.4f}, "
                    f"kept_teacher_lp_mean={kept_teacher_lp['mean']:.4f}, "
                    f"filtered_teacher_lp_mean={filtered_teacher_lp['mean']:.4f}")

    except Exception as e:
        logger.warning(f"[OPD KL Filter] Failed to log to wandb: {e}")

