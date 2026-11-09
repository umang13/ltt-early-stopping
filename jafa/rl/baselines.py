import numpy as np
import torch


def parse_int_list(value, n_features=None):
    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    out = []
    for raw in raw_values:
        if raw == "":
            continue
        item = int(raw)
        if n_features is not None and (item < 0 or item > n_features):
            raise ValueError(f"value {item} outside [0, {n_features}]")
        out.append(item)
    if not out:
        raise ValueError("expected at least one integer value")
    return out


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    out = [float(raw) for raw in raw_values if raw != ""]
    if not out:
        raise ValueError("expected at least one float value")
    return out


def _restore_model_mode(step_runner, was_training):
    if not was_training:
        return
    train_mode = getattr(step_runner, "train_mode", None)
    if train_mode is not None:
        train_mode()
    else:
        step_runner.model.train()


def _summary(correct, n_patches, returns_mean=None):
    n_patches = np.asarray(n_patches, dtype=np.float64)
    return {
        "accuracy": float(np.mean(correct)),
        "mean_patches": float(np.mean(n_patches)),
        "median_patches": float(np.median(n_patches)),
        "min_patches": float(np.amin(n_patches)),
        "max_patches": float(np.amax(n_patches)),
        "returns_mean": returns_mean,
    }


def policy_validation_summary(correct, acquired, returns):
    n_acquired = np.sum(acquired, axis=1)
    return {
        "accuracy": float(np.mean(correct)),
        "mean_patches": float(np.mean(n_acquired)),
        "median_patches": float(np.median(n_acquired)),
        "min_patches": float(np.amin(n_acquired)),
        "max_patches": float(np.amax(n_acquired)),
        "returns_mean": float(np.mean(returns)),
    }


def policy_validation_metrics(correct, acquired, returns):
    summary = policy_validation_summary(correct, acquired, returns)
    return {
        "rl/val_accuracy": summary["accuracy"],
        "rl/n_acquired_mean": summary["mean_patches"],
        "rl/n_acquired_median": summary["median_patches"],
        "rl/n_acquired_min": summary["min_patches"],
        "rl/n_acquired_max": summary["max_patches"],
        "rl/returns_mean": summary["returns_mean"],
    }


def _fixed_count_mask(batch_size, n_features, visible_count, rng):
    mask = np.zeros((batch_size, n_features), dtype=np.uint8)
    for i in range(batch_size):
        visible = rng.permutation(n_features)[:visible_count]
        mask[i, visible] = 1
    return mask


def evaluate_fixed_visible_counts(step_runner, data, visible_counts, args):
    was_training = step_runner.model.training
    step_runner.model.eval()
    results = {}

    with torch.no_grad():
        for visible_count in visible_counts:
            rng = np.random.default_rng(args.random_seed + visible_count)
            data.reset()
            correct = []
            total = 0
            while total < data.n_data:
                batch_size = min(args.n_envs, data.n_data - total)
                imgs, _, labs, _ = data.next_batch(batch_size)
                mask = _fixed_count_mask(
                    batch_size, data.n_features, visible_count, rng
                )
                imgs = torch.from_numpy(imgs).to(args.device)
                labs_t = torch.from_numpy(labs).to(args.device)
                mask_t = torch.from_numpy(mask).to(args.device)
                logits, _ = step_runner.model(imgs, mask_t)
                pred = logits.argmax(dim=1)
                correct.extend((pred == labs_t).detach().cpu().numpy().tolist())
                total += batch_size
            results[visible_count] = _summary(
                correct, np.full(len(correct), visible_count, dtype=np.float64)
            )

    _restore_model_mode(step_runner, was_training)
    return results


def evaluate_random_fixed_from_initial(step_runner, data, visible_counts, args):
    was_training = step_runner.model.training
    step_runner.model.eval()
    results = {}

    with torch.no_grad():
        for visible_count in visible_counts:
            if visible_count < args.initial_visible_count:
                continue
            rng = np.random.default_rng(args.random_seed + 700000 + visible_count)
            data.reset()
            correct = []
            n_patches = []
            total = 0
            while total < data.n_data:
                batch_size = min(args.n_envs, data.n_data - total)
                imgs, mask, labs, _ = data.next_batch(batch_size)
                mask = mask.copy()
                for i in range(batch_size):
                    n_add = visible_count - int(mask[i].sum())
                    if n_add <= 0:
                        continue
                    unseen = np.flatnonzero(mask[i] == 0)
                    chosen = rng.choice(unseen, size=n_add, replace=False)
                    mask[i, chosen] = 1

                imgs = torch.from_numpy(imgs).to(args.device)
                labs_t = torch.from_numpy(labs).to(args.device)
                mask_t = torch.from_numpy(mask).to(args.device)
                logits, _ = step_runner.model(imgs, mask_t)
                pred = logits.argmax(dim=1)
                correct.extend((pred == labs_t).detach().cpu().numpy().tolist())
                n_patches.extend(mask.sum(axis=1).tolist())
                total += batch_size
            results[visible_count] = _summary(correct, n_patches)

    _restore_model_mode(step_runner, was_training)
    return results


def evaluate_random_sequential_confidence(step_runner, data, thresholds, args):
    was_training = step_runner.model.training
    step_runner.model.eval()
    results = {}

    with torch.no_grad():
        for threshold in thresholds:
            rng = np.random.default_rng(args.random_seed + 900000)
            data.reset()
            correct = []
            n_patches = []
            total = 0
            while total < data.n_data:
                batch_size = min(args.n_envs, data.n_data - total)
                imgs, mask, labs, _ = data.next_batch(batch_size)
                done = np.zeros(batch_size, dtype=bool)
                final_pred = np.zeros(batch_size, dtype=np.int64)

                while not np.all(done):
                    imgs_t = torch.from_numpy(imgs).to(args.device)
                    mask_t = torch.from_numpy(mask).to(args.device)
                    logits, _ = step_runner.model(imgs_t, mask_t)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    conf = conf.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()

                    visible = mask.sum(axis=1)
                    should_stop = (conf >= threshold) | (visible >= data.n_features)
                    newly_done = (~done) & should_stop
                    final_pred[newly_done] = pred[newly_done]
                    done[newly_done] = True

                    active = np.flatnonzero(~done)
                    for idx in active:
                        unseen = np.flatnonzero(mask[idx] == 0)
                        if unseen.size == 0:
                            final_pred[idx] = pred[idx]
                            done[idx] = True
                            continue
                        chosen = rng.choice(unseen)
                        mask[idx, chosen] = 1

                correct.extend((final_pred == labs).tolist())
                n_patches.extend(mask.sum(axis=1).tolist())
                total += batch_size
            results[threshold] = _summary(correct, n_patches)

    _restore_model_mode(step_runner, was_training)
    return results


def evaluate_policy_prefix_visible_counts(step_runner, data, visible_counts, args):
    was_training = step_runner.model.training
    step_runner.model.eval()
    visible_counts = sorted(set(int(count) for count in visible_counts))
    max_count = min(max(visible_counts), data.n_features)
    visible_set = set(visible_counts)
    correct_by_count = {count: [] for count in visible_counts}

    with torch.no_grad():
        data.reset()
        total = 0
        while total < data.n_data:
            batch_size = min(args.n_envs, data.n_data - total)
            imgs, mask, labs, _ = data.next_batch(batch_size)
            mask = mask.copy()

            imgs_t = torch.from_numpy(imgs).to(args.device)

            budget_count = int(mask.sum(axis=1)[0])
            if budget_count > max_count:
                total += batch_size
                continue

            stopped = np.zeros(batch_size, dtype=bool)
            stopped_pred = np.full(batch_size, -1, dtype=np.int64)

            while budget_count <= max_count:
                active = np.flatnonzero(~stopped)
                if active.size:
                    mask_t = torch.from_numpy(mask).to(args.device)
                    logits, q_values = step_runner.model(imgs_t, mask_t)
                    pred = logits.argmax(dim=1).detach().cpu().numpy()

                    patch_q_values = q_values[:, : data.n_features].clone()
                    patch_q_values[mask_t.bool()] = -torch.inf
                    best_patch_values, actions_t = patch_q_values.max(dim=1)
                    actions = actions_t.detach().cpu().numpy()

                    stop_values = q_values[:, data.n_features]
                    should_stop = (stop_values > best_patch_values).detach().cpu().numpy()
                    if budget_count == 0:
                        should_stop[:] = False

                    newly_stopped = (~stopped) & should_stop
                    stopped_pred[newly_stopped] = pred[newly_stopped]
                    stopped[newly_stopped] = True
                else:
                    pred = stopped_pred.copy()
                    actions = None

                if budget_count in visible_set:
                    eval_pred = pred.copy()
                    eval_pred[stopped] = stopped_pred[stopped]
                    correct_by_count[budget_count].extend(
                        (eval_pred == labs).tolist()
                    )

                if budget_count >= max_count:
                    break

                active = np.flatnonzero(~stopped)
                if active.size:
                    mask[active, actions[active]] = 1
                budget_count += 1

            total += batch_size

    _restore_model_mode(step_runner, was_training)
    return {
        count: _summary(correct, np.full(len(correct), count, dtype=np.float64))
        for count, correct in correct_by_count.items()
        if correct
    }
