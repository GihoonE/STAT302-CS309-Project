# tools/run_pipeline.py
import argparse
from pathlib import Path
import pandas as pd

from experiment_part23.run_pipeline import run_pipeline_one


def _parse_floats_csv(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _extract_ratio_and_acc(item):
    """
    item이 dict / tuple / list 어떤 형태든 ratio와 accuracy를 최대한 추출.
    return: (ratio_float or None, acc_float or None)
    """
    if isinstance(item, dict):
        r = _as_float(item.get("ratio"))
        acc = None
        for k in ("test_acc", "test_accuracy", "accuracy", "acc"):
            if k in item:
                acc = _as_float(item.get(k))
                if acc is not None:
                    break
        return r, acc

    if isinstance(item, (tuple, list)) and len(item) >= 1:
        r = _as_float(item[0])

        if len(item) >= 2 and isinstance(item[1], dict):
            acc = None
            for k in ("test_acc", "test_accuracy", "accuracy", "acc"):
                if k in item[1]:
                    acc = _as_float(item[1].get(k))
                    if acc is not None:
                        break
            return r, acc

        if len(item) >= 2:
            acc = _as_float(item[1])
            return r, acc

        return r, None

    return None, None


def _pick_acc_at_ratio(cnn_results, target_ratio=0.8):
    """
    dict / list[dict] / list[tuple] / tuple(...) 전부 방어적으로 처리.
    """
    if cnn_results is None:
        return None

    # cnn_results 자체가 tuple인 경우: (rows, ...) 같은 형태
    if isinstance(cnn_results, tuple):
        if len(cnn_results) >= 1 and isinstance(cnn_results[0], list):
            cnn_results = cnn_results[0]
        elif len(cnn_results) >= 1 and isinstance(cnn_results[0], dict):
            cnn_results = cnn_results[0]
        else:
            return None

    if isinstance(cnn_results, list):
        cand = []
        for item in cnn_results:
            r, acc = _extract_ratio_and_acc(item)
            if r is not None and acc is not None:
                cand.append((r, acc))
        if not cand:
            return None
        best_r, best_acc = min(cand, key=lambda t: abs(t[0] - float(target_ratio)))
        return float(best_acc)

    if isinstance(cnn_results, dict):
        cand = []
        for k, v in cnn_results.items():
            r = _as_float(k)
            if r is None:
                continue
            acc = None
            if isinstance(v, dict):
                for kk in ("test_acc", "test_accuracy", "accuracy", "acc"):
                    if kk in v:
                        acc = _as_float(v.get(kk))
                        if acc is not None:
                            break
            else:
                acc = _as_float(v)
            if acc is not None:
                cand.append((r, acc))
        if not cand:
            return None
        best_r, best_acc = min(cand, key=lambda t: abs(t[0] - float(target_ratio)))
        return float(best_acc)

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_key", required=True, choices=["sports_ball", "animals", "mnist"])
    ap.add_argument("--out_root", default="outputs/part23")

    # pipeline knobs
    ap.add_argument("--cgan_epochs", type=int, default=30)
    ap.add_argument("--sample_every", type=int, default=5)

    ap.add_argument("--run_fid_kid", action="store_true", default=True)
    ap.add_argument("--no_fid_kid", action="store_false", dest="run_fid_kid")
    ap.add_argument("--verbose", type=int, default=1)

    # reuse
    ap.add_argument("--reuse_if_exists", action="store_true", default=True)
    ap.add_argument("--no_reuse", action="store_false", dest="reuse_if_exists")

    # sweep
    ap.add_argument("--label_noise_p", type=float, default=None,
                    help="Single run. If label_noise_ps is provided, this is ignored.")
    ap.add_argument("--label_noise_ps", type=str, default=None,
                    help='Sweep: comma-separated, e.g. "0.0,0.1,0.2,0.3"')

    # accuracy pick
    ap.add_argument("--acc_ratio", type=float, default=0.8,
                    help="Which ratio's test accuracy to record")

    # ratios for CNN
    ap.add_argument("--cnn_ratios", type=str, default="1.0,0.8,0.7,0.6,0.5",
                    help='comma-separated ratios')

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cnn_ratios = _parse_floats_csv(args.cnn_ratios)

    # which p's to run?
    if args.label_noise_ps:
        ps = _parse_floats_csv(args.label_noise_ps)
    else:
        p = args.label_noise_p
        if p is None:
            p = 0.2
        ps = [p]

    rows = []
    for p in ps:
        res = run_pipeline_one(
            args.dataset_key,
            out_root=str(out_root),
            cgan_epochs=args.cgan_epochs,
            sample_every=args.sample_every,
            label_noise_p=float(p),
            cnn_ratios=cnn_ratios,
            run_fid_kid=args.run_fid_kid,
            verbose=args.verbose,
            reuse_if_exists=args.reuse_if_exists,
        )

        acc_base = _pick_acc_at_ratio(res.get("cnn_results_baseline"), target_ratio=args.acc_ratio)
        acc_ln = _pick_acc_at_ratio(res.get("cnn_results_label_noise"), target_ratio=args.acc_ratio)

        row = {
            "dataset_key": args.dataset_key,
            "label_noise_p": float(p),
            "acc_ratio": float(args.acc_ratio),

            "acc_baseline_fake": acc_base,
            "acc_label_noise_fake": acc_ln,

            "fid_baseline": res.get("fid_baseline"),
            "kid_baseline_mean": res.get("kid_baseline_mean"),
            "kid_baseline_std": res.get("kid_baseline_std"),

            "fid_label_noise": res.get("fid_label_noise"),
            "kid_label_noise_mean": res.get("kid_label_noise_mean"),
            "kid_label_noise_std": res.get("kid_label_noise_std"),

            "fake_epoch_dir_baseline": res.get("fake_epoch_dir_baseline"),
            "fake_epoch_dir_label_noise": res.get("fake_epoch_dir_label_noise"),
            "cnn_csv_baseline": res.get("cnn_csv_baseline"),
            "cnn_csv_label_noise": res.get("cnn_csv_label_noise"),
        }
        rows.append(row)

    sweep_dir = out_root / args.dataset_key / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "label_noise_sweep.csv"

    df_new = pd.DataFrame(rows)

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.sort_values(["dataset_key", "label_noise_p", "acc_ratio"])
        df_all = df_all.drop_duplicates(subset=["dataset_key", "label_noise_p", "acc_ratio"], keep="last")
        df_all.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    print("Saved:", csv_path)
    print(df_new)


if __name__ == "__main__":
    main()
