# tools/run_pipeline.py
import argparse
from pathlib import Path
import pandas as pd

from colab.run_pipeline import run_pipeline_one


def _parse_floats_csv(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _pick_acc_at_ratio(cnn_results, target_ratio=0.8):
    """
    run_ratio_experiments() 반환 형태가 프로젝트마다 달라서 방어적으로 뽑음.
    우선순위: ratio 가장 가까운 row -> test_acc/test_accuracy/accuracy/acc
    """
    if cnn_results is None:
        return None

    # case A) list[dict]
    if isinstance(cnn_results, list):
        if not cnn_results:
            return None
        best = min(cnn_results, key=lambda r: abs(float(r.get("ratio", -1)) - target_ratio))
        for k in ("test_acc", "test_accuracy", "accuracy", "acc"):
            if k in best and best[k] is not None:
                return float(best[k])
        return None

    # case B) dict (ratio->metrics)
    if isinstance(cnn_results, dict):
        keys = list(cnn_results.keys())
        if not keys:
            return None
        rk = min(keys, key=lambda k: abs(float(k) - target_ratio))
        m = cnn_results[rk]
        if isinstance(m, dict):
            for k in ("test_acc", "test_accuracy", "accuracy", "acc"):
                if k in m and m[k] is not None:
                    return float(m[k])
        return None

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_key", required=True, choices=["sports_ball", "animals", "mnist"])
    ap.add_argument("--out_root", default="results")

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

            # accuracy (둘 다 저장)
            "acc_baseline_fake": acc_base,
            "acc_label_noise_fake": acc_ln,

            # baseline metrics
            "fid_baseline": res.get("fid_baseline"),
            "kid_baseline_mean": res.get("kid_baseline_mean"),
            "kid_baseline_std": res.get("kid_baseline_std"),

            # label-noise metrics
            "fid_label_noise": res.get("fid_label_noise"),
            "kid_label_noise_mean": res.get("kid_label_noise_mean"),
            "kid_label_noise_std": res.get("kid_label_noise_std"),

            # dirs
            "fake_epoch_dir_baseline": res.get("fake_epoch_dir_baseline"),
            "fake_epoch_dir_label_noise": res.get("fake_epoch_dir_label_noise"),
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
