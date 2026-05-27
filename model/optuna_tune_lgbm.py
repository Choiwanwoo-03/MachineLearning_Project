"""
Optuna LGBM 하이퍼파라미터 튜닝 — REB 디플레이션 타깃 기준
==========================================================
실행: venv/bin/python optuna_tune_lgbm.py 2>&1 | tee data/logs/optuna.log

저장: data/models/lgbm_best_params.json
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "pipeline"), str(_ROOT / "model")]:
    if _p not in sys.path: sys.path.insert(0, _p)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

from suwon_pipeline import split_temporal, FEATURE_COLS

N_TRIALS = 30  # 30 trials ≈ 30~45분
SEED = 42


def main():
    print(f"=== Optuna LGBM 튜닝 시작 (n_trials={N_TRIALS}) ===")
    feat = pd.read_parquet("data/features/suwon_features.parquet")
    train, val, _ = split_temporal(feat)

    # REB 디플레이션 타깃 (V9 표준)
    if "reb_idx" not in feat.columns:
        raise ValueError("reb_idx 컬럼 없음 — Phase 3 부터 다시 실행 필요")

    avail = [c for c in FEATURE_COLS if c in train.columns]
    Xt = train[avail]
    Xv = val[avail]
    yt = (np.log(train["price_per_pyeong"]) - np.log(train["reb_idx"].astype(float)))
    yv = (np.log(val["price_per_pyeong"]) - np.log(val["reb_idx"].astype(float)))
    mask_t = yt.notna(); mask_v = yv.notna()
    Xt, yt = Xt.loc[mask_t], yt.loc[mask_t]
    Xv, yv = Xv.loc[mask_v], yv.loc[mask_v]
    print(f"Train {len(Xt):,} / Val {len(Xv):,} / Features {len(avail)}")

    sw = np.where(train.loc[mask_t, "age"].fillna(99).to_numpy() < 5, 2.0, 1.0).astype("float32")

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=2000,
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            objective="huber",
            alpha=trial.suggest_float("huber_alpha", 0.7, 0.99),
            random_state=SEED,
            verbose=-1,
            subsample_freq=5,
        )
        m = LGBMRegressor(**params)
        m.fit(
            Xt, yt, sample_weight=sw,
            eval_set=[(Xv, yv)],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        pred = m.predict(Xv)
        # 평가지표: MAE on log_resid (REB 디플레이션된)
        return mean_absolute_error(yv, pred)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="lgbm_v9_reb")
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\n=== 완료 ({time.time()-t0:.1f}s) ===")
    print(f"Best MAE (log space): {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # alpha 키 변환 (LGBM 인터페이스용)
    final = dict(study.best_params)
    final["alpha"] = final.pop("huber_alpha")
    final.update({"n_estimators": 3000, "objective": "huber",
                  "subsample_freq": 5, "random_state": SEED, "verbose": -1})
    out = {"best_params": final, "best_mae_logspace": float(study.best_value),
           "n_trials": N_TRIALS}
    with open("data/models/lgbm_best_params.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"저장: data/models/lgbm_best_params.json")


if __name__ == "__main__":
    main()
