from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from .replay import public_view

COMMON_NUMERIC = ["yield_rel", "yield_rank"]


def design():
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                COMMON_NUMERIC,
            ),
            (
                "class",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
                ["grain_class"],
            ),
        ],
        sparse_threshold=0,
    )


@dataclass
class RecoveryModel:
    kind: str
    parameter: float = 10.0

    def fit(self, frame, recovery):
        frame = public_view(frame)
        recovery = pd.Series(recovery).reindex(frame.row_id).to_numpy(float)
        if not np.isfinite(recovery).all() or np.any((recovery < 0) | (recovery > 100)):
            raise ValueError("Training labels must be measured recovery percentages")
        self.training_max_year = int(frame.year.max())
        self.training_ids = set(frame.row_id)
        self.training_candidates = set(frame.candidate_id)
        self.global_mean = float(np.mean(recovery))
        self.class_column = (
            "source_class"
            if self.kind == "class_history" and "source_class" in frame
            else "grain_class"
        )
        self.class_means = (
            pd.DataFrame({"grain_class": frame[self.class_column], "hr": recovery})
            .groupby("grain_class")
            .hr.mean()
            .to_dict()
        )
        self.encoder = design()
        x = self.encoder.fit_transform(frame)
        self.class_slice = self.encoder.output_indices_["class"]
        self.mean_class_design = x[:, self.class_slice].mean(axis=0)
        self.diagnostics = {
            "kind": self.kind,
            "parameter": self.parameter,
            "training_rows": len(frame),
            "training_pools": frame.pool_id.nunique(),
            "training_max_year": self.training_max_year,
            "features": COMMON_NUMERIC + ["grain_class"],
        }
        if self.kind == "mixed":
            x = np.column_stack([np.ones(len(x)), x])
            codes, group_names = pd.factorize(frame.pool_id, sort=True)
            counts = np.bincount(codes).astype(float)
            sum_x = np.zeros((len(counts), x.shape[1]))
            np.add.at(sum_x, codes, x)
            sum_y = np.bincount(codes, weights=recovery)
            xx = x.T @ x
            xy = x.T @ recovery
            yy = recovery @ recovery

            def components(log_scales):
                sigma2, tau2 = np.exp(2 * np.asarray(log_scales))
                adjustment = tau2 / (sigma2 * (sigma2 + counts * tau2))
                information = xx / sigma2 - sum_x.T @ (sum_x * adjustment[:, None])
                information += np.eye(x.shape[1]) * 1e-8
                rhs = xy / sigma2 - sum_x.T @ (sum_y * adjustment)
                beta = np.linalg.solve(information, rhs)
                quad = yy / sigma2 - np.dot(adjustment, sum_y**2) - np.dot(rhs, beta)
                logdet = np.sum(
                    (counts - 1) * np.log(sigma2) + np.log(sigma2 + counts * tau2)
                )
                sign, logdet_x = np.linalg.slogdet(information)
                if sign <= 0:
                    return np.inf, None
                objective = 0.5 * (
                    logdet + logdet_x + quad + (len(x) - x.shape[1]) * np.log(2 * np.pi)
                )
                return objective, (beta, np.linalg.inv(information), sigma2, tau2)

            opt = minimize(
                lambda z: components(z)[0],
                np.log(
                    [max(np.std(recovery) * 0.8, 0.2), max(np.std(recovery) * 0.3, 0.2)]
                ),
                method="L-BFGS-B",
                bounds=[(np.log(0.05), np.log(50))] * 2,
                options={"maxiter": 200, "ftol": 1e-10},
            )
            objective, fitted = components(opt.x)
            if fitted is None or not np.isfinite(objective):
                raise RuntimeError("Mixed recovery model failed")
            self.beta, self.beta_cov, self.sigma2, self.tau2 = fitted
            self.diagnostics.update(
                {
                    "optimizer_success": bool(opt.success),
                    "optimizer_message": str(opt.message),
                    "reml_objective": float(objective),
                    "residual_sd": float(np.sqrt(self.sigma2)),
                    "pool_sd": float(np.sqrt(self.tau2)),
                    "fixed_coefficients": self.beta.tolist(),
                }
            )
            fitted_mean = x @ self.beta
        elif self.kind == "ridge":
            self.regressor = Ridge(alpha=self.parameter).fit(x, recovery)
            fitted_mean = self.regressor.predict(x)
        elif self.kind == "trees":
            self.regressor = ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=int(self.parameter),
                max_features=1.0,
                random_state=91021,
                n_jobs=1,
            ).fit(x, recovery)
            fitted_mean = self.regressor.predict(x)
        elif self.kind in ("class_history", "yield"):
            fitted_mean = np.array(
                [
                    self.class_means.get(c, self.global_mean)
                    for c in frame[self.class_column]
                ]
            )
        else:
            raise ValueError("Unknown recovery model")
        residual = recovery - fitted_mean
        self.residual_sd = max(float(np.std(residual, ddof=1)), 0.1)
        pool_means = (
            pd.DataFrame({"pool": frame.pool_id, "residual": residual})
            .groupby("pool")
            .residual.mean()
        )
        self.pool_sd = float(pool_means.std()) if len(pool_means) > 1 else 0.0
        self.diagnostics.update(
            {
                "training_residual_rmse": float(np.sqrt(np.mean(residual**2))),
                "historical_grain_classes": sorted(self.class_means),
                "uncertainty_note": "Historical candidate residual and pool variability; not a calibrated confidence interval.",
            }
        )
        return self

    def predict(self, frame, enforce_time=True):
        frame = public_view(frame)
        if enforce_time and (
            int(frame.year.min()) <= self.training_max_year
            or set(frame.row_id) & self.training_ids
        ):
            raise ValueError(
                "Prediction requires strictly later harvest years and unseen rows"
            )
        if self.kind == "yield":
            return pd.Series(100.0, index=frame.row_id), pd.Series(
                0.0, index=frame.row_id
            )
        if self.kind == "class_history":
            return pd.Series(
                [
                    self.class_means.get(c, self.global_mean)
                    for c in frame[self.class_column]
                ],
                index=frame.row_id,
            ), pd.Series(self.residual_sd, index=frame.row_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            x = self.encoder.transform(frame)
        unseen = ~frame.grain_class.isin(self.class_means).to_numpy()
        x[unseen, self.class_slice] = self.mean_class_design
        if self.kind == "mixed":
            x = np.column_stack([np.ones(len(x)), x])
            mean = x @ self.beta
            sd = np.sqrt(
                self.sigma2 + self.tau2 + np.einsum("ij,jk,ik->i", x, self.beta_cov, x)
            )
        else:
            mean = self.regressor.predict(x)
            variance = np.zeros(len(x))
            if self.kind == "trees":
                variance = np.var(
                    np.stack([tree.predict(x) for tree in self.regressor.estimators_]),
                    axis=0,
                )
            sd = np.sqrt(self.residual_sd**2 + self.pool_sd**2 + variance)
        return pd.Series(np.clip(mean, 0, 100), index=frame.row_id), pd.Series(
            np.maximum(sd, 0.1), index=frame.row_id
        )


MODEL_MENU = {
    "yield": ("yield", 0),
    "class_history": ("class_history", 0),
    "mixed": ("mixed", 0),
    "ridge_1": ("ridge", 1),
    "ridge_10": ("ridge", 10),
    "ridge_100": ("ridge", 100),
    "trees_5": ("trees", 5),
    "trees_20": ("trees", 20),
}
