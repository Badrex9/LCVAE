# Nouvelle version inspirée de TVAE avec gestion intelligente des entiers

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.mixture import GaussianMixture
from sklearn import __version__ as sklearn_version
from packaging import version
from scipy.stats import norm

def bool_to_int(x):
    return x.astype(int)

def int_to_bool(x):
    return x >= 0.5


class GMMCDFEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.models_ = {}
        self.col_components_ = {}

    def fit(self, X, y=None):
        self.columns = X.columns
        for col in self.columns:
            gmm = GaussianMixture(n_components=self.n_components, random_state=42)
            gmm.fit(X[[col]])
            self.models_[col] = gmm
        return self

    def transform(self, X):
        results = []
        for col in self.columns:
            gmm = self.models_[col]
            probs = gmm.predict_proba(X[[col]])
            components = np.argmax(probs, axis=1)
            means = gmm.means_.flatten()[components]
            stds = np.sqrt(gmm.covariances_.flatten()[components])
            z = (X[col] - means) / stds
            u = norm.cdf(z).clip(1e-6, 1 - 1e-6).reshape(-1, 1)
            comp_one_hot = np.eye(self.n_components)[components]
            self.col_components_[col] = comp_one_hot.shape[1]
            results.append(np.hstack([comp_one_hot, u]))
        return np.hstack(results)

    def inverse_transform(self, X):
        df_inv = pd.DataFrame()
        idx = 0
        for col in self.columns:
            gmm = self.models_[col]
            k = self.col_components_[col]
            comp_one_hot = X[:, idx:idx + k]
            u = X[:, idx + k]
            comp_idx = np.argmax(comp_one_hot, axis=1)
            means = gmm.means_.flatten()[comp_idx]
            stds = np.sqrt(gmm.covariances_.flatten()[comp_idx])
            z = norm.ppf(u)
            z = np.clip(z, -10, 10)
            x = z * stds + means
            x = np.maximum(x, 0)
            df_inv[col] = x
            idx += k + 1
        return df_inv

class TypedDataTransformer:
    def __init__(self, df: pd.DataFrame, n_gmm_components=10, max_categorical_cardinality=20):
        self.columns = df.columns
        self.dtypes = df.dtypes
        self.max_card = max_categorical_cardinality

        self.bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Gestion spécifique des entiers : catégoriel ou numérique
        self.int_cols = df.select_dtypes(include=["int64"]).columns.tolist()
        self.int_as_cat = [col for col in self.int_cols if df[col].nunique() <= self.max_card]
        self.int_as_num = list(set(self.int_cols) - set(self.int_as_cat))

        self.cat_cols += self.int_as_cat

        self.float_cols = df.select_dtypes(include=["float64"]).columns.tolist()
        self.num_cols = self.float_cols + self.int_as_num

        if version.parse(sklearn_version) >= version.parse("1.2"):
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        bool_transformer = FunctionTransformer(
            func=bool_to_int,
            inverse_func=int_to_bool,
            check_inverse=False
        )

        self.scaler = GMMCDFEncoder(n_components=n_gmm_components)

        self.transformer = ColumnTransformer([
            ("bool", bool_transformer, self.bool_cols),
            ("cat", encoder, self.cat_cols),
            ("num", self.scaler, self.num_cols),
        ], verbose_feature_names_out=False)

        self._fitted = False

    def fit(self, df: pd.DataFrame):
        self.transformer.fit(df)
        self.scaler = self.transformer.named_transformers_["num"]
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("You must call fit() before transform().")
        return self.transformer.transform(df)

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def inverse_transform(self, X) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("You must call fit() before inverse_transform().")

        start = 0
        df_inv = pd.DataFrame(index=range(X.shape[0]))

        if self.bool_cols:
            end = start + len(self.bool_cols)
            df_inv[self.bool_cols] = X[:, start:end] >= 0.5
            start = end

        if self.cat_cols:
            encoder = self.transformer.named_transformers_["cat"]
            cat_array_len = encoder.transform(pd.DataFrame([{col: encoder.categories_[i][0] for i, col in enumerate(self.cat_cols)}])).shape[1]
            cat_array = X[:, start:start + cat_array_len]
            cat_df = pd.DataFrame(encoder.inverse_transform(cat_array), columns=self.cat_cols)
            df_inv[self.cat_cols] = cat_df
            start += cat_array.shape[1]

        if self.num_cols:
            num_array = X[:, start:]
            df_decoded = self.scaler.inverse_transform(num_array)
            for col in self.num_cols:
                if col in self.int_as_num:
                    df_decoded[col] = df_decoded[col].round().astype(int)
                else:
                    df_decoded[col] = df_decoded[col].astype(float)
            df_inv[self.num_cols] = df_decoded[self.num_cols]

        return df_inv[self.columns]
