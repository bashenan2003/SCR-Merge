"""GPyTorch Bayesian surrogate for online validation score prediction."""

import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


class GPyTorchSurrogate:
    """Bayesian Gaussian Process surrogate for per-trajectory validation scores.

    Predicts score vector [score_k, ...] given a candidate merge vector.
    Used in online mode to accelerate Module 3 candidate screening.
    """

    def __init__(self, input_dim: int, num_tasks: int = 1):
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self._gpytorch = None
        self._model = None
        self._likelihood = None
        self._train_x = None
        self._train_y = None
        self._initialized = False

    def _ensure_gpytorch(self) -> None:
        if self._gpytorch is not None:
            return

        # Try local source first
        gp_base = Path("E:/资料备份/论文写作/第3篇/我的方案/代码/SkillSCR/models")
        gp_path = gp_base / "GPyTorch"
        if gp_path.exists() and str(gp_path.parent) not in sys.path:
            sys.path.insert(0, str(gp_path.parent))

        try:
            import gpytorch
            self._gpytorch = gpytorch
        except ImportError:
            try:
                import gpytorch as gpt
                self._gpytorch = gpt
            except ImportError:
                raise ImportError("GPyTorch not available. Install with: pip install gpytorch")

    def _build_model(self, train_x: np.ndarray, train_y: np.ndarray):
        gp = self._gpytorch
        train_x_t = gp.torch.from_numpy(train_x).float()
        train_y_t = gp.torch.from_numpy(train_y).float()

        class ExactGPModel(gp.models.ExactGP):
            def __init__(self, train_inputs, train_targets, likelihood):
                super().__init__(train_inputs, train_targets, likelihood)
                self.mean_module = gp.means.ConstantMean()
                self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gp.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = gp.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x_t, train_y_t, likelihood)
        return model, likelihood

    def train(self, X: np.ndarray, y: np.ndarray, n_iter: int = 100) -> None:
        """Train GP surrogate on (merge_vector, score) pairs.

        Args:
            X: Training inputs of shape (n_samples, input_dim) or (n_samples,).
            y: Training targets of shape (n_samples,) or (n_samples, num_tasks).
            n_iter: Number of training iterations.
        """
        self._ensure_gpytorch()

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self._train_x = X
        self._train_y = y
        self.input_dim = X.shape[1]
        self.num_tasks = y.shape[1]

        self._model, self._likelihood = self._build_model(X, y[:, 0])

        self._model.train()
        self._likelihood.train()
        optimizer = self._gpytorch.torch.optim.Adam(self._model.parameters(), lr=0.1)
        mll = self._gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        train_x_t = self._gpytorch.torch.from_numpy(X).float()
        train_y_t = self._gpytorch.torch.from_numpy(y[:, 0]).float()

        for _ in range(n_iter):
            optimizer.zero_grad()
            output = self._model(train_x_t)
            loss = -mll(output, train_y_t)
            loss.backward()
            optimizer.step()

        self._initialized = True

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Predict (mean, std) for a single candidate vector.

        Args:
            x: Candidate vector of shape (input_dim,) or (input_dim, 1).

        Returns:
            Tuple of (predicted_mean, predicted_std).
        """
        if not self._initialized:
            return (0.5, 1.0)

        self._model.eval()
        self._likelihood.eval()

        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        x_t = self._gpytorch.torch.from_numpy(x).float()

        with self._gpytorch.torch.no_grad():
            pred = self._model(x_t)
            mean = pred.mean.item()
            std = pred.stddev.item()

        return (mean, std)

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict (means, stds) for multiple candidate vectors."""
        if not self._initialized:
            return (np.full(len(X), 0.5), np.full(len(X), 1.0))

        self._model.eval()
        self._likelihood.eval()

        X_t = self._gpytorch.torch.from_numpy(np.asarray(X, dtype=np.float64)).float()

        with self._gpytorch.torch.no_grad():
            pred = self._model(X_t)
            means = pred.mean.numpy()
            stds = pred.stddev.numpy()

        return (means, stds)

    def is_confident(self, x: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if surrogate prediction is sufficiently confident."""
        _, std = self.predict(x)
        return std < threshold

    @property
    def is_ready(self) -> bool:
        return self._initialized
