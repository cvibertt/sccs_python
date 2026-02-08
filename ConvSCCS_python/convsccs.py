import numpy as np
from scipy.special import logsumexp

from penalties import prox_tv_group_lasso, tv_norm_1d


class ConvSCCS:
    """
    Discrete-time ConvSCCS with convolutional drug effects and TV + group-lasso penalty.
    """

    def __init__(
        self,
        p,
        gamma_tv=0.0,
        gamma_gl=0.0,
        max_iter=200,
        step_size=1.0,
        prox_iters=25,
        tol=1e-6,
        verbose=False,
        use_phi=True,
    ):
        self.p = int(p)
        self.gamma_tv = float(gamma_tv)
        self.gamma_gl = float(gamma_gl)
        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.prox_iters = int(prox_iters)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.use_phi = bool(use_phi)

        self.phi_ = None
        self.alpha_ = None
        self.theta_ = None
        self.history_ = []

    def _convolve_exposure(self, x, theta):
        return np.convolve(x, theta, mode="full")[: len(x)]

    def _neg_loglik_and_grad(self, phi, theta, alpha, y_list, x_list, age_list):
        d, p = theta.shape
        K = len(phi)

        nll = 0.0
        grad_phi = np.zeros_like(phi)
        grad_theta = np.zeros_like(theta)
        grad_alpha = None
        if age_list is not None:
            grad_alpha = np.zeros_like(alpha)

        for idx, (y, X) in enumerate(zip(y_list, x_list)):
            if y.shape[0] != K or X.shape != (d, K):
                raise ValueError("Inconsistent shapes for y or X.")

            eta = phi.copy()
            if age_list is not None:
                A = age_list[idx]
                if A.shape[0] != K:
                    raise ValueError("Inconsistent age design length.")
                eta += A @ alpha
            for j in range(d):
                eta += self._convolve_exposure(X[j], theta[j])

            n_i = y.sum()
            if n_i == 0:
                continue

            lse = logsumexp(eta)
            nll -= y @ eta - n_i * lse

            probs = np.exp(eta - lse)
            r = y - n_i * probs
            grad_phi -= r
            if age_list is not None:
                grad_alpha -= A.T @ r

            # Gradient for theta_j via lagged dot-products
            for j in range(d):
                xj = X[j]
                for l in range(p):
                    if l >= K:
                        break
                    grad_theta[j, l] -= np.dot(r[l:], xj[: K - l])

        return nll, grad_phi, grad_theta, grad_alpha

    def _penalty(self, theta):
        pen = 0.0
        for j in range(theta.shape[0]):
            pen += self.gamma_tv * tv_norm_1d(theta[j])
            pen += self.gamma_gl * np.linalg.norm(theta[j])
        return pen

    def _prox_theta(self, theta, step):
        out = np.zeros_like(theta)
        for j in range(theta.shape[0]):
            out[j] = prox_tv_group_lasso(
                theta[j],
                lam_tv=self.gamma_tv * step,
                lam_gl=self.gamma_gl * step,
                iters=self.prox_iters,
            )
        return out

    def fit(self, y_list, x_list, age_design_list=None, phi_init=None, theta_init=None, alpha_init=None):
        if len(y_list) == 0:
            raise ValueError("y_list is empty.")
        if len(y_list) != len(x_list):
            raise ValueError("y_list and x_list must have same length.")
        if age_design_list is not None and len(age_design_list) != len(y_list):
            raise ValueError("age_design_list and y_list must have same length.")

        d, K = x_list[0].shape
        p = self.p
        if p <= 0:
            raise ValueError("p must be positive.")

        phi = np.zeros(K) if phi_init is None else phi_init.copy()
        theta = np.zeros((d, p)) if theta_init is None else theta_init.copy()
        alpha = None
        if age_design_list is not None:
            g = age_design_list[0].shape[1]
            alpha = np.zeros(g) if alpha_init is None else alpha_init.copy()

        y_phi = phi.copy()
        y_theta = theta.copy()
        y_alpha = alpha.copy() if alpha is not None else None
        t = 1.0
        step = self.step_size

        for it in range(self.max_iter):
            smooth_val, grad_phi, grad_theta, grad_alpha = self._neg_loglik_and_grad(
                y_phi, y_theta, y_alpha, y_list, x_list, age_design_list
            )

            # Backtracking line search on smooth part
            step_bt = step
            while True:
                if self.use_phi:
                    phi_new = y_phi - step_bt * grad_phi
                else:
                    phi_new = y_phi
                theta_new = self._prox_theta(y_theta - step_bt * grad_theta, step_bt)
                if y_alpha is not None:
                    alpha_new = y_alpha - step_bt * grad_alpha
                else:
                    alpha_new = None

                smooth_new, _, _, _ = self._neg_loglik_and_grad(
                    phi_new, theta_new, alpha_new, y_list, x_list, age_design_list
                )
                obj_new = smooth_new + self._penalty(theta_new)

                # Quadratic upper bound
                dphi = phi_new - y_phi
                dtheta = theta_new - y_theta
                dalpha = alpha_new - y_alpha if y_alpha is not None else 0.0
                quad = (
                    smooth_val
                    + np.sum(grad_phi * dphi)
                    + np.sum(grad_theta * dtheta)
                    + (np.sum(dphi * dphi) + np.sum(dtheta * dtheta) + np.sum(dalpha * dalpha)) / (2 * step_bt)
                )

                if obj_new <= quad + 1e-9:
                    break
                step_bt *= 0.5
                if step_bt < 1e-8:
                    break

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y_phi = phi_new + ((t - 1.0) / t_new) * (phi_new - phi)
            y_theta = theta_new + ((t - 1.0) / t_new) * (theta_new - theta)
            if y_alpha is not None:
                y_alpha = alpha_new + ((t - 1.0) / t_new) * (alpha_new - alpha)
            t = t_new

            phi, theta = phi_new, theta_new
            alpha = alpha_new
            obj = obj_new
            self.history_.append(obj)

            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"iter={it} obj={obj:.6f} step={step_bt:.3e}")

            if it > 0 and abs(self.history_[-2] - obj) < self.tol:
                break

        self.phi_ = phi
        self.alpha_ = alpha
        self.theta_ = theta
        return self

    def bootstrap_ci(
        self,
        y_list,
        x_list,
        age_design_list=None,
        n_boot=50,
        ci=0.95,
        random_state=None,
        max_iter=None,
    ):
        if n_boot <= 1:
            raise ValueError("n_boot must be greater than 1.")
        if not (0.0 < ci < 1.0):
            raise ValueError("ci must be between 0 and 1.")

        rng = np.random.default_rng(random_state)
        n = len(y_list)
        d, p = x_list[0].shape[0], self.p

        theta_boot = np.zeros((n_boot, d, p))
        alpha_boot = None
        if age_design_list is not None:
            g = age_design_list[0].shape[1]
            alpha_boot = np.zeros((n_boot, g))

        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            y_b = [y_list[i] for i in idx]
            x_b = [x_list[i] for i in idx]
            age_b = [age_design_list[i] for i in idx] if age_design_list is not None else None

            model_b = ConvSCCS(
                p=self.p,
                gamma_tv=self.gamma_tv,
                gamma_gl=self.gamma_gl,
                max_iter=self.max_iter if max_iter is None else max_iter,
                step_size=self.step_size,
                prox_iters=self.prox_iters,
                tol=self.tol,
                verbose=False,
                use_phi=self.use_phi,
            )
            model_b.fit(y_b, x_b, age_design_list=age_b)
            theta_boot[b] = model_b.theta_
            if alpha_boot is not None and model_b.alpha_ is not None:
                alpha_boot[b] = model_b.alpha_

            if self.verbose and (b + 1) % max(1, n_boot // 5) == 0:
                print(f"bootstrap {b + 1}/{n_boot}")

        alpha = 1.0 - ci
        theta_lo = np.quantile(theta_boot, alpha / 2.0, axis=0)
        theta_hi = np.quantile(theta_boot, 1.0 - alpha / 2.0, axis=0)
        out = {"ci_level": ci, "theta_ci": (theta_lo, theta_hi)}

        if alpha_boot is not None:
            alpha_lo = np.quantile(alpha_boot, alpha / 2.0, axis=0)
            alpha_hi = np.quantile(alpha_boot, 1.0 - alpha / 2.0, axis=0)
            out["alpha_ci"] = (alpha_lo, alpha_hi)

        return out
