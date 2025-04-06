import numpy as np
from numba import njit
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import time

@njit
def cubic_bspline_basis(t, i, k, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    d1 = knots[i + k] - knots[i]
    d2 = knots[i + k + 1] - knots[i + 1]
    c1 = 0.0 if d1 == 0 else ((t - knots[i]) / d1) * cubic_bspline_basis(t, i, k - 1, knots)
    c2 = 0.0 if d2 == 0 else ((knots[i + k + 1] - t) / d2) * cubic_bspline_basis(t, i + 1, k - 1, knots)
    return c1 + c2

@njit
def cubic_bspline_eval(x_control, y_control, x_data, degree=3):
    n = len(x_control) - 1
    m = len(x_data)
    y_out = np.zeros(m)
    knot_len = degree + (n - 1) + degree
    knots = np.zeros(knot_len)
    for i in range(degree):
        knots[i] = x_control[0]
    for i in range(n - 1):
        knots[degree + i] = x_control[i + 1]
    for i in range(degree):
        knots[degree + (n - 1) + i] = x_control[-1]
    for j in range(m):
        t = min(max(x_data[j], x_control[0]), x_control[-1])
        for i in range(n + 1):
            y_out[j] += y_control[i] * cubic_bspline_basis(t, i, degree, knots)
    return y_out

@njit
def cubic_bspline_gradient(x_control, y_control, x_data, y_data, ti, class_1_pseudo, class_weights, degree=3):
    n = len(x_control)
    m = len(x_data)
    grad = np.zeros(n)
    knot_len = degree + (n - 1) + degree
    knots = np.zeros(knot_len)
    for i in range(degree):
        knots[i] = x_control[0]
    for i in range(n - 1):
        knots[degree + i] = x_control[i + 1]
    for i in range(degree):
        knots[degree + (n - 1) + i] = x_control[-1]
    spline_vals = cubic_bspline_eval(x_control, y_control, x_data)
    z = 1 - ti * class_1_pseudo * (y_data - spline_vals)
    errors = z > 0
    n_err = errors.sum()
    if n_err == 0:
        return grad
    for i in range(m):
        if not errors[i]:
            continue
        t = min(max(x_data[i], x_control[0]), x_control[-1])
        factor = (class_weights[i] / n_err) * ti[i] * class_1_pseudo
        for k in range(n):
            grad[k] += factor * cubic_bspline_basis(t, k, degree, knots)
    return grad

class FastCubicSMPA:
    def __init__(self, n_control_points=6, max_iter=50, lambda_reg=0.0001, verbose=False):
        self.n_control_points = n_control_points
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.verbose = verbose

    def _to_numpy(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif hasattr(data, 'to_numpy'):
            return data.to_numpy()
        elif hasattr(data, 'numpy'):
            return data.detach().cpu().numpy()
        else:
            try:
                return np.array(data)
            except:
                raise ValueError(f"Cannot convert {type(data)} to NumPy array")

    def fit(self, X, y):
        X = self._to_numpy(X)
        y = self._to_numpy(y)
        self.x_controls = []
        self.y_controls = []
        n_features = X.shape[1] - 1

        # Compute class weights (inverse frequency)
        class_counts = np.bincount(y)
        class_weights = 1.0 / class_counts
        sample_weights = np.array([class_weights[yi] for yi in y])

        for j in range(n_features):
            x_min, x_max = X[:, j].min(), X[:, j].max()
            x_control = np.linspace(x_min, x_max, self.n_control_points)
            y_control = np.zeros(self.n_control_points)

            def loss(y_ctrl):
                spline_vals = cubic_bspline_eval(x_control, y_ctrl, X[:, j])
                total_spline = np.sum([cubic_bspline_eval(self.x_controls[i], self.y_controls[i], X[:, i])
                                      for i in range(len(self.y_controls))], axis=0) + spline_vals
                displacements = X[:, -1] - total_spline
                pseudo_labels = np.where(y == 1, 1 if displacements.mean() > 0 else -1,
                                       -1 if displacements.mean() > 0 else 1)
                z = 1 - pseudo_labels * displacements
                hinge_loss = np.mean(sample_weights * np.maximum(0, z))
                y_diff = y_ctrl[1:] - y_ctrl[:-1]
                x_diff = x_control[1:] - x_control[:-1]
                smoothness_penalty = np.mean((y_diff / x_diff) ** 2)
                total_loss = hinge_loss + self.lambda_reg * smoothness_penalty
                if self.verbose:
                    print(f"Feature {j}, Loss: {total_loss:.6f}")
                return total_loss

            def grad(y_ctrl):
                spline_vals = cubic_bspline_eval(x_control, y_ctrl, X[:, j])
                total_spline = np.sum([cubic_bspline_eval(self.x_controls[i], self.y_controls[i], X[:, i])
                                      for i in range(len(self.y_controls))], axis=0) + spline_vals
                displacements = X[:, -1] - total_spline
                pseudo_labels = np.where(y == 1, 1 if displacements.mean() > 0 else -1,
                                       -1 if displacements.mean() > 0 else 1)
                grad_hinge = cubic_bspline_gradient(x_control, y_ctrl, X[:, j], X[:, -1], pseudo_labels,
                                                  1 if displacements.mean() > 0 else -1, sample_weights)
                y_diff = y_ctrl[1:] - y_ctrl[:-1]
                x_diff = x_control[1:] - x_control[:-1]
                grad_smooth = np.zeros_like(y_ctrl)
                grad_smooth[:-1] -= 2 * self.lambda_reg * (y_diff / x_diff**2)
                grad_smooth[1:] += 2 * self.lambda_reg * (y_diff / x_diff**2)
                return grad_hinge + grad_smooth

            result = minimize(loss, y_control, jac=grad, method='L-BFGS-B',
                            options={'maxiter': self.max_iter, 'gtol': 1e-6})
            self.x_controls.append(x_control)
            self.y_controls.append(result.x)

        displacements = self.predict_displacements(X)
        self.class_1_pseudo = 1 if displacements.mean() > 0 else -1

    def predict_displacements(self, X):
        X = self._to_numpy(X)
        total_spline = np.sum([cubic_bspline_eval(self.x_controls[i], self.y_controls[i], X[:, i])
                              for i in range(len(self.x_controls))], axis=0)
        return X[:, -1] - total_spline

    def predict(self, X):
        displacements = self.predict_displacements(X)
        return (displacements > 0).astype(int) if self.class_1_pseudo > 0 else (displacements <= 0).astype(int)

if __name__ == "__main__":
    X = np.random.rand(231, 3)
    y = np.random.randint(0, 2, 231)
    grid_search = run_grid_search(X, y)