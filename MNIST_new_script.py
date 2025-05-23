import numpy as np
import torch
import itertools
from torch import nn
from torch.optim import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from joblib import Parallel, delayed, dump, load
import time
import os
import gc
import time

class OptimizedDifferentiablePchip(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = nn.Parameter(y)
        self.n = len(x) - 1
        self.d = None  # To store precomputed derivatives

    def _compute_derivatives(self, y):
        dy = y[1:] - y[:-1]
        dx = self.x[1:] - self.x[:-1]
        slopes = dy / dx
        d = torch.zeros_like(y)
        for i in range(1, len(y) - 1):
            if slopes[i - 1] * slopes[i] > 0:
                w1 = 2 * dx[i] + dx[i - 1]
                w2 = dx[i] + 2 * dx[i - 1]
                d[i] = (w1 + w2) / (w1 / slopes[i - 1] + w2 / slopes[i])
        d[0] = slopes[0]
        d[-1] = slopes[-1]
        return d

    def update_derivatives(self):
        """Compute and store derivatives for the current y values."""
        self.d = self._compute_derivatives(self.y)

    def forward(self, t):
        if self.d is None:
            self.update_derivatives()  # Fallback in case not precomputed
        t = t.contiguous()
        idx = torch.clamp(torch.searchsorted(self.x, t) - 1, 0, self.n - 1)
        x0 = self.x[idx]
        x1 = self.x[idx + 1]
        y0 = self.y[idx]
        y1 = self.y[idx + 1]
        t_norm = (t - x0) / (x1 - x0)
        d0 = self.d[idx]
        d1 = self.d[idx + 1]
        t2 = t_norm * t_norm
        t3 = t2 * t_norm
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t_norm
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        dx_segment = x1 - x0
        return h00 * y0 + h10 * dx_segment * d0 + h01 * y1 + h11 * dx_segment * d1

class OptimizedPyTorchGradientSMPA(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.05, epochs=100, random_state=7, verbose=False,
                 lambda_reg=0.0001, patience=10, decay_factor=0.9, min_learning_rate=1e-6,
                 n_control_points=6, device=None, track_history=False, optimizer_type='adam',
                 scheduler_type='reduce_on_plateau'):
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        self.lambda_reg = lambda_reg
        self.patience = patience
        self.decay_factor = decay_factor
        self.min_learning_rate = min_learning_rate
        self.n_control_points = n_control_points
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.track_history = track_history
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _to_tensor(self, data, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=dtype, non_blocking=True)
        return torch.tensor(data, dtype=dtype, device=self.device)

    def _calculate_class_means(self, X, y):
        mask_1 = y == 1
        self.m1 = torch.mean(X[mask_1], dim=0)
        self.m0 = torch.mean(X[~mask_1], dim=0)

    def _initialize_control_points(self, X):
        n_features = X.shape[1] - 1
        self.spline_models = nn.ModuleList()
        for i in range(n_features):
            x_min, x_max = X[:, i].min().item(), X[:, i].max().item()
            control_x = torch.linspace(x_min, x_max, self.n_control_points, device=self.device)
            y_min, y_max = X[:, -1].min().item(), X[:, -1].max().item()
            y_mid = (self.m0[-1] + self.m1[-1]) / 2
            y_range = y_max - y_min
            control_y = torch.empty(self.n_control_points, device=self.device).uniform_(
                y_mid - y_range * 0.05, y_mid + y_range * 0.05
            )
            spline = OptimizedDifferentiablePchip(control_x, control_y).to(self.device)
            self.spline_models.append(spline)
        self.initial_control_points = [(m.x.clone(), m.y.clone()) for m in self.spline_models]

    def _calculate_displacement(self, X):
        total_spline = sum(spline(X[:, i]) for i, spline in enumerate(self.spline_models))
        return X[:, -1] - total_spline

    def _update_pseudo_labels(self, X, y):
        m1_displacement = self._calculate_displacement(self.m1.unsqueeze(0))[0]
        self.class_1_pseudo = 1 if m1_displacement > 0 else -1
        self.class_0_pseudo = -self.class_1_pseudo
        return torch.where(y == 1, self.class_1_pseudo, self.class_0_pseudo)

    def _create_optimizer_and_scheduler(self):
        params = [p for spline in self.spline_models for p in spline.parameters()]
        if self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.initial_learning_rate)
        else:
            optimizer = torch.optim.SGD(params, lr=self.initial_learning_rate)
        if self.scheduler_type.lower() == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.decay_factor,
                patience=self.patience, min_lr=self.min_learning_rate)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.patience, gamma=self.decay_factor
            )
        return optimizer, scheduler

    def fit(self, X, y):
        try:
            l = np.unique(y)
            if len(l) != 2:
                raise ValueError("Algorithm for binary classification only.")
            if X.shape[1] < 2:
                raise ValueError("At least 2 features required")

            self.label_mapping = {l[0]: 0, l[1]: 1}
            y = np.where(y == l[0], 0, 1)

            X_tensor = self._to_tensor(X)
            y_tensor = self._to_tensor(y, dtype=torch.long)

            with torch.no_grad():
                self._calculate_class_means(X_tensor, y_tensor)
                self._initialize_control_points(X_tensor)

            optimizer, scheduler = self._create_optimizer_and_scheduler()

            best_error = float('inf')
            best_control_ys = [spline.y.clone() for spline in self.spline_models]
            best_class_1_pseudo = None

            if self.track_history:
                self.error_history_ = []
                self.control_point_history = [self.initial_control_points]

            for epoch in range(self.epochs):
                # Update derivatives for all PCHIP splines
                for spline in self.spline_models:
                    spline.update_derivatives()

                # Compute total spline evaluations once per epoch
                total_spline = sum(spline(X_tensor[:, i]) for i, spline in enumerate(self.spline_models))
                displacements = X_tensor[:, -1] - total_spline

                # Update pseudo labels and compute errors
                pseudo_labels = self._update_pseudo_labels(X_tensor, y_tensor)
                errors = displacements * pseudo_labels <= 0
                error_count = errors.sum().item()

                if self.verbose and epoch % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}: Errors = {error_count}, LR = {current_lr:.6f}")

                if error_count < best_error:
                    best_error = error_count
                    best_control_ys = [spline.y.clone() for spline in self.spline_models]
                    best_class_1_pseudo = self.class_1_pseudo
                    self.best_epoch = epoch
                    if error_count == 0 and epoch > 10:
                        if self.verbose:
                            print(f"Perfect separation achieved at epoch {epoch}")
                        break

                if self.track_history:
                    self.error_history_.append(error_count)
                    self.control_point_history.append(
                        [(s.x.clone().cpu().numpy(), s.y.clone().detach().cpu().numpy())
                         for s in self.spline_models]
                    )

                if error_count == 0:
                    continue

                # Compute loss using precomputed displacements
                error_indices = torch.where(errors)[0]
                displacements_err = displacements[error_indices]
                y_err = y_tensor[error_indices]
                ti = torch.where(y_err == 1, 1, -1)
                loss = torch.mean(torch.relu(1.0 - ti * self.class_1_pseudo * displacements_err))

                if self.lambda_reg > 0:
                    smoothness_penalty = 0
                    for spline in self.spline_models:
                        y_diff = spline.y[1:] - spline.y[:-1]
                        x_diff = spline.x[1:] - spline.x[:-1]
                        smoothness_penalty += torch.mean((y_diff / (x_diff + 1e-8))**2)
                    loss += self.lambda_reg * smoothness_penalty

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if scheduler is not None:  # Fixed scheduler check
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(error_count)
                    else:
                        scheduler.step()
                    if optimizer.param_groups[0]['lr'] <= self.min_learning_rate:
                        if self.verbose:
                            print(f"Minimum learning rate reached at epoch {epoch}")
                        break

            for spline, best_y in zip(self.spline_models, best_control_ys):
                spline.y.data = best_y
            self.class_1_pseudo = best_class_1_pseudo
        except Exception as e:
            print(f"Error in SMPA fit: {str(e)}", flush=True)
            import traceback
            traceback.print_exc(flush=True)
            raise
        return self

    def predict(self, X):
        X_tensor = self._to_tensor(X)
        displacements = self._calculate_displacement(X_tensor)
        predictions = torch.where(displacements > 0,
                                  torch.tensor(1 if self.class_1_pseudo > 0 else 0, device=self.device),
                                  torch.tensor(0 if self.class_1_pseudo > 0 else 1, device=self.device))
        pred_numpy = predictions.cpu().numpy()
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        return np.array([reverse_mapping[p] for p in pred_numpy])

    def predict_proba(self, X):
      X_tensor = self._to_tensor(X)
      displacements = self._calculate_displacement(X_tensor)
      raw_probs = 1 / (1 + torch.exp(-displacements * 0.5))  # Sigmoid, no pseudo-label scaling

      # Map probabilities to internal class indices (0 and 1)
      probs = torch.zeros(X.shape[0], 2, device=self.device)  # [n_samples, 2] for classes 0 and 1
      if self.class_1_pseudo > 0:
          # Positive displacement means class 1
          probs[:, 1] = raw_probs  # P(class 1)
          probs[:, 0] = 1 - raw_probs  # P(class 0)
      else:
          # Negative displacement means class 1
          probs[:, 0] = raw_probs  # P(class 0)
          probs[:, 1] = 1 - raw_probs  # P(class 1)

      return probs.cpu().detach().numpy()  # [n_samples, 2], columns: P(class_0), P(class_1)


# Initialize your SMPA model
smpa_model = OptimizedPyTorchGradientSMPA(
    learning_rate=0.05,
    epochs=200,
    random_state=4,
    verbose=True,
    n_control_points=12,
    track_history=True
)

from sklearn.preprocessing import StandardScaler

class OvOSMPAWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.05, epochs=100, random_state=7, verbose=False,
                 lambda_reg=0.0001, patience=10, decay_factor=0.9, min_learning_rate=1e-6,
                 n_control_points=6, device=None, track_history=False,
                 optimizer_type='adam', scheduler_type='reduce_on_plateau',
                 model_save_dir='smpa_models'):
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        self.lambda_reg = lambda_reg
        self.patience = patience
        self.decay_factor = decay_factor
        self.min_learning_rate = min_learning_rate
        self.n_control_points = n_control_points
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.track_history = track_history
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.model_save_dir = model_save_dir

    def _save_classifier(self, clf_info, filename):
        model = clf_info['model']
        state_dict = model.spline_models.state_dict()
        for key, tensor in state_dict.items():
            if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                raise ValueError(f"Invalid values in state_dict for key {key}")
        control_points = []
        for spline in model.spline_models:
            control_points.append({
                'x': spline.x.cpu().numpy().astype(np.float32),
                'y': spline.y.cpu().detach().numpy().astype(np.float32)
            })
        m0 = model.m0.cpu().numpy().astype(np.float32) if model.m0 is not None else None
        m1 = model.m1.cpu().numpy().astype(np.float32) if model.m1 is not None else None
        if m0 is not None and (np.any(np.isnan(m0)) or np.any(np.isinf(m0))):
            raise ValueError("Invalid values in m0")
        if m1 is not None and (np.any(np.isnan(m1)) or np.any(np.isinf(m1))):
            raise ValueError("Invalid values in m1")
        save_dict = {
            'features': clf_info['features'],
            'label_mapping': model.label_mapping,
            'class_1_pseudo': model.class_1_pseudo,
            'class_0_pseudo': model.class_0_pseudo,
            'm0': m0,
            'm1': m1,
            'control_points': control_points,
            'state_dict': {k: v.cpu().numpy().astype(np.float32) for k, v in state_dict.items()},
            'model_params': {
                'learning_rate': model.learning_rate,
                'epochs': model.epochs,
                'random_state': model.random_state,
                'lambda_reg': model.lambda_reg,
                'patience': model.patience,
                'decay_factor': model.decay_factor,
                'min_learning_rate': model.min_learning_rate,
                'n_control_points': model.n_control_points,
                'optimizer_type': model.optimizer_type,
                'scheduler_type': model.scheduler_type,
                'track_history': model.track_history
            }
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dump(save_dict, filename)

    def _load_classifier(self, filename):
        try:
            save_dict = load(filename)
        except Exception as e:
            raise ValueError(f"Failed to load model from {filename}: {str(e)}")
        model = OptimizedPyTorchGradientSMPA(
            learning_rate=save_dict['model_params']['learning_rate'],
            epochs=save_dict['model_params']['epochs'],
            random_state=save_dict['model_params']['random_state'],
            verbose=False,
            lambda_reg=save_dict['model_params']['lambda_reg'],
            patience=save_dict['model_params']['patience'],
            decay_factor=save_dict['model_params']['decay_factor'],
            min_learning_rate=save_dict['model_params']['min_learning_rate'],
            n_control_points=save_dict['model_params']['n_control_points'],
            device=self.device,
            track_history=save_dict['model_params']['track_history'],
            optimizer_type=save_dict['model_params']['optimizer_type'],
            scheduler_type=save_dict['model_params']['scheduler_type']
        )
        model.spline_models = nn.ModuleList()
        for cp in save_dict['control_points']:
            x = torch.tensor(cp['x'], dtype=torch.float32, device=self.device)
            y = torch.tensor(cp['y'], dtype=torch.float32, device=self.device, requires_grad=True)
            spline = OptimizedDifferentiablePchip(x, y).to(self.device)
            model.splineHooks.append(spline)
        state_dict = {k: torch.tensor(v, dtype=torch.float32, device=self.device)
                     for k, v in save_dict['state_dict'].items()}
        try:
            model.spline_models.load_state_dict(state_dict)
        except Exception as e:
            raise ValueError(f"Failed to load state_dict: {str(e)}")
        model.label_mapping = save_dict['label_mapping']
        model.class_1_pseudo = save_dict['class_1_pseudo']
        model.class_0_pseudo = save_dict['class_0_pseudo']
        model.m0 = torch.tensor(save_dict['m0'], dtype=torch.float32, device=self.device) if save_dict['m0'] is not None else None
        model.m1 = torch.tensor(save_dict['m1'], dtype=torch.float32, device=self.device) if save_dict['m1'] is not None else None
        return {
            'model': model,
            'features': save_dict['features']
        }

    def train_pair(self, pair, X, y, counter):
        model_filename = os.path.join(self.model_save_dir, f"model_{pair[0]}_vs_{pair[1]}.joblib")
        if os.path.exists(model_filename):
            if self.verbose:
                print(f"Loading existing classifier for {pair[0]} vs {pair[1]}")
            try:
                return pair, self._load_classifier(model_filename)
            except Exception as e:
                print(f"Error loading model for {pair}: {str(e)}, retraining...")
        if self.verbose:
            print(f"Training {counter:02d}/{self.total_combos:02d} classifier: {pair[0]} vs {pair[1]}")
        mask = np.isin(y, [pair[0], pair[1]])
        X_pair = X[mask].astype(np.float32)  # Downcast to float32
        y_pair = y[mask]
        variances = np.var(X_pair, axis=0)
        valid_features = variances > 0
        X_pair_filtered = X_pair[:, valid_features]
        unique_labels = np.unique(y_pair)
        if len(unique_labels) != 2:
            print(f"Warning: Expected 2 classes but found {len(unique_labels)} for pair {pair}")
            return pair, None
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.05],
            'epochs': [200],
            'n_control_points': [7, 10],  # Reduced from [7, 10, 15]
            'optimizer_type': ['adam']
        }
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        best_accuracy = -1
        best_model = None
        best_params = None
        for params in param_combinations:  # Sequential grid search
            classifier = OptimizedPyTorchGradientSMPA(
                learning_rate=params['learning_rate'],
                epochs=params['epochs'],
                random_state=self.random_state,
                verbose=False,
                lambda_reg=self.lambda_reg,
                patience=self.patience,
                decay_factor=self.decay_factor,
                min_learning_rate=self.min_learning_rate,
                n_control_points=params['n_control_points'],
                device=self.device,
                track_history=self.track_history,
                optimizer_type=params['optimizer_type'],
                scheduler_type=self.scheduler_type
            )
            classifier.fit(X_pair_filtered, y_pair)
            preds = classifier.predict(X_pair_filtered)
            accuracy = accuracy_score(y_pair, preds)
            if self.verbose:
                print(f"  Params: {params}, Accuracy: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = classifier
                best_params = params
            if accuracy == 1:
                print(f"1.00 accuracy on training data, breaking for pair {pair}")
                break
            # Clean up
            del classifier
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        if best_model is None:
            print(f"Warning: No valid model trained for {pair}")
            return pair, None
        if self.verbose:
            print(f"  Best params for {pair}: {best_params}, Accuracy: {best_accuracy:.4f}")
        clf_info = {
            'model': best_model,
            'features': valid_features
        }
        try:
            self._save_classifier(clf_info, model_filename)
            if self.verbose:
                print(f"  Saved model for {pair} to {model_filename}")
        except Exception as e:
            print(f"Error saving model for {pair}: {str(e)}")
            raise
        # Clean up
        del X_pair, X_pair_filtered, y_pair
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return pair, clf_info

    def fit(self, X, y):
        os.makedirs(self.model_save_dir, exist_ok=True)
        class_labels = np.unique(y)
        class_combos = list(itertools.combinations(class_labels, r=2))
        self.classifiers = {}
        n_classes = len(class_labels)
        self.total_combos = n_classes * (n_classes - 1) // 2
        results = Parallel(n_jobs=10, backend='loky', verbose=10)(  # Reduced from 10
            delayed(self.train_pair)(pair, X, y, i + 1) for i, pair in enumerate(class_combos)
        )
        for result in results:
            if not isinstance(result, tuple) or len(result) != 2:
                print(f"Warning: Invalid result format for pair: {result}")
                continue
            pair, clf_info = result
            if clf_info is not None:
                if not isinstance(clf_info, dict) or 'model' not in clf_info or 'features' not in clf_info:
                    print(f"Warning: Invalid clf_info for pair {pair}: {clf_info}")
                    continue
                self.classifiers[pair] = clf_info
            else:
                print(f"Warning: No classifier info for pair {pair}")
        if not self.classifiers:
            raise ValueError("No classifiers trained!")
        self.class_labels_ = class_labels
        return self

    def predict(self, X):
        prob_scores = np.zeros((X.shape[0], len(self.class_labels_)), dtype=np.float32)
        label_to_idx = {label: idx for idx, label in enumerate(self.class_labels_)}
        for pair, clf_info in self.classifiers.items():
            if not isinstance(clf_info, dict):
                print(f"Error: Invalid clf_info for pair {pair}: {clf_info}")
                continue
            model = clf_info.get('model')
            features = clf_info.get('features')
            if model is None or features is None:
                print(f"Error: Missing model or features for pair {pair}")
                continue
            X_filtered = X[:, features].astype(np.float32)
            probs = model.predict_proba(X_filtered)
            label_0 = list(model.label_mapping.keys())[list(model.label_mapping.values()).index(0)]
            label_1 = list(model.label_mapping.keys())[list(model.label_mapping.values()).index(1)]
            idx_0 = label_to_idx[label_0]
            idx_1 = label_to_idx[label_1]
            prob_scores[:, idx_0] += probs[:, 0]
            prob_scores[:, idx_1] += probs[:, 1]
        return self.class_labels_[np.argmax(prob_scores, axis=1)]

# Main execution
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train = train_dataset.data.numpy().reshape(60000, -1).astype(np.float32)  # Downcast
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().reshape(10000, -1).astype(np.float32)
y_test = test_dataset.targets.numpy()

rng = np.random.RandomState(12)
shuffle_idx = rng.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

OvA_SMPA = OvOSMPAWrapper(
    learning_rate=0.08,
    epochs=200,
    random_state=12,
    verbose=True,
    n_control_points=10,
    optimizer_type='adam',
    track_history=False,  # Disabled
    model_save_dir='smpa_models_mnist'
)

start_time = time.time()
OvA_SMPA.fit(X_train, y_train)
train_time = time.time() - start_time

pred_time_start = time.time()
y_pred = OvA_SMPA.predict(X_test)
predict_time = time.time() - pred_time_start

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))