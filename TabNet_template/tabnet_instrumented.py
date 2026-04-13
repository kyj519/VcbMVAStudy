import numpy as np
import scipy
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Sampler

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.utils import SparseTorchDataset, TorchDataset, create_dataloaders


class _ReplacementEpochBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        y_train,
        *,
        mode: str,
        num_samples: int,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.y_arr = np.asarray(y_train, dtype=np.int64)
        if self.y_arr.size == 0:
            raise ValueError("Cannot build epoch batch sampler from an empty training set.")

        self.mode = str(mode).strip()
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._iteration = 0

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.mode not in {"uniform", "class_balanced"}:
            raise ValueError(
                f"Unsupported epoch_sampling_mode={self.mode!r}. "
                "Use 'class_balanced' or 'uniform'."
            )

        self.present_classes = np.unique(self.y_arr)
        self.class_indices = {
            int(cls): np.flatnonzero(self.y_arr == cls) for cls in self.present_classes
        }

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        remaining = self.num_samples
        class_prob = np.full(self.present_classes.size, 1.0 / self.present_classes.size)

        for _ in range(len(self)):
            current_batch = min(self.batch_size, remaining)
            if current_batch <= 0:
                break

            if self.mode == "uniform":
                indices = rng.integers(
                    0, self.y_arr.size, size=current_batch, endpoint=False, dtype=np.int64
                )
                yield indices.tolist()
            else:
                draws_by_class = rng.multinomial(current_batch, class_prob)
                parts = []
                for cls, n_draws in zip(self.present_classes, draws_by_class):
                    if n_draws <= 0:
                        continue
                    pool = self.class_indices[int(cls)]
                    picked = pool[
                        rng.integers(
                            0, pool.size, size=int(n_draws), endpoint=False, dtype=np.int64
                        )
                    ]
                    parts.append(picked)

                if not parts:
                    yield []
                else:
                    batch_idx = np.concatenate(parts, axis=0)
                    rng.shuffle(batch_idx)
                    yield batch_idx.tolist()
            remaining -= current_batch


class InstrumentedTabNetClassifier(TabNetClassifier):
    def _build_valid_dataloaders(self, eval_set):
        valid_dataloaders = []
        if len(eval_set[0]) == 2:
            for X, y in eval_set:
                if scipy.sparse.issparse(X):
                    valid_dataloaders.append(
                        DataLoader(
                            SparseTorchDataset(X.astype(np.float32), y),
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=False,
                        )
                    )
                else:
                    valid_dataloaders.append(
                        DataLoader(
                            TorchDataset(X.astype(np.float32), y),
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=False,
                        )
                    )
        else:
            for X, y, w in eval_set:
                if scipy.sparse.issparse(X):
                    valid_dataloaders.append(
                        DataLoader(
                            SparseTorchDataset(X.astype(np.float32), y, w),
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=False,
                        )
                    )
                else:
                    valid_dataloaders.append(
                        DataLoader(
                            TorchDataset(X.astype(np.float32), y, w),
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=False,
                        )
                    )
        return valid_dataloaders

    def _construct_loaders(self, X_train, y_train, eval_set, w_train=None):
        y_train_mapped = self.prepare_target(y_train)
        if len(eval_set[0]) == 2:
            for i, (X, y) in enumerate(eval_set):
                eval_set[i] = (X, self.prepare_target(y))
        else:
            for i, (X, y, w) in enumerate(eval_set):
                eval_set[i] = (X, self.prepare_target(y), w)

        epoch_train_event_count = getattr(self, "epoch_train_event_count", None)
        if epoch_train_event_count is None:
            return create_dataloaders(
                X_train,
                y_train_mapped,
                eval_set,
                self.updated_weights,
                self.batch_size,
                self.num_workers,
                self.drop_last,
                self.pin_memory,
                train_weight=w_train,
            )

        epoch_sampling_mode = str(
            getattr(self, "epoch_sampling_mode", "class_balanced")
        ).strip()
        batch_sampler = _ReplacementEpochBatchSampler(
            y_train_mapped,
            mode=epoch_sampling_mode,
            num_samples=int(epoch_train_event_count),
            batch_size=int(self.batch_size),
            drop_last=False,
            seed=int(getattr(self, "epoch_sampling_seed", 42)),
        )

        if scipy.sparse.issparse(X_train):
            train_dataset = SparseTorchDataset(
                X_train.astype(np.float32), y_train_mapped, w_train
            )
        else:
            train_dataset = TorchDataset(X_train.astype(np.float32), y_train_mapped, w_train)

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        print(
            f"[train] epoch sampler enabled: mode={epoch_sampling_mode}, "
            f"events/epoch={int(epoch_train_event_count):,}, replacement=True (numpy batch sampler)"
        )
        return train_dataloader, self._build_valid_dataloaders(eval_set)

    def _train_batch(self, X, y, w=None):
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).float()
        if w is not None:
            w = w.to(self.device).float()

        if self.augmentations is not None:
            X, y = self.augmentations(X, y)

        for param in self.network.parameters():
            param.grad = None

        output, M_loss = self.network(X)
        if w is None:
            task_loss = self.compute_loss(output, y)
        else:
            task_loss = self.compute_loss(output, y, w)

        sparse_penalty = -self.lambda_sparse * M_loss
        loss = task_loss + sparse_penalty

        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = float(loss.detach().cpu().item())
        batch_logs["task_loss"] = float(task_loss.detach().cpu().item())
        batch_logs["M_loss"] = float(M_loss.detach().cpu().item())
        batch_logs["sparse_penalty"] = float(sparse_penalty.detach().cpu().item())
        return batch_logs
