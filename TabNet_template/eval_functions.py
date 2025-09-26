import sys, os
import numpy as np
import torch
from pytorch_tabnet.metrics import Metric
from helpers import compute_class_counts
from sklearn.metrics import mean_squared_error, roc_auc_score

sys.path.append(os.environ["DIR_PATH"])
sys.path.append(os.environ["DIR_PATH"] + "/Class-balanced-loss-pytorch")
from class_balanced_loss import CB_loss

class ClassBalancedFocalLoss:
    """
    TabNet의 loss_fn으로 넘길 수 있는 콜러블.
    - counts: 학습 세트에서 계산한 클래스별 가중합
    - num_classes, gamma, device 파라미터를 내부에 바인딩
    """
    def __init__(self,
                 counts: np.ndarray,
                 num_classes: int,
                 gamma: float = 0.0,
                 device: torch.device = torch.device("cpu"),
                 force_cpu: bool = False):
        self.counts = np.asarray(counts, dtype=np.float64)
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.device = device
        self.force_cpu = force_cpu

    def __call__(self, y_pred, y_true, y_weight=None):
        # CB_loss는 내부에서 log-softmax/softmax를 처리한다는 가정 하에 원래 인자 형태 유지
        return CB_loss(
            y_true,
            y_pred,
            self.counts,
            no_of_classes=self.num_classes,
            device=self.device,
            gamma=self.gamma,
            evt_weight=y_weight,
            force_use_cpu=self.force_cpu,
        )


class CBFocalLossMetric(Metric):
    """
    검증용 metric: y_true, y_w로 '그때그때' counts 계산.
    - TabNet이 metric registry 스캔할 때 인자 없이 생성해도 안전해야 함.
    """
    def __init__(self,
                 num_classes: int | None = None,
                 gamma: float = 0.0,
                 device: torch.device = torch.device("cpu")):
        self._name = "CBFocalLoss[val]"
        self._maximize = False
        self.num_classes = None if num_classes is None else int(num_classes)
        self.gamma = float(gamma)
        self.device = device

    def __call__(self, y_true, y_score, y_w):
        import numpy as np
        from helpers import compute_class_counts
        from eval_functions import CB_loss  # 당신 코드의 CB_loss

        # y_score: softmax 확률 → log-softmax
        y_score = np.asarray(y_score, dtype=np.float64)
        y_score = np.clip(y_score, 1e-12, 1.0)
        log_prob = np.log(y_score)

        y_true = np.asarray(y_true, dtype=int)
        w = None if y_w is None else np.asarray(y_w, dtype=np.float64)

        # 필요시 num_classes 유추
        num_classes = self.num_classes or (int(np.max(y_true)) + 1)
        counts_val = compute_class_counts(y_true, w, num_classes=num_classes)

        loss = CB_loss(
            torch.tensor(y_true, dtype=torch.long, device="cpu"),
            torch.tensor(log_prob, dtype=torch.float32, device="cpu"),
            counts_val,
            no_of_classes=num_classes,
            device=torch.device("cpu"),
            gamma=self.gamma,
            evt_weight=(None if w is None else torch.tensor(w, dtype=torch.float32, device="cpu")),
            force_use_cpu=True,
        )
        return float(loss.detach().cpu().numpy())


class WeightedAUC(Metric):
    def __init__(self):
        self._name = "WeightedAUC"
        self._maximize = True

    def __call__(self, y_true, y_score, y_w):
        # Compute the AUC
        y_score = y_score[:, 1]
        y = np.array(y_true)
        score = np.array(y_score)
        weight = np.array(y_w)

        score = score[weight > 0]
        y = y[weight > 0]
        weight = weight[weight > 0]

        sorted_index = score.argsort()

        score = score[sorted_index]
        y = y[sorted_index]
        weight = weight[sorted_index]

        unique_score, unique_score_indices = np.unique(score, return_index=True)
        y = y[unique_score_indices]
        weight = weight[unique_score_indices]

        roc_auc = roc_auc_score(y, unique_score, sample_weight=weight)
        return roc_auc



class MaxSignificance(Metric):
    """
    Metric: score∈[0,1]에서 상위 컷(threshold)을 스캔하여 최대 유의도 반환.
      - bins: 100 -> np.linspace(0,1,101)
      - mode: 'asimov' | 's_sqrt_sb' | 's_sqrt_b'
      - clamp_nonneg: 누적 s,b<0를 0으로 클램프(음수 가중치 대응)
    반환값: 최대 Z (float)
    추가 정보: self.best_threshold, self.best_s, self.best_b 에 저장
    """
    def __init__(self, bins=100, mode='asimov', clamp_nonneg=False, clip01=False):
        self._name = f"MaxSignificance[{mode}]"
        self._maximize = True
        self.bins = int(bins)
        self.mode = mode
        self.clamp_nonneg = clamp_nonneg
        self.clip01 = clip01

        # extras
        self.best_threshold = None
        self.best_s = None
        self.best_b = None
        self.curve_ = None    # Z(threshold) 배열
        self.edges_ = None    # bin edges

    def _asimov_z(self, s, b):
        z = np.zeros_like(s, dtype=np.float64)
        m1 = (s > 0) & (b > 0)
        z[m1] = np.sqrt(2.0 * ((s[m1] + b[m1]) * np.log(1.0 + s[m1] / b[m1]) - s[m1]))
        m2 = (s > 0) & (b == 0)
        z[m2] = np.sqrt(2.0 * s[m2])  # B≈0 극한
        return z

    def _z_from_sb(self, s, b):
        if self.mode == 's_sqrt_sb':
            denom = np.sqrt(s + b)
        else:  # 's_sqrt_b'
            denom = np.sqrt(b)
        z = np.divide(s, denom, out=np.zeros_like(s, dtype=np.float64), where=denom > 0)
        return z

    def __call__(self, y_true, y_score, y_w):
        # y_score: (N,2) → 양성 확률 칼럼 사용, or (N,)
        ys = np.asarray(y_score)
        if ys.ndim == 2 and ys.shape[1] >= 2:
            score = ys[:, 0]
            score = score.astype(np.float32, copy=False)
        else:
            raise ValueError("Invalid shape for y_score")
            score = ys.astype(np.float32, copy=False)
        y = np.asarray(y_true, dtype=np.int8)
        w = np.asarray(y_w, dtype=np.float64)

        # 안전 필터 + score 클리핑
        m = np.isfinite(score) & np.isfinite(y) & np.isfinite(w)
        score, y, w = score[m], y[m], w[m]
        if self.clip01:
            score = np.clip(score, 0.0, 1.0)

        # --- binning: [0,1]을 100등분 ---
        edges = np.linspace(0.0, 1.0, self.bins + 1)
        s_mask =  (y == 0)
        s_hist, _ = np.histogram(score[s_mask], bins=edges, weights=w[s_mask])
        b_hist, _ = np.histogram(score[~s_mask], bins=edges, weights=w[~s_mask])

        # 위에서부터 누적(고점수 → 저점수)
        s_cum = np.cumsum(s_hist[::-1])[::-1].astype(np.float64, copy=False)
        b_cum = np.cumsum(b_hist[::-1])[::-1].astype(np.float64, copy=False)

        if self.clamp_nonneg:
            # 음수 가중치로 인해 누적이 음수가 되는 비물리 상황 방지
            s_cum = np.maximum(s_cum, 0.0)
            b_cum = np.maximum(b_cum, 0.0)

        # --- Z 계산 ---
        if self.mode == 'asimov':
            z = self._asimov_z(s_cum, b_cum)
        elif self.mode in ('s_sqrt_sb', 's_sqrt_b'):
            z = self._z_from_sb(s_cum, b_cum)
        else:
            raise ValueError("mode must be 'asimov', 's_sqrt_sb', or 's_sqrt_b'")

        if z.size == 0:
            return np.nan

        # 최대점 선택
        idx = int(np.nanargmax(z))
        best_z = float(z[idx])

        # 리포트용 속성 보관(컷은 'score >= threshold')
        self.best_threshold = float(edges[idx])
        self.best_s = float(s_cum[idx])
        self.best_b = float(b_cum[idx])
        self.curve_ = z
        self.edges_ = edges
        print(f"MaxSignificance[{self.mode}] = best_z : {best_z}, best_threshold : {self.best_threshold}")
        return best_z

# --------------------------------
# Factory functions
# --------------------------------

def make_cb_focal_metric_cls(num_classes: int, gamma: float, device: torch.device):
    class _CBFocalLossMetricConfigured(CBFocalLossMetric):
        def __init__(self):
            super().__init__(num_classes=num_classes, gamma=gamma, device=device)
    _CBFocalLossMetricConfigured.__name__ = f"CBFocalLossMetric_nc{num_classes}_g{gamma}"
    return _CBFocalLossMetricConfigured

def make_maxsig_metric_cls(bins=100, mode='asimov', clamp_nonneg=False, clip01=False):
    class _MaxSigConfigured(MaxSignificance):
        def __init__(self):
            super().__init__(bins=bins, mode=mode, clamp_nonneg=clamp_nonneg, clip01=clip01)
    _MaxSigConfigured.__name__ = f"MaxSig_{mode}_b{bins}"
    return _MaxSigConfigured