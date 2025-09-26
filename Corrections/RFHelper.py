import numpy as np
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import awkward as ak
import ROOT

RFProvider = Optional[Callable[[Dict[str, ak.Array], str, str, str], np.ndarray]]

class _RFTable:
    __slots__ = ("xedges", "yedges", "B", "C", "L", "use_abs_eta")
    def __init__(self, hB, hC, hL, use_abs_eta=True):
        # 축 엣지
        ax = hB.GetXaxis() if hB else (hC.GetXaxis() if hC else hL.GetXaxis())
        ay = hB.GetYaxis() if hB else (hC.GetYaxis() if hC else hL.GetYaxis())
        nx, ny = ax.GetNbins(), ay.GetNbins()
        self.xedges = np.array([ax.GetBinLowEdge(i) for i in range(1, nx+2)], dtype=np.float32)
        self.yedges = np.array([ay.GetBinLowEdge(i) for i in range(1, ny+2)], dtype=np.float32)
        # 콘텐츠 (nx, ny)
        def to_arr(h):
            if h is None:
                return None
            arr = np.empty((nx, ny), dtype=np.float32)
            for ix in range(1, nx+1):
                # 열 방향 루프가 내부에서 연속 접근
                for iy in range(1, ny+1):
                    arr[ix-1, iy-1] = h.GetBinContent(ix, iy)
            return arr
        self.B = to_arr(hB)
        self.C = to_arr(hC)
        self.L = to_arr(hL)
        self.use_abs_eta = use_abs_eta

    def eval_flat(self, pt: np.ndarray, eta: np.ndarray, hadflav: np.ndarray) -> np.ndarray:
        # 클램프(상한은 바로 아래쪽 값으로)
        x_hi = np.nextafter(self.xedges[-1], -np.inf)
        y_hi = np.nextafter(self.yedges[-1], -np.inf)
        x = np.clip(pt.astype(np.float32), self.xedges[0], x_hi)
        y_in = np.abs(eta, dtype=np.float32) if self.use_abs_eta else eta.astype(np.float32)
        y = np.clip(y_in, self.yedges[0], y_hi)

        # bin 찾기 (0..nx-1), (0..ny-1)
        ix = np.searchsorted(self.xedges, x, side="right") - 1
        iy = np.searchsorted(self.yedges, y, side="right") - 1

        # flavor 선택: 5->B, 4->C, else L
        af = np.abs(hadflav)
        # 미리 결과 배열 준비
        out = np.ones_like(x, dtype=np.float32)

        # B
        mB = (af == 5)
        if mB.any():
            out[mB] = self.B[ix[mB], iy[mB]] if self.B is not None else 1.0
        # C
        mC = (af == 4)
        if mC.any():
            out[mC] = self.C[ix[mC], iy[mC]] if self.C is not None else 1.0
        # L
        mL = (~mB & ~mC)
        if mL.any():
            out[mL] = self.L[ix[mL], iy[mL]] if self.L is not None else 1.0

        return out
    
def _load_table(path: str, sample: str, tag_kind: str, mode: Optional[int]) -> _RFTable:
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {path}")
    mode_suffix = f"_{mode}" if (mode is not None and mode >= 0) else ""
    dir_ = f"D/{sample}{mode_suffix}"
    base = f"Ratio_D_{sample}{mode_suffix}_{tag_kind}_Nominal_"
    hB = f.Get(f"{dir_}/{base}B")
    hC = f.Get(f"{dir_}/{base}C")
    hL = f.Get(f"{dir_}/{base}L")
    # 디스크 연결 분리(복사) 필요 없음: 우리가 bin content만 읽어 배열로 복사
    tbl = _RFTable(hB, hC, hL, use_abs_eta=True)
    f.Close()
    return tbl

_TABLE_CACHE: Dict[Tuple[str, str, Optional[int], str], _RFTable] = {}

def _get_table(corrections_root_dir: str, load_btag: bool, short: str, year: str, mode: Optional[int]) -> _RFTable:
    tag_kind = "B_Tag" if load_btag else "C_Tag"
    subdir = "bTag" if load_btag else "cTag"
    key = (short, year, mode, tag_kind)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    path = f"{corrections_root_dir}/{subdir}/Vcb_Tagging_RF_Flavor_{year}.root"
    tbl = _load_table(path, short, tag_kind, mode)
    _TABLE_CACHE[key] = tbl
    return tbl


def make_rf_provider_fast(
    load_btag: bool,
    corrections_root_dir: str = "/data6/Users/yeonjoon/VcbMVAStudy/Corrections",
) -> RFProvider:
    """
    고속 버전:
      - TH2D를 NumPy 테이블로 캐시
      - per-jet binning vectorize -> per-event product(ak.prod)
      - TTLJ는 event별 whatMode(2/4/45) 지원 (없으면 45)
    """
    def provider(arrays: Dict[str, ak.Array], short_mc: str, year_tag: str) -> np.ndarray:
        n_events = len(next(iter(arrays.values()))) if arrays else 0
        if n_events == 0 or not year_tag or short_mc == "DATA":
            return np.ones(n_events, dtype=np.float32)

        jet_pt   = arrays.get("Jet_Pt")
        jet_eta  = arrays.get("Jet_Eta")
        jet_flav = arrays.get("Jet_Flavor")
        if jet_pt is None or jet_eta is None or jet_flav is None:
            return np.ones(n_events, dtype=np.float32)

        # 이벤트별 젯 개수
        counts = ak.to_numpy(ak.num(jet_pt, axis=1))
        if counts.size == 0:
            return np.ones(n_events, dtype=np.float32)

        # 플랫 배열
        pt_flat   = ak.to_numpy(ak.flatten(jet_pt, axis=None))
        eta_flat  = ak.to_numpy(ak.flatten(jet_eta, axis=None))
        flav_flat = ak.to_numpy(ak.flatten(jet_flav, axis=None))

        if short_mc == "TTLJ":
            # event mode -> per-jet broadcast
            wm = arrays.get("whatMode")
            if wm is None:
                # 전부 45로
                tbl = _get_table(corrections_root_dir, load_btag, "TTLJ", year_tag, 45)
                sf_flat = tbl.eval_flat(pt_flat, eta_flat, flav_flat)
            else:
                wm_ev = ak.to_numpy(ak.values_astype(wm, np.int32))
                # per-jet 모드
                wm_per_jet = ak.to_numpy(ak.flatten(ak.broadcast_arrays(wm, jet_pt)[0], axis=None)).astype(np.int32)
                sf_flat = np.ones_like(pt_flat, dtype=np.float32)
                # 모드별 부분 벡터 처리
                for m in (2, 4, 45):
                    mask = (wm_per_jet == m)
                    if mask.any():
                        tbl = _get_table(corrections_root_dir, load_btag, "TTLJ", year_tag, m)
                        sf_flat[mask] = tbl.eval_flat(pt_flat[mask], eta_flat[mask], flav_flat[mask])
                # 예상 외 모드는 45로 폴백
                mask_else = ~((wm_per_jet == 2) | (wm_per_jet == 4) | (wm_per_jet == 45))
                if mask_else.any():
                    tbl = _get_table(corrections_root_dir, load_btag, "TTLJ", year_tag, 45)
                    sf_flat[mask_else] = tbl.eval_flat(pt_flat[mask_else], eta_flat[mask_else], flav_flat[mask_else])
        else:
            tbl = _get_table(corrections_root_dir, load_btag, short_mc, year_tag, None)
            sf_flat = tbl.eval_flat(pt_flat, eta_flat, flav_flat)

        # per-event 곱
        sf_nested = ak.unflatten(sf_flat, counts)
        ev_w = ak.prod(sf_nested, axis=1, mask_identity=True)
        ev_w = ak.fill_none(ev_w, 1.0)
        return np.asarray(ev_w, dtype=np.float32)

    return provider

