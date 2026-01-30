import numpy as np
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import awkward as ak
import ROOT

# provider 시그니처: (arrays, short_mc, year_tag, syst_name)
RFProvider = Optional[Callable[[Dict[str, ak.Array], str, str, Optional[str]], np.ndarray]]

class _RFTable:
    __slots__ = ("xedges", "yedges", "B", "C", "L", "use_abs_eta")
    def __init__(self, hB, hC, hL, use_abs_eta=False):
        # 축 엣지
        ax = hB.GetXaxis() if hB else (hC.GetXaxis() if hC else hL.GetXaxis())
        ay = hB.GetYaxis() if hB else (hC.GetYaxis() if hC else hL.GetYaxis())
        nx, ny = ax.GetNbins(), ay.GetNbins()
        self.xedges = np.array([ax.GetBinLowEdge(i) for i in range(1, nx+2)], dtype=np.float32)
        self.yedges = np.array([ay.GetBinLowEdge(i) for i in range(1, ny+2)], dtype=np.float32)

        def to_arr(h):
            if h is None:
                return None
            arr = np.empty((nx, ny), dtype=np.float32)
            # (ix,iy) 순서 유지
            for ix in range(1, nx+1):
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

        nx = len(self.xedges) - 1
        ny = len(self.yedges) - 1
        ix = np.searchsorted(self.xedges, x, side="right") - 1
        iy = np.searchsorted(self.yedges, y, side="right") - 1
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        af = np.abs(hadflav)
        out = np.ones_like(x, dtype=np.float32)

        mB = (af == 5)
        if mB.any():
            out[mB] = self.B[ix[mB], iy[mB]] if self.B is not None else 1.0
        mC = (af == 4)
        if mC.any():
            out[mC] = self.C[ix[mC], iy[mC]] if self.C is not None else 1.0
        mL = (~mB & ~mC)
        if mL.any():
            out[mL] = self.L[ix[mL], iy[mL]] if self.L is not None else 1.0

        return out


def _load_table(path: str, sample: str, tag_kind: str, mode: Optional[int], syst: str = "Nominal") -> _RFTable:
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {path}")
    mode_suffix = f"_{mode}" if (mode is not None and mode >= 0) else ""
    dir_ = f"D/{sample}{mode_suffix}"
    base = f"Ratio_D_{sample}{mode_suffix}_{tag_kind}_{syst}_"
    print(f"Loading RF table from {path}:{dir_} with base {base}")
    print(f"  B: {dir_}/{base}B")
    print(f"  C: {dir_}/{base}C")
    print(f"  L: {dir_}/{base}L")
    hB = f.Get(f"{dir_}/{base}B")
    hC = f.Get(f"{dir_}/{base}C")
    hL = f.Get(f"{dir_}/{base}L")
    tbl = _RFTable(hB, hC, hL, use_abs_eta=False)
    f.Close()
    return tbl


# 캐시 키에 syst 추가!
_TABLE_CACHE: Dict[Tuple[str, str, Optional[int], str, str], _RFTable] = {}
# key = (short, year, mode, tag_kind, syst)


def _get_table(corrections_root_dir: str, load_btag: bool, short: str, year: str, mode: Optional[int], syst: str = "Nominal") -> _RFTable:
    tag_kind = "B_Tag" if load_btag else "C_Tag"
    subdir = "bTag" if load_btag else "cTag"
    key = (short, year, mode, tag_kind, syst)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    path = f"{corrections_root_dir}/{subdir}/Vcb_Tagging_RF_Flavor_{year}.root"
    print(f"Loading RF table for {short} {year} mode={mode} tag={tag_kind} syst={syst}")
    tbl = _load_table(path, short, tag_kind, mode, syst)
    _TABLE_CACHE[key] = tbl
    return tbl


def _warm_load_all_systs_for(corrections_root_dir: str, load_btag: bool, short: str, year: str, mode: Optional[int], syst_list: Sequence[str]) -> None:
    """해당 (short, year, mode, tag)에 대해 syst_list 전부를 캐시에 선로딩."""
    for s in syst_list:
        _get_table(corrections_root_dir, load_btag, short, year, mode, s)


def make_rf_provider_fast(
    load_btag: bool,
    corrections_root_dir: str = "/data6/Users/yeonjoon/VcbMVAStudy/Corrections",
    syst_list: Sequence[str] = ("Nominal",),
) -> RFProvider:
    """
    고속 버전 (멀티-시스테매틱 선로딩 지원):
      - TH2D를 NumPy 테이블로 캐시 (시스테매틱별로 별도 캐시)
      - 최초로 만나는 (short_mc, year_tag, mode) 조합에 대해 syst_list를 **모두 로드**해서 캐시에 넣음
      - provider 호출 시 4번째 인자 `syst_name`으로 사용할 시스테매틱 지정 (None/미지정이면 "Nominal")
      - per-jet binning vectorize -> per-event product(ak.prod)
      - TTLJ는 event별 whatMode(2/4/45) 지원 (없으면 에러)
    """
    # 정규화된 목록
    syst_list = tuple(dict.fromkeys(syst_list))  # 중복 제거, 순서 유지
    if "Nominal" not in syst_list:
        syst_list = ("Nominal",) + syst_list

    def provider(arrays: Dict[str, ak.Array], short_mc: str, year_tag: str, syst_name: Optional[str] = None) -> np.ndarray:
        syst_sel = (syst_name or "Nominal")

        n_events = len(next(iter(arrays.values()))) if arrays else 0
        if n_events == 0 or not year_tag or short_mc == "DATA":
            return np.ones(n_events, dtype=np.float32)

        jet_pt   = arrays.get("Jet_Pt")
        jet_eta  = arrays.get("Jet_Eta")
        jet_flav = arrays.get("Jet_Flavor")
        if jet_pt is None or jet_eta is None or jet_flav is None:
            return np.ones(n_events, dtype=np.float32)

        counts = ak.to_numpy(ak.num(jet_pt, axis=1))
        if counts.size == 0:
            return np.ones(n_events, dtype=np.float32)

        pt_flat   = ak.to_numpy(ak.flatten(jet_pt, axis=None))
        eta_flat  = ak.to_numpy(ak.flatten(jet_eta, axis=None))
        flav_flat = ak.to_numpy(ak.flatten(jet_flav, axis=None))

        if short_mc == "TTLJ":
            decay_mode = arrays.get("decay_mode")
            if decay_mode is None:
                raise RuntimeError("TTLJ requires 'decay_mode' branch for decay_mode")

            wm = ak.zeros_like(decay_mode)  # default 0
            wm = ak.where((decay_mode//10)%10 == 2, 2, wm)
            wm = ak.where((decay_mode//10)%10 == 4, 4, wm)
            wm = ak.where(decay_mode == 45, 45, wm)
            if ak.any(wm==0):
                raise RuntimeError("TTLJ decay_mode has unexpected value (not 2,4,45)")

            wm_per_jet = ak.to_numpy(ak.flatten(ak.broadcast_arrays(wm, jet_pt)[0], axis=None)).astype(np.int32)

            # 최초 호출 시: 해당 (short_mc, year_tag, mode={2,4,45})에 대해 syst_list 전체 선로딩
            for m in (2, 4, 45):
                _warm_load_all_systs_for(corrections_root_dir, load_btag, "TTLJ", year_tag, m, syst_list)

            sf_flat = np.ones_like(pt_flat, dtype=np.float32)
            # 모드별 부분 벡터 처리 (선택된 syst_sel 사용)
            for m in (2, 4, 45):
                mask = (wm_per_jet == m)
                if mask.any():
                    tbl = _get_table(corrections_root_dir, load_btag, "TTLJ", year_tag, m, syst_sel)
                    sf_flat[mask] = tbl.eval_flat(pt_flat[mask], eta_flat[mask], flav_flat[mask])

            # 방어
            mask_else = ~((wm_per_jet == 2) | (wm_per_jet == 4) | (wm_per_jet == 45))
            if mask_else.any():
                raise RuntimeError("TTLJ decay_mode has unexpected value (not 2,4,45)")

        else:
            # 최초 호출 시: 해당 (short_mc, year_tag, mode=None)에 대해 syst_list 전체 선로딩
            _warm_load_all_systs_for(corrections_root_dir, load_btag, short_mc, year_tag, None, syst_list)
            tbl = _get_table(corrections_root_dir, load_btag, short_mc, year_tag, None, syst_sel)
            sf_flat = tbl.eval_flat(pt_flat, eta_flat, flav_flat)

        # per-event 곱
        sf_nested = ak.unflatten(sf_flat, counts)
        ev_w = ak.prod(sf_nested, axis=1, mask_identity=True)
        ev_w = ak.fill_none(ev_w, 1.0)
        return np.asarray(ev_w, dtype=np.float32)

    return provider