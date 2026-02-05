import argparse
import logging
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import unittest


FLAG_COL = "FLAG"
UE_BEAM_COL = "UE_Beam[5:0]十进制"
BS_BEAM_COL = "BS_Beam[5:0]十进制"
RSS_COL = "RSS十进制"
CLK_COL = "CLK十进制"
OUT_COL = "Corrected_BS_Beam"
OUT_UE = "UE_Beam"
OUT_BS = "BS_Beam"
OUT_RSS = "RSS值"
OUT_CLK = "CLK值"

CYCLE = 61000
TOL = 500
MOD_BASE = 64


def _validate_columns(df: pd.DataFrame) -> None:
    required = [FLAG_COL, UE_BEAM_COL, BS_BEAM_COL, RSS_COL, CLK_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺失列: {missing}")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in [FLAG_COL, UE_BEAM_COL, BS_BEAM_COL, RSS_COL, CLK_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[[FLAG_COL, UE_BEAM_COL, BS_BEAM_COL, RSS_COL, CLK_COL]].isna().any().any():
        logging.warning("存在无法转换为数值的单元格，已填充为NaN")
    return df


def _detect_groups(df: pd.DataFrame) -> pd.Series:
    s = df[UE_BEAM_COL]
    boundary = (s.shift(1) > s) | s.shift(1).isna()
    gid = boundary.cumsum() - 1
    return gid.astype(int)


def _identify_baselines(group: pd.DataFrame) -> List[Tuple[int, int]]:
    prev_flag = group[FLAG_COL].shift(1)
    prev_rss = group[RSS_COL].shift(1)
    mask = (group[FLAG_COL] == 1) & (prev_flag == 0) & (group[RSS_COL] == prev_rss)
    idx = group.index[mask].tolist()
    baselines: List[Tuple[int, int]] = []
    for i in idx:
        prev_i = group.index[group.index.get_loc(i) - 1]
        clk_b = int(group.at[prev_i, CLK_COL])
        bs_b = int(group.at[i, BS_BEAM_COL])
        baselines.append((clk_b, bs_b))
    return baselines


def _choose_correction(clk_diff: int, baselines: List[Tuple[int, int]]) -> Optional[int]:
    best = None
    best_resid = None
    for clk_b, bs_b in baselines:
        d = clk_diff
        k = int(round(d / CYCLE))
        resid = abs(d - k * CYCLE)
        if resid <= TOL:
            corrected = (bs_b + k) % MOD_BASE
            if best is None or resid < best_resid:
                best = corrected
                best_resid = resid
    return best


def _correct_group(group: pd.DataFrame, baselines: List[Tuple[int, int]]) -> pd.DataFrame:
    out = group.copy()
    out[OUT_COL] = np.nan
    corrected_count = 0
    for i, row in out.iterrows():
        flag = int(row[FLAG_COL]) if not pd.isna(row[FLAG_COL]) else 0
        if flag == 1:
            out.at[i, OUT_COL] = int(row[BS_BEAM_COL])
            continue
        if not baselines:
            out.at[i, OUT_COL] = int(row[BS_BEAM_COL])
            continue
        clk = int(row[CLK_COL]) if not pd.isna(row[CLK_COL]) else None
        if clk is None:
            out.at[i, OUT_COL] = int(row[BS_BEAM_COL])
            continue
        candidates = []
        for clk_b, bs_b in baselines:
            d = clk - clk_b
            k = int(round(d / CYCLE))
            resid = abs(d - k * CYCLE)
            if resid <= TOL:
                candidates.append((resid, (bs_b + k) % MOD_BASE))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            out.at[i, OUT_COL] = int(candidates[0][1])
            corrected_count += 1
        else:
            out.at[i, OUT_COL] = int(row[BS_BEAM_COL])
    logging.info(
        "组索引范围[%s-%s] 基准数量=%d 修正数量=%d 总行数=%d",
        str(group.index.min()),
        str(group.index.max()),
        len(baselines),
        corrected_count,
        len(group),
    )
    return out


def process_excel(input_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")
    df = pd.read_excel(input_path, engine="openpyxl")
    _validate_columns(df)
    df = _coerce_types(df)
    gid = _detect_groups(df)
    df["_gid"] = gid
    pieces = []
    for g, sub in df.groupby("_gid", sort=True):
        baselines = _identify_baselines(sub)
        if not baselines:
            logging.warning("组%d 未识别到基准值", g)
        corrected = _correct_group(sub, baselines)
        pieces.append(corrected)
    res = pd.concat(pieces).sort_index()
    res = res.drop(columns=["_gid"])
    with pd.ExcelWriter(input_path, engine="openpyxl", mode="w") as writer:
        res.to_excel(writer, index=False)
    logging.info("已写回修正文件: %s", input_path)

def _filter_group_corrected(group: pd.DataFrame, baselines: List[Tuple[int, int]]) -> pd.DataFrame:
    rows = []
    corrected_count = 0
    for i, row in group.iterrows():
        flag = int(row[FLAG_COL]) if not pd.isna(row[FLAG_COL]) else 0
        if flag == 1:
            continue
        if not baselines:
            continue
        clk = int(row[CLK_COL]) if not pd.isna(row[CLK_COL]) else None
        if clk is None:
            continue
        candidates = []
        for clk_b, bs_b in baselines:
            d = clk - clk_b
            k = int(round(d / CYCLE))
            resid = abs(d - k * CYCLE)
            if resid <= TOL:
                candidates.append((resid, (bs_b + k) % MOD_BASE))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            corrected_bs = int(candidates[0][1])
            rows.append(
                {
                    OUT_UE: int(row[UE_BEAM_COL]),
                    OUT_BS: corrected_bs,
                    OUT_RSS: int(row[RSS_COL]),
                    OUT_CLK: int(row[CLK_COL]),
                }
            )
            corrected_count += 1
    logging.info(
        "过滤组索引范围[%s-%s] 基准数量=%d 输出修正数量=%d",
        str(group.index.min()),
        str(group.index.max()),
        len(baselines),
        corrected_count,
    )
    return pd.DataFrame(rows, columns=[OUT_UE, OUT_BS, OUT_RSS, OUT_CLK])

def process_excel_filtered(input_path: str, output_path: Optional[str] = None) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")
    df = pd.read_excel(input_path, engine="openpyxl")
    _validate_columns(df)
    df = _coerce_types(df)
    gid = _detect_groups(df)
    df["_gid"] = gid
    pieces = []
    for g, sub in df.groupby("_gid", sort=True):
        baselines = _identify_baselines(sub)
        if not baselines:
            logging.warning("组%d 未识别到基准值，已跳过组内所有预测行", g)
        filtered = _filter_group_corrected(sub, baselines)
        if not filtered.empty:
            pieces.append(filtered)
    if not pieces:
        raise ValueError("未产生任何可修正数据行，输出为空")
    res = pd.concat(pieces, ignore_index=True)
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_filtered.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        res.to_excel(writer, index=False)
    logging.info("已生成过滤后的修正文件: %s 行数=%d", output_path, len(res))
    return output_path

class TestBSBeamCorrection(unittest.TestCase):
    def _make_group(self):
        rows = []
        clk0 = 1_000_000
        ue_seq = list(range(0, 64))
        rss = 42
        rows.append({FLAG_COL: 0, UE_BEAM_COL: ue_seq[0], BS_BEAM_COL: 10, RSS_COL: rss, CLK_COL: clk0})
        rows.append({FLAG_COL: 1, UE_BEAM_COL: ue_seq[1], BS_BEAM_COL: 12, RSS_COL: rss, CLK_COL: clk0 + 100})
        rows.append({FLAG_COL: 0, UE_BEAM_COL: ue_seq[2], BS_BEAM_COL: 99, RSS_COL: rss, CLK_COL: clk0 + CYCLE + 50})
        rows.append({FLAG_COL: 0, UE_BEAM_COL: ue_seq[3], BS_BEAM_COL: 99, RSS_COL: rss, CLK_COL: clk0 + 2 * CYCLE - 480})
        rows.append({FLAG_COL: 0, UE_BEAM_COL: ue_seq[4], BS_BEAM_COL: 99, RSS_COL: rss, CLK_COL: clk0 + 3 * CYCLE + 600})
        rows.append({FLAG_COL: 0, UE_BEAM_COL: ue_seq[5], BS_BEAM_COL: 99, RSS_COL: rss, CLK_COL: clk0 - CYCLE + 100})
        df = pd.DataFrame(rows)
        return df

    def test_baseline_identification(self):
        df = self._make_group()
        baselines = _identify_baselines(df)
        self.assertEqual(len(baselines), 1)
        clk_b, bs_b = baselines[0]
        self.assertEqual(clk_b, df.iloc[0][CLK_COL])
        self.assertEqual(bs_b, df.iloc[1][BS_BEAM_COL])

    def test_correction_logic(self):
        df = self._make_group()
        baselines = _identify_baselines(df)
        out = _correct_group(df, baselines)
        self.assertEqual(int(out.iloc[1][OUT_COL]), int(df.iloc[1][BS_BEAM_COL]))
        self.assertEqual(int(out.iloc[2][OUT_COL]), (12 + 1) % MOD_BASE)
        self.assertEqual(int(out.iloc[3][OUT_COL]), (12 + 2) % MOD_BASE)

    def test_boundary_tolerance(self):
        clk0 = 5_000_000
        rss = 7
        df = pd.DataFrame(
            [
                {FLAG_COL: 0, UE_BEAM_COL: 0, BS_BEAM_COL: 3, RSS_COL: rss, CLK_COL: clk0},
                {FLAG_COL: 1, UE_BEAM_COL: 1, BS_BEAM_COL: 8, RSS_COL: rss, CLK_COL: clk0 + 10},
                {FLAG_COL: 0, UE_BEAM_COL: 2, BS_BEAM_COL: 0, RSS_COL: rss, CLK_COL: clk0 + CYCLE + TOL},
                {FLAG_COL: 0, UE_BEAM_COL: 3, BS_BEAM_COL: 0, RSS_COL: rss, CLK_COL: clk0 + CYCLE + TOL + 1},
            ]
        )
        baselines = _identify_baselines(df)
        out = _correct_group(df, baselines)
        self.assertEqual(int(out.iloc[2][OUT_COL]), (8 + 1) % MOD_BASE)
        self.assertEqual(int(out.iloc[3][OUT_COL]), int(df.iloc[3][BS_BEAM_COL]))

    def test_negative_diff(self):
        clk0 = 7_000_000
        rss = 13
        df = pd.DataFrame(
            [
                {FLAG_COL: 0, UE_BEAM_COL: 0, BS_BEAM_COL: 60, RSS_COL: rss, CLK_COL: clk0},
                {FLAG_COL: 1, UE_BEAM_COL: 1, BS_BEAM_COL: 5, RSS_COL: rss, CLK_COL: clk0 + 1},
                {FLAG_COL: 0, UE_BEAM_COL: 2, BS_BEAM_COL: 0, RSS_COL: rss, CLK_COL: clk0 - CYCLE + 10},
            ]
        )
        baselines = _identify_baselines(df)
        out = _correct_group(df, baselines)
        self.assertEqual(int(out.iloc[2][OUT_COL]), (5 - 1) % MOD_BASE)

    def test_filter_only_corrected_rows(self):
        clk0 = 2_000_000
        rss = 21
        df = pd.DataFrame(
            [
                {FLAG_COL: 0, UE_BEAM_COL: 0, BS_BEAM_COL: 10, RSS_COL: rss, CLK_COL: clk0},
                {FLAG_COL: 1, UE_BEAM_COL: 1, BS_BEAM_COL: 12, RSS_COL: rss, CLK_COL: clk0 + 50},
                {FLAG_COL: 0, UE_BEAM_COL: 2, BS_BEAM_COL: 99, RSS_COL: rss, CLK_COL: clk0 + CYCLE + 20},
                {FLAG_COL: 0, UE_BEAM_COL: 3, BS_BEAM_COL: 99, RSS_COL: rss, CLK_COL: clk0 + CYCLE + TOL + 10},
            ]
        )
        baselines = _identify_baselines(df)
        filtered = _filter_group_corrected(df, baselines)
        self.assertEqual(list(filtered.columns), [OUT_UE, OUT_BS, OUT_RSS, OUT_CLK])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(int(filtered.iloc[0][OUT_BS]), (12 + 1) % MOD_BASE)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=r"D:\Code\SLAMPro\UE21\debugDoc\Serial Debug 2026-01-27 115303.xlsx")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--run-tests", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBSBeamCorrection)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    try:
        process_excel_filtered(args.input, args.output)
    except Exception as e:
        logging.exception("处理失败: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
