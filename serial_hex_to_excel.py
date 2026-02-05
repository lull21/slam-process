import logging
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(r"D:\Code\SLAMPro\UE19\debugDoc")
DATE_STR = "2026-01-20 094112"

def _to_hex(v: int) -> str:
    return f"0x{v:02X}"

def _top2(v: int) -> int:
    return (v >> 6) & 0x3

def _lsb6(v: int) -> int:
    return v & 0x3F

def _read_bytes(path: Path, logger: logging.Logger) -> List[int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = text.split()
    result: List[int] = []
    for idx, t in enumerate(tokens):
        s = t.strip()
        if not s:
            continue
        if s.lower().startswith("0x"):
            s = s[2:]
        try:
            v = int(s, 16) & 0xFF
            result.append(v)
        except ValueError:
            logger.error(f"无效字节，位置token[{idx}]='{t}'，已跳过")
    return result

def _parse_groups(bytes_list: List[int], logger: logging.Logger) -> Tuple[List[Tuple[str, int, str, int, str, str, str, int]], int, int]:
    rows: List[Tuple[str, int, str, int, str, str, str, int]] = []
    valid = 0
    discarded = 0
    i = 0
    n = len(bytes_list)
    while i < n:
        if i + 4 >= n:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：字节数量不足5个，剩余{n - i}字节")
            i += 1
            continue
        g0, g1, g2, g3, g4 = bytes_list[i:i+5]
        if _top2(g0) != 0b01:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：UE_Beam位[7:6]不为01，值={_to_hex(g0)}")
            i += 1
            continue
        if _top2(g1) not in (0b00, 0b11):
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：BS_Beam位[7:6]不为00或11，值={_to_hex(g1)}")
            i += 1
            continue
        if _top2(g2) != 0b10:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：RSS0位[7:6]不为10，值={_to_hex(g2)}")
            i += 1
            continue
        if _top2(g3) != 0b10:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：RSS1位[7:6]不为10，值={_to_hex(g3)}")
            i += 1
            continue
        if _top2(g4) != 0b10:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：RSS2位[7:6]不为10，值={_to_hex(g4)}")
            i += 1
            continue
        ue_hex = _to_hex(g0)
        ue_dec = _lsb6(g0)
        bs_hex = _to_hex(g1)
        bs_dec = 65 if _top2(g1) == 0b11 else _lsb6(g1)
        rss0_hex = _to_hex(g2)
        rss1_hex = _to_hex(g3)
        rss2_hex = _to_hex(g4)
        low6 = _lsb6(g2)
        mid6 = _lsb6(g3)
        high6 = _lsb6(g4)
        rss_val = (high6 << 12) | (mid6 << 6) | low6
        rows.append((ue_hex, ue_dec, bs_hex, bs_dec, rss0_hex, rss1_hex, rss2_hex, rss_val))
        valid += 1
        i += 5
    return rows, valid, discarded

def _ensure_openpyxl() -> None:
    try:
        import openpyxl  # noqa
    except Exception:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "openpyxl"])

def _write_excel(rows: List[Tuple[str, int, str, int, str, str, str, int]], out_path: Path) -> None:
    _ensure_openpyxl()
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Parsed"
    ws.append([
        "UE_Beam原始16进制值",
        "UE_Beam[5:0]十进制",
        "BS_Beam原始16进制值",
        "BS_Beam[5:0]十进制",
        "RSS0原始16进制值",
        "RSS1原始16进制值",
        "RSS2原始16进制值",
        "RSS十进制"
    ])
    for r in rows:
        ws.append(list(r))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(out_path.as_posix())
    except PermissionError:
        alt = out_path.with_name(out_path.stem + "_out.xlsx")
        wb.save(alt.as_posix())

def main() -> None:
    in_path = BASE_DIR / f"Serial Debug {DATE_STR}.txt"
    out_path = BASE_DIR / f"Serial Debug {DATE_STR}.xlsx"
    if not in_path.exists():
        print(f"输入文件不存在: {in_path}")
        return
    log_path = in_path.with_suffix(".log")
    logger = logging.getLogger("serial_hex_to_excel")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path.as_posix(), mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers.clear()
    logger.addHandler(file_handler)
    bytes_list = _read_bytes(in_path, logger)
    rows, valid, discarded = _parse_groups(bytes_list, logger)
    _write_excel(rows, out_path)
    summary = f"有效组数={valid} 丢弃组数={discarded} 日志={log_path}"
    logger.info(summary)
    print(summary)

if __name__ == "__main__":
    main()

