import logging
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(r"D:\Code\FFTsize")
DATE_STR = "2026-01-23 105949"

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

def _parse_groups(bytes_list: List[int], logger: logging.Logger) -> Tuple[List[Tuple[int, int, int, int, str, str, str, str, str]], int, int]:
    rows: List[Tuple[int, int, int, int, str, str, str, str, str]] = []
    valid = 0
    discarded = 0
    i = 0
    n = len(bytes_list)
    while i < n:
        if i + 5 >= n:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：字节数量不足6个，剩余{n - i}字节")
            i += 1
            continue
        flag, ue, bs, rss0, rss1, rss2 = bytes_list[i:i+6]
        if flag not in (0xCC, 0x33):
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：FLAG无效，值={_to_hex(flag)}")
            i += 1
            continue
        if _top2(ue) != 0b01:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：UE_Beam位[7:6]不为01，值={_to_hex(ue)}")
            i += 1
            continue
        if not (bs == 0xFF or _top2(bs) == 0b00):
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：BS_Beam位模式错误，值={_to_hex(bs)}")
            i += 1
            continue
        if _top2(rss0) != 0b10:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：RSS0位[7:6]不为10，值={_to_hex(rss0)}")
            i += 1
            continue
        if _top2(rss1) != 0b10:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：RSS1位[7:6]不为10，值={_to_hex(rss1)}")
            i += 1
            continue
        if _top2(rss2) != 0b10:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：RSS2位[7:6]不为10，值={_to_hex(rss2)}")
            i += 1
            continue
        flag_out = 1 if flag == 0xCC else 0
        ue_dec = _lsb6(ue)
        bs_dec = _lsb6(bs)
        low6 = _lsb6(rss0)
        mid6 = _lsb6(rss1)
        high6 = _lsb6(rss2)
        rss_val = (high6 << 12) | (mid6 << 6) | low6
        rows.append((
            flag_out,
            ue_dec,
            bs_dec,
            rss_val,
            _to_hex(ue),
            _to_hex(bs),
            _to_hex(rss0),
            _to_hex(rss1),
            _to_hex(rss2),
        ))
        valid += 1
        i += 6
    return rows, valid, discarded

def _ensure_openpyxl() -> None:
    try:
        import openpyxl  # noqa
    except Exception:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "openpyxl"])

def _write_excel(rows: List[Tuple[int, int, int, int, str, str, str, str, str]], out_path: Path) -> None:
    _ensure_openpyxl()
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Parsed"
    ws.append([
        "FLAG",
        "UE_Beam[5:0]十进制",
        "BS_Beam[5:0]十进制",
        "RSS十进制",
        "UE_Beam原始16进制值",
        "BS_Beam原始16进制值",
        "RSS0原始16进制值",
        "RSS1原始16进制值",
        "RSS2原始16进制值"
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

