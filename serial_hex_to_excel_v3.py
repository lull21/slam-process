import logging
from pathlib import Path
import re
import sys
from typing import List, Tuple

def _to_hex(v: int) -> str:
    return f"0x{v:02X}"

def _top2(v: int) -> int:
    return (v >> 6) & 0x3

def _lsb6(v: int) -> int:
    return v & 0x3F

_HEX_BYTE_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{2}$")

def _read_bytes(path: Path, logger: logging.Logger) -> List[int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = text.split()
    result: List[int] = []
    for idx, t in enumerate(tokens):
        s = t.strip()
        if not s:
            continue
        if not _HEX_BYTE_RE.fullmatch(s):
            logger.debug(f"跳过非16进制单字节token[{idx}]='{t}'")
            continue
        if s.lower().startswith("0x"):
            s = s[2:]
        try:
            v = int(s, 16) & 0xFF
            result.append(v)
        except ValueError:
            logger.debug(f"无效字节，位置token[{idx}]='{t}'，已跳过")
    return result

def _fmt_bytes(seq: List[int]) -> str:
    return " ".join(_to_hex(v) for v in seq)

def _parse_groups(bytes_list: List[int], logger: logging.Logger) -> Tuple[List[Tuple[int, int, int, int, int]], int, int]:
    rows: List[Tuple[int, int, int, int, int]] = []
    valid = 0
    discarded = 0
    i = 0
    n = len(bytes_list)
    while i < n:
        flag = bytes_list[i]
        if flag not in (0xCC, 0x33):
            i += 1
            continue

        if i + 11 > n:
            discarded += 1
            logger.error(
                f"丢弃组，起始索引{i}，原因：字节数量不足11个，剩余{n - i}字节，数据={_fmt_bytes(bytes_list[i:n])}"
            )
            break

        ue, bs, clk0, clk1, clk2, clk3, clk4, rss0, rss1, rss2 = bytes_list[i + 1 : i + 11]

        if _top2(ue) != 0b00:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：UE_Beam位[7:6]不为00，UE={_to_hex(ue)}，数据={_fmt_bytes(bytes_list[i:i+11])}")
            i += 1
            continue
        if _top2(bs) != 0b11:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：BS_Beam位[7:6]不为11，BS={_to_hex(bs)}，数据={_fmt_bytes(bytes_list[i:i+11])}")
            i += 1
            continue

        clk_bytes = [clk0, clk1, clk2, clk3, clk4]
        bad_clk = next((b for b in clk_bytes if _top2(b) != 0b01), None)
        if bad_clk is not None:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：CLK字节位[7:6]不为01，BAD={_to_hex(bad_clk)}，数据={_fmt_bytes(bytes_list[i:i+11])}")
            i += 1
            continue

        rss_bytes = [rss0, rss1, rss2]
        bad_rss = next((b for b in rss_bytes if _top2(b) != 0b10), None)
        if bad_rss is not None:
            discarded += 1
            logger.error(f"丢弃组，起始索引{i}，原因：RSS字节位[7:6]不为10，BAD={_to_hex(bad_rss)}，数据={_fmt_bytes(bytes_list[i:i+11])}")
            i += 1
            continue

        flag_out = 1 if flag == 0xCC else 0
        ue_dec = _lsb6(ue)
        bs_dec = _lsb6(bs)

        clk_val = 0
        for idx, b in enumerate(clk_bytes):
            clk_val |= (_lsb6(b) << (6 * idx))

        rss_val = (_lsb6(rss0) << 0) | (_lsb6(rss1) << 6) | (_lsb6(rss2) << 12)

        rows.append((flag_out, ue_dec, bs_dec, rss_val, clk_val))
        valid += 1
        i += 11

    return rows, valid, discarded

def _write_excel(rows: List[Tuple[int, int, int, int, int]], out_path: Path) -> None:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Parsed"
    ws.append([
        "FLAG",
        "UE_Beam[5:0]十进制",
        "BS_Beam[5:0]十进制",
        "RSS十进制",
        "CLK十进制",
    ])
    for r in rows:
        ws.append(list(r))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))

def main() -> None:
    default_in = Path(r"D:\Code\SLAMPro\UE21\debugDoc\Serial Debug 2026-01-27 115303.txt")
    default_out = Path(r"D:\Code\SLAMPro\UE21\debugDoc\Serial Debug 2026-01-27 115303.xlsx")

    in_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else default_in
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else default_out

    if not in_path.exists():
        print(f"输入文件不存在: {in_path}")
        return

    log_path = out_path.with_suffix(".log")
    logger = logging.getLogger("serial_hex_to_excel")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path.as_posix(), mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    stream_handler.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    try:
        import openpyxl  # noqa: F401
    except Exception as e:
        logger.error(f"缺少openpyxl，无法生成Excel：{e}")
        print("缺少openpyxl，请先执行：pip install openpyxl")
        return

    bytes_list = _read_bytes(in_path, logger)
    rows, valid, discarded = _parse_groups(bytes_list, logger)
    _write_excel(rows, out_path)
    summary = f"有效组数={valid} 丢弃组数={discarded} 输出={out_path} 日志={log_path}"
    logger.info(summary)
    print(summary)

if __name__ == "__main__":
    main()

