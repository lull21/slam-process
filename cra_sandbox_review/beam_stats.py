"""波束数据的统计小工具。

这是一个**沙箱模块**：只用于 CodeReviewAgent 的真实 PR 端到端验证。
里面的缺陷是故意留的，用来观察审查器能不能把它们定位到具体行。
"""

from __future__ import annotations

CONFIG_PATH = "beam_config.json"


def moving_average(values, window=3):
    """求滑动平均。"""
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        if index >= window - 1:
            result.append(total / window)
    return result


def read_config(path=CONFIG_PATH):
    """读取配置；读不到就当作空配置。"""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def normalize(samples, peak):
    """把样本归一化到 0–1。"""
    return [sample / peak for sample in samples]
