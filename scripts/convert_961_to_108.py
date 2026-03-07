#!/usr/bin/env python3
"""本脚本用于将961版问卷字段映射为108列标准结构并记录转换审计。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from qp_io import write_dict_csv


def _norm(s: str) -> str:
    s = str(s or "").strip()
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", "", s)
    return s


def _is_other(s: str) -> bool:
    return _norm(s).startswith("其他")


def _split_multi(raw: str) -> List[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    out = []
    for x in raw.split("┋"):
        t = x.strip()
        if t:
            out.append(t)
    return out


def _build_alias(keys: Iterable[str]) -> Dict[str, str]:
    out = {}
    for k in keys:
        out[_norm(k)] = k
    return out


Q9_OPTIONS = [
    "社交媒体（抖音、小红书、微信、微博等）",
    "旅游平台（携程、美团等）",
    "亲友/同事推荐",
    "线下门店/街区路过发现",
    "新闻/公众号推文",
    "电视/广播",
    "其他",
]

Q10_OPTIONS = [
    "了解中医药文化、学习相关知识",
    "体验中医药非遗项目",
    "打卡休闲、品尝特色美食 / 养生茶饮",
    "购买中医药文创、保健品及药食同源产品",
    "慕名而来（叶开泰品牌吸引力）",
    "体验中医健康调理服务",
    "陪同家人 / 朋友",
    "其他",
]

Q13_OPTIONS = [
    "中医药诊疗/理疗服务",
    "中医药文化参观/研学",
    "药膳/养生餐体验",
    "药食同源产品购买（如黑芝麻丸、养生茶饮等）",
    "中药材/文创产品购买",
    "主题活动/课程参与（如八段锦、太极拳、香囊制作等）",
    "其他",
]

Q14_OPTIONS = [
    "无（未遇到任何问题）",
    "非遗体验项目排队时间过长、体验感不佳",
    "工作人员专业度不足（不熟悉中医药知识、服务态度一般）",
    "配套设施不完善（休息区少、卫生间不便、停车位不足等）",
    "中医药文化展示/讲解不够详细、缺乏吸引力",
    "美食/文创/保健品价格偏高、品质不符预期",
    "街区环境整洁度不足、卫生状况有待改善",
    "指示标识不清晰（找不到景点、体验区等）",
    "交通不便",
    "其他",
]

Q15_OPTIONS = [
    "不够了解街区特色 / 首次听说",
    "对中医药文化、非遗项目兴趣不大",
    "距离较远、交通不便",
    "没有合适的时间",
    "认为街区缺乏吸引力",
    "消费预算考虑",
    "已有替代目的地",
    "担心体验感不佳（如项目单一、服务差等）",
    "其他",
]

Q22_OPTIONS = [
    "中医药主题沉浸式研学营",
    "时令节气养生体验活动",
    "非遗中药炮制技艺体验课",
    "药膳食疗定制服务",
    "中医药健康管理私人顾问",
    "中医药科普互动课堂（如药材辨识、养生讲座）",
    "传统功法体验（如八段锦、太极拳深度学习）",
    "中医药主题文化节、展演活动",
    "其他",
]

Q23_OPTIONS = [
    "产品/服务折扣券",
    "餐饮套餐优惠",
    "会员积分 / 储值福利",
    "节假日限定活动",
    "联名周边 / 伴手礼赠送",
    "团购 / 家庭套票",
    "其他",
]

Q9_ALIAS = _build_alias(Q9_OPTIONS)
Q10_ALIAS = _build_alias(Q10_OPTIONS)
Q13_ALIAS = _build_alias(Q13_OPTIONS)
Q14_ALIAS = _build_alias(Q14_OPTIONS)
Q15_ALIAS = _build_alias(Q15_OPTIONS)
Q22_ALIAS = _build_alias(Q22_OPTIONS)
Q23_ALIAS = _build_alias(Q23_OPTIONS)


SINGLE_MAPS = {
    7: {"男": 1, "女": 2},
    8: {"18岁以下": 1, "26-45岁": 2, "18-25岁": 3, "46-64岁": 4, "65岁及以上": 5},
    9: {"初中及以下": 1, "中专/高中": 2, "大专": 3, "本科": 4, "硕士及以上": 5},
    10: {
        "学生": 1,
        "企业/公司职员": 2,
        "公务员/事业单位人员": 3,
        "事业单位人员/公务员": 3,
        "自由职业者": 4,
        "个体经营者": 5,
        "服务业从业者": 6,
        "离退休人员": 7,
        "其他": 8,
    },
    11: {"3000元以下": 1, "3001-5000元": 2, "5001-8000元": 3, "8001-15000元": 4, "15000元以上": 5},
    12: {"经常（每月≥3次）": 1, "偶尔（每月1-2次）": 2, "很少（每年几次）": 3, "从不": 4},
    13: {"非常了解": 1, "比较了解": 2, "一般": 3, "不太了解": 4, "完全不了解": 5},
    14: {"是": 1, "否": 2},
    17: {"1小时以内": 1, "1-2小时": 2, "2-3小时": 3, "3小时以上": 4},
    18: {"100元以内": 1, "101-300元": 2, "301-500元": 3, "501-1000元": 4, "1000元以上": 5},
    60: {"非常不愿意": 1, "不愿意": 2, "一般": 3, "愿意": 4, "非常愿意": 5},
    61: {"一定不会": 1, "可能不会": 2, "不确定": 3, "可能会": 4, "一定会": 5},
}

LIKERT_AGREE = {"完全不同意": 1, "不同意": 2, "不一定": 3, "同意": 4, "完全同意": 5}
LIKERT_IMPORTANCE = {"非常不重要": 1, "不重要": 2, "一般": 3, "重要": 4, "非常重要": 5}
LIKERT_SATISFACTION = {"很不满意": 1, "不满意": 2, "一般": 3, "满意": 4, "很满意": 5}
LIKERT_COGNITION = {"非常不认同": 1, "不认同": 2, "一般": 3, "认同": 4, "非常认同": 5}


TARGET_HEADERS_108 = [
    "1. 您的性别是?（单选）",
    "2. 您的年龄段是?（单选）",
    "3. 您的教育程度是?（单选）",
    "4. 您的职业是?（单选）",
    "5. 您的月收入范围是?（单选）",
    "6. 您平时是否有购买中医药保健品、养生茶饮或接受中医调理的习惯?（单选）",
    "7. 您对“中医药+文旅”融合模式的了解程度是?（单选）",
    "8. 您是否听说或到访过武汉“叶开泰中医药文化街区” ?（单选）",
    *Q9_OPTIONS,
    *Q10_OPTIONS,
    "11. 您在街区的停留时长大约是?（单选）",
    "12. 您本次在街区的消费金额大约是?（单选）",
    *Q13_OPTIONS,
    *Q14_OPTIONS,
    *Q15_OPTIONS,
    "① 中医药文化的专业性与体验深度（艾灸、推拿等）",
    "② 非遗项目的独特性与参与感（如中医养生锤、草本饮品的学习与制作等）",
    "③ 非遗传承人互动体验机会",
    "①体验特色药膳 / 药食同源产品",
    "②产品丰富度与独特性",
    "③消费价格合理性",
    "① 交通便利性与停车便利",
    "② 环境舒适度与文化氛围",
    "③ 配套设施完善度（休息区、卫生间等）",
    "④ 周边配套（购物、其他景点联动）",
    "①宣传推广的真实性（无夸大宣传）",
    "②品牌知名度与口碑",
    "③该题请选择“完全不同意”",
    "④小红书/抖音等线上宣传种草",
    "①丰富的中医药文化展示和非遗体验项目",
    "②环境舒适度与卫生状况",
    "③便捷的交通、充足的停车位",
    "④亲友推荐/正面评价多",
    "⑤特色美食、文创产品的种类及品质",
    "⑥提供个性化中医体质辨识、养生咨询等服务",
    "⑦服务专业度与态度",
    "⑧产品价格与优惠力度",
    "⑨中医药服务专业度",
    "⑩线上线下宣传推广",
    "①丰富的中医药文化展示和非遗体验项目",
    "②环境舒适度与卫生状况",
    "③便捷的交通、充足的停车位",
    "④亲友推荐/正面评价多",
    "⑤特色美食、文创产品的种类及品质",
    "⑥提供个性化中医体质辨识、养生咨询等服务",
    "⑦服务专业度与态度",
    "⑧产品价格与优惠力度",
    "⑨中医药服务专业度",
    "⑩线上线下宣传推广",
    "①我对中医药文化（尤其是叶开泰文化街区所传承与展现的文化内涵）非常感兴趣",
    "②在游览叶开泰文化街区前，我对“中医药+文旅”模式非常了解",
    "③游览叶开泰文化街区可以提升我对“中医药+文旅”模式的理解",
    "④我非常愿意学习和了解中医药文化的相关知识与技艺",
    "20. 若有机会，您是否愿意前往叶开泰中医药文化街区游览?（单选）",
    "21. 您是否会向亲友推荐叶开泰中医药文化街区?（单选）",
    *Q22_OPTIONS,
    *Q23_OPTIONS,
    "24. 关于叶开泰“药食同源”产品（如黑芝麻丸、养生茶饮、膏方等），您有哪些看法和建议?（如口感、包装、价格、购买意愿等）",
]


@dataclass
class ConversionResult:
    headers_108: List[str]
    rows_108: List[List[str]]
    unknown_rows: List[dict]
    audit: dict


def _map_single(
    out_row: List[str],
    row_idx_1b: int,
    src_col_1b: int,
    src_header: str,
    src_val: str,
    dst_col_1b: int,
    mapping: Dict[str, int],
    unknown_rows: List[dict],
) -> None:
    raw = str(src_val or "").strip()
    if not raw:
        return
    key = _norm(raw)
    matched = None
    for k, v in mapping.items():
        if _norm(k) == key:
            matched = v
            break
    if matched is None and _is_other(raw):
        for k, v in mapping.items():
            if _norm(k) == _norm("其他"):
                matched = v
                break
    if matched is None:
        unknown_rows.append(
            {
                "row_id": row_idx_1b,
                "source_col_idx": src_col_1b,
                "source_header": src_header,
                "source_value": raw,
                "target_col_idx": dst_col_1b,
                "reason": "single_unmapped",
            }
        )
        return
    out_row[dst_col_1b - 1] = str(matched)


def _map_multi(
    out_row: List[str],
    row_idx_1b: int,
    src_col_1b: int,
    src_header: str,
    src_val: str,
    dst_start_1b: int,
    options: Sequence[str],
    alias_map: Dict[str, str],
    unknown_rows: List[dict],
) -> int:
    other_text_count = 0
    for j in range(len(options)):
        out_row[dst_start_1b - 1 + j] = "0"
    tokens = _split_multi(src_val)
    if not tokens:
        return other_text_count
    selected = set()
    for tok in tokens:
        if _norm(tok) == _norm("(跳过)"):
            continue
        if _is_other(tok):
            other_col = dst_start_1b + len(options) - 1
            selected.add(other_col)
            text = tok.strip()
            if _norm(text) != _norm("其他"):
                out_row[other_col - 1] = f"1^{text}"
                other_text_count += 1
            else:
                out_row[other_col - 1] = "1"
            continue
        key = _norm(tok)
        if key not in alias_map:
            unknown_rows.append(
                {
                    "row_id": row_idx_1b,
                    "source_col_idx": src_col_1b,
                    "source_header": src_header,
                    "source_value": tok,
                    "target_col_idx": f"{dst_start_1b}-{dst_start_1b+len(options)-1}",
                    "reason": "multi_unmapped",
                }
            )
            continue
        label = alias_map[key]
        idx = options.index(label)
        selected.add(dst_start_1b + idx)
    for c in selected:
        if not out_row[c - 1].startswith("1^"):
            out_row[c - 1] = "1"
    return other_text_count


def convert_961_to_108(headers_64: Sequence[str], rows_64: Sequence[Sequence[str]]) -> ConversionResult:
    if len(headers_64) < 64:
        raise ValueError(f"Expected >=64 columns in new questionnaire, got {len(headers_64)}")

    rows_out: List[List[str]] = []
    unknown_rows: List[dict] = []
    other_text_count = 0
    branch_conflict_count = 0

    for i, row in enumerate(rows_64, start=1):
        r = list(row) + [""] * (64 - len(row))
        out = [""] * 108

        # Q1-Q8 -> C001-C008
        _map_single(out, i, 7, headers_64[6], r[6], 1, SINGLE_MAPS[7], unknown_rows)
        _map_single(out, i, 8, headers_64[7], r[7], 2, SINGLE_MAPS[8], unknown_rows)
        _map_single(out, i, 9, headers_64[8], r[8], 3, SINGLE_MAPS[9], unknown_rows)
        _map_single(out, i, 10, headers_64[9], r[9], 4, SINGLE_MAPS[10], unknown_rows)
        _map_single(out, i, 11, headers_64[10], r[10], 5, SINGLE_MAPS[11], unknown_rows)
        _map_single(out, i, 12, headers_64[11], r[11], 6, SINGLE_MAPS[12], unknown_rows)
        _map_single(out, i, 13, headers_64[12], r[12], 7, SINGLE_MAPS[13], unknown_rows)
        _map_single(out, i, 14, headers_64[13], r[13], 8, SINGLE_MAPS[14], unknown_rows)

        # Q9/Q10/Q13
        other_text_count += _map_multi(out, i, 15, headers_64[14], r[14], 9, Q9_OPTIONS, Q9_ALIAS, unknown_rows)
        other_text_count += _map_multi(out, i, 16, headers_64[15], r[15], 16, Q10_OPTIONS, Q10_ALIAS, unknown_rows)
        _map_single(out, i, 17, headers_64[16], r[16], 24, SINGLE_MAPS[17], unknown_rows)
        _map_single(out, i, 18, headers_64[17], r[17], 25, SINGLE_MAPS[18], unknown_rows)
        other_text_count += _map_multi(out, i, 19, headers_64[18], r[18], 26, Q13_OPTIONS, Q13_ALIAS, unknown_rows)

        # Branch by Q8
        q8 = out[7]
        q14_tokens = _split_multi(r[19])
        q15_tokens = _split_multi(r[20])
        q14_effective = [x for x in q14_tokens if _norm(x) != _norm("(跳过)")]
        q15_effective = [x for x in q15_tokens if _norm(x) != _norm("(跳过)")]

        if q8 == "1":
            if q15_effective:
                branch_conflict_count += 1
            other_text_count += _map_multi(out, i, 20, headers_64[19], r[19], 33, Q14_OPTIONS, Q14_ALIAS, unknown_rows)
            for c in range(43, 52):
                out[c - 1] = "-3"
        elif q8 == "2":
            if q14_effective:
                branch_conflict_count += 1
            for c in range(33, 43):
                out[c - 1] = "-3"
            other_text_count += _map_multi(out, i, 21, headers_64[20], r[20], 43, Q15_OPTIONS, Q15_ALIAS, unknown_rows)
        else:
            for c in range(33, 52):
                out[c - 1] = ""
            if q14_effective or q15_effective:
                branch_conflict_count += 1

        # 16-1..16-4 -> C052..C065 (agree scale)
        for src in range(22, 36):
            dst = 52 + (src - 22)
            _map_single(out, i, src, headers_64[src - 1], r[src - 1], dst, LIKERT_AGREE, unknown_rows)

        # 17 -> C066..C075 (importance)
        for src in range(36, 46):
            dst = 66 + (src - 36)
            _map_single(out, i, src, headers_64[src - 1], r[src - 1], dst, LIKERT_IMPORTANCE, unknown_rows)

        # 18 -> C076..C085 (satisfaction)
        for src in range(46, 56):
            dst = 76 + (src - 46)
            _map_single(out, i, src, headers_64[src - 1], r[src - 1], dst, LIKERT_SATISFACTION, unknown_rows)

        # 19 -> C086..C089 (cognition)
        for src in range(56, 60):
            dst = 86 + (src - 56)
            _map_single(out, i, src, headers_64[src - 1], r[src - 1], dst, LIKERT_COGNITION, unknown_rows)

        # Q20/Q21
        _map_single(out, i, 60, headers_64[59], r[59], 90, SINGLE_MAPS[60], unknown_rows)
        _map_single(out, i, 61, headers_64[60], r[60], 91, SINGLE_MAPS[61], unknown_rows)

        # Q22/Q23
        other_text_count += _map_multi(out, i, 62, headers_64[61], r[61], 92, Q22_OPTIONS, Q22_ALIAS, unknown_rows)
        other_text_count += _map_multi(out, i, 63, headers_64[62], r[62], 101, Q23_OPTIONS, Q23_ALIAS, unknown_rows)

        # Q24 open text
        open_text = str(r[63] or "").strip()
        out[107] = open_text if open_text else "无"
        rows_out.append(out)

    unknown_count = len(unknown_rows)
    n_rows = len(rows_out)
    denom = max(1, n_rows * 108)
    unknown_rate = unknown_count / denom
    branch_rate = branch_conflict_count / max(1, n_rows)
    conversion_integrity = max(0.0, min(1.0, 1.0 - unknown_rate - 0.5 * branch_rate))

    audit = {
        "n_rows": n_rows,
        "unknown_value_count": unknown_count,
        "unknown_value_rate": unknown_rate,
        "other_text_count": other_text_count,
        "branch_conflict_count": branch_conflict_count,
        "branch_conflict_rate": branch_rate,
        "conversion_integrity": conversion_integrity,
    }

    return ConversionResult(
        headers_108=list(TARGET_HEADERS_108),
        rows_108=rows_out,
        unknown_rows=unknown_rows,
        audit=audit,
    )


def export_conversion_artifacts(out_dir: Path, result: ConversionResult) -> None:
    out_dir = Path(out_dir)
    tdir = out_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(
        tdir / "unknown_value_log.csv",
        ["row_id", "source_col_idx", "source_header", "source_value", "target_col_idx", "reason"],
        result.unknown_rows,
    )
    (out_dir / "conversion_audit.json").write_text(json.dumps(result.audit, ensure_ascii=False, indent=2), encoding="utf-8")
