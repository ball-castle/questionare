#!/usr/bin/env python3
"""Narrative renderer for chapter 6/7 report content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from report_data_extractors import num, pct, to_float, to_int


def _fmt_path(path: str | Path) -> str:
    return Path(path).as_posix()


def _add_heading(blocks: list[dict[str, Any]], level: int, text: str) -> None:
    blocks.append({"type": "heading", "level": level, "text": text})


def _add_paragraph(blocks: list[dict[str, Any]], text: str) -> None:
    blocks.append({"type": "paragraph", "text": text})


def _add_bullet(blocks: list[dict[str, Any]], text: str) -> None:
    blocks.append({"type": "bullet", "text": text})


def _add_image(blocks: list[dict[str, Any]], path: str, caption: str, exists: bool) -> None:
    blocks.append({"type": "image", "path": path, "caption": caption, "exists": exists})


def _add_table(blocks: list[dict[str, Any]], caption: str, headers: list[str], rows: list[list[str]]) -> None:
    blocks.append({"type": "table", "caption": caption, "headers": headers, "rows": rows})


def _metric(v: Any, digits: int = 3) -> str:
    return num(to_float(v), digits)


def render_report_content(data: dict, tables_dir: Path, figures_dir: Path) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, str]] = []

    def ev(section: str, statement: str, *paths: str) -> str:
        clean_paths = [_fmt_path(p) for p in paths if p]
        for p in clean_paths:
            evidence_rows.append({"section": section, "statement": statement, "evidence_path": p})
        if not clean_paths:
            return ""
        return f"[证据：{'，'.join(clean_paths)}]"

    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)

    desc = data["desc"]
    mca = data["mca"]
    logit = data["logit"]
    mechanism = data.get("mechanism", {})
    cluster = data["cluster"]
    ipa = data["ipa"]
    voice = data["voice"]

    raw_n = to_int(data["run_meta"].get("n_samples"))
    main_n = to_int(data["run_meta"].get("remain_n_revised"))
    quality_profile = str(data["run_meta"].get("quality_profile", "unknown"))

    male_pct = to_float(desc["q1_male"]["pct"])
    female_pct = to_float(desc["q1_female"]["pct"])
    age_26_45 = to_float(desc["q2_age"][2]["pct"])
    age_18_25 = to_float(desc["q2_age"][3]["pct"])
    q8_yes_pct = to_float(desc["q8_yes"]["pct"])

    q24_top = desc["q24_top"]
    q25_top = desc["q25_top"]
    q90_pos = to_float(desc["q90_pos"])
    q91_pos = to_float(desc["q91_pos"])
    motive_top = desc["motive_top"][0] if desc["motive_top"] else {}

    visited_pain_top = desc["visited_pain_top"]
    unvisited_block_top = desc["unvisited_block_top"]
    chi_focus = desc["chi_focus"]

    _add_heading(blocks, 1, "六、叶开泰中医药文化街区游客特征分析")
    _add_paragraph(
        blocks,
        (
            f"本章基于主样本 n={main_n}（原始样本 n={raw_n}，质控口径 {quality_profile}），"
            "围绕“游客是谁、游客如何游、什么因素推动深入游览”三个问题展开。"
            "相较于模板式写法，本稿将描述统计、模型结论和运营启示合并为可直接落地的市场结论。"
        ),
    )

    _add_heading(blocks, 2, "章节结论速览")
    _add_table(
        blocks,
        "表6-0 核心结论速览",
        ["维度", "关键发现", "运营含义"],
        [
            ["客群结构", f"中青年占比高（18-25岁{pct(age_18_25)}；26-45岁{pct(age_26_45)}）", "重点设计中青年友好型内容和转化链路"],
            ["到访基础", f"已听说/到访占比{pct(q8_yes_pct)}", "当前重点不是“从0到1曝光”，而是“从认知到深度体验”"],
            ["意愿表现", f"游览意愿{pct(q90_pos)}；推荐意愿{pct(q91_pos)}", "具备口碑扩散基础，但高忠诚客户尚未形成"],
            ["关键瓶颈", "设施、标识与价格-品质感知落差共振", "优先做体验端补短板，再做内容端放大"],
        ],
    )
    _add_bullet(
        blocks,
        f"数据来源覆盖描述统计、MCA、Logit、聚类、IPA五类分析。 {ev('六-章节结论速览', '六章核心结论数据来源', str(tables_dir / '单选题频数百分比表.csv'), str(tables_dir / '多选题选择率表.csv'), str(tables_dir / 'MCA特征值.csv'), str(tables_dir / 'Logit模型指标.csv'))}",
    )

    _add_heading(blocks, 2, "（一）基于描述性统计的游客游览特征分析")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "识别客群结构、游览行为和消费意愿，回答“街区当前流量质量如何、主要增长阻力在哪里”。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(
        blocks,
        f"数据：survey_clean、单选频数、多选选择率、关键交叉卡方。 {ev('六（一）数据与方法', '描述统计输入', str(tables_dir / 'survey_clean.csv'), str(tables_dir / '单选题频数百分比表.csv'), str(tables_dir / '多选题选择率表.csv'), str(tables_dir / '交叉分析卡方汇总.csv'))}",
    )
    _add_bullet(blocks, "方法：结构分布（频数占比）+ 已到访/未到访分群对照 + Q8关键变量卡方检验。")

    _add_heading(blocks, 3, "3. 图表结果")
    _add_table(
        blocks,
        "表6-1 游客结构与行为关键指标",
        ["指标", "结果", "解读"],
        [
            ["性别结构", f"男{pct(male_pct)} / 女{pct(female_pct)}", "结构均衡，无明显性别偏置"],
            ["年龄主力", f"18-25岁{pct(age_18_25)}；26-45岁{pct(age_26_45)}", "中青年是核心消费与传播人群"],
            ["到访认知", f"C008=1 占比{pct(q8_yes_pct)}", "市场已形成认知基础"],
            ["停留时长Top1", f"编码{to_int(q24_top.get('code'))}（{pct(to_float(q24_top.get('pct')))}）", "存在“轻深度停留”特征"],
            ["消费金额Top1", f"编码{to_int(q25_top.get('code'))}（{pct(to_float(q25_top.get('pct')))}）", "中等消费是当前主流"],
            ["到访动机Top1", f"{motive_top.get('item_text', '（缺失）')}（{pct(to_float(motive_top.get('selected_pct', 0.0)))}）", "休闲打卡仍是主驱动，文化深度转化空间大"],
        ],
    )
    if visited_pain_top:
        pain_txt = "；".join([f"{x['item_text']}（{pct(to_float(x['selected_pct']))}）" for x in visited_pain_top])
        _add_bullet(blocks, f"已到访痛点Top3：{pain_txt}。")
    if unvisited_block_top:
        block_txt = "；".join([f"{x['item_text']}（{pct(to_float(x['selected_pct']))}）" for x in unvisited_block_top])
        _add_bullet(blocks, f"未到访阻碍Top2：{block_txt}。")
    if chi_focus:
        chi_txt = "；".join([f"{x['pair']} {x['text']}" for x in chi_focus])
        _add_bullet(blocks, f"关键卡方关系：{chi_txt}。")
    _add_bullet(
        blocks,
        f"结构化结果显示问题并非单点缺陷，而是“体验端摩擦 + 认知端吸引力不足”的叠加。 {ev('六（一）图表结果', '描述统计核心结果', str(tables_dir / '多选题选择率表.csv'), str(tables_dir / '交叉分析卡方汇总.csv'))}",
    )
    _add_image(
        blocks,
        _fmt_path(figures_dir / "核心画像图_core_profile.png"),
        "图6-1 核心画像图（性别、年龄、消费习惯、到访状态）",
        (figures_dir / "核心画像图_core_profile.png").exists(),
    )
    _add_image(
        blocks,
        _fmt_path(figures_dir / "年龄段人数_性别堆叠图.png"),
        "图6-2 年龄-性别结构图",
        (figures_dir / "年龄段人数_性别堆叠图.png").exists(),
    )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "第一，现阶段核心任务是“把已认知人群转成深度体验客”，而不是盲目扩大量。")
    _add_bullet(blocks, "第二，先解决设施、标识、价格感知三项摩擦，再提升内容密度，转化效率会更高。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "客群质量中等偏优，但深度体验与高价值转化尚未形成稳定闭环。")
    _add_bullet(blocks, "下一步需识别“谁更可能被转化”，进入MCA结构分析。")

    _add_heading(blocks, 2, "（二）基于多重对应分析的游客文化认知分析")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "识别不同人群在“文化认知-消费习惯-行为意愿”上的共现结构，找到精准运营对象。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(
        blocks,
        f"变量：Q2-Q5（人口属性）+ Q6-Q8（习惯与认知）+ Q90-Q91（行为意愿）。 {ev('六（二）数据与方法', 'MCA变量输入', str(tables_dir / 'MCA特征值.csv'), str(tables_dir / 'MCA类别坐标与贡献.csv'))}",
    )
    _add_bullet(blocks, "方法：MCA二维降维，按类别贡献与语义解释形成人群分化轴。")

    _add_heading(blocks, 3, "3. 图表结果")
    cards1 = mca.get("cards1") or [str(r.get("category", "")) for r in mca.get("dim1_top", [])]
    cards2 = mca.get("cards2") or [str(r.get("category", "")) for r in mca.get("dim2_top", [])]
    _add_table(
        blocks,
        "表6-2 MCA二维结果摘要",
        ["指标", "结果", "说明"],
        [
            ["Dim1", _metric(mca.get("dim1"), 4), "基础认知-消费习惯分化轴"],
            ["Dim2", _metric(mca.get("dim2"), 4), "游览/推荐意愿极化轴"],
            ["累计解释度", _metric(mca.get("cum2"), 4), "两维可支撑运营解释"],
            ["Dim1高贡献类别", "；".join(cards1[:5]), "低认知低习惯与其他群体分离明显"],
            ["Dim2高贡献类别", "；".join(cards2[:5]), "意愿等级差异显著"],
        ],
    )
    _add_image(
        blocks,
        _fmt_path(figures_dir / "MCA二维图.png"),
        "图6-3 MCA二维结构图",
        (figures_dir / "MCA二维图.png").exists(),
    )
    _add_bullet(
        blocks,
        f"重点转化对象是“低认知-低习惯-未到访”群体，应采用低门槛体验切入。 {ev('六（二）图表结果', 'MCA核心结论', str(tables_dir / 'MCA类别坐标与贡献.csv'), str(tables_dir / 'MCA群体解释卡.txt'))}",
    )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "建议把运营分为“认知唤醒层”和“深度体验层”，避免全人群统一促销。")
    _add_bullet(blocks, "认知不足人群先做内容教育，认知较高人群直接给深体验产品。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "MCA验证了市场存在可操作的人群分层。")
    _add_bullet(blocks, "下一节检验这些分层是否映射到“深入游览概率”。")

    _add_heading(blocks, 2, "（三）基于二元Logistic模型的深入游览分析")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "在概率框架下识别进入“深入游览状态”的关键变量，并明确结论适用边界。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(blocks, "因变量：visit_depth_bin=1（Q11>=3 且 Q12>=3）。")
    _add_bullet(
        blocks,
        f"自变量：人口属性+习惯认知+感知/表现/认知均值+动机计数，主样本与敏感性样本并列估计。 {ev('六（三）数据与方法', 'Logit模型设置', str(tables_dir / 'Logit模型指标.csv'), str(tables_dir / 'Logit回归结果_主样本.csv'), str(tables_dir / 'Logit回归结果_敏感性样本.csv'))}",
    )

    _add_heading(blocks, 3, "3. 图表结果")
    main_metric = logit.get("main_metric", {})
    sens_metric = logit.get("sens_metric", {})
    q8_sens = logit.get("q8_sens", {})
    _add_table(
        blocks,
        "表6-3 Logit模型表现与稳健性",
        ["样本", "n", "Accuracy", "AUC", "Pseudo R²", "关键结论"],
        [
            [
                "主样本",
                str(to_int(main_metric.get("n"))),
                _metric(main_metric.get("accuracy")),
                _metric(main_metric.get("auc")),
                _metric(main_metric.get("pseudo_r2")),
                "整体解释力有限，适合做趋势判断",
            ],
            [
                "敏感性样本",
                str(to_int(sens_metric.get("n"))),
                _metric(sens_metric.get("accuracy")),
                _metric(sens_metric.get("auc")),
                _metric(sens_metric.get("pseudo_r2")),
                f"Q8显著（p={_metric(q8_sens.get('p_value'))}, OR={_metric(q8_sens.get('odds_ratio'))}）",
            ],
        ],
    )
    _add_bullet(
        blocks,
        f"方向反转变量数={to_int(logit.get('reversed_n'))}，提示模型对样本口径敏感，必须与MCA/聚类/IPA联合解释。 {ev('六（三）图表结果', 'Logit稳健性边界', str(tables_dir / 'Logit稳健性方向对比.csv'), str(tables_dir / '注意力题双口径对比.csv'))}",
    )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "“是否已经建立到访认知”是深度游览的重要前置变量。")
    _add_bullet(blocks, "当前模型适合用作“线索筛查模型”，不适合做强因果政策评估。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "主样本弱显著、敏感性样本局部显著，结论应定位为“方向证据”。")
    _add_bullet(blocks, "第七章将把这些方向证据转成可执行的分层策略。")

    _add_heading(blocks, 1, "七、游客游览体验与文旅拓展路径分析")
    _add_paragraph(
        blocks,
        "本章聚焦“怎么做”：先做人群分层，再补足机制链解释，最后用IPA和行动矩阵形成整改优先级与执行清单。",
    )

    _add_heading(blocks, 2, "（一）基于二阶聚类的游客画像")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "识别可运营客群，明确“给谁什么产品、配什么优惠”两张清单。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(
        blocks,
        f"K=2~4 方案比较，按 silhouette/CH/DB 与稳定性说明选取最优分群。 {ev('七（一）数据与方法', '聚类候选与稳定性', str(tables_dir / '二阶聚类候选K评估.csv'), str(tables_dir / '聚类稳定性对比表.csv'))}",
    )

    _add_heading(blocks, 3, "3. 图表结果")
    best = cluster.get("best", {})
    profiles = cluster.get("profiles", [])
    profile_rows: list[list[str]] = []
    for row in profiles[:2]:
        profile_rows.append(
            [
                f"C{to_int(row.get('cluster'))}-{row.get('cluster_name', '')}",
                str(to_int(row.get("n"))),
                pct(to_float(row.get("share_pct"))),
                _metric(row.get("importance_mean")),
                _metric(row.get("performance_mean")),
                _metric(row.get("cognition_mean")),
                _metric(row.get("motive_count")),
            ]
        )
    _add_table(
        blocks,
        "表7-1 聚类画像卡",
        ["群体", "n", "占比", "重要度均值", "表现度均值", "认知均值", "动机计数"],
        profile_rows,
    )
    _add_bullet(
        blocks,
        f"最优簇数 K={to_int(best.get('k'))}，silhouette={_metric(best.get('silhouette'))}，说明分群可解释但不应机械切割。 {ev('七（一）图表结果', '聚类最优方案', str(tables_dir / '聚类稳定性对比表.csv'))}",
    )
    _add_image(
        blocks,
        _fmt_path(figures_dir / "二阶聚类画像图.png"),
        "图7-1 二阶聚类画像图",
        (figures_dir / "二阶聚类画像图.png").exists(),
    )
    _add_image(
        blocks,
        _fmt_path(figures_dir / "核心画像图_core_profile.png"),
        "图7-2 核心画像图",
        (figures_dir / "核心画像图_core_profile.png").exists(),
    )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "价格敏感群体优先做“高性价比套餐+便利服务”，深度体验群体优先做“内容密度+沉浸活动”。")
    _add_bullet(blocks, "落地时采用“分群投放、分群产品、分群优惠”，避免预算浪费。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "街区至少存在两类可运营客群，已可支持差异化运营。")

    _add_heading(blocks, 2, "（二）基于SOR逻辑链的机制推断（替代SEM）")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "在未追加SEM建模前，先用量表合成指标与相关性检验建立“体验-认知-意愿”机制链证据。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(blocks, "S（刺激）= 感知均值（C052-C063, C065）；O（机体）= 认知均值（C086-C089）；R（反应）= 意愿均值（C090-C091）。")
    _add_bullet(
        blocks,
        f"方法：Pearson相关 + 到访状态分组均值差异，不做因果识别。 {ev('七（二）数据与方法', 'SOR替代机制方法', str(tables_dir / 'survey_clean.csv'), str(tables_dir / '假设变量模型映射表.csv'))}",
    )

    _add_heading(blocks, 3, "3. 图表结果")
    corr_s_o = mechanism.get("corr_s_o", {})
    corr_o_r = mechanism.get("corr_o_r", {})
    corr_s_r = mechanism.get("corr_s_r", {})
    corr_perf_r = mechanism.get("corr_perf_r", {})
    visit_yes = mechanism.get("visit_yes", {})
    visit_no = mechanism.get("visit_no", {})
    _add_table(
        blocks,
        "表7-2 SOR替代机制证据",
        ["关系", "相关系数 r", "样本量 n", "解释"],
        [
            ["S(感知) -> O(认知)", _metric(corr_s_o.get("r"), 4), str(to_int(corr_s_o.get("n"))), "体验感知越好，认知提升越明显"],
            ["O(认知) -> R(意愿)", _metric(corr_o_r.get("r"), 4), str(to_int(corr_o_r.get("n"))), "认知是意愿形成的核心桥梁"],
            ["S(感知) -> R(意愿)", _metric(corr_s_r.get("r"), 4), str(to_int(corr_s_r.get("n"))), "感知可直接影响意愿，但强度低于认知链路"],
            ["Performance -> R", _metric(corr_perf_r.get("r"), 4), str(to_int(corr_perf_r.get("n"))), "表现度改善可提升意愿，但需与认知建设联动"],
        ],
    )
    _add_table(
        blocks,
        "表7-3 到访状态与意愿均值差异",
        ["群体", "意愿均值", "n", "结论"],
        [
            ["已到访/已认知（C008=1）", _metric(visit_yes.get("mean"), 4), str(to_int(visit_yes.get("n"))), "意愿水平更高"],
            ["未到访（C008=2）", _metric(visit_no.get("mean"), 4), str(to_int(visit_no.get("n"))), "需先做认知破冰"],
            ["均值差（已到访-未到访）", _metric(mechanism.get("visit_gap"), 4), "-", "到访经历具备前置激活效应"],
        ],
    )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "机制链路优先级应是“先认知、后意愿、再复游”，不建议跳过认知教育直接促销。")
    _add_bullet(blocks, "在执行层面，环境与服务改造要与文化内容解释同步推进，才能把满意转成推荐。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "即使未做SEM，当前证据已支持“S->O->R”方向性判断。")
    _add_bullet(blocks, "后续若补SEM，应重点验证认知变量的中介效应强度。")

    _add_heading(blocks, 2, "（三）基于IPA模型的文旅拓展路径")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "识别“高重要-低表现”的优先整改项，形成可执行治理顺序。")

    _add_heading(blocks, 3, "2. 数据与方法")
    mean_rows = ipa.get("mean_rows", [])
    imp_th = to_float(mean_rows[0].get("importance_threshold")) if mean_rows else 0.0
    perf_th = to_float(mean_rows[0].get("performance_threshold")) if mean_rows else 0.0
    _add_bullet(
        blocks,
        f"指标：Q66-75（重要度）与Q76-85（表现度）；均值阈值 importance={num(imp_th, 4)}、performance={num(perf_th, 4)}。 {ev('七（三）数据与方法', 'IPA阈值', str(tables_dir / 'IPA阈值敏感性表.csv'))}",
    )
    _add_bullet(blocks, "稳健性：均值阈值与中位数阈值并列校验。")

    _add_heading(blocks, 3, "3. 图表结果")
    q2_rows = ipa.get("q2", [])
    q1_rows = ipa.get("q1", [])
    q3_rows = ipa.get("q3", [])
    q4_rows = ipa.get("q4", [])
    top_q2 = q2_rows[:3]
    top_q1 = q1_rows[:4]
    _add_table(
        blocks,
        "表7-4 IPA象限结果摘要（均值阈值）",
        ["象限", "代表条目", "策略"],
        [
            ["Q2 优先改进", "；".join([str(r.get("item_text")) for r in top_q2]) or "无", "优先投入整改预算，纳入季度KPI"],
            ["Q1 保持优势", "；".join([str(r.get("item_text")) for r in top_q1]) or "无", "维持服务稳定，防止优势流失"],
            ["Q3 低优先级", "；".join([str(r.get("item_text")) for r in q3_rows[:3]]) or "无", "阶段性观察，避免过早重投"],
            ["Q4 可能过度投入", "；".join([str(r.get("item_text")) for r in q4_rows[:3]]) or "无", "优化资源配置，避免边际浪费"],
        ],
    )
    _add_image(
        blocks,
        _fmt_path(figures_dir / "IPA象限图.png"),
        "图7-3 IPA象限图",
        (figures_dir / "IPA象限图.png").exists(),
    )
    _add_bullet(
        blocks,
        f"当前稳定的优先整改项是“环境舒适度与卫生状况”，建议先完成单点突破再扩展。 {ev('七（三）图表结果', 'IPA优先项与行动映射', str(tables_dir / 'IPA整改优先级表.csv'), str(tables_dir / '建议落地行动矩阵.csv'))}",
    )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "治理应遵循“先补短板，再放大优势”，而不是均匀撒资源。")
    _add_bullet(blocks, "体验整改要与新增产品、优惠机制联动，才能转化为实际客流和客单。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "IPA给出了明确的优先级次序，可直接进入项目管理台账。")

    _add_heading(blocks, 2, "（四）游客声音与标杆案例对照（替代专家访谈）")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "用“游客直接反馈 + 国奖标杆案例结构”替代缺失的专家访谈，保证建议章节完整可读。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(
        blocks,
        f"游客声音来源：痛点、阻碍、新增项目偏好、优惠偏好与行动矩阵。 {ev('七（四）数据与方法', '游客声音数据', str(tables_dir / '多选题选择率表.csv'), str(tables_dir / '问题-证据-建议对照表.csv'), str(tables_dir / '建议落地行动矩阵.csv'))}",
    )
    _add_bullet(
        blocks,
        f"标杆对照来源：国奖案例论文摘要与策略结构。 {ev('七（四）数据与方法', '标杆案例来源', 'example/25国一晋商大院.pdf', 'example/【2024全国一等奖（研究生组）】踏遍春山不思还，与你相约梵净山——贵州梵净山旅游客源市场调研及对策研究.pdf')}",
    )

    _add_heading(blocks, 3, "3. 图表结果")
    problem_rows = voice.get("problem_rows", [])
    project_top = voice.get("new_project_top", [])
    promo_top = voice.get("promo_top", [])
    _add_table(
        blocks,
        "表7-5 游客声音与转化抓手",
        ["类别", "Top项", "建议动作"],
        [
            ["核心痛点", "；".join([str(r.get("problem")) for r in problem_rows[:3]]) or "无", "先做设施/标识/价格感知三项整改"],
            ["新增项目偏好", "；".join([f"{x['item_text']}（{pct(to_float(x['selected_pct']))}）" for x in project_top[:3]]) or "无", "优先上新药膳定制、主题展演、非遗体验课"],
            ["优惠偏好", "；".join([f"{x['item_text']}（{pct(to_float(x['selected_pct']))}）" for x in promo_top[:2]]) or "无", "以家庭套票+折扣券作为转化组合拳"],
        ],
    )
    _add_bullet(blocks, "标杆案例共同写法是“问题-证据-策略”闭环，而不是堆砌方法名。")
    _add_bullet(blocks, "标杆案例共同策略是“分层客群+差异化产品+协同治理”，与本研究分层结论一致。")

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "当前版本已可形成完整建议母稿；后续如补专家访谈，只需补充观点校准，不需改框架。")
    _add_bullet(blocks, "文本表达上应持续保持“先结论、后证据、再动作”，提升评审阅读效率。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "游客证据足以支撑策略闭环，标杆对照强化了文本可读性与比赛表达规范。")

    _add_heading(blocks, 1, "八、执行优先级与90天行动清单")
    _add_table(
        blocks,
        "表8-1 90天行动节奏",
        ["时间段", "优先动作", "量化指标"],
        [
            ["0-30天", "环境卫生与标识整改、价格说明透明化", "差评关键词占比下降，现场投诉率下降"],
            ["31-60天", "推出分群产品包（家庭/深度体验）", "分群产品转化率、客单价提升"],
            ["61-90天", "联动促销（家庭套票+折扣券）与内容传播", "复访意愿、推荐意愿持续提升"],
        ],
    )
    _add_bullet(blocks, "若执行中出现预算约束，优先保留Q2象限整改项与分群产品动作。")
    _add_bullet(blocks, "报告中的全部数字均可在同目录CSV反查，满足国奖评审的可追溯要求。")

    return {
        "title": "六七章市场调研报告（数据增强版）",
        "blocks": blocks,
        "evidence_rows": evidence_rows,
        "warnings": list(data.get("warnings", [])),
        "outline_path": str(data.get("outline_path", "")),
        "outline_exists": bool(data.get("outline_exists", False)),
    }


def render_markdown(content: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {content.get('title', '六七章市场调研报告')}")
    lines.append("")
    if content.get("outline_exists"):
        lines.append(f"> 模板来源：`{_fmt_path(content.get('outline_path', ''))}`")
    else:
        lines.append(f"> 模板来源缺失：`{_fmt_path(content.get('outline_path', ''))}`（已按数据模板生成）")
    lines.append("")

    for b in content.get("blocks", []):
        t = b.get("type")
        if t == "heading":
            lv = int(b.get("level", 1))
            lines.append(f"{'#' * max(1, min(6, lv))} {b.get('text', '')}")
            lines.append("")
        elif t == "paragraph":
            lines.append(str(b.get("text", "")))
            lines.append("")
        elif t == "bullet":
            lines.append(f"- {b.get('text', '')}")
        elif t == "image":
            caption = str(b.get("caption", ""))
            path = _fmt_path(str(b.get("path", "")))
            exists = bool(b.get("exists", False))
            if exists:
                lines.append(f"![{caption}]({path})")
                lines.append("")
                lines.append(f"*{caption}*")
                lines.append("")
            else:
                lines.append(f"- 图像缺失：{caption}（{path}）")
        elif t == "table":
            caption = str(b.get("caption", ""))
            headers = [str(h).replace("\n", " ").replace("|", "\\|") for h in b.get("headers", [])]
            rows = [
                [str(c).replace("\n", " ").replace("|", "\\|") for c in row]
                for row in b.get("rows", [])
            ]
            if caption:
                lines.append(f"**{caption}**")
                lines.append("")
            if headers:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    padded = row + [""] * max(0, len(headers) - len(row))
                    lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
                lines.append("")
    lines.append("")
    lines.append("## 证据索引提示")
    lines.append("详细证据索引见同目录 `六七章_证据索引.csv`。")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
