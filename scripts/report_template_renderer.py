#!/usr/bin/env python3
"""Template renderer for chapter 6/7 report content."""

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

    q24_top_code = to_int(desc["q24_top"]["code"])
    q24_top_pct = to_float(desc["q24_top"]["pct"])
    q25_top_code = to_int(desc["q25_top"]["code"])
    q25_top_pct = to_float(desc["q25_top"]["pct"])
    q90_pos = to_float(desc["q90_pos"])
    q91_pos = to_float(desc["q91_pos"])

    motive_top = desc["motive_top"][0] if desc["motive_top"] else {}
    motive_text = str(motive_top.get("item_text", "（缺失）"))
    motive_pct = to_float(motive_top.get("selected_pct", 0.0))

    visited_pain_top = desc["visited_pain_top"]
    unvisited_block_top = desc["unvisited_block_top"]
    chi_focus = desc["chi_focus"]

    _add_heading(blocks, 1, "六、叶开泰中医药文化街区游客特征分析")
    _add_paragraph(
        blocks,
        (
            f"本章以“行为事实—认知结构—深入游览机制”为主线展开。"
            f"分析采用统一口径 raw={raw_n}、main={main_n}（{quality_profile}），"
            "并在关键模型处并列敏感性样本，保证结论可追溯。"
        ),
    )

    _add_heading(blocks, 2, "（一）基于描述性统计的游客游览特征分析")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "识别街区游客的人口结构、游览行为、消费梯度、意愿水平与主要痛点，为后续MCA与Logit提供事实基础。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(blocks, f"数据：survey_clean.csv、单选题频数百分比表.csv、多选题选择率表.csv、交叉分析卡方汇总.csv、交叉分析列联明细.csv {ev('六（一）数据与方法', '描述统计与交叉分析数据来源', str(tables_dir / 'survey_clean.csv'), str(tables_dir / '单选题频数百分比表.csv'), str(tables_dir / '多选题选择率表.csv'), str(tables_dir / '交叉分析卡方汇总.csv'), str(tables_dir / '交叉分析列联明细.csv'))}")
    _add_bullet(blocks, "方法：频数-占比描述、到访/未到访分组比较、Q8关键交叉卡方检验。")

    _add_heading(blocks, 3, "3. 图表结果")
    _add_bullet(
        blocks,
        (
            f"样本结构上，性别基本均衡（男{pct(male_pct)}，女{pct(female_pct)}），年龄以18-25岁（{pct(age_18_25)}）与26-45岁（{pct(age_26_45)}）为主；"
            f"到访认知“是”占{pct(q8_yes_pct)}。"
            f" {ev('六（一）图表结果', '样本结构与到访认知', str(tables_dir / '单选题频数百分比表.csv'))}"
        ),
    )
    _add_bullet(
        blocks,
        (
            f"游览行为与消费上，Q24停留时长Top1为编码{q24_top_code}（{pct(q24_top_pct)}），"
            f"Q25消费金额Top1为编码{q25_top_code}（{pct(q25_top_pct)}）；到访动机Top1为“{motive_text}”（{pct(motive_pct)}）。"
            f" {ev('六（一）图表结果', '停留时长、消费梯度与到访动机', str(tables_dir / '单选题频数百分比表.csv'), str(tables_dir / '多选题选择率表.csv'))}"
        ),
    )
    _add_bullet(
        blocks,
        (
            f"意愿与口碑指标：游览意愿（Q90=4+5）{pct(q90_pos)}，推荐意愿（Q91=4+5）{pct(q91_pos)}。"
            f" {ev('六（一）图表结果', '游览意愿与推荐意愿', str(tables_dir / '单选题频数百分比表.csv'))}"
        ),
    )
    if visited_pain_top:
        pain_txt = "；".join([f"{x['item_text']}（{pct(to_float(x['selected_pct']))}）" for x in visited_pain_top])
        _add_bullet(
            blocks,
            f"已到访群体主要痛点Top3：{pain_txt}。 {ev('六（一）图表结果', '已到访痛点Top3', str(tables_dir / '多选题选择率表.csv'))}",
        )
    if unvisited_block_top:
        block_txt = "；".join([f"{x['item_text']}（{pct(to_float(x['selected_pct']))}）" for x in unvisited_block_top])
        _add_bullet(
            blocks,
            f"未到访群体阻碍Top2：{block_txt}。 {ev('六（一）图表结果', '未到访阻碍Top2', str(tables_dir / '多选题选择率表.csv'))}",
        )
    if chi_focus:
        chi_txt = "；".join([f"{x['pair']} {x['text']}" for x in chi_focus])
        _add_bullet(
            blocks,
            f"Q8关键交叉检验显示：{chi_txt}。 {ev('六（一）图表结果', 'Q8关键卡方关系', str(tables_dir / '交叉分析卡方汇总.csv'))}",
        )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "街区当前呈现“已认知但未深耕”的特征，应将中青年流量转化为深度体验客。")
    _add_bullet(blocks, "治理重点应并行覆盖体验端（设施/标识/价格-品质感知）和认知端（兴趣与内容吸引力）。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "客群结构中青年化，行为呈中等时长与中等消费并存，意愿指标处于中上区间。")
    _add_bullet(blocks, "体验端与认知端障碍并存，说明后续策略需“服务优化+内容引流”双轮驱动。")
    _add_bullet(blocks, "下一节以MCA识别高认知-高意愿组合及其分化轴。")

    _add_heading(blocks, 2, "（二）基于多重对应分析的游客文化认知分析")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "识别人口属性、消费习惯认知与行为意愿之间的潜在类别耦合关系，形成可运营人群分化轴。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(blocks, f"变量范围：人口属性（Q2-Q5）+消费习惯认知（Q6-Q8）+行为意愿（Q90-Q91）。 {ev('六（二）数据与方法', 'MCA输入变量范围', str(tables_dir / 'MCA类别坐标与贡献.csv'))}")
    _add_bullet(blocks, "方法：MCA二维降维与类别贡献解释。")

    _add_heading(blocks, 3, "3. 图表结果")
    _add_bullet(
        blocks,
        f"维度解释力：Dim1={num(to_float(mca['dim1']), 4)}，Dim2={num(to_float(mca['dim2']), 4)}，累计解释度={num(to_float(mca['cum2']), 4)}。 {ev('六（二）图表结果', 'MCA维度解释力', str(tables_dir / 'MCA特征值.csv'))}",
    )
    cards1 = mca.get("cards1") or [str(r.get("category", "")) for r in mca.get("dim1_top", [])]
    cards2 = mca.get("cards2") or [str(r.get("category", "")) for r in mca.get("dim2_top", [])]
    _add_bullet(
        blocks,
        f"维度1（基础认知-消费习惯分化轴）Top贡献类别：{'；'.join(cards1[:5])}。 {ev('六（二）图表结果', 'MCA维度1高贡献类别', str(tables_dir / 'MCA类别坐标与贡献.csv'), str(tables_dir / 'MCA群体解释卡.txt'))}",
    )
    _add_bullet(
        blocks,
        f"维度2（游览与推荐意愿极化轴）Top贡献类别：{'；'.join(cards2[:5])}。 {ev('六（二）图表结果', 'MCA维度2高贡献类别', str(tables_dir / 'MCA类别坐标与贡献.csv'), str(tables_dir / 'MCA群体解释卡.txt'))}",
    )
    _add_image(
        blocks,
        _fmt_path(figures_dir / "MCA二维图.png"),
        "图6-1 MCA二维结构图",
        (figures_dir / "MCA二维图.png").exists(),
    )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "“低认知-低习惯-未到访”群体是首要转化对象，应优先设计低门槛沉浸体验。")
    _add_bullet(blocks, "运营策略应采用认知分层，而非统一促销投放。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "MCA确认了可解释的人群分化轴，支持后续概率模型验证。")
    _add_bullet(blocks, "下一节通过Logit验证分化是否体现在“深入游览状态”概率差异。")

    _add_heading(blocks, 2, "（三）基于二元 Logistic 回归模型的游客深入游览分析")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "在概率框架下检验“深入游览状态”的影响因素，并识别模型解释边界。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(blocks, "因变量：visit_depth_bin=1（Q11>=3 且 Q12>=3）。")
    _add_bullet(blocks, "自变量：Q2/Q6/Q7/Q8编码变量 + 感知/表现/认知均值 + 动机数量。")
    _add_bullet(blocks, f"并列主样本与敏感性样本估计。 {ev('六（三）数据与方法', 'Logit模型与样本口径', str(tables_dir / 'Logit模型指标.csv'), str(tables_dir / 'Logit回归结果_主样本.csv'), str(tables_dir / 'Logit回归结果_敏感性样本.csv'))}")

    _add_heading(blocks, 3, "3. 图表结果")
    main_metric = logit.get("main_metric", {})
    sens_metric = logit.get("sens_metric", {})
    _add_bullet(
        blocks,
        (
            f"主样本（n={to_int(main_metric.get('n'))}）：accuracy={num(to_float(main_metric.get('accuracy')))}, "
            f"AUC={num(to_float(main_metric.get('auc')))}, pseudo R²={num(to_float(main_metric.get('pseudo_r2')))}；"
            "主效应整体不显著，解释力有限。 "
            f"{ev('六（三）图表结果', '主样本Logit指标', str(tables_dir / 'Logit模型指标.csv'), str(tables_dir / 'Logit回归结果_主样本.csv'))}"
        ),
    )
    _add_bullet(
        blocks,
        (
            f"敏感性样本（n={to_int(sens_metric.get('n'))}）：accuracy={num(to_float(sens_metric.get('accuracy')))}, "
            f"AUC={num(to_float(sens_metric.get('auc')))}, pseudo R²={num(to_float(sens_metric.get('pseudo_r2')))}；"
            f"Q8_visit_status_code 显著（{num(to_float(logit['q8_sens']['p_value']), 3)}，OR={num(to_float(logit['q8_sens']['odds_ratio']), 3)}）。 "
            f"{ev('六（三）图表结果', '敏感性样本Logit指标及Q8效应', str(tables_dir / 'Logit模型指标.csv'), str(tables_dir / 'Logit回归结果_敏感性样本.csv'))}"
        ),
    )
    dual_summary = logit.get("dual_summary", [])
    if dual_summary:
        best_dual = dual_summary[0]
        _add_bullet(
            blocks,
            (
                f"稳健性边界：方向反转变量数={to_int(logit.get('reversed_n'))}；"
                f"注意力题双口径中“{best_dual['calibration']}”表现较优（AUC={num(to_float(best_dual['auc']))}）。 "
                f"{ev('六（三）图表结果', 'Logit方向反转与注意力题双口径稳健性', str(tables_dir / 'Logit稳健性方向对比.csv'), str(tables_dir / '注意力题双口径对比.csv'))}"
            ),
        )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "深入游览更可能由“到访经历+认知基础”驱动，而非单一满意度项直接驱动。")
    _add_bullet(blocks, "Logit结论应作为趋势证据，与MCA、聚类、IPA联合解释，不做强因果外推。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "敏感性样本中到访状态变量显著，具备前置条件意义。")
    _add_bullet(blocks, "主样本整体解释力弱，边界清晰。")
    _add_bullet(blocks, "第七章将转入分层运营与治理优先级落地。")

    _add_heading(blocks, 1, "七、叶开泰中医药文化街区游客游览体验及文旅拓展路径分析")
    _add_paragraph(
        blocks,
        (
            "本章采用“画像识别—机制假设—优先级治理—多方意见整合”路径："
            "先做二阶聚类分层，再给出SEM机制框架占位，随后用IPA确定优先整改项，"
            "最后整合游客结构化意见与专家占位意见形成行动闭环。"
        ),
    )

    _add_heading(blocks, 2, "（一）基于二阶聚类的游客画像")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "识别不同需求结构游客群体，形成差异化产品供给与优惠机制。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(blocks, f"候选簇：K=2~4；评价指标：silhouette、CH、DB。 {ev('七（一）数据与方法', '聚类候选K与稳定性', str(tables_dir / '二阶聚类候选K评估.csv'), str(tables_dir / '聚类稳定性对比表.csv'))}")
    _add_bullet(blocks, "采用两阶段聚类并按稳定性指标选择最终簇数。")

    _add_heading(blocks, 3, "3. 图表结果")
    best = cluster.get("best", {})
    _add_bullet(
        blocks,
        f"聚类合理性：K={to_int(best.get('k'))} 的 silhouette={num(to_float(best.get('silhouette')))}，当前分层可解释但分离度中等。 {ev('七（一）图表结果', '聚类最优K与silhouette', str(tables_dir / '聚类稳定性对比表.csv'), str(tables_dir / '二阶聚类候选K评估.csv'))}",
    )
    profiles = cluster.get("profiles", [])
    for row in profiles[:2]:
        cluster_id = to_int(row.get("cluster"))
        profile_ev = ev("七（一）图表结果", f"聚类画像 C{cluster_id}", str(tables_dir / "二阶聚类画像卡.csv"))
        _add_bullet(
            blocks,
            (
                f"C{cluster_id} {row.get('cluster_name')}：n={to_int(row.get('n'))}，占比{pct(to_float(row.get('share_pct')))}；"
                f"importance_mean={num(to_float(row.get('importance_mean')))}，performance_mean={num(to_float(row.get('performance_mean')))}，"
                f"cognition_mean={num(to_float(row.get('cognition_mean')))}，motive_count={num(to_float(row.get('motive_count')))}。 "
                f"{profile_ev}"
            ),
        )
    _add_bullet(
        blocks,
        f"策略映射由“建议落地行动矩阵”承接，支持分层供给与定向促销。 {ev('七（一）图表结果', '聚类到行动矩阵映射', str(tables_dir / '建议落地行动矩阵.csv'))}",
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
    _add_bullet(blocks, "价格优惠敏感型优先强调性价比与便利感知，文化深度体验型优先强调内容密度与深度活动。")
    _add_bullet(blocks, "应避免“统一优惠泛投放”，改为“画像定向投放”。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "街区至少存在两类可运营客群。")
    _add_bullet(blocks, "分层供给与分层促销是提升转化效率的关键抓手。")

    _add_heading(blocks, 2, "（二）基于结构方程模型的游客行为意愿影响因素分析【完整框架版】")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "验证“体验评价如何通过文化认知影响行为意愿”的机制路径，为治理优先级提供理论证据。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(blocks, "理论框架：SOR（Stimulus-Organism-Response）。")
    _add_bullet(blocks, "S：体验评价维度（C052-C065）；O：文化认知（C086-C089）；R：行为意愿（C090、C091）。")
    _add_bullet(blocks, f"假设映射详见表。 {ev('七（二）数据与方法', 'SEM假设变量映射', str(tables_dir / '假设变量模型映射表.csv'))}")

    _add_heading(blocks, 3, "3. 图表结果位（占位）")
    _add_bullet(blocks, "图7-? 结构路径图：【待补：SEM结果】")
    _add_bullet(blocks, "表7-? 模型拟合指标：【待补：SEM结果】")
    _add_bullet(blocks, "表7-? 路径系数与显著性：【待补：SEM结果】")
    _add_bullet(blocks, "表7-? 中介效应（Bootstrap）：【待补：SEM结果】")

    _add_heading(blocks, 3, "4. 解释与启示（模板）")
    _add_paragraph(blocks, "结果显示，___体验通过___认知对___意愿产生显著中介作用，说明提升策略应优先作用于___环节。【待补：SEM结果】")
    _add_paragraph(blocks, "若___路径不显著，说明单纯提升___难以直接驱动行为意愿，需要与___联动。【待补：SEM结果】")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "本节仅提供机制框架位，待SEM结果到位后替换定量内容。")
    _add_bullet(blocks, "在结果补齐前，不输出任何SEM定量结论。")

    _add_heading(blocks, 2, "（三）基于IPA模型的文旅拓展路径探究")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "通过“重要度-表现度”差异识别优先改进事项，形成可执行整改清单。")

    _add_heading(blocks, 3, "2. 数据与方法")
    mean_rows = ipa.get("mean_rows", [])
    imp_th = to_float(mean_rows[0].get("importance_threshold")) if mean_rows else 0.0
    perf_th = to_float(mean_rows[0].get("performance_threshold")) if mean_rows else 0.0
    _add_bullet(blocks, f"指标：Q66-75（重要度）对 Q76-85（表现度）；均值阈值 importance={num(imp_th, 4)}，performance={num(perf_th, 4)}。 {ev('七（三）数据与方法', 'IPA均值阈值', str(tables_dir / 'IPA阈值敏感性表.csv'))}")
    _add_bullet(blocks, "稳健性：均值阈值与中位数阈值双检验。")

    _add_heading(blocks, 3, "3. 图表结果")
    q2_rows = ipa.get("q2", [])
    q1_rows = ipa.get("q1", [])
    q3_rows = ipa.get("q3", [])
    q4_rows = ipa.get("q4", [])
    if q2_rows:
        q2_txt = "；".join([f"{r.get('item_text')}（重要度{num(to_float(r.get('importance_mean')))}, 表现度{num(to_float(r.get('performance_mean')))}）" for r in q2_rows[:3]])
        _add_bullet(blocks, f"优先改进（Q2）：{q2_txt}。 {ev('七（三）图表结果', 'IPA优先改进象限', str(tables_dir / 'IPA整改优先级表.csv'), str(tables_dir / 'IPA阈值敏感性表.csv'))}")
    if q1_rows:
        _add_bullet(blocks, f"保持优势（Q1）主要包含：{'；'.join([str(r.get('item_text')) for r in q1_rows[:4]])}。")
    if q3_rows:
        _add_bullet(blocks, f"低优先级（Q3）主要包含：{'；'.join([str(r.get('item_text')) for r in q3_rows[:4]])}。")
    if q4_rows:
        _add_bullet(blocks, f"可能过度投入（Q4）主要包含：{'；'.join([str(r.get('item_text')) for r in q4_rows[:4]])}。")
    _add_bullet(blocks, f"行动转化由“问题-证据-建议对照表”和“建议落地行动矩阵”承接。 {ev('七（三）图表结果', 'IPA到行动矩阵转化', str(tables_dir / '问题-证据-建议对照表.csv'), str(tables_dir / '建议落地行动矩阵.csv'))}")
    _add_image(
        blocks,
        _fmt_path(figures_dir / "IPA象限图.png"),
        "图7-3 IPA象限图",
        (figures_dir / "IPA象限图.png").exists(),
    )

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "治理资源应优先投向“高重要-低表现”单点突破，而非平均分配。")
    _add_bullet(blocks, "供给扩展与促销设计需同步推进，避免“只改体验不改转化”。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "优先改进清单以Q2象限为核心。")
    _add_bullet(blocks, "优势项需维持稳定，避免治理资源错配。")

    _add_heading(blocks, 2, "（四）游客和专家意见（双轨模板）")
    _add_heading(blocks, 3, "1. 研究目的")
    _add_paragraph(blocks, "整合“结构化游客声音+专家判断”，形成策略闭环与建议章节母稿。")

    _add_heading(blocks, 3, "2. 数据与方法")
    _add_bullet(blocks, "游客意见：痛点多选、未到访阻碍、新增项目偏好、优惠偏好。")
    _add_bullet(blocks, f"专家意见：访谈纪要占位，待补文本证据。 {ev('七（四）数据与方法', '游客意见与行动映射数据来源', str(tables_dir / '多选题选择率表.csv'), str(tables_dir / '问题-证据-建议对照表.csv'), str(tables_dir / '建议落地行动矩阵.csv'))}")

    _add_heading(blocks, 3, "3. 图表结果")
    if voice.get("problem_rows"):
        _add_bullet(blocks, f"游客结构化问题（Top）：{'；'.join([str(r.get('problem')) for r in voice['problem_rows'][:3]])}。")
    if voice.get("new_project_top"):
        project_txt = "；".join([f"{x['item_text']}（{pct(to_float(x['selected_pct']))}）" for x in voice["new_project_top"]])
        _add_bullet(blocks, f"新增项目偏好（Top）：{project_txt}。")
    if voice.get("promo_top"):
        promo_txt = "；".join([f"{x['item_text']}（{pct(to_float(x['selected_pct']))}）" for x in voice["promo_top"]])
        _add_bullet(blocks, f"优惠偏好（Top）：{promo_txt}。")
    _add_paragraph(blocks, "专家意见（身份）：受访专家为___（机构/职称）。【待补：专家访谈纪要】")
    _add_paragraph(blocks, "专家意见（观点）：专家指出___是街区高质量发展的关键约束。【待补：专家访谈纪要】")
    _add_paragraph(blocks, "专家意见（印证）：该观点与___模型结果形成印证。【待补：专家访谈纪要】")
    _add_paragraph(blocks, "专家意见（建议）：建议优先推进___、___、___。【待补：专家访谈纪要】")

    _add_heading(blocks, 3, "4. 解释与启示")
    _add_bullet(blocks, "游客结构化需求与统计模型结果同向，可直接进入执行层。")
    _add_bullet(blocks, "专家意见用于补足机制解释与政策表达，提升文本论证完整性。")

    _add_heading(blocks, 3, "5. 本节小结")
    _add_bullet(blocks, "“问题-证据-建议”闭环可直接作为后续建议章节母稿。")
    _add_bullet(blocks, "专家材料到位后仅替换占位段落，不改变整体结构。")

    _add_heading(blocks, 1, "附录：写作质量验收清单")
    _add_bullet(blocks, "数值追溯：正文关键数值均可在CSV定位。")
    _add_bullet(blocks, "口径一致：统一使用 raw=961、main=880，杜绝旧口径残留。")
    _add_bullet(blocks, "统计合规：主样本Logit不显著项不得写成显著影响。")
    _add_bullet(blocks, "稳健性呈现：敏感性样本与阈值敏感性并列报告。")
    _add_bullet(blocks, "占位合规：SEM与专家意见必须保留待补标记。")
    _add_bullet(blocks, "结构一致：每个二级小节执行“目的-方法-结果-启示-小结”五单元。")

    return {
        "title": "六七章自动生成报告（主稿）",
        "blocks": blocks,
        "evidence_rows": evidence_rows,
        "warnings": list(data.get("warnings", [])),
        "outline_path": str(data.get("outline_path", "")),
        "outline_exists": bool(data.get("outline_exists", False)),
    }


def render_markdown(content: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {content.get('title', '六七章自动生成报告')}")
    lines.append("")
    if content.get("outline_exists"):
        lines.append(f"> 模板来源：`{_fmt_path(content.get('outline_path', ''))}`")
    else:
        lines.append(f"> 模板来源缺失：`{_fmt_path(content.get('outline_path', ''))}`（已按内置模板生成）")
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
    lines.append("")
    lines.append("## 证据索引提示")
    lines.append("详细证据索引见同目录 `六七章_证据索引.csv`。")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
