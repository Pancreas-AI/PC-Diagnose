from __future__ import annotations

import math
import os
import shutil
import warnings
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("PC_DIAGNOSE_CYTOLOGY_WORKBOOK", ROOT / "private_data" / "cytology_source.xlsx"))
OUT = SCRIPT_DIR / "specified_models_result6"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
DATA_DIR = OUT / "data"

for directory in [OUT, FIG_DIR, TABLE_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


COLUMN_RENAMES = {
    "CA19-9（0-无 1-＜100 2-＞100）": "CA19-9三分类",
    "CA19-9（0-不升高 1-升高）": "CA19-9升高",
    "CA19-9（0-＜100 1-＞100）": "CA19-9>100",
    "是否有血管包绕": "血管包绕",
    "病灶长径（cm）": "病灶长径(cm)",
    "针数": "针数",
    "细胞穿刺结果（1-无法诊断 2-良性 3-非典型 4 -neoplastic 5-可疑恶性 6-恶性": "首次细胞学结果",
    "最终诊断（0-非胰腺癌 1-胰腺癌）": "最终诊断",
}

CYTOLOGY_LABELS = {
    1: "无法诊断",
    2: "良性",
    3: "非典型",
    4: "neoplastic",
    5: "可疑恶性",
    6: "恶性",
}

MODEL_SPECS = {
    "模型1：CA19-9升高 vs 不升高": [
        "CA19-9升高vs不升高",
        "病灶长径每增加1cm",
        "淋巴结肿大vs无",
        "血管包绕vs无",
        "针数每增加1针",
    ],
    "模型2：CA19-9>100 vs <100": [
        "CA19-9>100 vs <100",
        "病灶长径每增加1cm",
        "淋巴结肿大vs无",
        "血管包绕vs无",
        "针数每增加1针",
    ],
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    for font_path in [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        path = Path(font_path)
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                prop = font_manager.FontProperties(fname=str(path))
                plt.rcParams["font.sans-serif"] = [prop.get_name(), "DejaVu Sans"]
                break
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 300


def clean_name(name: object) -> str:
    return COLUMN_RENAMES.get(str(name).strip(), str(name).strip())


def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def format_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.1%}"


def or_ci(beta: float, se: float) -> tuple[float, float, float]:
    return math.exp(beta), math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se)


def load_data() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(DATA_PATH, dtype=object)
    raw.columns = [clean_name(c) for c in raw.columns]
    df = raw[~raw["姓名"].astype(str).str.contains("首次穿刺", na=False)].copy()
    for col in df.columns:
        if col != "姓名":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["首次细胞学结果"].isin([1, 2, 3, 4, 5, 6])].copy()
    df["细胞学阳性_恶性6"] = df["首次细胞学结果"].eq(6).astype(int)
    df["细胞学结果标签"] = df["首次细胞学结果"].map(CYTOLOGY_LABELS)
    metadata = {
        "source_file": str(DATA_PATH),
        "raw_n": int(len(raw)),
        "valid_n": int(len(df)),
        "positive_n": int(df["细胞学阳性_恶性6"].sum()),
        "not_positive_n": int((1 - df["细胞学阳性_恶性6"]).sum()),
    }
    return df, metadata


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["CA19-9升高vs不升高"] = df["CA19-9升高"]
    features["CA19-9>100 vs <100"] = df["CA19-9>100"]
    features["病灶长径每增加1cm"] = df["病灶长径(cm)"]
    features["淋巴结肿大vs无"] = df["肿大淋巴结"]
    features["血管包绕vs无"] = df["血管包绕"]
    features["针数每增加1针"] = df["针数"]
    return features.astype(float)


def missing_table(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "变量": features.columns,
            "可用例数": features.notna().sum().values,
            "缺失例数": features.isna().sum().values,
            "缺失比例": features.isna().mean().values,
        }
    )
    out["缺失比例格式"] = out["缺失比例"].map(format_pct)
    return out


def cytology_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df.groupby(["首次细胞学结果", "细胞学结果标签"], observed=False)
        .size()
        .reset_index(name="例数")
        .sort_values("首次细胞学结果")
    )
    dist["比例"] = dist["例数"] / len(df)
    dist["结局分组"] = np.where(dist["首次细胞学结果"].eq(6), "阳性", "未达阳性")
    return dist


def fit_logistic(y: pd.Series, features: pd.DataFrame, variables: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.concat([y.rename("y"), features[variables]], axis=1).dropna()
    model = sm.Logit(data["y"], sm.add_constant(data[variables], has_constant="add")).fit(disp=False, maxiter=300)
    rows = []
    for var in variables:
        beta = model.params[var]
        se = model.bse[var]
        or_value, low, high = or_ci(beta, se)
        rows.append(
            {
                "模型": model_name,
                "变量": var,
                "N": len(data),
                "阳性例数": int(data["y"].sum()),
                "未达阳性例数": int((1 - data["y"]).sum()),
                "OR": or_value,
                "95%CI下限": low,
                "95%CI上限": high,
                "P值": model.pvalues[var],
                "OR(95%CI)": f"{or_value:.2f} ({low:.2f}-{high:.2f})",
                "P值格式": format_p(model.pvalues[var]),
            }
        )
    info = pd.DataFrame(
        [
            {
                "模型": model_name,
                "N": len(data),
                "阳性例数": int(data["y"].sum()),
                "未达阳性例数": int((1 - data["y"]).sum()),
                "纳入变量": "、".join(variables),
                "AIC": model.aic,
                "LogLik": model.llf,
            }
        ]
    )
    return pd.DataFrame(rows), info


def savefig(name: str) -> None:
    path = FIG_DIR / name
    plt.tight_layout(pad=1.25)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_forest(table: pd.DataFrame, title: str, filename: str) -> None:
    plot_df = table.sort_values("OR").copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    y = np.arange(len(plot_df))
    colors = np.where(plot_df["P值"] < 0.05, "#C74B4B", "#5C89A8")
    ax.errorbar(
        plot_df["OR"],
        y,
        xerr=[plot_df["OR"] - plot_df["95%CI下限"], plot_df["95%CI上限"] - plot_df["OR"]],
        fmt="o",
        color="#333333",
        ecolor="#666666",
        elinewidth=1.4,
        capsize=3,
        zorder=1,
    )
    ax.scatter(plot_df["OR"], y, s=70, c=colors, edgecolor="#333333", zorder=2)
    ax.axvline(1, linestyle="--", color="#555555", lw=1.2)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["变量"], fontsize=11)
    ax.set_xlabel("OR（对数坐标）")
    ax.set_title(title, fontsize=17, weight="bold")
    finite_high = plot_df["95%CI上限"].replace(np.inf, np.nan).dropna()
    finite_low = plot_df["95%CI下限"].replace(0, np.nan).dropna()
    xmax = min(max(finite_high.max() * 1.25, 4), 30) if not finite_high.empty else 10
    xmin = max(finite_low.min() / 1.5, 0.05) if not finite_low.empty else 0.05
    ax.set_xlim(xmin, xmax)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            xmax / 1.03,
            i,
            f"{row['OR']:.2f} ({row['95%CI下限']:.2f}-{row['95%CI上限']:.2f}), P={format_p(row['P值'])}",
            va="center",
            ha="right",
            fontsize=10,
        )
    savefig(filename)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    tmp = df[columns].copy()
    if "比例" in tmp.columns:
        tmp["比例"] = tmp["比例"].map(format_pct)
    if "缺失比例" in tmp.columns:
        tmp["缺失比例"] = tmp["缺失比例"].map(format_pct)
    return tmp.to_markdown(index=False)


def write_markdown_report(
    metadata: dict,
    dist: pd.DataFrame,
    miss: pd.DataFrame,
    model1: pd.DataFrame,
    model2: pd.DataFrame,
    info: pd.DataFrame,
) -> None:
    model1_n = int(info.loc[info["模型"].eq("模型1：CA19-9升高 vs 不升高"), "N"].iloc[0])
    model2_n = int(info.loc[info["模型"].eq("模型2：CA19-9>100 vs <100"), "N"].iloc[0])
    lesion1 = model1.loc[model1["变量"].eq("病灶长径每增加1cm")].iloc[0]
    lesion2 = model2.loc[model2["变量"].eq("病灶长径每增加1cm")].iloc[0]
    ca1 = model1.loc[model1["变量"].eq("CA19-9升高vs不升高")].iloc[0]
    ca2 = model2.loc[model2["变量"].eq("CA19-9>100 vs <100")].iloc[0]

    report = f"""# 首次细胞学恶性阳性（6）指定多因素 Logistic 回归分析报告

## 1. 分析口径

数据来源：本地临床数据（不随仓库上传）。

结局定义：首次细胞学结果 `6=恶性` 定义为阳性；`1=无法诊断`、`2=良性`、`3=非典型`、`4=neoplastic`、`5=可疑恶性` 均定义为未达阳性。

有效细胞学病例 {metadata['valid_n']} 例，其中阳性 {metadata['positive_n']} 例，未达阳性 {metadata['not_positive_n']} 例。

本次只做你指定的两个多因素 Logistic 回归模型，不额外加入其他变量。

## 2. 结局分布与缺失情况

{markdown_table(dist, ['首次细胞学结果', '细胞学结果标签', '例数', '比例', '结局分组'])}

{markdown_table(miss, ['变量', '可用例数', '缺失例数', '缺失比例'])}

病灶长径缺失 1 例，因此两个模型的完整病例均为 {model1_n} 例。

## 3. 模型1：CA19-9升高 vs 不升高

纳入变量：CA19-9升高vs不升高、病灶长径（每增加1cm）、淋巴结肿大、血管包绕、针数（每增加1针）。完整病例 N={model1_n}。

![模型1森林图](figures/fig_01_model1_ca199_elevated_forest.png)

{markdown_table(model1, ['变量', 'N', '阳性例数', '未达阳性例数', 'OR(95%CI)', 'P值格式'])}

## 4. 模型2：CA19-9>100 vs <100

纳入变量：CA19-9>100 vs <100、病灶长径（每增加1cm）、淋巴结肿大、血管包绕、针数（每增加1针）。完整病例 N={model2_n}。

![模型2森林图](figures/fig_02_model2_ca199_gt100_forest.png)

{markdown_table(model2, ['变量', 'N', '阳性例数', '未达阳性例数', 'OR(95%CI)', 'P值格式'])}

## 5. 结果解读

1. 两个模型中，病灶长径每增加 1cm 与首次细胞学恶性阳性概率升高相关：模型1 OR={lesion1['OR']:.2f}，P={format_p(lesion1['P值'])}；模型2 OR={lesion2['OR']:.2f}，P={format_p(lesion2['P值'])}。
2. 模型1中，CA19-9升高 vs 不升高的 OR={ca1['OR']:.2f}，P={format_p(ca1['P值'])}，提示方向为阳性率升高，但未达到 0.05 显著性。
3. 模型2中，CA19-9>100 vs <100 的 OR={ca2['OR']:.2f}，P={format_p(ca2['P值'])}，未显示明确独立关联。
4. 淋巴结肿大 OR 均大于 1，但 P 值未达到 0.05；血管包绕未显示明确关联。
5. 针数每增加 1 针的 OR 小于 1，但未达到 0.05 显著性；该变量仍应谨慎解释，可能反映取材困难病例需要更多针，而不是“针数增加导致未达阳性”。

## 6. 统计说明

本分析为回顾性观察数据的多因素 Logistic 回归，只能表述为“相关因素”或“提示阳性率差异”，不能直接证明因果关系。
"""
    (OUT / "首次细胞学恶性阳性_指定多因素回归报告.md").write_text(report, encoding="utf-8")


def wrap_text(text: str, width: int = 42) -> list[str]:
    lines = []
    for paragraph in str(text).split("\n"):
        paragraph = paragraph.strip()
        while len(paragraph) > width:
            cut = paragraph.rfind("，", 0, width)
            if cut < width * 0.55:
                cut = paragraph.rfind("。", 0, width)
            if cut < width * 0.55:
                cut = width
            lines.append(paragraph[: cut + 1].strip())
            paragraph = paragraph[cut + 1 :].strip()
        if paragraph:
            lines.append(paragraph)
    return lines


def pdf_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    y = 0.94
    ax.text(0.06, y, title, fontsize=18, weight="bold", color="#1F4E6E", transform=ax.transAxes)
    y -= 0.065
    for paragraph in paragraphs:
        if paragraph == "":
            y -= 0.025
            continue
        for line in wrap_text(paragraph, 42):
            ax.text(0.08, y, line, fontsize=11.3, transform=ax.transAxes, va="top")
            y -= 0.032
        y -= 0.015
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pdf_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.05, 0.96, title, fontsize=17, weight="bold", color="#1F4E6E", transform=ax.transAxes)
    img = mpimg.imread(image_path)
    ax_img = fig.add_axes([0.05, 0.12, 0.90, 0.76])
    ax_img.imshow(img)
    ax_img.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pdf_table_page(pdf: PdfPages, title: str, table_df: pd.DataFrame, columns: list[str]) -> None:
    display = table_df[columns].copy().astype(str)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=16, weight="bold", color="#1F4E6E", transform=ax.transAxes)
    table = ax.table(
        cellText=display.values,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.03, 0.12, 0.94, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DA")
        if row == 0:
            cell.set_facecolor("#DCEBF3")
            cell.set_text_props(weight="bold", color="#1F3D52")
        elif row % 2 == 0:
            cell.set_facecolor("#F7FAFC")
        if col == 0:
            cell.set_text_props(ha="left")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(metadata: dict, dist: pd.DataFrame, miss: pd.DataFrame, model1: pd.DataFrame, model2: pd.DataFrame) -> None:
    pdf_path = OUT / "首次细胞学恶性阳性_指定多因素回归报告.pdf"
    miss_pdf = miss.copy()
    miss_pdf["缺失比例"] = miss_pdf["缺失比例"].map(format_pct)
    with PdfPages(pdf_path) as pdf:
        pdf_text_page(
            pdf,
            "首次细胞学恶性阳性（6）指定多因素回归报告",
            [
                f"数据来源：本地临床数据。有效细胞学病例{metadata['valid_n']}例，阳性{metadata['positive_n']}例，未达阳性{metadata['not_positive_n']}例。",
                "结局定义：首次细胞学结果6=恶性定义为阳性；1-5均定义为未达阳性。",
                "模型1纳入：CA19-9升高vs不升高、病灶长径每增加1cm、淋巴结肿大、血管包绕、针数每增加1针。",
                "模型2纳入：CA19-9>100 vs <100、病灶长径每增加1cm、淋巴结肿大、血管包绕、针数每增加1针。",
                "本分析为回顾性观察数据，只能说明相关或提示阳性率差异，不能直接证明因果。",
            ],
        )
        pdf_table_page(pdf, "结局分布", dist, ["首次细胞学结果", "细胞学结果标签", "例数", "比例", "结局分组"])
        pdf_table_page(pdf, "模型变量缺失情况", miss_pdf, ["变量", "可用例数", "缺失例数", "缺失比例"])
        pdf_image_page(pdf, "模型1森林图", FIG_DIR / "fig_01_model1_ca199_elevated_forest.png")
        pdf_table_page(pdf, "模型1结果", model1, ["变量", "N", "阳性例数", "未达阳性例数", "OR(95%CI)", "P值格式"])
        pdf_image_page(pdf, "模型2森林图", FIG_DIR / "fig_02_model2_ca199_gt100_forest.png")
        pdf_table_page(pdf, "模型2结果", model2, ["变量", "N", "阳性例数", "未达阳性例数", "OR(95%CI)", "P值格式"])


def build_outputs() -> None:
    configure_style()
    df, metadata = load_data()
    features = derive_features(df)
    y = df["细胞学阳性_恶性6"]

    dist = cytology_distribution(df)
    miss = missing_table(features)
    model1, info1 = fit_logistic(y, features, MODEL_SPECS["模型1：CA19-9升高 vs 不升高"], "模型1：CA19-9升高 vs 不升高")
    model2, info2 = fit_logistic(y, features, MODEL_SPECS["模型2：CA19-9>100 vs <100"], "模型2：CA19-9>100 vs <100")
    info = pd.concat([info1, info2], ignore_index=True)

    cleaned = pd.concat([df, features], axis=1)
    cleaned.to_csv(DATA_DIR / "cleaned_specified_models_result6.csv", index=False)
    dist.to_csv(TABLE_DIR / "table_00_cytology_distribution.csv", index=False)
    miss.to_csv(TABLE_DIR / "table_01_missing.csv", index=False)
    model1.to_csv(TABLE_DIR / "table_02_model1_ca199_elevated.csv", index=False)
    model2.to_csv(TABLE_DIR / "table_03_model2_ca199_gt100.csv", index=False)
    info.to_csv(TABLE_DIR / "table_04_model_info.csv", index=False)

    with pd.ExcelWriter(OUT / "specified_multivariable_models_result6.xlsx", engine="openpyxl") as writer:
        dist.to_excel(writer, sheet_name="cytology_distribution", index=False)
        miss.to_excel(writer, sheet_name="missing", index=False)
        model1.to_excel(writer, sheet_name="model1_ca199_elevated", index=False)
        model2.to_excel(writer, sheet_name="model2_ca199_gt100", index=False)
        info.to_excel(writer, sheet_name="model_info", index=False)

    plot_forest(model1, "模型1：CA19-9升高 vs 不升高", "fig_01_model1_ca199_elevated_forest.png")
    plot_forest(model2, "模型2：CA19-9>100 vs <100", "fig_02_model2_ca199_gt100_forest.png")
    write_markdown_report(metadata, dist, miss, model1, model2, info)
    write_pdf_report(metadata, dist, miss, model1, model2)

    for png in FIG_DIR.glob("*.png"):
        shutil.copy2(png, OUT / png.name)

    print(f"Generated outputs in: {OUT}")
    print(f"PDF: {OUT / '首次细胞学恶性阳性_指定多因素回归报告.pdf'}")


if __name__ == "__main__":
    build_outputs()
