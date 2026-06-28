from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", APP_DIR))
    return base.joinpath(*parts)


def first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


TRAINING_DATA_PATH = first_existing_path(
    [
        resource_path("data", "analysis_cohort_first_puncture_undiagnosed.csv"),
        APP_DIR / "data" / "analysis_cohort_first_puncture_undiagnosed.csv",
        PROJECT_ROOT
        / "pancreatic_cancer_eus_model_analysis"
        / "data"
        / "analysis_cohort_first_puncture_undiagnosed.csv",
    ]
)
PERFORMANCE_TABLE_PATH = first_existing_path(
    [
        resource_path("data", "table_05_model_performance_5fold_oof.csv"),
        APP_DIR / "data" / "table_05_model_performance_5fold_oof.csv",
        PROJECT_ROOT
        / "pancreatic_cancer_eus_model_analysis"
        / "tables"
        / "table_05_model_performance_5fold_oof.csv",
    ]
)

TARGET_COL = "最终诊断"
HIGH_SENSITIVITY_THRESHOLD = 0.4398864929884982
SOFT_VOTING_WEIGHTS = {
    "Logistic回归": 2,
    "SVM": 1,
    "决策树": 1,
    "神经网络": 1,
}
BASE_MODEL_ORDER = list(SOFT_VOTING_WEIGHTS.keys())

TECHNICAL_FEATURES = {"穿刺针型号", "穿刺针数", "抽吸方式"}
CYTOLOGY_FEATURES = {"首次细胞学结果"}

MODEL_FEATURES = [
    "性别",
    "年龄",
    "体型",
    "症状",
    "体重下降",
    "吸烟",
    "饮酒",
    "胆胰疾病既往史",
    "胆胰疾病家族史",
    "CEA升高",
    "CA19-9",
    "黄疸/TBil>34",
    "IgG4升高",
    "肿大淋巴结",
    "血管包绕",
    "病灶长径(cm)",
    "胰管扩张",
    "胆管扩张",
    "穿刺/病灶部位",
]

NUMERIC_FEATURES = {"年龄", "病灶长径(cm)"}
TREE_CORE_FEATURES = [
    "年龄",
    "体重下降",
    "胆胰疾病既往史",
    "CA19-9",
    "黄疸/TBil>34",
    "IgG4升高",
    "肿大淋巴结",
    "血管包绕",
    "胰管扩张",
    "胆管扩张",
    "穿刺/病灶部位",
]

FIELD_DEFINITIONS = [
    {
        "key": "性别",
        "label": "性别",
        "type": "category",
        "options": [("1", "男"), ("2", "女")],
    },
    {"key": "年龄", "label": "年龄", "type": "number", "unit": "岁"},
    {
        "key": "体型",
        "label": "体型",
        "type": "category",
        "options": [("0", "正常/偏瘦"), ("1", "超重"), ("2", "肥胖")],
    },
    {
        "key": "症状",
        "label": "症状",
        "type": "category",
        "options": [("0", "无"), ("1", "有")],
    },
    {
        "key": "体重下降",
        "label": "体重下降",
        "type": "category",
        "options": [("0", "无"), ("1", "<10kg"), ("2", ">=10kg")],
    },
    {
        "key": "吸烟",
        "label": "吸烟",
        "type": "category",
        "options": [("0", "否"), ("1", "是")],
    },
    {
        "key": "饮酒",
        "label": "饮酒",
        "type": "category",
        "options": [("0", "否"), ("1", "是")],
    },
    {
        "key": "胆胰疾病既往史",
        "label": "胆胰疾病既往史",
        "type": "category",
        "options": [("0", "无"), ("1", "有")],
    },
    {
        "key": "胆胰疾病家族史",
        "label": "胆胰疾病家族史",
        "type": "category",
        "options": [("0", "无"), ("1", "有")],
    },
    {
        "key": "CEA升高",
        "label": "CEA升高",
        "type": "category",
        "options": [("0", "否"), ("1", "是")],
    },
    {
        "key": "CA19-9",
        "label": "CA19-9",
        "type": "category",
        "options": [("0", "正常"), ("1", "<100"), ("2", ">=100")],
    },
    {
        "key": "黄疸/TBil>34",
        "label": "黄疸/TBil>34",
        "type": "category",
        "options": [("0", "无"), ("1", "有")],
    },
    {
        "key": "IgG4升高",
        "label": "IgG4升高",
        "type": "category",
        "options": [("0", "否"), ("1", "是")],
    },
    {
        "key": "肿大淋巴结",
        "label": "肿大淋巴结",
        "type": "category",
        "options": [("0", "无"), ("1", "有")],
    },
    {
        "key": "血管包绕",
        "label": "血管包绕",
        "type": "category",
        "options": [("0", "无"), ("1", "有")],
    },
    {"key": "病灶长径(cm)", "label": "病灶长径", "type": "number", "unit": "cm"},
    {
        "key": "胰管扩张",
        "label": "胰管扩张",
        "type": "category",
        "options": [("0", "无"), ("1", "有")],
    },
    {
        "key": "胆管扩张",
        "label": "胆管扩张",
        "type": "category",
        "options": [("0", "无"), ("1", "有")],
    },
    {
        "key": "穿刺/病灶部位",
        "label": "病灶部位",
        "type": "category",
        "options": [
            ("1", "胰头"),
            ("2", "钩突"),
            ("3", "胰颈"),
            ("4", "胰体"),
            ("5", "胰尾"),
            ("6", "其他"),
            ("1+3", "胰头+胰颈"),
            ("3+4", "胰颈+胰体"),
        ],
    },
]

VALUE_LABELS = {
    item["key"]: {code: label for code, label in item.get("options", [])}
    for item in FIELD_DEFINITIONS
    if item["type"] == "category"
}

COLUMN_RENAMES = {
    "性别（1-男 2-女）": "性别",
    "体重（0-正常/偏瘦 1-超重 2-肥胖）": "体型",
    "症状（0-无 1-有）": "症状",
    "体重下降（0-无 1-10kg以内 2-10kg及以上）": "体重下降",
    "吸烟（1-是 0-否）": "吸烟",
    "饮酒（1-是 0-否）": "饮酒",
    "既往史（0-无 1-有）": "胆胰疾病既往史",
    "胆胰疾病家族史（1-有 0-无）": "胆胰疾病家族史",
    "CEA（0-不高 1-高）": "CEA升高",
    "CA19-9（0-无 1-＜100 2-＞100）": "CA19-9",
    "CA19-9（0-无 1-<100 2->100）": "CA19-9",
    "黄疸（0-无 1-有，34为界）": "黄疸/TBil>34",
    "黄疸/TBil＞34": "黄疸/TBil>34",
    "黄疸/TBil>34": "黄疸/TBil>34",
    "IgG4（0-不高 1-高）": "IgG4升高",
    "肿大淋巴结": "肿大淋巴结",
    "是否有血管包绕": "血管包绕",
    "病灶长径（cm）": "病灶长径(cm)",
    "病灶长径": "病灶长径(cm)",
    "胰管扩张": "胰管扩张",
    "胆管扩张": "胆管扩张",
    "针（型号，数字越大针越细，22、25较常用）": "穿刺针型号",
    "针数": "穿刺针数",
    "抽吸方式（1-负压 2-负压+湿法）": "抽吸方式",
    "穿刺部位（1-胰头 2-钩突 3-胰颈 4-胰体 5-胰尾 6-其他）": "穿刺/病灶部位",
    "病灶部位": "穿刺/病灶部位",
    "部位": "穿刺/病灶部位",
    "细胞穿刺结果（1-无法诊断 2-良性 3-非典型 4 -neoplastic 5-可疑恶性 6-恶性": "首次细胞学结果",
    "最终诊断（0-非胰腺癌 1-胰腺癌）": "最终诊断",
}

TEXT_ALIASES = {
    "": np.nan,
    "nan": np.nan,
    "none": np.nan,
    "null": np.nan,
    "na": np.nan,
    "n/a": np.nan,
    "缺失": np.nan,
    "未知": np.nan,
    "未查": np.nan,
    "未检测": np.nan,
    "无资料": np.nan,
}

CATEGORY_ALIASES = {
    "性别": {
        "男": "1",
        "男性": "1",
        "m": "1",
        "male": "1",
        "女": "2",
        "女性": "2",
        "f": "2",
        "female": "2",
    },
    "体型": {
        "正常": "0",
        "正常/偏瘦": "0",
        "偏瘦": "0",
        "超重": "1",
        "肥胖": "2",
    },
    "体重下降": {
        "无": "0",
        "否": "0",
        "没有": "0",
        "<10kg": "1",
        "＜10kg": "1",
        "10kg以内": "1",
        "小于10kg": "1",
        ">=10kg": "2",
        "≥10kg": "2",
        "＞10kg": "2",
        ">10kg": "2",
        "10kg及以上": "2",
        "大于等于10kg": "2",
    },
    "CA19-9": {
        "无": "0",
        "正常": "0",
        "不高": "0",
        "阴性": "0",
        "<100": "1",
        "＜100": "1",
        "低于100": "1",
        "升高<100": "1",
        ">=100": "2",
        "≥100": "2",
        ">100": "2",
        "＞100": "2",
        "大于100": "2",
        "高于100": "2",
    },
    "穿刺/病灶部位": {
        "胰头": "1",
        "头部": "1",
        "胰头部": "1",
        "钩突": "2",
        "胰腺钩突": "2",
        "胰颈": "3",
        "胰体": "4",
        "胰尾": "5",
        "其他": "6",
        "胰头+胰颈": "1+3",
        "胰头胰颈": "1+3",
        "胰颈+胰体": "3+4",
        "胰颈胰体": "3+4",
    },
}

YES_NO_ALIASES = {
    "是": "1",
    "有": "1",
    "阳性": "1",
    "高": "1",
    "升高": "1",
    "true": "1",
    "yes": "1",
    "y": "1",
    "否": "0",
    "无": "0",
    "没有": "0",
    "阴性": "0",
    "不高": "0",
    "正常": "0",
    "false": "0",
    "no": "0",
    "n": "0",
}

YES_NO_FEATURES = {
    "症状",
    "吸烟",
    "饮酒",
    "胆胰疾病既往史",
    "胆胰疾病家族史",
    "CEA升高",
    "黄疸/TBil>34",
    "IgG4升高",
    "肿大淋巴结",
    "血管包绕",
    "胰管扩张",
    "胆管扩张",
}


@dataclass
class PredictionResult:
    probability: float
    confidence: float
    diagnosis: str
    risk_level: str
    screen_flag: str
    submodel_probabilities: dict[str, float]
    submodel_decisions: dict[str, str]
    submodel_consensus: str
    soft_voting_formula: str
    recommendation: str
    rule_basis: str
    positive_basis: list[str]
    negative_basis: list[str]
    missing_fields: list[str]
    normalized_record: dict[str, Any]
    report: str


def clean_name(name: Any) -> str:
    text = str(name).strip()
    text = text.replace("＜", "<").replace("＞", ">")
    return COLUMN_RENAMES.get(text, text)


def normalize_category(value: Any) -> str:
    if pd.isna(value):
        return "缺失"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).strip()


def normalize_feature_value(feature: str, value: Any) -> Any:
    if value is None or pd.isna(value):
        return np.nan

    if feature in NUMERIC_FEATURES:
        if isinstance(value, str):
            value = value.strip().replace("cm", "").replace("岁", "")
        return pd.to_numeric(value, errors="coerce")

    text = str(value).strip()
    text_normalized = text.lower()
    if text_normalized in TEXT_ALIASES:
        return np.nan

    numeric_value = pd.to_numeric(text.replace(",", ""), errors="coerce")
    if feature == "CA19-9" and not pd.isna(numeric_value):
        if numeric_value in {0, 1, 2}:
            return str(int(numeric_value))
        if numeric_value <= 37:
            return "0"
        if numeric_value < 100:
            return "1"
        return "2"
    if feature == "黄疸/TBil>34" and not pd.isna(numeric_value):
        if numeric_value in {0, 1}:
            return str(int(numeric_value))
        return "1" if numeric_value > 34 else "0"

    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    text = text.replace("（", "(").replace("）", ")").replace("＜", "<").replace("＞", ">")
    if feature in CATEGORY_ALIASES and text in CATEGORY_ALIASES[feature]:
        return CATEGORY_ALIASES[feature][text]
    if feature in YES_NO_FEATURES and text_normalized in YES_NO_ALIASES:
        return YES_NO_ALIASES[text_normalized]
    if feature in VALUE_LABELS:
        label_to_code = {label: code for code, label in VALUE_LABELS[feature].items()}
        if text in label_to_code:
            return label_to_code[text]
    return text


def value_label(feature: str, value: Any) -> str:
    key = normalize_category(value)
    if key == "缺失":
        return "缺失"
    return VALUE_LABELS.get(feature, {}).get(key, key)


def split_columns(features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def prepare_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for feature in features:
        X[feature] = df[feature] if feature in df.columns else np.nan

    numeric, categorical = split_columns(features)
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical:
        X[col] = X[col].map(normalize_category)
        X[col] = X[col].mask(X[col].eq("缺失"), np.nan)
    return X


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric, categorical = split_columns(features)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="缺失")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def base_estimator(model_name: str, features: list[str]) -> Pipeline:
    model_features = features
    if model_name == "决策树":
        model_features = [c for c in TREE_CORE_FEATURES if c in features]

    if model_name == "Logistic回归":
        clf = LogisticRegression(
            max_iter=5000,
            penalty="l2",
            C=0.5,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
    elif model_name == "SVM":
        clf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "决策树":
        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=3,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "神经网络":
        clf = MLPClassifier(
            hidden_layer_sizes=(6,),
            activation="relu",
            solver="lbfgs",
            alpha=0.05,
            max_iter=5000,
            random_state=42,
        )
    else:
        raise ValueError(model_name)

    return Pipeline([("preprocess", make_preprocessor(model_features)), ("model", clf)])


def ensemble_estimator(features: list[str]) -> VotingClassifier:
    estimators = [
        ("logit", base_estimator("Logistic回归", features)),
        ("svm", base_estimator("SVM", features)),
        ("tree", base_estimator("决策树", features)),
        ("mlp", base_estimator("神经网络", features)),
    ]
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[SOFT_VOTING_WEIGHTS[name] for name in BASE_MODEL_ORDER],
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_name(c) for c in out.columns]
    if out.columns.has_duplicates:
        merged: dict[str, pd.Series] = {}
        for column in dict.fromkeys(out.columns):
            subset = out.loc[:, out.columns == column]
            if isinstance(subset, pd.Series) or subset.shape[1] == 1:
                merged[column] = subset.squeeze(axis=1)
            else:
                merged[column] = subset.bfill(axis=1).iloc[:, 0]
        out = pd.DataFrame(merged, index=out.index)
    return out


def load_patient_file(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return standardize_columns(pd.read_excel(file_path, dtype=object))

    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, dtype=object, sep=None, engine="python", encoding=encoding)
            return standardize_columns(df)
        except Exception as exc:  # pragma: no cover - keeps UI tolerant of local encodings.
            last_error = exc
    raise ValueError(f"无法读取文件：{file_path.name}。最后错误：{last_error}")


def coerce_patient_record(record: dict[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(record, pd.Series):
        raw = record.to_dict()
    else:
        raw = dict(record)

    cleaned = {clean_name(key): value for key, value in raw.items()}
    normalized = {}
    for feature in MODEL_FEATURES:
        normalized[feature] = normalize_feature_value(feature, cleaned.get(feature, np.nan))

    for key in ["姓名", "ID号", "时间"]:
        if key in cleaned:
            normalized[key] = cleaned.get(key)
    return normalized


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


class PancreaticCancerDiagnosisModel:
    def __init__(
        self,
        training_data_path: str | Path = TRAINING_DATA_PATH,
        performance_table_path: str | Path = PERFORMANCE_TABLE_PATH,
    ) -> None:
        self.training_data_path = Path(training_data_path)
        self.performance_table_path = Path(performance_table_path)
        if not self.training_data_path.exists():
            raise FileNotFoundError(f"找不到训练队列：{self.training_data_path}")

        self.training_df = standardize_columns(pd.read_csv(self.training_data_path, dtype=object))
        for feature in MODEL_FEATURES:
            self.training_df[feature] = self.training_df[feature].map(
                lambda value, f=feature: normalize_feature_value(f, value)
            )
        self.y = self.training_df[TARGET_COL].astype(int).to_numpy()
        self.model = ensemble_estimator(MODEL_FEATURES)
        self.model.fit(prepare_X(self.training_df, MODEL_FEATURES), self.y)
        self.performance = self._load_performance()

    def _load_performance(self) -> dict[str, Any]:
        if not self.performance_table_path.exists():
            return {}
        table = pd.read_csv(self.performance_table_path)
        match = table[
            (table["特征集"] == "clinical_lab_imaging") & (table["模型"] == "集成模型-软投票")
        ]
        if match.empty:
            return {}
        row = match.iloc[0].to_dict()
        return {
            "AUC": float(row.get("AUC", np.nan)),
            "敏感度": float(row.get("敏感度", np.nan)),
            "特异度": float(row.get("特异度", np.nan)),
            "阈值": float(row.get("阈值", 0.5)),
            "高敏感度阈值": float(row.get("90%敏感度阈值", HIGH_SENSITIVITY_THRESHOLD)),
            "高敏感度_特异度": float(row.get("90%敏感度_特异度", np.nan)),
            "高敏感度_NPV": float(row.get("90%敏感度_NPV", np.nan)),
        }

    @property
    def summary_text(self) -> str:
        if not self.performance:
            return "主模型：临床+实验室+影像软投票集成模型。"
        return (
            "主模型：临床+实验室+影像软投票集成模型；"
            f"内部5折AUC {self.performance['AUC']:.3f}，"
            f"0.5阈值敏感度 {self.performance['敏感度']:.3f}，"
            f"特异度 {self.performance['特异度']:.3f}。"
        )

    def predict_one(self, record: dict[str, Any] | pd.Series) -> PredictionResult:
        normalized = coerce_patient_record(record)
        row = pd.DataFrame([{feature: normalized.get(feature, np.nan) for feature in MODEL_FEATURES}])
        X = prepare_X(row, MODEL_FEATURES)
        submodel_probabilities = self._submodel_probabilities(X)
        probability = self._soft_voting_probability(submodel_probabilities)
        confidence = max(probability, 1 - probability)
        diagnosis = self._diagnosis_text(probability)
        risk_level = self._risk_level(probability)
        screen_flag = "阳性" if probability >= HIGH_SENSITIVITY_THRESHOLD else "阴性"
        submodel_decisions = {
            name: self._single_model_decision(probability)
            for name, probability in submodel_probabilities.items()
        }
        submodel_consensus = self._submodel_consensus(submodel_probabilities)
        soft_voting_formula = self._soft_voting_formula(submodel_probabilities)
        recommendation = self._recommendation(probability)
        rule_basis = self._rule_basis(normalized)
        positive_basis, negative_basis = self._feature_basis(normalized)
        missing_fields = [
            feature
            for feature in MODEL_FEATURES
            if feature not in normalized or pd.isna(normalized.get(feature))
        ]
        report = self._build_report(
            normalized,
            probability,
            confidence,
            diagnosis,
            risk_level,
            screen_flag,
            submodel_probabilities,
            submodel_decisions,
            submodel_consensus,
            soft_voting_formula,
            recommendation,
            rule_basis,
            positive_basis,
            negative_basis,
            missing_fields,
        )
        return PredictionResult(
            probability=probability,
            confidence=confidence,
            diagnosis=diagnosis,
            risk_level=risk_level,
            screen_flag=screen_flag,
            submodel_probabilities=submodel_probabilities,
            submodel_decisions=submodel_decisions,
            submodel_consensus=submodel_consensus,
            soft_voting_formula=soft_voting_formula,
            recommendation=recommendation,
            rule_basis=rule_basis,
            positive_basis=positive_basis,
            negative_basis=negative_basis,
            missing_fields=missing_fields,
            normalized_record=normalized,
            report=report,
        )

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        standardized = standardize_columns(df)
        rows = []
        for idx, row in standardized.iterrows():
            result = self.predict_one(row)
            name = row.get("姓名", "")
            patient_id = row.get("ID号", "")
            rows.append(
                {
                    "序号": idx + 1,
                    "姓名": "" if pd.isna(name) else name,
                    "ID号": "" if pd.isna(patient_id) else patient_id,
                    "胰腺癌概率": round(result.probability, 4),
                    "胰腺癌概率(%)": pct(result.probability),
                    "判定可信度": round(result.confidence, 4),
                    "判定可信度(%)": pct(result.confidence),
                    "诊断结论": result.diagnosis,
                    "风险等级": result.risk_level,
                    "高敏感度筛查": result.screen_flag,
                    "子模型一致性": result.submodel_consensus,
                    "Logistic回归概率": round(result.submodel_probabilities["Logistic回归"], 4),
                    "Logistic回归概率(%)": pct(result.submodel_probabilities["Logistic回归"]),
                    "Logistic回归判断": result.submodel_decisions["Logistic回归"],
                    "SVM概率": round(result.submodel_probabilities["SVM"], 4),
                    "SVM概率(%)": pct(result.submodel_probabilities["SVM"]),
                    "SVM判断": result.submodel_decisions["SVM"],
                    "决策树概率": round(result.submodel_probabilities["决策树"], 4),
                    "决策树概率(%)": pct(result.submodel_probabilities["决策树"]),
                    "决策树判断": result.submodel_decisions["决策树"],
                    "神经网络概率": round(result.submodel_probabilities["神经网络"], 4),
                    "神经网络概率(%)": pct(result.submodel_probabilities["神经网络"]),
                    "神经网络判断": result.submodel_decisions["神经网络"],
                    "软投票公式": result.soft_voting_formula,
                    "简化规则辅助解释": result.rule_basis,
                    "建议": result.recommendation,
                    "缺失字段": "；".join(result.missing_fields),
                    "诊断报告": result.report,
                }
            )
        return pd.DataFrame(rows)

    def _diagnosis_text(self, probability: float) -> str:
        if probability >= 0.5:
            return "集成模型倾向：胰腺癌高风险"
        if probability >= HIGH_SENSITIVITY_THRESHOLD:
            return "集成模型提示：临界风险，高敏感度阈值阳性"
        return "集成模型倾向：暂未达到胰腺癌高风险阈值"

    def _single_model_decision(self, probability: float) -> str:
        if probability >= 0.5:
            return "单独判断：高风险"
        if probability >= HIGH_SENSITIVITY_THRESHOLD:
            return "单独判断：临界风险"
        return "单独判断：未达高风险阈值"

    def _submodel_probabilities(self, X: pd.DataFrame) -> dict[str, float]:
        return {
            name: float(estimator.predict_proba(X)[0, 1])
            for name, estimator in zip(BASE_MODEL_ORDER, self.model.estimators_)
        }

    def _soft_voting_probability(self, submodel_probabilities: dict[str, float]) -> float:
        numerator = sum(
            SOFT_VOTING_WEIGHTS[name] * submodel_probabilities[name] for name in BASE_MODEL_ORDER
        )
        denominator = sum(SOFT_VOTING_WEIGHTS.values())
        return float(numerator / denominator)

    def _submodel_consensus(self, submodel_probabilities: dict[str, float]) -> str:
        high_count = sum(probability >= 0.5 for probability in submodel_probabilities.values())
        borderline_count = sum(
            HIGH_SENSITIVITY_THRESHOLD <= probability < 0.5
            for probability in submodel_probabilities.values()
        )
        if high_count == len(submodel_probabilities):
            return "4/4 子模型均为高风险"
        if high_count:
            return f"{high_count}/4 子模型为高风险，{borderline_count}/4 为临界风险"
        if borderline_count:
            return f"0/4 子模型为高风险，{borderline_count}/4 为临界风险"
        return "0/4 子模型为高风险"

    def _soft_voting_formula(self, submodel_probabilities: dict[str, float]) -> str:
        p_logit = submodel_probabilities["Logistic回归"]
        p_svm = submodel_probabilities["SVM"]
        p_tree = submodel_probabilities["决策树"]
        p_mlp = submodel_probabilities["神经网络"]
        final_probability = self._soft_voting_probability(submodel_probabilities)
        return (
            "P集成 = (2×PLogistic + 1×PSVM + 1×P决策树 + 1×P神经网络) / 5 "
            f"= (2×{pct(p_logit)} + {pct(p_svm)} + {pct(p_tree)} + {pct(p_mlp)}) / 5 "
            f"= {pct(final_probability)}"
        )

    def _risk_level(self, probability: float) -> str:
        if probability >= 0.70:
            return "高风险"
        if probability >= 0.50:
            return "中高风险"
        if probability >= HIGH_SENSITIVITY_THRESHOLD:
            return "临界风险"
        if probability >= 0.25:
            return "中低风险"
        return "低风险"

    def _recommendation(self, probability: float) -> str:
        if probability >= 0.70:
            return "建议尽快结合影像、病理及MDT意见，优先考虑复穿、进一步分期或积极诊疗路径。"
        if probability >= 0.50:
            return "建议结合临床判断，考虑短期复查、重复取材或MDT讨论。"
        if probability >= HIGH_SENSITIVITY_THRESHOLD:
            return "建议保持警惕；如影像或症状进展，宜短期复查或进一步取材。"
        return "建议常规随访并结合其他临床证据；若出现新危险特征，应重新评估。"

    def _rule_basis(self, record: dict[str, Any]) -> str:
        ca199 = normalize_category(record.get("CA19-9"))
        igg4 = normalize_category(record.get("IgG4升高"))
        pdil = normalize_category(record.get("胰管扩张"))
        vascular = normalize_category(record.get("血管包绕"))

        if ca199 == "2":
            return "简化规则树：CA19-9 >=100，训练队列中 20/25 为胰腺癌，约80%。"
        if ca199 in {"0", "1"} and igg4 == "1":
            return "简化规则树：CA19-9未重度升高且IgG4升高，训练队列中 0/12 为胰腺癌。"
        if ca199 in {"0", "1"} and igg4 == "0" and (pdil == "1" or vascular == "1"):
            return "简化规则树：CA19-9未重度升高、IgG4不高且有胰管扩张或血管包绕，训练队列中 16/28 为胰腺癌，约57%。"
        if ca199 in {"0", "1"} and igg4 == "0" and pdil == "0" and vascular == "0":
            return "简化规则树：CA19-9未重度升高、IgG4不高且无胰管扩张和血管包绕，训练队列中 3/15 为胰腺癌，约20%。"
        return "简化规则树：CA19-9、IgG4、胰管扩张或血管包绕存在缺失，无法完整套用规则树。"

    def _feature_basis(self, record: dict[str, Any]) -> tuple[list[str], list[str]]:
        positive: list[str] = []
        negative: list[str] = []

        age = pd.to_numeric(record.get("年龄"), errors="coerce")
        lesion = pd.to_numeric(record.get("病灶长径(cm)"), errors="coerce")
        if not pd.isna(age) and age >= 60:
            positive.append(f"年龄 {age:.0f} 岁")
        if normalize_category(record.get("体重下降")) in {"1", "2"}:
            positive.append(f"体重下降：{value_label('体重下降', record.get('体重下降'))}")
        elif normalize_category(record.get("体重下降")) == "0":
            negative.append("无体重下降")
        if normalize_category(record.get("CA19-9")) == "2":
            positive.append("CA19-9 >=100")
        elif normalize_category(record.get("CA19-9")) == "0":
            negative.append("CA19-9正常")
        if normalize_category(record.get("黄疸/TBil>34")) == "1":
            positive.append("黄疸/TBil>34")
        if normalize_category(record.get("肿大淋巴结")) == "1":
            positive.append("肿大淋巴结")
        if normalize_category(record.get("血管包绕")) == "1":
            positive.append("血管包绕")
        if normalize_category(record.get("胰管扩张")) == "1":
            positive.append("胰管扩张")
        if normalize_category(record.get("胆管扩张")) == "1":
            positive.append("胆管扩张")
        if normalize_category(record.get("穿刺/病灶部位")) in {"1", "2"}:
            positive.append(f"病灶位于{value_label('穿刺/病灶部位', record.get('穿刺/病灶部位'))}")
        if not pd.isna(lesion) and lesion >= 3:
            positive.append(f"病灶长径 {lesion:.1f} cm")
        if normalize_category(record.get("IgG4升高")) == "1":
            negative.append("IgG4升高，需注意自身免疫性胰腺炎等鉴别")

        if not positive:
            positive.append("未录得明确的高危特征组合")
        if not negative:
            negative.append("未录得明确降低风险或提示鉴别诊断的特征")
        return positive, negative

    def _build_report(
        self,
        record: dict[str, Any],
        probability: float,
        confidence: float,
        diagnosis: str,
        risk_level: str,
        screen_flag: str,
        submodel_probabilities: dict[str, float],
        submodel_decisions: dict[str, str],
        submodel_consensus: str,
        soft_voting_formula: str,
        recommendation: str,
        rule_basis: str,
        positive_basis: list[str],
        negative_basis: list[str],
        missing_fields: list[str],
    ) -> str:
        name = record.get("姓名")
        patient_id = record.get("ID号")
        patient_line = []
        if name is not None and not pd.isna(name) and str(name).strip():
            patient_line.append(f"姓名：{name}")
        if patient_id is not None and not pd.isna(patient_id) and str(patient_id).strip():
            patient_line.append(f"ID号：{patient_id}")
        patient_text = "；".join(patient_line) if patient_line else "未填写姓名/ID"

        feature_text = "；".join(
            f"{feature}={self._display_value(feature, record.get(feature))}" for feature in MODEL_FEATURES
        )
        missing_text = "；".join(missing_fields) if missing_fields else "无"
        high_sens_text = (
            f"高敏感度参考阈值 {HIGH_SENSITIVITY_THRESHOLD:.3f}，当前为{screen_flag}。"
        )
        submodel_lines = "\n".join(
            f"- {name}：胰腺癌概率 {pct(submodel_probabilities[name])}，{submodel_decisions[name]}"
            for name in BASE_MODEL_ORDER
        )

        return (
            "胰腺癌辅助诊断报告\n"
            "====================\n"
            f"{patient_text}\n\n"
            "一、集成模型主结论\n"
            f"诊断结论：{diagnosis}\n"
            f"风险等级：{risk_level}\n"
            f"集成模型估计胰腺癌概率：{pct(probability)}\n"
            f"判定可信度：{pct(confidence)}\n"
            f"{high_sens_text}\n\n"
            "二、子模型单独判断\n"
            f"{submodel_lines}\n"
            f"子模型一致性：{submodel_consensus}。\n\n"
            "三、软投票规则\n"
            "本软件先让四个子模型分别输出“胰腺癌概率”，再按权重加权平均。"
            "Logistic回归权重为2，SVM、决策树、神经网络权重各为1；"
            "最终集成概率以加权平均结果为准。\n"
            f"{soft_voting_formula}。\n"
            "主诊断结论使用上述集成概率，而不是单独决策树结果。\n\n"
            "四、诊断依据\n"
            f"1. 辅助规则解释：{rule_basis}\n"
            f"2. 支持风险升高的特征：{'；'.join(positive_basis)}。\n"
            f"3. 提示风险降低或需鉴别的特征：{'；'.join(negative_basis)}。\n\n"
            f"建议：{recommendation}\n\n"
            "模型说明：本报告使用已构建的临床+实验室+影像软投票集成模型；"
            "技术变量（针型号、针数、抽吸方式）和首次细胞学结果不进入主模型。"
            "该结果仅用于临床辅助决策，不能替代病理诊断或医生综合判断。\n\n"
            f"缺失字段：{missing_text}\n"
            f"录入特征：{feature_text}\n"
        )

    def _display_value(self, feature: str, value: Any) -> str:
        if value is None or pd.isna(value):
            return "缺失"
        if feature in NUMERIC_FEATURES:
            numeric = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric):
                return "缺失"
            if feature == "年龄":
                return f"{numeric:.0f}岁"
            return f"{numeric:.1f}cm"
        return value_label(feature, value)


def create_import_template(path: str | Path) -> Path:
    out = Path(path)
    columns = ["姓名", "ID号", *MODEL_FEATURES]
    sample = {
        "姓名": "示例患者",
        "ID号": "P001",
        "性别": "1",
        "年龄": 62,
        "体型": "0",
        "症状": "1",
        "体重下降": "1",
        "吸烟": "0",
        "饮酒": "0",
        "胆胰疾病既往史": "0",
        "胆胰疾病家族史": "0",
        "CEA升高": "0",
        "CA19-9": "2",
        "黄疸/TBil>34": "1",
        "IgG4升高": "0",
        "肿大淋巴结": "0",
        "血管包绕": "1",
        "病灶长径(cm)": 3.2,
        "胰管扩张": "1",
        "胆管扩张": "0",
        "穿刺/病灶部位": "1",
    }
    pd.DataFrame([sample], columns=columns).to_csv(out, index=False, encoding="utf-8-sig")
    return out
