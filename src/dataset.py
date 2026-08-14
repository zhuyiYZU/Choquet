from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .task_schemas import get_num_classes


CLICKBAIT_DESC = (
    "clickbait detection: decide whether a headline uses misleading curiosity gap, "
    "sensational wording, or click-inducing intention"
)
SENTIMENT_DESC = (
    "implicit sentiment detection: infer positive or negative feeling from subtle "
    "implication, emotion, sarcasm, and subjective attitude"
)


def _clickbait_rows():
    positive = [
        "You won't believe what happened after this nurse opened the old box",
        "Shocking secret doctors do not want you to know",
        "This one weird trick will change how you sleep forever",
        "10 unbelievable photos that prove history was hiding the truth",
        "Must see: the hidden reason your phone battery dies so fast",
        "Everyone is sharing this mysterious kitchen hack",
        "The ending of this rescue story will leave you speechless",
        "Do not miss the secret message behind this celebrity post",
        "What happened next shocked the entire neighborhood",
        "赶紧看 这个秘密技巧让所有人震惊",
        "万万没想到 这张照片背后竟然有这样的真相",
        "一定要知道 医生从不主动告诉你的生活秘密",
        "点击查看 这个普通家庭发现了惊人的东西",
        "震惊 这个决定竟然改变了整个城市",
        "Nobody expected the final detail in this strange apartment tour",
        "The hidden warning sign every traveler must know before booking",
        "She opened the letter and what she found was unbelievable",
        "A simple mistake that could secretly cost you thousands",
        "This viral story has an ending you need to see",
        "Why experts are suddenly talking about this bizarre habit",
    ]
    negative = [
        "City council approves the annual public transport budget",
        "Researchers publish a new report on urban air quality",
        "Local library extends opening hours for students",
        "New recycling rules will begin next month",
        "Hospital announces schedule for free health screenings",
        "University team releases data from a climate survey",
        "The central bank leaves interest rates unchanged",
        "School board discusses teacher training plan",
        "Community center hosts weekend job fair",
        "市政府公布下一季度公共交通预算",
        "研究人员发布城市空气质量年度报告",
        "图书馆将在考试周延长开放时间",
        "医院公布免费体检活动安排",
        "公司发布第三季度财务摘要",
        "County officials review flood prevention measures",
        "Museum opens a new archive of local photographs",
        "Scientists compare soil samples from three regions",
        "Transit agency updates the bus maintenance schedule",
        "Public school enrollment numbers remain stable",
        "New park benches installed along the river path",
    ]
    rows = []
    for text in positive:
        rows.append(("clickbait", CLICKBAIT_DESC, text, 1))
    for text in negative:
        rows.append(("clickbait", CLICKBAIT_DESC, text, 0))
    return rows


def _sentiment_rows():
    positive = [
        "I expected another boring update, but this one quietly made my day better",
        "The small kindness from the staff stayed with me all afternoon",
        "I did not say much, yet I kept smiling after the meeting",
        "The room felt lighter once she started explaining the plan",
        "I thought it would be ordinary, however it left a warm impression",
        "The wait was long, but the final meal felt worth every minute",
        "My old phone somehow made the trip feel easier than expected",
        "He simply listened, and that was exactly what I needed",
        "The ending was gentle in a way I did not expect",
        "这个结果比我想象中更让人安心",
        "虽然过程很慢 但最后的体验让我很感动",
        "她没有说太多 却让整个下午都变得明亮",
        "本来不抱希望 其实效果出乎意料地好",
        "服务员的小举动让我记了很久",
        "The presentation was modest, but it solved the real problem",
        "I walked in tired and somehow left feeling encouraged",
        "The repair took patience, but now the machine feels dependable",
        "Nobody made a big promise, which made the result even nicer",
        "It was quiet praise, but everyone understood the respect",
        "The plain design grew on me after a day of use",
    ]
    negative = [
        "The service was so fast that I barely had time to regret ordering",
        "Amazing, my laptop crashed right before the meeting again",
        "I just love when the app deletes my work without warning",
        "The hotel called it cozy, which apparently means no room for luggage",
        "He promised a simple process; I spent the afternoon fixing errors",
        "The update was impressive if the goal was to hide every useful button",
        "I smiled through dinner because arguing would have taken more energy",
        "The package arrived early, unfortunately to the wrong address",
        "What a delightful surprise to find another unexplained fee",
        "真不错 又一次在关键时刻崩溃了",
        "客服的耐心大概都用在让我继续等待上了",
        "所谓升级 其实只是把问题藏得更深",
        "这家店的惊喜是账单比菜品更有存在感",
        "我当然开心 毕竟谁不喜欢白忙一下午呢",
        "The instructions were clear, except for the part where nothing matched",
        "I appreciate paying extra for the privilege of being ignored",
        "The chair looked elegant and felt like a punishment",
        "They fixed the bug by removing the feature I needed",
        "The tour was memorable mostly because we got lost twice",
        "I was touched by how confidently wrong the advice was",
    ]
    rows = []
    for text in positive:
        rows.append(("implicit_sentiment", SENTIMENT_DESC, text, 1))
    for text in negative:
        rows.append(("implicit_sentiment", SENTIMENT_DESC, text, 0))
    return rows


def generate_toy_dataframe(repeats: int = 2, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic toy dataset.

    The demo intentionally uses templated data: the goal is to validate the
    mechanism, not to benchmark real-world generalization.
    """
    rng = np.random.default_rng(seed)
    base_rows = _clickbait_rows() + _sentiment_rows()
    modifiers = [
        "",
        " today",
        " in a short report",
        " with comments from readers",
        " after a public discussion",
    ]
    rows = []
    for _ in range(repeats):
        for task_name, task_desc, text, label in base_rows:
            suffix = rng.choice(modifiers)
            rows.append(
                {
                    "task_name": task_name,
                    "task_description": task_desc,
                    "text": f"{text}{suffix}",
                    "label": int(label),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["task_name", "text"]).reset_index(drop=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _read_csv(data_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(data_path)


def _combined_text_from_title_content(df: pd.DataFrame) -> pd.Series:
    parts = []
    if "title" in df.columns:
        parts.append(df["title"].fillna("").astype(str))
    if "content" in df.columns:
        parts.append(df["content"].fillna("").astype(str))
    if not parts:
        raise ValueError(
            "Dataset must contain either a 'text' column, or at least one of "
            "'title'/'content' so text can be built in memory."
        )
    text = parts[0]
    for part in parts[1:]:
        text = (text + "\n" + part).str.strip()
    return text.str.strip()


def normalize_dataset(df: pd.DataFrame, source: Path | str = "<memory>") -> pd.DataFrame:
    """Validate and normalize a user dataset without writing back to disk.

    Required semantic fields for the model are task_name, task_description,
    text, and label. User files may provide text directly, or provide
    title/content columns; in that case text is assembled only in memory.
    """
    df = df.copy()
    source = str(source)
    if "task_name" not in df.columns:
        raise ValueError(f"{source} is missing required column 'task_name'.")
    if "label" not in df.columns:
        raise ValueError(f"{source} is missing required column 'label'.")

    if "task_description" not in df.columns:
        df["task_description"] = df["task_name"].fillna("").astype(str)

    if "text" not in df.columns:
        df["text"] = _combined_text_from_title_content(df)
    else:
        df["text"] = df["text"].fillna("").astype(str).str.strip()

    df["task_name"] = df["task_name"].fillna("").astype(str).str.strip()
    df["task_description"] = df["task_description"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    missing_task = df["task_name"].eq("")
    missing_text = df["text"].eq("")
    expected_classes = [
        get_num_classes(task_name, task_description)
        for task_name, task_description in zip(df["task_name"], df["task_description"])
    ]
    invalid_label = pd.Series(
        [
            label < 0 or label >= num_classes
            for label, num_classes in zip(df["label"], expected_classes)
        ],
        index=df.index,
    )
    if missing_task.any() or missing_text.any() or invalid_label.any():
        raise ValueError(
            f"{source} has invalid rows: empty task_name={int(missing_task.sum())}, "
            f"empty text={int(missing_text.sum())}, out-of-range label={int(invalid_label.sum())}."
        )

    if df.empty:
        raise ValueError(f"{source} contains no rows.")
    return df.reset_index(drop=True)


def infer_dataset_num_classes(df: pd.DataFrame) -> int:
    """Infer one output width and reject incompatible tasks in the same run."""
    class_counts = {
        get_num_classes(task_name, task_description)
        for task_name, task_description in zip(df["task_name"], df["task_description"])
    }
    if len(class_counts) != 1:
        raise ValueError(
            "One model run requires all tasks to use the same number of classes; "
            f"found class counts {sorted(class_counts)}. Run binary, tmnnews, and "
            "snippets datasets separately."
        )
    return class_counts.pop()


def load_dataset(data_path: Path, allow_generate_demo: bool = False) -> pd.DataFrame:
    """Load a dataset exactly from data_path, without overwriting user files.

    The old demo helper generated toy data whenever a file had fewer than 80
    rows or used title/content instead of text. That was unsafe for real data.
    This loader only generates demo data when the target path is missing and
    allow_generate_demo=True.
    """
    data_path = Path(data_path)
    if data_path.exists():
        return normalize_dataset(_read_csv(data_path), data_path)
    if allow_generate_demo:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_toy_dataframe(repeats=2)
        df.to_csv(data_path, index=False, encoding="utf-8")
        return normalize_dataset(df, data_path)
    raise FileNotFoundError(
        f"Dataset not found: {data_path}. Put your prepared CSV at this path "
        "or set DATA_PATH to the correct file."
    )


def ensure_toy_data(data_path: Path, allow_generate_demo: bool = False) -> pd.DataFrame:
    """Backward-compatible wrapper for code that previously loaded toy data."""
    return load_dataset(data_path, allow_generate_demo=allow_generate_demo)



def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split an already-loaded dataframe, with graceful small-sample fallback."""
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to create train/valid/test splits.")
    stratify_key = df["task_name"].astype(str) + "_" + df["label"].astype(str)
    stratify = stratify_key if stratify_key.value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify,
    )
    relative_valid = valid_ratio / (1.0 - train_ratio)
    temp_key = temp_df["task_name"].astype(str) + "_" + temp_df["label"].astype(str)
    temp_stratify = temp_key if temp_key.value_counts().min() >= 2 else None
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=relative_valid,
        random_state=seed,
        stratify=temp_stratify,
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def load_and_split(
    data_path: Path,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    allow_generate_demo: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataset(data_path, allow_generate_demo=allow_generate_demo)
    return split_dataframe(df, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed)
