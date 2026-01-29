from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("tweet_pipeline")


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@dataclass(frozen=True)
class Config:
    """Runtime configuration."""

    base_dir: Path
    input_xlsx: Path

    baier_env_lex: Path
    baier_soc_lex: Path
    baier_gov_lex: Path
    lm_unc_lex: Path
    lexicon_term_col: str = "term"

    out_dir: Path = Path("analysis_data")
    save_intermediate: bool = False
    save_topic_report: bool = False

    min_text_len_chars: int = 20

    n_topics: int = 15
    n_top_words: int = 15
    cv_max_df: float = 0.95
    cv_min_df: int = 10
    cv_stop_words: str | None = "english"
    random_state: int = 42

    eps: float = 1e-6

    out_structured_name: str = "dss_tweets_analysis_structured.csv"
    out_topics_name: str = "lda_topics_top_words.txt"

    out_intermediate_en: str = "dss_tweets_en_clean_basic.csv"
    out_intermediate_eng: str = "dss_tweets_en_with_engagement.csv"
    out_intermediate_topic: str = "dss_tweets_en_with_topics.csv"
    out_intermediate_esg_ur: str = "dss_tweets_en_with_topics_sustain_esg_ur_lm.csv"

# Text and language helpers

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def basic_clean_text(text: str) -> str:
    """Lowercase; remove urls/@/#; keep a-z0-9 whitespace; compress spaces."""
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _HASHTAG_RE.sub(" ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def detect_lang_safe(text: str, seed: int = 42) -> str:
    """Detect language for one text; return 'empty' or 'error' if needed."""
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = seed
    t = str(text).strip()
    if not t:
        return "empty"
    try:
        return detect(t)
    except Exception:
        return "error"


def simple_tokenize_alpha(text: str) -> list[str]:
    """Tokenize into lowercase a-z tokens only."""
    if pd.isna(text):
        return []
    return re.findall(r"[a-z]+", str(text).lower())

# Lexicon helpers

def load_term_set(path: Path, term_col: str = "term") -> set[str]:
    """Load lexicon CSV into a lowercase term set."""
    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {path}")
    df = pd.read_csv(path)
    if term_col not in df.columns:
        raise ValueError(f"Missing column '{term_col}' in lexicon: {path}")
    terms = set(df[term_col].astype(str).str.strip().str.lower())
    terms.discard("")
    return terms


def count_hits_from_clean_text(clean_text: str, term_set: set[str]) -> int:
    """Token-exact match using clean_text.split()."""
    if pd.isna(clean_text):
        return 0
    t = str(clean_text).strip().lower()
    if not t:
        return 0
    toks = t.split()
    return sum(1 for tok in toks if tok in term_set)


# Engagement helpers

def to_numeric_fill0(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Convert columns to numeric; invalid -> NaN -> 0."""
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing required engagement column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

# Topic modeling (LDA)

def add_lda_topics(
    df: pd.DataFrame,
    text_col: str,
    n_topics: int,
    n_top_words: int,
    max_df: float,
    min_df: int,
    stop_words: str | None,
    random_state: int,
) -> tuple[pd.DataFrame, str]:
    """Fit LDA and assign a dominant topic id per document."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    if text_col not in df.columns:
        raise ValueError(f"Missing '{text_col}' for topic modeling.")

    docs = df[text_col].fillna("").astype(str).tolist()

    safe_min_df = min_df
    if len(docs) < min_df:
        safe_min_df = max(1, min(2, len(docs)))
        LOGGER.warning(
            "cv_min_df (%s) > number of docs (%s). Using cv_min_df=%s.",
            min_df,
            len(docs),
            safe_min_df,
        )

    vectorizer = CountVectorizer(
        max_df=max_df,
        min_df=safe_min_df,
        stop_words=stop_words,
    )
    dtm = vectorizer.fit_transform(docs)

    if dtm.shape[1] == 0:
        raise ValueError(
            "Document-term matrix has 0 terms. "
            "Try lowering cv_min_df or check text cleaning results."
        )

    LOGGER.info("Document-term matrix: %s docs × %s terms", dtm.shape[0], dtm.shape[1])

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
    )
    lda.fit(dtm)

    doc_topic_dist = lda.transform(dtm)
    out = df.copy()
    out["topic"] = np.argmax(doc_topic_dist, axis=1).astype(int)

    feature_names = vectorizer.get_feature_names_out()
    lines: list[str] = ["=== LDA Topics (top words) ==="]
    for topic_idx, topic_vec in enumerate(lda.components_):
        top_indices = topic_vec.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        lines.append(f"Topic {topic_idx:02d}: {', '.join(top_words)}")
    report = "\n".join(lines)

    return out, report

# Sentiment (VADER)

def get_vader_analyzer():
    """Create a VADER sentiment analyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError as e:
        raise ImportError(
            "VADER not found. Install it:\n"
            "  pip install vaderSentiment"
        ) from e

# Final engineering helpers

def esg_pattern(e: int, s: int, g: int) -> str:
    """Map (E,S,G) binary flags to a compact label."""
    total = int(e) + int(s) + int(g)
    if total == 0:
        return "none"
    if total >= 2:
        return "multiple_dimensions"
    if int(e) == 1:
        return "E_only"
    if int(s) == 1:
        return "S_only"
    return "G_only"


def safe_log(series: pd.Series, eps: float, add_one: bool = False) -> pd.Series:
    """Safe log transform."""
    x = pd.to_numeric(series, errors="coerce")
    if add_one:
        return np.log(x.fillna(0) + 1.0)
    return np.log(x.clip(lower=eps))


def ensure_out_dir(path: Path) -> None:
    """Create output folder if missing."""
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a CSV in utf-8."""
    df.to_csv(path, index=False)
    LOGGER.info("Saved: %s", path)

# Pipeline steps

def load_raw_excel(input_xlsx: Path) -> pd.DataFrame:
    """Load raw Excel."""
    if not input_xlsx.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_xlsx}")
    LOGGER.info("Loading raw Excel: %s", input_xlsx)
    df = pd.read_excel(input_xlsx)
    LOGGER.info("Raw shape: %s", df.shape)
    LOGGER.info("Raw columns: %s", list(df.columns))
    return df


def select_and_filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep core columns if present; drop empty text."""
    keep_cols = [
        "user_name", "text", "user_id", "date",
        "comment", "good", "share", "view",
        "detail_url", "comment_id",
    ]
    cols_existing = [c for c in keep_cols if c in df.columns]
    if "text" not in cols_existing:
        raise ValueError("Input file missing required column: 'text'")

    out = df[cols_existing].copy()
    out = out.dropna(subset=["text"])
    out = out[out["text"].astype(str).str.strip() != ""].copy()
    LOGGER.info("After dropping empty text: %s", out.shape)
    return out


def clean_and_filter_language(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clean text, detect language, keep English, apply length filter."""
    out = df.copy()

    LOGGER.info("Cleaning text...")
    out["clean_text"] = out["text"].apply(basic_clean_text)
    out["text_len"] = out["clean_text"].str.len()

    LOGGER.info("Language detection (langdetect)...")
    out["lang"] = out["clean_text"].apply(lambda x: detect_lang_safe(x, seed=cfg.random_state))

    before = len(out)
    out = out[out["lang"] == "en"].copy()
    LOGGER.info("Filter lang=='en': %s (from %s)", len(out), before)

    before = len(out)
    out = out[out["text_len"] >= cfg.min_text_len_chars].copy()
    LOGGER.info("Filter text_len>=%s: %s (from %s)", cfg.min_text_len_chars, len(out), before)

    out["token_count"] = out["clean_text"].astype(str).str.split().str.len()
    return out


def add_engagement_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Build engagement variables."""
    required = ["comment", "good", "share", "view"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing engagement column in input Excel: {c}")

    out = df.copy()
    out = to_numeric_fill0(out, required)

    out["engagement_raw"] = out[["comment", "good", "share"]].sum(axis=1)
    out["log_engagement_A"] = np.log1p(out["engagement_raw"])

    out["eng_rate_view"] = out["engagement_raw"] / (out["view"] + 1)
    out["log_eng_rate_view"] = np.log1p(out["engagement_raw"]) - np.log1p(out["view"] + 1)
    out["log_view_plus1"] = np.log(out["view"] + 1)
    return out


def add_esg_vars(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """ESG hits/binary/density using Baier lexicons."""
    LOGGER.info("Loading ESG lexicons...")
    env_terms = load_term_set(cfg.baier_env_lex, term_col=cfg.lexicon_term_col)
    soc_terms = load_term_set(cfg.baier_soc_lex, term_col=cfg.lexicon_term_col)
    gov_terms = load_term_set(cfg.baier_gov_lex, term_col=cfg.lexicon_term_col)

    env_terms.update({"sustainable", "sustainably"})

    LOGGER.info("Computing ESG hits/binary/density...")
    out = df.copy()
    out["E_hits"] = out["clean_text"].apply(lambda x: count_hits_from_clean_text(x, env_terms))
    out["S_hits"] = out["clean_text"].apply(lambda x: count_hits_from_clean_text(x, soc_terms))
    out["G_hits"] = out["clean_text"].apply(lambda x: count_hits_from_clean_text(x, gov_terms))

    out["E_binary"] = (out["E_hits"] > 0).astype(int)
    out["S_binary"] = (out["S_hits"] > 0).astype(int)
    out["G_binary"] = (out["G_hits"] > 0).astype(int)

    denom = out["token_count"].replace(0, np.nan)
    out["E_density"] = (out["E_hits"] / denom).fillna(0.0)
    out["S_density"] = (out["S_hits"] / denom).fillna(0.0)
    out["G_density"] = (out["G_hits"] / denom).fillna(0.0)

    out["ESG_hits_total"] = out["E_hits"] + out["S_hits"] + out["G_hits"]
    out["ESG_binary_any"] = (out["ESG_hits_total"] > 0).astype(int)
    return out


def add_uncertainty_vars(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Uncertainty hits/binary/density using LM uncertainty lexicon."""
    LOGGER.info("Loading LM uncertainty lexicon...")
    lm_terms = load_term_set(cfg.lm_unc_lex, term_col=cfg.lexicon_term_col)

    LOGGER.info("Computing uncertainty hits/binary/density...")
    out = df.copy()
    tokens_alpha = out["clean_text"].astype(str).apply(simple_tokenize_alpha)

    def count_ur_hits(token_list: list[str]) -> int:
        return sum(1 for t in token_list if t in lm_terms)

    out["UR_hits_LM"] = tokens_alpha.apply(count_ur_hits)
    out["UR_binary_LM"] = (out["UR_hits_LM"] > 0).astype(int)
    out["UR_density_LM"] = (out["UR_hits_LM"] / out["token_count"].replace(0, np.nan)).fillna(0.0)
    return out


def add_sentiment_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VADER sentiment scores on clean text."""
    LOGGER.info("Computing VADER sentiment...")
    analyzer = get_vader_analyzer()

    out = df.copy()
    scores = out["clean_text"].fillna("").astype(str).apply(analyzer.polarity_scores)
    scores_df = pd.DataFrame(list(scores))

    out["sentiment_neg"] = scores_df["neg"]
    out["sentiment_neu"] = scores_df["neu"]
    out["sentiment_pos"] = scores_df["pos"]
    out["sentiment_compound"] = scores_df["compound"]
    return out


def build_structured_dataset(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rename columns and build analysis-ready variables."""
    rename_map = {
        "comment_id": "tweet_id",
        "date": "tweet_date",
        "detail_url": "tweet_url",
        "text": "tweet_text_raw",
        "clean_text": "tweet_text_clean",
        "good": "likes",
        "comment": "replies",
        "share": "reposts",
        "view": "views",
        "text_len": "tweet_length_chars",
        "token_count": "tweet_length_tokens",
        "topic": "dss_topic",
        "E_hits": "environmental_term_count",
        "S_hits": "social_term_count",
        "G_hits": "governance_term_count",
        "E_binary": "environmental_framing",
        "S_binary": "social_framing",
        "G_binary": "governance_framing",
        "E_density": "environmental_term_density",
        "S_density": "social_term_density",
        "G_density": "governance_term_density",
        "ESG_hits_total": "esg_term_count_total",
        "ESG_binary_any": "sustainability_framing_any_esg",
        "UR_hits_LM": "uncertainty_term_count",
        "UR_binary_LM": "uncertainty_framing",
        "UR_density_LM": "uncertainty_term_density",
        "lang": "language",
    }

    out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

    for c in ["likes", "replies", "reposts", "views", "tweet_length_tokens", "tweet_length_chars"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["engagement_count"] = out["likes"].fillna(0) + out["replies"].fillna(0) + out["reposts"].fillna(0)
    out["engagement_rate"] = out["engagement_count"] / out["views"].replace({0: np.nan})
    out["engagement_rate_log"] = np.log(out["engagement_rate"].fillna(0) + cfg.eps)
    out["views_log_offset"] = safe_log(out["views"].fillna(0), eps=cfg.eps, add_one=True)

    out["tweet_length_log"] = np.log(out["tweet_length_tokens"].clip(lower=1))

    out["esg_dimension_pattern"] = out.apply(
        lambda r: esg_pattern(
            int(r.get("environmental_framing", 0)),
            int(r.get("social_framing", 0)),
            int(r.get("governance_framing", 0)),
        ),
        axis=1,
    )

    preferred_cols = [
        "tweet_id", "user_id", "user_name", "tweet_date", "tweet_url", "language",
        "dss_topic",
        "tweet_text_raw", "tweet_text_clean",
        "tweet_length_chars", "tweet_length_tokens", "tweet_length_log",
        "likes", "replies", "reposts", "views",
        "engagement_count", "engagement_rate", "engagement_rate_log", "views_log_offset",
        "sustainability_framing_any_esg",
        "environmental_framing", "social_framing", "governance_framing",
        "environmental_term_count", "social_term_count", "governance_term_count",
        "environmental_term_density", "social_term_density", "governance_term_density",
        "esg_term_count_total",
        "esg_dimension_pattern",
        "uncertainty_framing", "uncertainty_term_count", "uncertainty_term_density",
        "sentiment_compound", "sentiment_neg", "sentiment_neu", "sentiment_pos",
    ]
    final_cols = [c for c in preferred_cols if c in out.columns]
    return out[final_cols].copy()

# CLI

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tweet analysis data pipeline (Excel -> structured CSV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-dir", type=str, default="Twitter_data",
                        help="Base folder that contains the input and lexicons.")
    parser.add_argument("--input", type=str, default="Twitter_clean.xlsx",
                        help="Input Excel filename under base-dir (or a full path).")

    parser.add_argument("--out-dir", type=str, default="analysis_data",
                        help="Output folder (relative to base-dir unless absolute).")

    parser.add_argument("--save-intermediate", action="store_true",
                        help="Save intermediate CSV files for debugging.")
    parser.add_argument("--save-topic-report", action="store_true",
                        help="Save LDA topic top-words report (txt).")

    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level: DEBUG/INFO/WARNING/ERROR.")

    parser.add_argument("--baier-env", type=str, default="baier_env_lexicon_clean.csv")
    parser.add_argument("--baier-soc", type=str, default="baier_social_lexicon_clean.csv")
    parser.add_argument("--baier-gov", type=str, default="baier_governance_lexicon_clean.csv")
    parser.add_argument("--lm-unc", type=str, default="lm_uncertainty_lexicon_clean.csv")
    parser.add_argument("--lexicon-term-col", type=str, default="term")

    parser.add_argument("--n-topics", type=int, default=15)
    parser.add_argument("--cv-min-df", type=int, default=10)
    parser.add_argument("--cv-max-df", type=float, default=0.95)
    parser.add_argument("--n-top-words", type=int, default=15)

    parser.add_argument("--min-text-len", type=int, default=20)

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    base_dir = Path(args.base_dir).expanduser().resolve()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = base_dir / input_path

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir

    def under_base(p: str) -> Path:
        pp = Path(p).expanduser()
        return pp if pp.is_absolute() else (base_dir / pp)

    return Config(
        base_dir=base_dir,
        input_xlsx=input_path,
        baier_env_lex=under_base(args.baier_env),
        baier_soc_lex=under_base(args.baier_soc),
        baier_gov_lex=under_base(args.baier_gov),
        lm_unc_lex=under_base(args.lm_unc),
        lexicon_term_col=args.lexicon_term_col,
        out_dir=out_dir,
        save_intermediate=args.save_intermediate,
        save_topic_report=args.save_topic_report,
        min_text_len_chars=args.min_text_len,
        n_topics=args.n_topics,
        n_top_words=args.n_top_words,
        cv_max_df=args.cv_max_df,
        cv_min_df=args.cv_min_df,
        random_state=42,
        eps=1e-6,
    )

# Main

def run_pipeline(cfg: Config) -> int:
    ensure_out_dir(cfg.out_dir)

    df = load_raw_excel(cfg.input_xlsx)
    df = select_and_filter_rows(df)

    df = clean_and_filter_language(df, cfg)
    if cfg.save_intermediate:
        save_csv(df, cfg.out_dir / cfg.out_intermediate_en)

    df = add_engagement_vars(df)
    if cfg.save_intermediate:
        save_csv(df, cfg.out_dir / cfg.out_intermediate_eng)

    LOGGER.info("Running LDA topic modeling...")
    df, topic_report = add_lda_topics(
        df=df,
        text_col="clean_text",
        n_topics=cfg.n_topics,
        n_top_words=cfg.n_top_words,
        max_df=cfg.cv_max_df,
        min_df=cfg.cv_min_df,
        stop_words=cfg.cv_stop_words,
        random_state=cfg.random_state,
    )
    if cfg.save_intermediate:
        save_csv(df, cfg.out_dir / cfg.out_intermediate_topic)

    if cfg.save_topic_report:
        topics_path = cfg.out_dir / cfg.out_topics_name
        topics_path.write_text(topic_report, encoding="utf-8")
        LOGGER.info("Saved LDA topic report: %s", topics_path)

    df = add_esg_vars(df, cfg)
    df = add_uncertainty_vars(df, cfg)
    if cfg.save_intermediate:
        save_csv(df, cfg.out_dir / cfg.out_intermediate_esg_ur)

    df = add_sentiment_vars(df)

    df_out = build_structured_dataset(df, cfg)
    out_structured = cfg.out_dir / cfg.out_structured_name
    save_csv(df_out, out_structured)

    LOGGER.info("Done ✅ Rows × Cols: %s × %s", df_out.shape[0], df_out.shape[1])
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    cfg = build_config(args)

    LOGGER.info("Base dir: %s", cfg.base_dir)
    LOGGER.info("Input: %s", cfg.input_xlsx)
    LOGGER.info("Output dir: %s", cfg.out_dir)

    try:
        return run_pipeline(cfg)
    except Exception as e:
        LOGGER.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
