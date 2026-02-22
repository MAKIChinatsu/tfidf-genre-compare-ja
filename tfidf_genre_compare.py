# -*- coding: utf-8 -*-
"""
TF-IDFジャンル内比較（日本語・分かち書き済みコーパス用）

"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 正規表現パターン（ノイズ判定・文字種判定）
# ============================================================

# 記号のみ（Unicode対応）
RE_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

# 数字のみ（半角・全角）
RE_NUM_ONLY = re.compile(r"^[0-9０-９]+$")

# 漢字1字（実用上の簡易判定）
RE_SINGLE_KANJI = re.compile(r"^[一-龥々〆ヶ]$")


# ============================================================
# 設定ファイル（JSON）関連のヘルパー
# ============================================================

def resolve_path(base_dir: Path, p: str) -> Path:
    """
    パスを解決する。
    - 絶対パスならそのまま
    - 相対パスなら config ファイルのある場所からの相対として解決
    """
    path = Path(p)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_json_config(config_path: Path) -> Dict[str, Any]:
    """JSON設定ファイルを読み込む。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    設定にデフォルト値を補う。
    config に書かれていない項目はここで既定値を使う。
    """
    defaults = {
        "top_n": 100,
        "allow_only_single_kanji": True,
        "remove_num_only": True,
        "remove_punct_only": True,
        "poetry_delimiter": "★",
        "tales_chunk_size_tokens": 500,
        "merge_short_last_chunk": True,
        "min_last_chunk_tokens": 150,
        # 安全のため既定では False（ローカル絶対パスを出力しない）
        "include_file_path_in_outputs": False,
    }
    merged = defaults.copy()
    merged.update(cfg)
    return merged


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    設定値の最低限の検証を行う。
    必須キーの有無、数値の妥当性などを確認する。
    """
    required_keys = [
        "poetry_files",
        "tales_files",
        "stopwords_file",
        "output_dir",
    ]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"設定ファイルに必須キーがありません: '{k}'")

    if not isinstance(cfg["poetry_files"], list) or len(cfg["poetry_files"]) == 0:
        raise ValueError("'poetry_files' は空でないリストである必要があります。")
    if not isinstance(cfg["tales_files"], list) or len(cfg["tales_files"]) == 0:
        raise ValueError("'tales_files' は空でないリストである必要があります。")

    if int(cfg["top_n"]) <= 0:
        raise ValueError("'top_n' は 1 以上である必要があります。")
    if int(cfg["tales_chunk_size_tokens"]) <= 0:
        raise ValueError("'tales_chunk_size_tokens' は 1 以上である必要があります。")
    if int(cfg["min_last_chunk_tokens"]) < 0:
        raise ValueError("'min_last_chunk_tokens' は 0 以上である必要があります。")


# ============================================================
# ストップワード読み込み
# ============================================================

def load_stopwords(path: Path) -> set:
    """
    ストップワードを1行1語として読み込む。
    BOM（UTF-8 with BOM）にも簡易対応。
    """
    stopwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lstrip("\ufeff")  # BOM対策
            if w:
                stopwords.add(w)
    return stopwords


# ============================================================
# 文字種判定ヘルパー
# ============================================================

def is_single_kanji(token: str) -> bool:
    """トークンが『漢字1字』かどうかを判定する。"""
    return bool(RE_SINGLE_KANJI.fullmatch(token))


# ============================================================
# 分かち書き済みテキスト用トークナイザ
# ============================================================

def build_tokenizer(
    stopwords_set: set,
    remove_punct_only: bool = True,
    remove_num_only: bool = True,
    allow_only_single_kanji: bool = True,
):
    """
    分かち書き済みテキスト（空白区切り）を前提に、
    TF-IDF用のトークナイザ関数を返す。

    実施する処理：
    1) ストップワード除去
    2) 記号だけのトークン除去
    3) 数字だけのトークン除去
    4) 一字トークンを制御（設定ONなら漢字のみ残す）
    """
    def tokenizer(text: str) -> List[str]:
        tokens = text.split()  # 分かち書き済みを想定
        cleaned = []

        for tok in tokens:
            t = tok.strip()
            if not t:
                continue

            # 1) ストップワード除去
            if t in stopwords_set:
                continue

            # 2) 記号だけのトークンを除去
            if remove_punct_only and RE_PUNCT_ONLY.fullmatch(t):
                continue

            # 3) 数字だけのトークンを除去
            if remove_num_only and RE_NUM_ONLY.fullmatch(t):
                continue

            # 4) 一字トークンの扱い（漢字のみ残す）
            if allow_only_single_kanji and len(t) == 1:
                if not is_single_kanji(t):
                    continue

            cleaned.append(t)

        return cleaned

    return tokenizer


# ============================================================
# 文書分割関数
# ============================================================

def split_by_delimiter(text: str, delimiter: str) -> List[str]:
    """
    指定区切りで分割し、空文書を除外する。
    例：詩コーパスの '★' 区切り（作品単位）
    """
    docs = []
    for p in text.split(delimiter):
        s = p.strip()
        if s:
            docs.append(s)
    return docs


def split_by_token_chunks(
    text: str,
    chunk_size_tokens: int = 500,
    merge_short_last_chunk: bool = True,
    min_last_chunk_tokens: int = 150,
) -> List[str]:
    """
    分かち書き済みテキストを、一定トークン数ごとの人工文書に分割する。
    例：童話コーパスを 500 トークン単位で分割

    末尾の余りチャンクが短すぎる場合は、直前チャンクに結合できる。
    """
    raw_tokens = [t for t in text.split() if t.strip()]
    if len(raw_tokens) == 0:
        return []

    chunks: List[List[str]] = []
    for i in range(0, len(raw_tokens), chunk_size_tokens):
        chunk = raw_tokens[i:i + chunk_size_tokens]
        if chunk:
            chunks.append(chunk)

    # 最後のチャンクが短すぎる場合、前のチャンクへ結合
    if (
        merge_short_last_chunk
        and len(chunks) >= 2
        and len(chunks[-1]) < min_last_chunk_tokens
    ):
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    return [" ".join(c) for c in chunks]


# ============================================================
# ジャンル単位のTF-IDF実行（mean集約）
# ============================================================

def run_tfidf_for_group_mean(
    group_name: str,
    input_files: List[Path],
    stopwords_set: set,
    split_mode: str,  # "delimiter" / "token_chunk" / "none"
    delimiter: Optional[str] = None,
    chunk_size_tokens: Optional[int] = None,
    merge_short_last_chunk: bool = True,
    min_last_chunk_tokens: int = 150,
    top_n: int = 100,
    remove_punct_only: bool = True,
    remove_num_only: bool = True,
    allow_only_single_kanji: bool = True,
    include_file_path_in_outputs: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1つのジャンル（詩 / 童話）について TF-IDF を計算し、
    各作家（=各入力ファイル）ごとに mean TF-IDF の上位語ランキングを作る。

    重要：
    - TF-IDFのfitはジャンルごとに行う（詩と童話を混ぜない）
    - 作家比較はジャンル内で行う
    """
    tokenizer = build_tokenizer(
        stopwords_set=stopwords_set,
        remove_punct_only=remove_punct_only,
        remove_num_only=remove_num_only,
        allow_only_single_kanji=allow_only_single_kanji,
    )

    documents: List[str] = []   # TF-IDFに渡す文書（作品単位 or チャンク単位）
    doc_labels: List[int] = []  # 各文書がどの作家ファイルに属するか（file_index）
    doc_records: List[Dict[str, Any]] = []  # 診断用情報（分割数・トークン数など）

    # ----------------------------
    # 入力ファイルを読み込み、分割
    # ----------------------------
    for file_index, file_path in enumerate(input_files):
        if not file_path.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 分割モードに応じて文書単位を作る
        if split_mode == "delimiter":
            if delimiter is None:
                raise ValueError("split_mode='delimiter' のときは delimiter を指定してください。")
            split_docs = split_by_delimiter(text, delimiter)

        elif split_mode == "token_chunk":
            if chunk_size_tokens is None or chunk_size_tokens <= 0:
                raise ValueError("split_mode='token_chunk' のときは chunk_size_tokens を正の整数で指定してください。")
            split_docs = split_by_token_chunks(
                text=text,
                chunk_size_tokens=chunk_size_tokens,
                merge_short_last_chunk=merge_short_last_chunk,
                min_last_chunk_tokens=min_last_chunk_tokens,
            )

        elif split_mode == "none":
            split_docs = [text.strip()] if text.strip() else []

        else:
            raise ValueError(f"未知の split_mode です: {split_mode}")

        if len(split_docs) == 0:
            raise ValueError(f"分割後の文書が0件になりました。区切りやファイル内容を確認してください: {file_path}")

        # 診断情報（文書ごと）
        for local_doc_id, doc_text in enumerate(split_docs, start=1):
            filtered_tokens = tokenizer(doc_text)

            rec = {
                "group": group_name,
                "file_index": file_index,
                "file_basename": file_path.name,
                "split_mode": split_mode,
                "doc_id_in_file": local_doc_id,  # 詩なら作品番号、童話ならチャンク番号
                "doc_char_len": len(doc_text),
                "raw_whitespace_token_count": len(doc_text.split()),
                "token_count_after_filter": len(filtered_tokens),
            }
            # 公開時の情報漏えい対策のため、既定ではフルパスを出力しない
            if include_file_path_in_outputs:
                rec["file_path"] = str(file_path)

            doc_records.append(rec)

            documents.append(doc_text)
            doc_labels.append(file_index)

    # ----------------------------
    # ジャンル内で TF-IDF を fit
    # ----------------------------
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=None,
        token_pattern=None,  # デフォルトの「2文字以上」制限を無効化（一字語対応）
        lowercase=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as e:
        raise ValueError(
            f"[{group_name}] TF-IDF計算に失敗しました。"
            f"フィルタ条件が厳しすぎる可能性があります。\n元エラー: {e}"
        )

    feature_names = vectorizer.get_feature_names_out()

    # ----------------------------
    # 作家（ファイル）ごとに mean で集約してランキング化
    # ----------------------------
    results: List[pd.DataFrame] = []

    for file_index, file_path in enumerate(input_files):
        # この作家に属する文書行だけ抽出
        row_indices = [i for i, label in enumerate(doc_labels) if label == file_index]
        file_tfidf_matrix = tfidf_matrix[row_indices, :]

        # 文書単位のTF-IDFを平均（作家ごとの代表値）
        mean_scores = np.asarray(file_tfidf_matrix.mean(axis=0)).ravel()
        n_docs_for_file = len(row_indices)

        ranking = pd.DataFrame({
            "term": feature_names,
            "score": mean_scores,
        })

        # スコア>0 の語だけ残し、上位N語を抽出
        ranking = ranking[ranking["score"] > 0].copy()
        ranking = ranking.sort_values(by="score", ascending=False).head(top_n).copy()

        ranking["rank"] = range(1, len(ranking) + 1)
        ranking["group"] = group_name
        ranking["metric"] = "mean"
        ranking["file_index"] = file_index
        ranking["file"] = file_path.name
        ranking["n_documents_in_file"] = n_docs_for_file
        ranking["split_mode"] = split_mode

        if include_file_path_in_outputs:
            ranking["file_path"] = str(file_path)

        # 列順を整える
        base_cols = [
            "group", "split_mode", "metric", "file_index", "file",
            "rank", "term", "score", "n_documents_in_file"
        ]
        if include_file_path_in_outputs:
            base_cols.append("file_path")

        ranking = ranking[base_cols]
        results.append(ranking)

    ranking_df = pd.concat(results, ignore_index=True)
    docinfo_df = pd.DataFrame(doc_records)

    return ranking_df, docinfo_df


# ============================================================
# メイン処理
# ============================================================

def main():
    # コマンドライン引数：設定ファイルを受け取る
    parser = argparse.ArgumentParser(
        description="分かち書き済み日本語コーパスのTF-IDFジャンル内比較（詩/童話）"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON設定ファイルのパス（例: config.example.json / config.local.json）",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    # 設定読み込み → デフォルト補完 → 検証
    raw_cfg = load_json_config(config_path)
    cfg = apply_defaults(raw_cfg)
    validate_config(cfg)

    # 相対パス解決の基準（configファイルの場所）
    base_dir = config_path.parent

    # パスを解決
    poetry_files = [resolve_path(base_dir, p) for p in cfg["poetry_files"]]
    tales_files = [resolve_path(base_dir, p) for p in cfg["tales_files"]]
    stopwords_path = resolve_path(base_dir, cfg["stopwords_file"])
    output_dir = resolve_path(base_dir, cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not stopwords_path.exists():
        raise FileNotFoundError(f"ストップワードファイルが見つかりません: {stopwords_path}")

    # 解析オプション
    top_n = int(cfg["top_n"])
    allow_only_single_kanji = bool(cfg["allow_only_single_kanji"])
    remove_num_only = bool(cfg["remove_num_only"])
    remove_punct_only = bool(cfg["remove_punct_only"])
    poetry_delimiter = str(cfg["poetry_delimiter"])
    tales_chunk_size_tokens = int(cfg["tales_chunk_size_tokens"])
    merge_short_last_chunk = bool(cfg["merge_short_last_chunk"])
    min_last_chunk_tokens = int(cfg["min_last_chunk_tokens"])
    include_file_path_in_outputs = bool(cfg.get("include_file_path_in_outputs", False))

    # ストップワード読み込み
    stopwords_set = load_stopwords(stopwords_path)

    # 設定内容を簡易表示（実行ログ）
    print(f"設定ファイル: {config_path}")
    print(f"読み込んだストップワード数: {len(stopwords_set)}")
    print(f"一字トークン方針: {'一字は漢字のみ許可' if allow_only_single_kanji else '制限なし'}")
    print(
        f"童話分割: {tales_chunk_size_tokens}トークン "
        f"(末尾短チャンク結合={merge_short_last_chunk}, 最小末尾長={min_last_chunk_tokens})"
    )
    print(f"出力順位: 上位{top_n}位")
    print("出力指標: mean")
    print(f"出力CSVにフルパスを含める: {include_file_path_in_outputs}")
    print()

    # ----------------------------
    # 詩グループ（★区切り）
    # ----------------------------
    poetry_ranking_df, poetry_docinfo_df = run_tfidf_for_group_mean(
        group_name="poetry",
        input_files=poetry_files,
        stopwords_set=stopwords_set,
        split_mode="delimiter",
        delimiter=poetry_delimiter,
        top_n=top_n,
        remove_punct_only=remove_punct_only,
        remove_num_only=remove_num_only,
        allow_only_single_kanji=allow_only_single_kanji,
        include_file_path_in_outputs=include_file_path_in_outputs,
    )

    # ----------------------------
    # 童話グループ（固定トークン数チャンク分割）
    # ----------------------------
    tales_ranking_df, tales_docinfo_df = run_tfidf_for_group_mean(
        group_name="tales",
        input_files=tales_files,
        stopwords_set=stopwords_set,
        split_mode="token_chunk",
        chunk_size_tokens=tales_chunk_size_tokens,
        merge_short_last_chunk=merge_short_last_chunk,
        min_last_chunk_tokens=min_last_chunk_tokens,
        top_n=top_n,
        remove_punct_only=remove_punct_only,
        remove_num_only=remove_num_only,
        allow_only_single_kanji=allow_only_single_kanji,
        include_file_path_in_outputs=include_file_path_in_outputs,
    )

    # ----------------------------
    # 結果を統合
    # ----------------------------
    combined_ranking_df = pd.concat([poetry_ranking_df, tales_ranking_df], ignore_index=True)
    combined_docinfo_df = pd.concat([poetry_docinfo_df, tales_docinfo_df], ignore_index=True)

    # 出力ファイル名のサフィックス（設定が分かるようにする）
    single_char_tag = "singlechar_kanji_only" if allow_only_single_kanji else "singlechar_all"
    suffix = f"mean_only_{single_char_tag}_tales_chunk{tales_chunk_size_tokens}_top{top_n}"

    ranking_out = output_dir / f"tfidf_genre_compare_{suffix}.csv"
    docinfo_out = output_dir / f"tfidf_docinfo_genre_compare_{suffix}.csv"
    poetry_out = output_dir / f"tfidf_poetry_{suffix}.csv"
    tales_out = output_dir / f"tfidf_tales_{suffix}.csv"

    # CSV保存
    combined_ranking_df.to_csv(ranking_out, index=False, encoding="utf-8-sig")
    combined_docinfo_df.to_csv(docinfo_out, index=False, encoding="utf-8-sig")
    poetry_ranking_df.to_csv(poetry_out, index=False, encoding="utf-8-sig")
    tales_ranking_df.to_csv(tales_out, index=False, encoding="utf-8-sig")

    # ----------------------------
    # 簡易診断表示（分割文書数など）
    # ----------------------------
    poetry_doc_counts = (
        poetry_docinfo_df.groupby(["file_index", "file_basename"])["doc_id_in_file"]
        .max()
        .reset_index()
        .rename(columns={"doc_id_in_file": "n_split_documents"})
    )

    tales_doc_counts = (
        tales_docinfo_df.groupby(["file_index", "file_basename"])["doc_id_in_file"]
        .max()
        .reset_index()
        .rename(columns={"doc_id_in_file": "n_split_documents"})
    )

    print("[詩グループ] 分割後文書数（作品数の近似）")
    print(poetry_doc_counts.to_string(index=False))

    print(f"\n[童話グループ] 分割後文書数（{tales_chunk_size_tokens}トークンチャンク数）")
    print(tales_doc_counts.to_string(index=False))

    tales_chunk_stats = (
        tales_docinfo_df.groupby(["file_index", "file_basename"])["raw_whitespace_token_count"]
        .agg(["count", "min", "median", "max", "mean"])
        .reset_index()
    )
    print("\n[童話グループ] チャンク長（生トークン数）の統計")
    print(tales_chunk_stats.to_string(index=False))

    print("\n保存完了:")
    print(f"  総合ランキング(mean): {ranking_out}")
    print(f"  文書診断情報        : {docinfo_out}")
    print(f"  詩グループ(mean)    : {poetry_out}")
    print(f"  童話グループ(mean)  : {tales_out}")


if __name__ == "__main__":
    main()