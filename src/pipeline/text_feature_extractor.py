"""
Text Feature Extractor Module
Extracts robust tabular NLP features from free-text columns:
- Character and word length counts
- Lexical diversity (TTR)
- Special character, digit, and Hangul/Jamo composition
- Cross-text lexical overlap & Jaccard similarity between pairs
"""
import re
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class TextFeatureExtractor:
    def __init__(self, max_features_per_text: int = 5, max_vocab_size: int = 500, min_df: int = 2):
        self.max_features_per_text = max_features_per_text
        self.max_vocab_size = max_vocab_size
        self.min_df = min_df
        self.text_cols_: List[str] = []
        self.extracted_cols_: List[str] = []
        self.fitted_vocab_: Dict[str, List[str]] = {}

    def identify_text_columns(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
        """Identifies text columns suitable for NLP feature extraction (average len > 10, non-trivial cardinality)."""
        exclude_cols = exclude_cols or []
        text_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                s = df[col].dropna().astype(str)
                if len(s) > 0 and s.str.len().mean() > 10:
                    text_cols.append(col)
        self.text_cols_ = text_cols
        return text_cols

    def fit(self, df: pd.DataFrame) -> "TextFeatureExtractor":
        """Fits vocabulary with strict upper bound (max_vocab_size) and min_df to prevent memory explosion."""
        self.fitted_vocab_ = {}
        for col in self.text_cols_:
            if col not in df.columns:
                continue
            s = df[col].fillna("").astype(str)
            token_counts = {}
            for text_val in s:
                tokens = re.findall(r"[a-zA-Z가-힣0-9]{2,}", text_val.lower())
                seen = set(tokens)
                for t in seen:
                    token_counts[t] = token_counts.get(t, 0) + 1

            # Filter by min_df and sort by document frequency descending, up to max_vocab_size
            frequent_tokens = [
                token for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
                if count >= self.min_df
            ][:min(self.max_vocab_size, 20)] # top 20 salient keywords per text column for tabular compactness
            self.fitted_vocab_[col] = frequent_tokens
        return self

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts numerical features from identified text columns."""
        feats = pd.DataFrame(index=df.index)
        if not self.text_cols_:
            return feats

        for col in self.text_cols_:
            if col not in df.columns:
                continue
            s = df[col].fillna("").astype(str)
            
            # 1. Character & Word lengths
            c_len = s.str.len().astype(float)
            w_cnt = s.str.split().str.len().fillna(0).astype(float)
            feats[f"{col}_char_len"] = c_len
            feats[f"{col}_word_count"] = w_cnt
            
            # 2. Average Word Length
            feats[f"{col}_avg_word_len"] = np.where(w_cnt > 0, c_len / np.maximum(w_cnt, 1.0), 0.0)

            # 3. Digit & Numeric ratio
            digit_cnt = s.apply(lambda x: sum(1 for c in x if c.isdigit())).astype(float)
            feats[f"{col}_digit_ratio"] = np.where(c_len > 0, digit_cnt / np.maximum(c_len, 1.0), 0.0)

            # 4. Korean Hangul Ratio & Character distribution
            hangul_cnt = s.apply(lambda x: sum(1 for c in x if '\uac00' <= c <= '\ud7a3')).astype(float)
            feats[f"{col}_hangul_ratio"] = np.where(c_len > 0, hangul_cnt / np.maximum(c_len, 1.0), 0.0)

            # 5. Question mark & punctuation indicator
            feats[f"{col}_has_qmark"] = s.apply(lambda x: 1.0 if '?' in x else 0.0)

            # 5-1. Salient Keywords Frequency (Bounded Vocabulary)
            vocab = self.fitted_vocab_.get(col, [])
            for token in vocab:
                clean_t = re.sub(r"[^a-zA-Z가-힣0-9]", "_", token)
                feats[f"{col}_kw_{clean_t}"] = s.str.lower().apply(lambda x, t=token.lower(): 1.0 if t in x else 0.0)

        # 6. Pairwise text overlaps (if multiple text columns exist, e.g. question vs answer/title)
        if len(self.text_cols_) >= 2:
            for i in range(len(self.text_cols_)):
                for j in range(i + 1, len(self.text_cols_)):
                    c1, c2 = self.text_cols_[i], self.text_cols_[j]
                    s1 = df[c1].fillna("").astype(str)
                    s2 = df[c2].fillna("").astype(str)
                    
                    # Length Ratio
                    feats[f"text_ratio_{c1}_div_{c2}"] = feats[f"{c1}_char_len"] / (feats[f"{c2}_char_len"] + 1.0)
                    
                    # Jaccard word-level overlap
                    def calc_jaccard(t1, t2):
                        w1 = set(t1.split())
                        w2 = set(t2.split())
                        if not w1 or not w2:
                            return 0.0
                        return float(len(w1 & w2)) / float(len(w1 | w2))
                        
                    feats[f"jaccard_overlap_{c1}_{c2}"] = [calc_jaccard(x, y) for x, y in zip(s1, s2)]

        self.extracted_cols_ = list(feats.columns)
        return feats
