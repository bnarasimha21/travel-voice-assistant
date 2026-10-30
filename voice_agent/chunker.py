"""Turn a stream of LLM text deltas into speakable chunks.

A chunk ends at a sentence end, or at a clause boundary (comma, semicolon,
colon) once it has enough words, or at a hard word limit. The first chunk
uses a lower clause threshold so the first audio starts early.

A sentence end needs to see the next character (a space) before it is
trusted, so "3.5" and "St. Louis" are not split while they are still
streaming in.
"""

from __future__ import annotations

import re

ABBREVIATIONS = {
    "st.", "dr.", "mr.", "mrs.", "ms.", "jr.", "sr.", "vs.", "mt.", "ft.", "ave.", "blvd.", "rd.",
    "no.", "approx.", "est.", "dept.", "inc.", "co.", "corp.", "ltd.", "prof.", "gen.", "sen.", "rep.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
}

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
_CLAUSE_END = re.compile(r"[,;:](?=\s)")


def _is_abbreviation(text_before: str) -> bool:
    match = re.search(r"(\S+)$", text_before)
    if not match:
        return False
    token = match.group(1).lstrip("(\"'").lower()
    if token in ABBREVIATIONS:
        return True
    if re.fullmatch(r"[a-z]\.", token):  # an initial, as in "J. Smith"
        return True
    return bool(re.fullmatch(r"(?:[a-z]\.){2,}", token))  # e.g. i.e. a.m.


def _word_count(text: str) -> int:
    return len(text.split())


class SentenceChunker:
    def __init__(self, first_clause_words: int = 5, clause_words: int = 12, max_words: int = 35):
        self.first_clause_words = first_clause_words
        self.clause_words = clause_words
        self.max_words = max_words
        self._buf = ""
        self._emitted = 0

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if chunk:
                out.append(chunk)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        """Whatever is left at the end of the stream, punctuation or not."""
        chunk, self._buf = self._buf.strip(), ""
        if chunk:
            self._emitted += 1
            return [chunk]
        return []

    def _find_cut(self) -> int | None:
        buf = self._buf
        nl = buf.find("\n")
        candidates: list[int] = []
        if nl > 0 and buf[:nl].strip():
            candidates.append(nl + 1)
        for m in _SENTENCE_END.finditer(buf):
            if m.group(0).startswith(".") and len(m.group(0).rstrip("\"')]")) == 1 and _is_abbreviation(buf[: m.end()]):
                continue
            candidates.append(m.end())
            break
        threshold = self.first_clause_words if self._emitted == 0 else self.clause_words
        for m in _CLAUSE_END.finditer(buf):
            if _word_count(buf[: m.end()]) >= threshold:
                candidates.append(m.end())
                break
        if candidates:
            return min(candidates)
        # Hard limit: cut after max_words complete words (a word is complete once whitespace follows it).
        words = list(re.finditer(r"\S+(?=\s)", buf))
        if len(words) >= self.max_words:
            return words[self.max_words - 1].end()
        return None
