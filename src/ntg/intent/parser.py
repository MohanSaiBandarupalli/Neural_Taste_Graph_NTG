"""Natural-language intent parsing for discovery requests.

Deterministic, lexicon-based and fully local: no viewing data leaves the
process, and the parse is inspectable. The output strictly separates

* ``explicit``   — constraints the viewer actually stated,
* ``unsupported``— stated constraints the catalogue cannot honour (e.g. runtime
                   or series format are absent from MovieLens) — surfaced to the
                   user instead of being silently ignored,

from ``inferred`` preferences, which the engine adds later from TasteTwin and
always labels as inferred.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# word/phrase -> (genre boosts, genre penalties, tags)
MOOD_LEXICON: dict[str, dict] = {
    "exciting": {"boost": ["Action", "Thriller", "Adventure"], "tags": ["action-packed", "suspenseful"]},
    "thrilling": {"boost": ["Thriller", "Action"], "tags": ["suspenseful"]},
    "action": {"boost": ["Action"]},
    "adventure": {"boost": ["Adventure"]},
    "funny": {"boost": ["Comedy"], "tags": ["quirky", "feel-good"]},
    "comedy": {"boost": ["Comedy"]},
    "laugh": {"boost": ["Comedy"]},
    "light": {"boost": ["Comedy", "Animation", "Romance"], "penalize": ["Horror", "War", "Film-Noir"], "tags": ["feel-good"]},
    "feel-good": {"boost": ["Comedy", "Romance", "Animation", "Musical"], "penalize": ["Horror"], "tags": ["feel-good"]},
    "uplifting": {"boost": ["Comedy", "Documentary", "Musical"], "tags": ["inspiring", "feel-good"]},
    "romantic": {"boost": ["Romance"], "tags": ["romantic"]},
    "romance": {"boost": ["Romance"]},
    "scary": {"boost": ["Horror", "Thriller"], "tags": ["dark", "suspenseful"]},
    "horror": {"boost": ["Horror"]},
    "mystery": {"boost": ["Mystery", "Thriller", "Crime"], "tags": ["suspenseful"]},
    "detective": {"boost": ["Mystery", "Crime"]},
    "crime": {"boost": ["Crime"]},
    "thoughtful": {"boost": ["Drama", "Documentary", "Sci-Fi"], "tags": ["cerebral"]},
    "smart": {"boost": ["Sci-Fi", "Documentary", "Mystery"], "tags": ["cerebral"]},
    "mind-bending": {"boost": ["Sci-Fi", "Mystery"], "tags": ["cerebral"]},
    "sci-fi": {"boost": ["Sci-Fi"]},
    "science fiction": {"boost": ["Sci-Fi"]},
    "space": {"boost": ["Sci-Fi"]},
    "fantasy": {"boost": ["Fantasy"]},
    "epic": {"boost": ["Adventure", "Fantasy", "War"], "tags": ["epic"]},
    "family": {"boost": ["Children", "Animation"], "penalize": ["Horror"], "tags": ["family"]},
    "kids": {"boost": ["Children", "Animation"], "penalize": ["Horror", "Crime"], "tags": ["family"]},
    "animated": {"boost": ["Animation"]},
    "documentary": {"boost": ["Documentary"]},
    "true story": {"boost": ["Documentary", "Drama"]},
    "musical": {"boost": ["Musical"]},
    "western": {"boost": ["Western"]},
    "war": {"boost": ["War"]},
    "drama": {"boost": ["Drama"]},
    "emotional": {"boost": ["Drama", "Romance"]},
    "noir": {"boost": ["Film-Noir", "Crime"]},
    "gritty": {"boost": ["Crime", "War"], "tags": ["gritty"]},
    "dark": {"boost": ["Horror", "Film-Noir", "Crime"], "tags": ["dark"]},
}
NEGATION = re.compile(r"\b(not too|not|no|without|nothing|isn't|less|avoid)\s+(\w[\w\- ]*)")
DURATION = re.compile(r"\b(short|not too long|under \d+|quick|long|lengthy|\d+\s*(min|minutes|hours?))\b")
FORMAT = re.compile(r"\b(series|show|tv|episode|season|miniseries)\b")
LANGUAGE = re.compile(r"\b(korean|french|spanish|japanese|hindi|german|italian|subtitled|foreign|english)\b")
DIFFERENT = re.compile(r"\b(completely different|something different|out of my comfort zone|surprise me|unexpected|new to me|never watch)\b")
FAMILIAR = re.compile(r"\b(comfort|familiar|like usual|my usual|rewatch|safe pick)\b")
EXPLORE = re.compile(r"\b(explore|discover|something new|branch out|fresh)\b")
OLD = re.compile(r"\b(classic|old|vintage|retro|golden age)\b")
NEW = re.compile(r"\b(new|recent|latest|modern)\b")
DECADE = re.compile(r"\b(19[2-9]0|20[0-2]0)s\b|\b([2-9]0)s\b")
LIKE = re.compile(r"\b(?:like|similar to)\s+[\"']?([A-Z][\w:'\- ]{2,60})")


@dataclass
class ParsedIntent:
    text: str
    boost_genres: list[str] = field(default_factory=list)
    avoid_genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    exploration: str | None = None
    anchor_title: str | None = None
    unsupported: list[dict] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["explicit"] = {
            k: d[k] for k in ("boost_genres", "avoid_genres", "tags", "avoid_tags", "year_min", "year_max",
                              "exploration", "anchor_title") if d[k] not in (None, [])
        }
        return d


def _apply(term: str, entry: dict, out: ParsedIntent, negated: bool) -> None:
    out.matched_terms.append(("not " if negated else "") + term)
    if negated:
        out.avoid_genres += entry.get("boost", [])
        out.avoid_tags += entry.get("tags", [])
    else:
        out.boost_genres += entry.get("boost", [])
        out.avoid_genres += entry.get("penalize", [])
        out.tags += entry.get("tags", [])


def parse_intent(text: str, current_year: int = 2024) -> ParsedIntent:
    raw = (text or "").strip()[:500]  # bounded input
    t = raw.lower()
    out = ParsedIntent(text=raw)
    negated_spans: list[tuple[int, int]] = []
    for m in NEGATION.finditer(t):
        tail = m.group(2)
        for term in sorted(MOOD_LEXICON, key=len, reverse=True):
            if re.match(rf"{re.escape(term)}\b", tail):
                _apply(term, MOOD_LEXICON[term], out, negated=True)
                negated_spans.append((m.start(2), m.start(2) + len(term)))
                break
    for term in sorted(MOOD_LEXICON, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(term)}\b", t):
            if any(a <= m.start() < b for a, b in negated_spans):
                continue
            _apply(term, MOOD_LEXICON[term], out, negated=False)
            negated_spans.append((m.start(), m.end()))
    if DIFFERENT.search(t):
        out.exploration = "unexpected"
    elif EXPLORE.search(t):
        out.exploration = "exploratory"
    elif FAMILIAR.search(t):
        out.exploration = "familiar"
    if OLD.search(t):
        out.year_max = 1989
    elif NEW.search(t) and not EXPLORE.search(t):
        out.year_min = current_year - 6
    m = DECADE.search(t)
    if m:
        dec = int(m.group(1)) if m.group(1) else (1900 + int(m.group(2)))
        out.year_min, out.year_max = dec, dec + 9
    lm = LIKE.search(raw)
    if lm:
        title = re.split(r"\s+(?:but|and|that|with|without|except|which|or)\b|,", lm.group(1))[0]
        out.anchor_title = title.strip().rstrip(".!?")
    if DURATION.search(t):
        out.unsupported.append({"constraint": "duration", "phrase": DURATION.search(t).group(0),
                                "reason": "runtime is not available in the MovieLens catalogue"})
    if FORMAT.search(t):
        out.unsupported.append({"constraint": "format", "phrase": FORMAT.search(t).group(0),
                                "reason": "the catalogue contains films only (no series or episodes)"})
    if LANGUAGE.search(t):
        out.unsupported.append({"constraint": "language", "phrase": LANGUAGE.search(t).group(0),
                                "reason": "the catalogue has no language metadata"})
    # de-duplicate; a genre both boosted and avoided is resolved in favour of the explicit avoidance
    out.avoid_genres = sorted(set(out.avoid_genres))
    out.boost_genres = [g for g in dict.fromkeys(out.boost_genres) if g not in out.avoid_genres]
    out.tags = [x for x in dict.fromkeys(out.tags) if x not in out.avoid_tags]
    out.avoid_tags = sorted(set(out.avoid_tags))
    return out
