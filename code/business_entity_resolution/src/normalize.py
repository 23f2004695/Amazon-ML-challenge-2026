"""Text normalization for business names and addresses.

Everything here is country-agnostic: the same rules run on US, India, France and any
other label. Country-specific knowledge is limited to generic lookup tables (legal
forms, street-type abbreviations, state names) that only ever *canonicalize* tokens.

Indic-script tokens are converted to Latin via a dictionary learned from the training
pairs (see build_dict.py); unknown tokens fall back to unidecode + letter squashing.
"""
import json
import os
import re
import unicodedata

from unidecode import unidecode

# ----------------------------------------------------------------------------- tables
LEGAL = {
    # english / us / india
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "pvt", "private", "llp", "plc", "lp", "pllc", "pc", "pa", "the",
    "and", "of", "ltda", "opc", "pte", "dba", "ms", "m", "s", "pra", "li", "prv",
    # french
    "sarl", "sas", "sasu", "eurl", "sa", "snc", "ei", "sci", "cie", "et", "des", "de",
    "du", "la", "le", "les", "d", "l", "gmbh", "scop", "selarl", "scm", "eirl",
}
# junk tokens injected as noise: we drop them from the core name
NOISE = {"www", "com", "net", "org", "in", "fr", "id", "http", "https", "na"}

STREET = {
    "street": "st", "str": "st", "strt": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "avenu": "ave", "drive": "dr", "drv": "dr", "lane": "ln", "place": "pl", "circle": "cir",
    "terrace": "ter", "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "court": "ct",
    "highway": "hwy", "parkway": "pkwy", "pky": "pkwy", "square": "sq", "trail": "trl",
    "way": "wy", "point": "pt", "mount": "mt", "fort": "ft", "saint": "st", "sainte": "ste",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "apartment": "apt", "appt": "apt", "suite": "ste",
    "floor": "fl", "flr": "fl", "building": "bldg", "near": "nr", "opposite": "opp",
    "behind": "bh", "sector": "sec", "nagar": "ngr", "colony": "col", "village": "vill",
    "district": "dist", "post": "po", "mountain": "mtn", "center": "ctr", "centre": "ctr",
    "heights": "hts", "junction": "jct", "expressway": "expy", "pike": "pk", "crescent": "cres",
    "rue": "r", "route": "rte", "chemin": "ch", "chem": "ch", "impasse": "imp", "allee": "all",
    "quai": "qu", "cours": "crs", "residence": "res", "faubourg": "fbg",
}
ADDR_DROP = {"n", "a", "na", "no", "hn", "h", "unit", "pmb", "apt", "ste", "fl", "bis",
             "ter", "c", "o", "s", "do", "house", "number", "num", "plot", "shop", "flat",
             "door", "nr", "opp", "bh", "ward", "the", "of", "de", "du", "la", "le", "les",
             "des", "d", "l", "et", "and", "at", "in", "po"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "maharashtra": "mh", "delhi": "dl", "new delhi": "dl", "uttar pradesh": "up",
    "karnataka": "ka", "tamil nadu": "tn", "gujarat": "gj", "west bengal": "wb",
    "telangana": "tg", "ts": "tg", "haryana": "hr", "rajasthan": "rj", "kerala": "kl",
    "bihar": "br", "madhya pradesh": "mp", "andhra pradesh": "ap", "punjab": "pb",
    "odisha": "od", "orissa": "od", "or": "od", "jharkhand": "jh", "chhattisgarh": "cg",
    "ct": "cg", "uttarakhand": "uk", "ut": "uk", "himachal pradesh": "hp", "goa": "ga",
    "assam": "as", "jammu and kashmir": "jk", "jammu kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "tripura": "tr", "meghalaya": "ml",
    "manipur": "mn", "nagaland": "nl", "mizoram": "mz", "sikkim": "sk",
    "arunachal pradesh": "ar",
    # native-script state names seen in the data
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka",
    "தமிழ்நாடு": "tn", "ગુજરાત": "gj", "পশ্চিমবঙ্গ": "wb", "తెలంగాణ": "tg",
    "हरियाणा": "hr", "राजस्थान": "rj", "കേരളം": "kl", "बिहार": "br",
    "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "od",
}
# state codes map to themselves
US_STATES.update({v: v for v in list(US_STATES.values())})
IN_STATES.update({v: v for v in list(IN_STATES.values()) if v not in US_STATES})
WORD_NUM = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
            "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
            "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
            "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
            "nineteenth": "19", "twentieth": "20", "thirtieth": "30", "fortieth": "40",
            "fiftieth": "50", "sixtieth": "60", "seventieth": "70", "eightieth": "80",
            "ninetieth": "90", "hundredth": "100", "one": "1", "two": "2", "three": "3"}
# a number right after one of these tokens is a unit / box / floor number, not the house number
UNIT_MARKERS = {"unit", "apt", "apartment", "suite", "ste", "fl", "flr", "floor", "pmb", "box",
                "bldg", "building", "room", "rm", "desk", "office", "offce", "appt"}

# ----------------------------------------------------------------------------- regexes
RE_INDIC = re.compile(r"[ऀ-෿]")
RE_NONLATIN = re.compile(r"[^\x00-ɏḀ-ỿ -⁯₠-⃏]")
RE_PHONE = re.compile(r"\+?\d[\d\-\s]{8,}\d")
RE_IDTAG = re.compile(r"\(\s*(id|ref|reg|no)\s*[:#.]?\s*[\w\-]+\s*\)", re.I)
RE_URL = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|us|co)\b", re.I)
RE_NONALNUM = re.compile(r"[^a-z0-9]+")
RE_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b)?|\b(?:[a-z]\.)(?:[a-z]\b)")  # s.a.s.u. / l.l.c / s.a
RE_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|e|eme|ème)\b")
# BER_ORD_STREET=1: a numbered street ("18th Street", "Eighth St", "5th Cross") keeps its number as an
# address token (letters only, so digit/letter splitting leaves it intact) instead of becoming a
# house-number candidate; otherwise every "Nth Street" collapses to "st" and "87TH STREET" reads as
# house number 87.
ORD_STREET = os.environ.get("BER_ORD_STREET", "0") == "1"
_STYPES = (r"(?:street|st|avenue|ave|av|road|rd|cross|main|place|pl|drive|dr|lane|ln|terrace|ter|court|ct|way|blvd|"
           r"boulevard|parkway|pkwy|block|sector|phase|stage|circle|cir|trail|trl|highway|hwy)\b")
RE_ORD_STREET = re.compile(r"\b0*(\d+)(?:st|nd|rd|th)\s+(?=" + _STYPES + ")")
_SPELLED = [k for k in WORD_NUM if k not in ("one", "two", "three")]
RE_SPELL_STREET = re.compile(r"\b(" + "|".join(_SPELLED) + r")\s+(?=" + _STYPES + ")")


def ord_token(n):
    return "ord" + "".join("abcdefghij"[int(c)] for c in n)
RE_DIGALPHA = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
RE_SPACES = re.compile(r"\s+")
RE_SQUASH = re.compile(r"(.)\1+")

_DICT = None


def load_indic_dict(path=None):
    """Load the learned Indic-token -> Latin-token dictionary (empty if not built yet)."""
    global _DICT
    if path is None:
        from config import work_path
        path = work_path("indic_dict.json")
    _DICT = json.load(open(path)) if os.path.exists(path) else {}
    return _DICT


OCR_MAP = str.maketrans("0123456789", "olzeasgtbg")
RE_ORDINAL_TOK = re.compile(r"^\d+(st|nd|rd|th)$")


def deocr(tok):
    """Repair OCR-style digits inside word tokens: 'c0mpany' -> 'company', '5chool' -> 'school',
    'consu1tants' -> 'consultants'. Pure numbers, ordinals ('2nd') and short tokens are kept."""
    if tok.isdigit() or tok.isalpha() or len(tok) < 3 or RE_ORDINAL_TOK.match(tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    if letters < 2 or letters <= len(tok) - letters:  # as many digits as letters ('24hr', 'b52'): keep
        return tok
    return tok.translate(OCR_MAP)


def squash(s):
    """Collapse repeated letters: 'limittedd' -> 'limited'. Used for fuzzy transliteration."""
    return RE_SQUASH.sub(r"\1", s)


def indic_token(tok):
    """Latinize a single Indic-script token via learned dictionary, else unidecode."""
    if _DICT is None:
        load_indic_dict()
    t = tok.strip(".,()[]-")
    hit = _DICT.get(t)
    if hit is not None:
        return hit
    return squash(unidecode(t).lower())


def latinize(s):
    """Convert any mix of scripts to lowercase ASCII."""
    s = unicodedata.normalize("NFKC", s)
    if RE_INDIC.search(s):
        s = " ".join(indic_token(t) if RE_INDIC.search(t) else t for t in s.split())
    return unidecode(s).lower()


# ----------------------------------------------------------------------------- names
def norm_name(raw):
    """Return (full_tokens, core_tokens, domain) for a business name.

    full_tokens: all normalized tokens in order.
    core_tokens: tokens minus legal forms / noise, de-duplicated, order preserved.
    domain:      the website body if the name contains a URL ('' otherwise).
    """
    s = raw or ""
    s = RE_PHONE.sub(" ", s)
    s = RE_IDTAG.sub(" ", s)
    s = latinize(s)
    domain = ""
    m = RE_URL.search(s)
    if m:
        domain = m.group(1).replace("-", "")
        # a pure-website name ('shcozy.com') becomes its domain body as a single token
        s = RE_URL.sub(" " + domain + " ", s)
    # dotted acronyms stay one token: 'S.A.S.U.' -> 'sasu', 'L.L.C.' -> 'llc'
    s = RE_DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = RE_NONALNUM.sub(" ", s)
    toks = [deocr(t) for t in s.split()]
    full = [t for t in toks if t not in NOISE]
    seen = set()
    core = []
    for t in full:
        if t in LEGAL or t in seen:
            continue
        seen.add(t)
        core.append(t)
    return full, core, domain


# ----------------------------------------------------------------------------- addresses
def _canon_state(comp):
    c = comp.strip().lower()
    if c in US_STATES:
        return US_STATES[c]
    if c in IN_STATES:
        return IN_STATES[c]
    return None


def norm_address(raw):
    """Return (tokens, numbers, state, house_number) for an address.

    tokens:  canonicalized alpha tokens (street types abbreviated, noise dropped).
    numbers: numeric tokens (leading zeros stripped, ordinals reduced to digits); house /
             premise numbers first, unit / box / floor numbers (after 'Unit', 'PMB', 'Apt',
             'Suite', 'Floor', ... or before 'floor') last.
    state:   canonical state code if a comma-component is a known state, else ''.
    house_number: the first non-unit number ('' when the address has none).
    """
    if not raw:
        return [], [], "", ""
    state = ""
    comps = []
    for comp in raw.split(","):
        c = comp.strip()
        if not c:
            continue
        st = _canon_state(c)
        if st is None and RE_INDIC.search(c):
            st = IN_STATES.get(c)
        if st is not None and (len(c) > 2 or c.isupper()):
            state = st
            continue
        comps.append(c)
    s = latinize(" , ".join(comps).replace("°", " "))
    s = s.replace("n/a", " ").replace("&", " and ")
    if ORD_STREET:
        s = RE_ORD_STREET.sub(lambda m: ord_token(m.group(1)) + " ", s)
        s = RE_SPELL_STREET.sub(lambda m: ord_token(WORD_NUM[m.group(1)]) + " ", s)
    s = RE_ORD.sub(r"\1", s)
    s = RE_NONALNUM.sub(" ", s)
    s = RE_DIGALPHA.sub(" ", s)
    toks, nums, units = [], [], []
    raw_toks = s.split()
    for i, t in enumerate(raw_toks):
        t = WORD_NUM.get(t, t)
        if t.isdigit():
            n = t.lstrip("0") or "0"
            prev = raw_toks[i - 1] if i else ""
            nxt = raw_toks[i + 1] if i + 1 < len(raw_toks) else ""
            (units if prev in UNIT_MARKERS or nxt in ("floor", "fl", "flr") else nums).append(n)
            continue
        t = STREET.get(t, t)
        if t in ADDR_DROP or len(t) == 1:
            continue
        toks.append(t)
    return toks, nums + units, state, (nums[0] if nums else "")


def skeleton(tok):
    """Consonant skeleton used as a typo/transliteration-tolerant key."""
    t = squash(tok)
    if len(t) <= 1:
        return t
    return t[0] + re.sub(r"[aeiouyhw]", "", t[1:])
