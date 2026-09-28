"""Hand-seeded variant tables (Layer 1). Only frequent, unambiguous forms live here;
data-mined variants (Layer 2) are learned from training pairs and merged at runtime."""

# token -> canonical legal-form token. Multi-token forms are handled by LEGAL_PHRASES first.
LEGAL = {
    "private": "pvt", "pvt": "pvt", "prv": "pvt", "pte": "pvt", "pvtltd": "pvt ltd",
    "limited": "ltd", "ltd": "ltd", "lmt": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cmpny": "co", "compagnie": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "gmbh": "gmbh",
}
LEGAL_PHRASES = [
    ("limited liability company", "llc"),
    ("limited liability partnership", "llp"),
    ("limited partnership", "lp"),
    ("public limited", "public ltd"),
    ("p ltd", "pvt ltd"),
]
LEGAL_FORMS = set(v for v in LEGAL.values() for v in v.split())

# Generic business words: kept in the name, but useful as a feature group.
ABBREV_NAME = {
    "tech": "technologies", "technology": "technologies", "intl": "international",
    "mfg": "manufacturing", "svcs": "services", "svc": "services", "service": "services",
    "bros": "brothers", "assoc": "associates", "hosp": "hospital", "mgmt": "management",
    "centre": "center", "ctr": "center", "dept": "department", "natl": "national",
    "st": "saint", "ste": "saint", "sainte": "saint", "snt": "saint",
    "univ": "university", "inst": "institute", "engg": "engineering", "infra": "infrastructure",
}

STREET = {
    "road": "rd", "rd": "rd", "street": "st", "st": "st", "str": "st",
    "avenue": "ave", "ave": "ave", "av": "ave", "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "boulevard": "blvd", "blvd": "blvd",
    "bd": "blvd", "bvd": "blvd", "place": "pl", "pl": "pl", "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "terrace": "ter",
    "square": "sq", "sq": "sq", "trail": "trl", "trl": "trl", "way": "way",
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "floor": "fl", "flr": "fl",
    "building": "bldg", "bldg": "bldg", "number": "no", "no": "no", "nr": "near", "near": "near",
    "opposite": "opp", "opp": "opp", "marg": "marg", "nagar": "nagar", "sector": "sec", "sec": "sec",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "rue": "rue", "r": "rue", "chemin": "chemin", "che": "chemin", "pla": "pl", "allee": "allee", "impasse": "imp", "imp": "imp",
    "route": "rte", "rte": "rte", "quai": "quai", "cours": "cours",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
# Multi-word regions are replaced as phrases (country-scoped in address.py).
REGION_PHRASES = {"US": US_STATES, "India": IN_STATES}

CITY_ALIASES = {
    "bengaluru": "bangalore", "mumbai": "mumbai", "bombay": "mumbai", "gurugram": "gurgaon",
    "chennai": "chennai", "madras": "chennai", "kolkata": "kolkata", "calcutta": "kolkata",
}

# Placeholder values that mean "missing" when they form a whole address component.
NULL_COMPONENTS = {"null", "<null>", "none", "n/a", "nil", "na", "-"}
