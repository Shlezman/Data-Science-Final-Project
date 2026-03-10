"""
processing_engine.tools
=======================
Domain-specific tools for the SentiSense ReAct agents.

Each agent in the pipeline is a ``create_react_agent`` that can choose
to call any of its bound tools before producing a final score.  Tools
are organised into:

* **Shared** — text processing utilities available to *every* agent.
* **Category-specific** — relevancy-domain tools (politics, economy, …).
* **Sentiment-specific** — financial / market signal detectors.

All tools are side-effect-free, deterministic, and operate on the
headline text locally (no network calls).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import regex  # enhanced Unicode support
from langchain_core.tools import tool


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SHARED TOOLS  — bound to every agent                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝


@tool
def clean_hebrew_text(text: str) -> str:
    """Clean and normalise a Hebrew text string for analysis.

    Performs Unicode NFC normalisation, strips Hebrew niqqud (vowel
    diacritics U+0591–U+05C7), and collapses whitespace.  Use this
    before analysing a headline to reduce noise.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u0591-\u05C7]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_HEBREW_TO_LATIN: dict[str, str] = {
    "א": "'", "ב": "b", "ג": "g", "ד": "d",
    "ה": "h", "ו": "v", "ז": "z", "ח": "ch",
    "ט": "t", "י": "y", "כ": "k", "ך": "k",
    "ל": "l", "מ": "m", "ם": "m", "נ": "n",
    "ן": "n", "ס": "s", "ע": "'", "פ": "p",
    "ף": "f", "צ": "ts", "ץ": "ts", "ק": "q",
    "ר": "r", "ש": "sh", "ת": "t",
}


@tool
def transliterate_hebrew(text: str) -> str:
    """Provide a rough Latin transliteration of Hebrew text.

    This is a simplified phonetic mapping useful for "sounding out"
    proper nouns or unfamiliar terms.  Non-Hebrew characters are
    passed through unchanged.
    """
    return "".join(_HEBREW_TO_LATIN.get(c, c) for c in text)


@tool
def count_headline_words(text: str) -> str:
    """Count words and Hebrew characters in the headline.

    Returns a summary string with word count, character count, and
    average word length.  Short headlines (< 5 words) are often
    breaking-news alerts; longer ones tend to carry more context.
    """
    words = text.split()
    hebrew_chars = sum(1 for c in text if "\u0590" <= c <= "\u05FF")
    avg_len = round(sum(len(w) for w in words) / max(len(words), 1), 1)
    return (
        f"Words: {len(words)} | "
        f"Hebrew characters: {hebrew_chars} | "
        f"Average word length: {avg_len}"
    )


_URGENCY_PATTERNS: list[tuple[str, str]] = [
    ("מבזק", "FLASH/BREAKING"),
    ("דחוף", "URGENT"),
    ("עכשיו", "NOW/HAPPENING"),
    ("פיגוע", "ATTACK"),
    ("אזעקה", "ALERT/SIREN"),
    ("חירום", "EMERGENCY"),
    ("פינוי", "EVACUATION"),
    ("ראשון לדווח", "FIRST TO REPORT"),
    ("בזמן אמת", "REAL-TIME"),
]


@tool
def detect_urgency_markers(text: str) -> str:
    """Detect urgency / breaking-news markers in a Hebrew headline.

    Scans for words like מבזק (flash), דחוף (urgent), חירום
    (emergency), etc.  Returns matched markers and their English
    translations, or "none" if no urgency markers found.
    """
    found = [(heb, eng) for heb, eng in _URGENCY_PATTERNS if heb in text]
    if not found:
        return "No urgency markers detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Urgency markers found ({len(found)}):\n" + "\n".join(lines)


@tool
def extract_quoted_text(text: str) -> str:
    """Extract quoted speech or phrases from the headline.

    Finds text enclosed in Hebrew quotation marks (״...״), standard
    double quotes, or guillemets (« »).  Quoted speech often carries
    strong sentiment or indicates official statements.
    """
    patterns = [
        r'[״"]([^״"]+)[״"]',
        r'"([^"]+)"',
        r"«([^»]+)»",
    ]
    quotes: list[str] = []
    for pat in patterns:
        quotes.extend(re.findall(pat, text))
    if not quotes:
        return "No quoted text found."
    return "Quoted text found:\n" + "\n".join(f'  - "{q}"' for q in quotes)


@tool
def extract_numbers_and_percentages(text: str) -> str:
    """Extract numeric values, percentages, and currency amounts.

    Scans for patterns like "0.25%", "50 מיליון", "$3.2B", "שקל",
    etc.  Numeric data in headlines often signals economic or
    statistical significance.
    """
    results: list[str] = []

    # Percentages
    pcts = regex.findall(r"[\d,.]+\s*%", text)
    if pcts:
        results.append(f"Percentages: {', '.join(pcts)}")

    # Currency amounts
    currency_pats = [
        (r"\$[\d,.]+\s*(?:מיליון|מיליארד|אלף|billion|million|B|M|K)?", "USD"),
        (r"[\d,.]+\s*(?:שקל|שקלים|₪|ש\"ח)", "ILS"),
        (r"[\d,.]+\s*(?:יורו|€|EUR)", "EUR"),
        (r"[\d,.]+\s*(?:דולר|דולרים)", "USD"),
    ]
    for pat, curr in currency_pats:
        matches = regex.findall(pat, text)
        if matches:
            results.append(f"{curr}: {', '.join(matches)}")

    # Large numbers with Hebrew magnitude words
    magnitudes = regex.findall(
        r"[\d,.]+\s*(?:מיליון|מיליארד|אלף|טריליון)", text
    )
    if magnitudes:
        results.append(f"Large numbers: {', '.join(magnitudes)}")

    # Plain numbers (standalone)
    plain = regex.findall(r"(?<!\S)[\d,.]{2,}(?!\S)", text)
    if plain:
        results.append(f"Other numbers: {', '.join(plain)}")

    if not results:
        return "No numeric values found."
    return "Numeric data extracted:\n" + "\n".join(f"  - {r}" for r in results)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  POLITICS & GOVERNMENT TOOLS                                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_POLITICAL_ENTITIES: dict[str, str] = {
    # Government bodies
    "כנסת": "Knesset (Parliament)",
    "ממשלה": "Government",
    "ראש הממשלה": "Prime Minister",
    "נשיא": "President",
    'בג"ץ': "Supreme Court (HCJ)",
    "בית המשפט העליון": "Supreme Court",
    "היועץ המשפטי": "Attorney General",
    "משרד המשפטים": "Ministry of Justice",
    "משרד החוץ": "Ministry of Foreign Affairs",
    "משרד הפנים": "Ministry of Interior",
    # Political roles
    "שר": "Minister",
    "סגן שר": "Deputy Minister",
    "חבר כנסת": 'MK (Member of Knesset)',
    'ח"כ': "MK (Member of Knesset)",
    "יושב ראש": "Chairman",
    "ועדה": "Committee",
    # Parties
    "ליכוד": "Likud",
    "יש עתיד": "Yesh Atid",
    "מחנה ממלכתי": "National Unity",
    "הציונות הדתית": "Religious Zionism",
    'ש"ס': "Shas",
    "יהדות התורה": "United Torah Judaism",
    "העבודה": "Labor",
    "מרצ": "Meretz",
    "הרשימה המשותפת": "Joint List",
    "ישראל ביתנו": "Yisrael Beiteinu",
    # International
    "או\"ם": "United Nations",
    "האיחוד האירופי": "European Union",
    "נאט\"ו": "NATO",
    "ארה\"ב": "United States",
    "הבית הלבן": "White House",
    "קונגרס": "Congress",
}

_LEGISLATIVE_KEYWORDS: list[tuple[str, str]] = [
    ("חקיקה", "legislation"),
    ("הצעת חוק", "bill/proposal"),
    ("חוק", "law"),
    ("תיקון", "amendment"),
    ("הצבעה", "vote"),
    ("אישור", "approval"),
    ("קואליציה", "coalition"),
    ("אופוזיציה", "opposition"),
    ("בחירות", "elections"),
    ("משאל עם", "referendum"),
    ("הפגנה", "protest/demonstration"),
    ("עצומה", "petition"),
    ("רפורמה", "reform"),
    ("מינוי", "appointment"),
    ("התפטרות", "resignation"),
    ("אי אמון", "no-confidence"),
    ("פיזור כנסת", "Knesset dissolution"),
]


@tool
def scan_political_entities(text: str) -> str:
    """Scan headline for political entities — politicians, parties,
    government bodies, and international political organisations.

    Returns each matched entity with its English translation.
    """
    found = [(heb, eng) for heb, eng in _POLITICAL_ENTITIES.items() if heb in text]
    if not found:
        return "No political entities detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Political entities ({len(found)}):\n" + "\n".join(lines)


@tool
def detect_legislative_activity(text: str) -> str:
    """Detect legislative and governmental activity keywords.

    Scans for terms like חקיקה (legislation), הצבעה (vote),
    קואליציה (coalition), בחירות (elections), etc.
    """
    found = [(heb, eng) for heb, eng in _LEGISLATIVE_KEYWORDS if heb in text]
    if not found:
        return "No legislative activity keywords detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Legislative keywords ({len(found)}):\n" + "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  ECONOMY & FINANCE TOOLS                                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_FINANCIAL_ENTITIES: dict[str, str] = {
    "בנק ישראל": "Bank of Israel (Central Bank)",
    "הבורסה": "Stock Exchange",
    "בורסה": "Stock Exchange",
    'ת"א 125': "TA-125 Index",
    'ת"א 35': "TA-35 Index",
    "רשות ניירות ערך": "Securities Authority",
    "רשות המסים": "Tax Authority",
    "משרד האוצר": "Ministry of Finance",
    "קרן המטבע": "IMF",
    "הבנק העולמי": "World Bank",
    "הפדרל ריזרב": "Federal Reserve",
    "פד": "Fed",
    "נגיד בנק ישראל": "Governor of Bank of Israel",
    "לאומי": "Bank Leumi",
    "הפועלים": "Bank Hapoalim",
    "דיסקונט": "Discount Bank",
    "מזרחי טפחות": "Mizrahi Tefahot",
}

_ECONOMIC_INDICATORS: list[tuple[str, str]] = [
    ("ריבית", "interest rate"),
    ("אינפלציה", "inflation"),
    ('תמ"ג', "GDP"),
    ("תוצר", "GDP/output"),
    ("צמיחה", "growth"),
    ("מיתון", "recession"),
    ("אבטלה", "unemployment"),
    ("תעסוקה", "employment"),
    ("יצוא", "exports"),
    ("יבוא", "imports"),
    ("גירעון", "deficit"),
    ("עודף", "surplus"),
    ("חוב", "debt"),
    ("מדד המחירים", "CPI"),
    ("יוקר המחיה", "cost of living"),
    ("שכר", "wages/salary"),
    ("מניות", "stocks/shares"),
    ('אג"ח', "bonds"),
    ("תשואה", "yield/return"),
    ("דיבידנד", "dividend"),
    ("הנפקה", "IPO/issuance"),
    ("מט\"ח", "foreign exchange"),
    ("שער", "exchange rate"),
    ("משכנתא", "mortgage"),
    ("נדל\"ן", "real estate"),
    ("השקעה", "investment"),
    ("הפרטה", "privatisation"),
    ("רגולציה", "regulation"),
    ("מונופול", "monopoly"),
]


@tool
def scan_financial_entities(text: str) -> str:
    """Scan for financial institutions, indices, and market entities.

    Detects mentions of Bank of Israel, stock exchanges, major banks,
    the TA-125/TA-35 indices, regulatory bodies, etc.
    """
    found = [(heb, eng) for heb, eng in _FINANCIAL_ENTITIES.items() if heb in text]
    if not found:
        return "No financial entities detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Financial entities ({len(found)}):\n" + "\n".join(lines)


@tool
def detect_economic_indicators(text: str) -> str:
    """Detect economic indicator keywords in the headline.

    Scans for terms like ריבית (interest rate), אינפלציה (inflation),
    צמיחה (growth), מיתון (recession), מניות (stocks), etc.
    """
    found = [(heb, eng) for heb, eng in _ECONOMIC_INDICATORS if heb in text]
    if not found:
        return "No economic indicator keywords detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Economic indicators ({len(found)}):\n" + "\n".join(lines)


@tool
def extract_economic_figures(text: str) -> str:
    """Extract financial figures with context — amounts, rate changes,
    and percentage movements.

    Identifies patterns like "העלאת הריבית ב-0.25%", "גייסה 50 מיליון",
    and labels them with their financial context (rate change, funding
    round, market movement, etc.).
    """
    results: list[str] = []

    # Rate changes
    rate = regex.findall(
        r"(?:העלא|הורד|שינוי|עדכון).*?(?:ריבית|מדד).*?([\d,.]+\s*%?)", text
    )
    if rate:
        results.append(f"Rate change: {', '.join(rate)}")

    # Market movements
    market = regex.findall(
        r"(?:עלי|ירד|נפל|זינק|קרס|עלות|ירידה).*?([\d,.]+\s*%)", text
    )
    if market:
        results.append(f"Market movement: {', '.join(market)}")

    # Fundraising / deals
    funds = regex.findall(
        r"(?:גייס|רכש|מכר|עסק).*?([\d,.]+\s*(?:מיליון|מיליארד|אלף)?)", text
    )
    if funds:
        results.append(f"Deal value: {', '.join(funds)}")

    if not results:
        return "No specific financial figures with context detected."
    return "Financial figures:\n" + "\n".join(f"  - {r}" for r in results)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SECURITY & MILITARY TOOLS                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_MILITARY_ENTITIES: dict[str, str] = {
    'צה"ל': "IDF (Israel Defense Forces)",
    "צהל": "IDF",
    "חיל האוויר": "Air Force",
    "חיל הים": "Navy",
    "חיל היבשה": "Ground Forces",
    'שב"כ': "Shin Bet (ISA)",
    "מוסד": "Mossad",
    "אמ\"ן": "Military Intelligence",
    "משטרה": "Police",
    'מג"ב': "Border Police",
    'זק"א': "ZAKA",
    'מד"א': "Magen David Adom",
    "כיפת ברזל": "Iron Dome",
    "חמאס": "Hamas",
    "חיזבאללה": "Hezbollah",
    "ג'יהאד איסלאמי": "Islamic Jihad",
    "דאעש": "ISIS/ISIL",
    "איראן": "Iran",
    "אירן": "Iran",
    "משמרות המהפכה": "IRGC",
    "חות'ים": "Houthis",
}

_CONFLICT_KEYWORDS: list[tuple[str, str]] = [
    ("מבצע", "operation"),
    ("תקיפה", "strike/attack"),
    ("הפצצה", "bombing"),
    ("ירי", "shooting/fire"),
    ("רקטות", "rockets"),
    ("טילים", "missiles"),
    ("רחפנים", "drones/UAVs"),
    ("פיגוע", "terror attack"),
    ("טרור", "terrorism"),
    ("חטיפה", "kidnapping"),
    ("הסלמה", "escalation"),
    ("הפסקת אש", "ceasefire"),
    ("עימות", "confrontation"),
    ("לחימה", "combat"),
    ("גבול", "border"),
    ("מנהרה", "tunnel"),
    ("חדירה", "infiltration"),
    ("יירוט", "interception"),
    ("כוננות", "alert/readiness"),
    ("גיוס", "draft/mobilisation"),
    ("נפגעים", "casualties"),
    ("חללים", "fallen soldiers"),
    ("פצועים", "wounded"),
    ("שבויים", "prisoners/captives"),
]


@tool
def scan_military_entities(text: str) -> str:
    """Scan for military organisations, armed groups, and defence
    systems.

    Detects IDF branches, intelligence agencies, militant groups
    (Hamas, Hezbollah, etc.), and defence systems (Iron Dome).
    """
    found = [(heb, eng) for heb, eng in _MILITARY_ENTITIES.items() if heb in text]
    if not found:
        return "No military entities detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Military entities ({len(found)}):\n" + "\n".join(lines)


@tool
def detect_conflict_signals(text: str) -> str:
    """Detect conflict and security-related keywords.

    Scans for terms like מבצע (operation), רקטות (rockets), הסלמה
    (escalation), הפסקת אש (ceasefire), נפגעים (casualties), etc.
    """
    found = [(heb, eng) for heb, eng in _CONFLICT_KEYWORDS if heb in text]
    if not found:
        return "No conflict-related keywords detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Conflict signals ({len(found)}):\n" + "\n".join(lines)


@tool
def assess_threat_level(text: str) -> str:
    """Classify the headline's security threat level based on keyword
    severity.

    Categories:
      CRITICAL — active attack, mass casualties, major operation
      HIGH     — escalation, rocket fire, military response
      MODERATE — border tension, alerts, military exercises
      LOW      — policy discussions, veteran affairs, defence budget
      NONE     — no security relevance detected
    """
    critical = {"פיגוע", "חללים", "הפצצה", "טבח", "פלישה"}
    high = {"רקטות", "טילים", "הסלמה", "תקיפה", "ירי", "חדירה", "יירוט"}
    moderate = {"מבצע", "כוננות", "גיוס", "גבול", "רחפנים", "מנהרה"}
    low = {"תקציב ביטחון", "ותיקי צבא", "תרגיל", "אימון"}

    words = set(text.split())
    headline_set = set()
    for w in words:
        headline_set.add(w)
    # Also check multi-word patterns against full text
    for kw in critical:
        if kw in text:
            return f"Threat level: CRITICAL (matched: '{kw}')"
    for kw in high:
        if kw in text:
            return f"Threat level: HIGH (matched: '{kw}')"
    for kw in moderate:
        if kw in text:
            return f"Threat level: MODERATE (matched: '{kw}')"
    for kw in low:
        if kw in text:
            return f"Threat level: LOW (matched: '{kw}')"
    return "Threat level: NONE — no security-related keywords detected."


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  HEALTH & MEDICINE TOOLS                                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_HEALTH_ENTITIES: dict[str, str] = {
    "משרד הבריאות": "Ministry of Health",
    "קופת חולים": "Health Fund (HMO)",
    "כללית": "Clalit Health",
    "מכבי": "Maccabi Health",
    "מאוחדת": "Meuhedet Health",
    "לאומית": "Leumit Health",
    "בית חולים": "Hospital",
    "WHO": "World Health Organization",
    "ארגון הבריאות": "WHO",
    'רמב"ם': "Rambam Hospital",
    "איכילוב": "Ichilov Hospital",
    "הדסה": "Hadassah Hospital",
    "שיבא": "Sheba Medical Center",
    "סורוקה": "Soroka Hospital",
}

_MEDICAL_KEYWORDS: list[tuple[str, str]] = [
    ("חיסון", "vaccine/vaccination"),
    ("מגפה", "pandemic/epidemic"),
    ("תרופה", "medicine/drug"),
    ("ניתוח", "surgery"),
    ("אבחון", "diagnosis"),
    ("טיפול", "treatment"),
    ("מחלה", "disease"),
    ("וירוס", "virus"),
    ("בקטריה", "bacteria"),
    ("סרטן", "cancer"),
    ("סוכרת", "diabetes"),
    ("קורונה", "COVID/coronavirus"),
    ("שפעת", "flu/influenza"),
    ("דלקת", "inflammation/infection"),
    ("אנטיביוטיקה", "antibiotics"),
    ("ניסוי קליני", "clinical trial"),
    ("תמותה", "mortality"),
    ("תחלואה", "morbidity"),
    ("אשפוז", "hospitalisation"),
    ("חדר מיון", "emergency room"),
    ("רופא", "doctor"),
    ("אחות", "nurse"),
    ("רפואה", "medicine (field)"),
    ("בריאות הנפש", "mental health"),
    ("שיקום", "rehabilitation"),
]


@tool
def scan_health_entities(text: str) -> str:
    """Scan for healthcare organisations, hospitals, and health funds.

    Detects Ministry of Health, major Israeli hospitals (Ichilov,
    Hadassah, Sheba, etc.), HMOs (Clalit, Maccabi), and WHO.
    """
    found = [(heb, eng) for heb, eng in _HEALTH_ENTITIES.items() if heb in text]
    if not found:
        return "No health entities detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Health entities ({len(found)}):\n" + "\n".join(lines)


@tool
def detect_medical_terms(text: str) -> str:
    """Detect medical and health-related terminology.

    Scans for terms like חיסון (vaccine), מגפה (pandemic), סרטן
    (cancer), ניסוי קליני (clinical trial), etc.
    """
    found = [(heb, eng) for heb, eng in _MEDICAL_KEYWORDS if heb in text]
    if not found:
        return "No medical terminology detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Medical terms ({len(found)}):\n" + "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SCIENCE & CLIMATE TOOLS                                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_SCIENTIFIC_KEYWORDS: list[tuple[str, str]] = [
    ("מחקר", "research/study"),
    ("מדענים", "scientists"),
    ("חוקרים", "researchers"),
    ("אוניברסיטה", "university"),
    ("מכון ויצמן", "Weizmann Institute"),
    ("טכניון", "Technion"),
    ("אקדמיה", "academia"),
    ("פרופסור", "professor"),
    ("דוקטור", "doctor (academic)"),
    ("מעבדה", "laboratory"),
    ("ניסוי", "experiment"),
    ("תגלית", "discovery"),
    ("פריצת דרך", "breakthrough"),
    ("פרס נובל", "Nobel Prize"),
    ("מאמר מדעי", "scientific paper"),
    ("גנום", "genome"),
    ("DNA", "DNA"),
    ("פיזיקה", "physics"),
    ("כימיה", "chemistry"),
    ("ביולוגיה", "biology"),
]

_CLIMATE_KEYWORDS: list[tuple[str, str]] = [
    ("אקלים", "climate"),
    ("התחממות", "warming"),
    ("פליטות", "emissions"),
    ("פחמן", "carbon"),
    ("גזי חממה", "greenhouse gases"),
    ("אנרגיה מתחדשת", "renewable energy"),
    ("סולארי", "solar"),
    ("רוח", "wind (energy)"),
    ("מיחזור", "recycling"),
    ("זיהום", "pollution"),
    ("סביבה", "environment"),
    ("בצורת", "drought"),
    ("שיטפון", "flood"),
    ("רעידת אדמה", "earthquake"),
    ("הכחדה", "extinction"),
    ("מגוון ביולוגי", "biodiversity"),
    ("קיימות", "sustainability"),
    ("COP", "COP (Climate Summit)"),
    ("הסכם פריז", "Paris Agreement"),
    ("ים המלח", "Dead Sea"),
    ("אלמוגים", "corals"),
]


@tool
def detect_scientific_terms(text: str) -> str:
    """Detect science and research-related keywords.

    Scans for terms like מחקר (research), מדענים (scientists), תגלית
    (discovery), אוניברסיטה (university), etc.
    """
    found = [(heb, eng) for heb, eng in _SCIENTIFIC_KEYWORDS if heb in text]
    if not found:
        return "No scientific keywords detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Scientific terms ({len(found)}):\n" + "\n".join(lines)


@tool
def detect_climate_indicators(text: str) -> str:
    """Detect climate, environment, and sustainability keywords.

    Scans for terms like אקלים (climate), פליטות (emissions), אנרגיה
    מתחדשת (renewable energy), קיימות (sustainability), etc.
    """
    found = [(heb, eng) for heb, eng in _CLIMATE_KEYWORDS if heb in text]
    if not found:
        return "No climate/environment keywords detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Climate indicators ({len(found)}):\n" + "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  TECHNOLOGY TOOLS                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_TECH_KEYWORDS: list[tuple[str, str]] = [
    ("הייטק", "hi-tech"),
    ("סטארטאפ", "startup"),
    ("בינה מלאכותית", "artificial intelligence"),
    ("AI", "AI"),
    ("סייבר", "cyber"),
    ("אפליקציה", "application/app"),
    ("טכנולוגיה", "technology"),
    ("דיגיטל", "digital"),
    ("ענן", "cloud"),
    ("רובוט", "robot"),
    ("אוטומציה", "automation"),
    ("בלוקצ'יין", "blockchain"),
    ("קריפטו", "crypto"),
    ("ביטקוין", "Bitcoin"),
    ("מחשוב קוונטי", "quantum computing"),
    ("למידת מכונה", "machine learning"),
    ("רשת עצבית", "neural network"),
    ("מציאות מדומה", "virtual reality"),
    ("מציאות רבודה", "augmented reality"),
    ("אינטרנט", "internet"),
    ("5G", "5G"),
    ("שבב", "chip/semiconductor"),
    ("מוליך למחצה", "semiconductor"),
    ("פינטק", "fintech"),
]

_TECH_COMPANIES: dict[str, str] = {
    "אפל": "Apple",
    "גוגל": "Google",
    "מיקרוסופט": "Microsoft",
    "מטא": "Meta",
    "אמזון": "Amazon",
    "אנבידיה": "NVIDIA",
    "אינטל": "Intel",
    'צ\'ק פוינט': "Check Point",
    "מובילאיי": "Mobileye",
    "וויקס": "Wix",
    "מאנדיי": "monday.com",
    "נייס": "NICE",
    "אלביט": "Elbit Systems",
    "טבע": "Teva Pharmaceutical",
    "סאפ": "SAP",
    "אורקל": "Oracle",
    "טסלה": "Tesla",
    "OpenAI": "OpenAI",
    "סמסונג": "Samsung",
    "וואווי": "Huawei",
}


@tool
def detect_tech_keywords(text: str) -> str:
    """Detect technology-related keywords and concepts.

    Scans for terms like בינה מלאכותית (AI), סייבר (cyber), סטארטאפ
    (startup), בלוקצ'יין (blockchain), שבב (chip), etc.
    """
    found = [(heb, eng) for heb, eng in _TECH_KEYWORDS if heb in text]
    if not found:
        return "No technology keywords detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Technology terms ({len(found)}):\n" + "\n".join(lines)


@tool
def scan_tech_companies(text: str) -> str:
    """Scan for major technology and hi-tech companies.

    Detects mentions of global tech giants (Apple, Google, NVIDIA)
    and prominent Israeli tech companies (Check Point, Mobileye,
    Wix, monday.com, etc.).
    """
    found = [(heb, eng) for heb, eng in _TECH_COMPANIES.items() if heb in text]
    if not found:
        return "No technology companies detected."
    lines = [f"  - '{heb}' → {eng}" for heb, eng in found]
    return f"Tech companies ({len(found)}):\n" + "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  SENTIMENT / MARKET SIGNAL TOOLS                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

_BULLISH_SIGNALS: list[tuple[str, str]] = [
    ("עליות", "gains/rises"),
    ("עלייה", "rise/increase"),
    ("זינוק", "surge/jump"),
    ("שיא", "record high"),
    ("צמיחה", "growth"),
    ("התאוששות", "recovery"),
    ("אופטימיות", "optimism"),
    ("שיפור", "improvement"),
    ("הסכם", "agreement/deal"),
    ("שלום", "peace"),
    ("הפחתת מס", "tax cut"),
    ("הקלה", "easing/relief"),
    ("גיוס הון", "fundraising"),
    ("רווח", "profit"),
    ("דיבידנד", "dividend"),
    ("ייצוב", "stabilisation"),
]

_BEARISH_SIGNALS: list[tuple[str, str]] = [
    ("ירידות", "declines/drops"),
    ("ירידה", "decline/drop"),
    ("נפילה", "fall/crash"),
    ("קריסה", "collapse"),
    ("מיתון", "recession"),
    ("משבר", "crisis"),
    ("אינפלציה", "inflation"),
    ("הסלמה", "escalation"),
    ("מלחמה", "war"),
    ("סנקציות", "sanctions"),
    ("העלאת מס", "tax hike"),
    ("פיטורים", "layoffs"),
    ("חדלות פירעון", "insolvency"),
    ("הפסד", "loss"),
    ("חוסר ודאות", "uncertainty"),
    ("פשיטת רגל", "bankruptcy"),
    ("אבטלה", "unemployment"),
    ("גירעון", "deficit"),
]


@tool
def detect_market_sentiment_signals(text: str) -> str:
    """Detect bullish and bearish market signal keywords.

    Classifies found keywords into BULLISH (positive market impact)
    and BEARISH (negative market impact) categories, and computes a
    raw signal balance.
    """
    bullish = [(heb, eng) for heb, eng in _BULLISH_SIGNALS if heb in text]
    bearish = [(heb, eng) for heb, eng in _BEARISH_SIGNALS if heb in text]

    if not bullish and not bearish:
        return "No market sentiment signals detected — headline appears market-neutral."

    lines: list[str] = []
    if bullish:
        lines.append(f"BULLISH signals ({len(bullish)}):")
        lines.extend(f"  + '{heb}' → {eng}" for heb, eng in bullish)
    if bearish:
        lines.append(f"BEARISH signals ({len(bearish)}):")
        lines.extend(f"  - '{heb}' → {eng}" for heb, eng in bearish)

    balance = len(bullish) - len(bearish)
    direction = "BULLISH" if balance > 0 else "BEARISH" if balance < 0 else "MIXED"
    lines.append(f"Raw signal balance: {balance:+d} ({direction})")
    return "\n".join(lines)


@tool
def assess_geopolitical_risk(text: str) -> str:
    """Assess geopolitical risk level from the headline.

    Evaluates whether the headline signals geopolitical instability
    (which is typically bearish for the TA-125) or stability/diplomacy
    (which is neutral-to-bullish).
    """
    instability = [
        ("מלחמה", "war"), ("הסלמה", "escalation"), ("טרור", "terrorism"),
        ("פיגוע", "attack"), ("סנקציות", "sanctions"), ("עימות", "confrontation"),
        ("גרעין", "nuclear"), ("טילים", "missiles"), ("רקטות", "rockets"),
        ("איום", "threat"), ("משבר", "crisis"), ("התנגשות", "clash"),
    ]
    stability = [
        ("הסכם שלום", "peace agreement"), ("שלום", "peace"),
        ("דיפלומטיה", "diplomacy"), ("משא ומתן", "negotiations"),
        ("הפסקת אש", "ceasefire"), ("נורמליזציה", "normalisation"),
        ("הסכם", "agreement"), ("שיתוף פעולה", "cooperation"),
    ]

    risk = [(heb, eng) for heb, eng in instability if heb in text]
    stab = [(heb, eng) for heb, eng in stability if heb in text]

    if not risk and not stab:
        return "No geopolitical signals detected."

    lines: list[str] = []
    if risk:
        lines.append(f"RISK/INSTABILITY signals ({len(risk)}):")
        lines.extend(f"  ⚠ '{heb}' → {eng}" for heb, eng in risk)
    if stab:
        lines.append(f"STABILITY signals ({len(stab)}):")
        lines.extend(f"  ✓ '{heb}' → {eng}" for heb, eng in stab)

    if risk and not stab:
        lines.append("Assessment: ELEVATED geopolitical risk (bearish pressure)")
    elif stab and not risk:
        lines.append("Assessment: POSITIVE diplomatic signal (bullish support)")
    else:
        lines.append("Assessment: MIXED geopolitical signals")
    return "\n".join(lines)


@tool
def detect_monetary_policy_signals(text: str) -> str:
    """Detect central bank and monetary policy signals.

    Identifies rate decisions, quantitative easing/tightening, and
    other monetary policy actions that directly impact market
    expectations.
    """
    dovish = [
        ("הורדת ריבית", "rate cut"),
        ("הקלה כמותית", "quantitative easing"),
        ("הרחבה מוניטרית", "monetary expansion"),
        ("הפחתה", "reduction"),
        ("הזרמה", "injection/liquidity"),
    ]
    hawkish = [
        ("העלאת ריבית", "rate hike"),
        ("הידוק מוניטרי", "monetary tightening"),
        ("צמצום", "contraction"),
        ("ריסון", "restraint"),
    ]

    d_found = [(heb, eng) for heb, eng in dovish if heb in text]
    h_found = [(heb, eng) for heb, eng in hawkish if heb in text]

    if not d_found and not h_found:
        return "No monetary policy signals detected."

    lines: list[str] = []
    if d_found:
        lines.append("DOVISH (expansionary) signals:")
        lines.extend(f"  ↓ '{heb}' → {eng}" for heb, eng in d_found)
    if h_found:
        lines.append("HAWKISH (contractionary) signals:")
        lines.extend(f"  ↑ '{heb}' → {eng}" for heb, eng in h_found)
    return "\n".join(lines)


@tool
def extract_impact_magnitude(text: str) -> str:
    """Estimate the magnitude of market impact from numeric data.

    Combines numeric extraction with directional keywords to assess
    whether the headline describes a small, moderate, or large market
    event.
    """
    # Find percentages
    pcts = regex.findall(r"([\d,.]+)\s*%", text)
    pct_vals = []
    for p in pcts:
        try:
            pct_vals.append(float(p.replace(",", "")))
        except ValueError:
            pass

    # Find magnitude words
    magnitude_words = {
        "חד": "sharp",
        "דרמטי": "dramatic",
        "היסטורי": "historic",
        "חריג": "unusual",
        "מפתיע": "surprising",
        "משמעותי": "significant",
        "קל": "slight",
        "מתון": "moderate",
        "שולי": "marginal",
    }
    found_mag = [(heb, eng) for heb, eng in magnitude_words.items() if heb in text]

    lines: list[str] = []
    if pct_vals:
        max_pct = max(pct_vals)
        if max_pct >= 5:
            lines.append(f"Percentage magnitude: LARGE ({max_pct}%)")
        elif max_pct >= 1:
            lines.append(f"Percentage magnitude: MODERATE ({max_pct}%)")
        else:
            lines.append(f"Percentage magnitude: SMALL ({max_pct}%)")

    if found_mag:
        lines.append("Magnitude descriptors: " + ", ".join(
            f"'{heb}' ({eng})" for heb, eng in found_mag
        ))

    if not lines:
        return "No quantifiable impact magnitude detected."
    return "Impact magnitude analysis:\n" + "\n".join(f"  - {l}" for l in lines)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  TOOL REGISTRIES  — used by agents.py to bind tools per agent        ║
# ╚═══════════════════════════════════════════════════════════════════════╝

SHARED_TOOLS = [
    clean_hebrew_text,
    transliterate_hebrew,
    count_headline_words,
    detect_urgency_markers,
    extract_quoted_text,
    extract_numbers_and_percentages,
]

TOOLS_BY_CATEGORY: dict[str, list] = {
    "politics_government": [
        scan_political_entities,
        detect_legislative_activity,
    ],
    "economy_finance": [
        scan_financial_entities,
        detect_economic_indicators,
        extract_economic_figures,
    ],
    "security_military": [
        scan_military_entities,
        detect_conflict_signals,
        assess_threat_level,
    ],
    "health_medicine": [
        scan_health_entities,
        detect_medical_terms,
    ],
    "science_climate": [
        detect_scientific_terms,
        detect_climate_indicators,
    ],
    "technology": [
        detect_tech_keywords,
        scan_tech_companies,
    ],
}

SENTIMENT_TOOLS = [
    detect_market_sentiment_signals,
    assess_geopolitical_risk,
    detect_monetary_policy_signals,
    extract_impact_magnitude,
    # Sentiment agent also gets financial entity/indicator detection
    scan_financial_entities,
    detect_economic_indicators,
]
