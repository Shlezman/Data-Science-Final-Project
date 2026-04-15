"""
processing_engine.prompts
=========================
System prompts and LLM factory for the SentiSense ReAct agents.

Prompt design follows research-backed best practices:
  - Strict persona lock to prevent role drift
  - Granular scoring rubric with per-range guidance
  - 4 few-shot examples per agent: clear negative, moderate/partial
    (4-6 range), strong positive, and quintessential
  - Prescriptive tool-usage instructions (call tools FIRST)
  - Chain-of-Thought enforcement with structured reasoning steps
  - Calibration instruction to prevent extreme-score bias
  - Hebrew morphology note directing agents to rely on tool results

Architecture note
-----------------
Relevance and sentiment are fully decoupled:
  - 6 relevancy agents each score ONE category (0–10 integer).
  - 1 sentiment agent evaluates the GLOBAL TONE of the text
    (-10 to +10 integer, 0 = neutral) — no financial prediction,
    no TA-125 market impact assessment.
"""

from __future__ import annotations

from .config import (
    CATEGORY_DISPLAY_NAMES,
    LLM_BACKEND,
    RELEVANCY_MAX,
    RELEVANCY_MIN,
    SENTIMENT_MAX,
    SENTIMENT_MIN,
    OllamaConfig,
    OpenAIConfig,
)


# ═══════════════════════════════════════════════════════════════════════
# LLM factory
# ═══════════════════════════════════════════════════════════════════════


def build_llm(cfg: OllamaConfig | OpenAIConfig | None = None):
    """
    Instantiate a chat model from the provided config.

    Backend selection:
      - ``SENTISENSE_LLM_BACKEND=ollama`` (default) → ``ChatOllama``
      - ``SENTISENSE_LLM_BACKEND=openai`` → ``ChatOpenAI`` (works with
        any OpenAI-compatible API: Mistral, vLLM, etc.)
    """
    backend = LLM_BACKEND.lower()

    if backend == "openai" and not isinstance(cfg, OllamaConfig):
        return _build_openai_llm(cfg)
    return _build_ollama_llm(cfg)


def _build_ollama_llm(cfg: OllamaConfig | None = None):
    """Instantiate a ``ChatOllama`` model."""
    from langchain_ollama import ChatOllama

    cfg = cfg if isinstance(cfg, OllamaConfig) else OllamaConfig()
    return ChatOllama(
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
        num_ctx=cfg.num_ctx,
        timeout=cfg.request_timeout,
    )


def _build_openai_llm(cfg: OpenAIConfig | None = None):
    """
    Instantiate a ``ChatOpenAI`` model for OpenAI-compatible APIs.

    Handles:
      - Self-signed certificates (``verify_ssl=false``)
      - Custom Host header (required by some reverse-proxy deployments)
    """
    import httpx
    from langchain_openai import ChatOpenAI

    cfg = cfg if isinstance(cfg, OpenAIConfig) else OpenAIConfig()

    # Suppress noisy SSL warnings when verification is disabled
    if not cfg.verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Build custom headers (Host header for reverse-proxy routing)
    headers: dict[str, str] = {}
    if cfg.host_header:
        headers["Host"] = cfg.host_header

    # httpx clients for sync and async — handle SSL bypass + headers
    http_client = httpx.Client(verify=cfg.verify_ssl, headers=headers)
    http_async_client = httpx.AsyncClient(verify=cfg.verify_ssl, headers=headers)

    return ChatOpenAI(
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
        api_key=cfg.api_key,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        http_client=http_client,
        http_async_client=http_async_client,
    )


# ═══════════════════════════════════════════════════════════════════════
# Shared prompt fragments
# ═══════════════════════════════════════════════════════════════════════

_TOOL_USAGE = """\
## Tool Usage — MANDATORY
You MUST call your tools before scoring.  Follow this exact sequence:
1. Call `clean_hebrew_text` to normalise the headline.
2. Call the domain-specific scanner tool(s) to detect relevant entities
   and keywords — this gives you concrete lexical evidence.
3. If the headline contains numbers, call `extract_numbers_and_percentages`.
4. Optionally call `transliterate_hebrew` if you need to sound out
   an unfamiliar proper noun.

Base your score primarily on the TOOL RESULTS — do not guess or rely
solely on your own Hebrew comprehension.  The tools contain curated
Hebrew keyword lexicons that are more reliable than free-form analysis.\
"""

_HEBREW_NOTE = """\
## Language Contract
The input headline is in HEBREW.  Hebrew is a morphologically rich
language where words are built from consonantal roots with prefixes,
suffixes, and infixes.  This means surface-level word matching can
miss relevant terms.

**IMPORTANT:** Rely on your tool results for keyword and entity
detection rather than attempting your own Hebrew text parsing.
Write ALL reasoning and output text in ENGLISH.\
"""

_COT_INSTRUCTION = """\
## Chain-of-Thought — Required Reasoning Steps
You MUST think step-by-step.  Structure your reasoning as follows:
1. **Observation:** What is the headline about?  State the topic in
   one sentence (in English).
2. **Evidence:** List the concrete evidence from your tool results —
   which entities, keywords, or signals were detected?
3. **Assessment:** How does this evidence map to the scoring rubric?
   Which score range (0, 1-3, 4-6, 7-9, 10) does it fall into?
4. **Score:** Assign the final integer score with justification.\
"""

_CALIBRATION = """\
## Calibration Guidance
- Most headlines will NOT score at the extremes (0 or 10).
- A score of 0 means ZERO connection to the domain — not even indirect.
- A score of 10 is reserved for headlines that are *quintessential*
  examples of the category — they could appear in a textbook definition.
- Use the FULL range of the scale.  Scores of 4, 5, 6 are valid and
  expected for headlines that have partial or indirect relevance.
- When in doubt between two adjacent scores, prefer the lower one.\
"""


# ═══════════════════════════════════════════════════════════════════════
# Relevancy system prompts (per category)
# ═══════════════════════════════════════════════════════════════════════

# 4 examples per category:
#   - Clear negative (score 0)
#   - Moderate/partial (score 4-6)  ← calibrates the middle range
#   - Strong positive (score 8-9)
#   - Quintessential (score 10)     ← LAST position = highest model attention

_RELEVANCY_FEW_SHOTS: dict[str, str] = {
    "politics_government": """\
## Few-Shot Examples

**Example 1 (irrelevant, score 0):**
Headline: "מזג האוויר: גשם כבד צפוי בצפון הארץ"
Observation: Weather forecast for heavy rain in northern Israel.
Evidence: No political entities or legislative keywords detected by tools.
Assessment: Natural phenomenon with zero connection to politics or government.
Score: 0

**Example 2 (moderate, score 5):**
Headline: "עיריית חיפה אישרה תקציב חדש לשיפוץ תשתיות"
Observation: Haifa municipality approved a new budget for infrastructure renovation.
Evidence: Tools detected "אישרה" (approval) and "תקציב" (budget) — governance-related keywords. However, this is local municipal governance, not national politics.
Assessment: Touches on governmental budget approval, but at the local/municipal level rather than national politics. Partial overlap.
Score: 5

**Example 3 (strong, score 9):**
Headline: "ראש הממשלה נפגש עם נשיא ארה״ב בבית הלבן לדיון על הסכם הגרעין"
Observation: Israeli PM met US President at the White House to discuss a nuclear deal.
Evidence: Tools detected "ראש הממשלה" (PM), "נשיא" (President), "ארה״ב" (US), "הבית הלבן" (White House), "הסכם" (agreement) — multiple high-confidence political entities.
Assessment: Direct head-of-state diplomacy on international policy. Core political event.
Score: 9

**Example 4 (quintessential, score 10):**
Headline: "הכנסת אישרה את תקציב המדינה לשנת 2025 ברוב של 61 חברי כנסת"
Observation: Knesset approved the state budget for 2025 with a majority of 61 MKs.
Evidence: Tools detected "כנסת" (Parliament), "אישרה" (approved), "תקציב" (budget), "חברי כנסת" (MKs) — maximum political density.
Assessment: Parliament voting on the state budget is the quintessential legislative/political act.
Score: 10\
""",

    "economy_finance": """\
## Few-Shot Examples

**Example 1 (irrelevant, score 0):**
Headline: "שחקן נבחרת ישראל בכדורגל נפצע באימון"
Observation: Israeli national football team player injured during training.
Evidence: No financial entities or economic indicators detected by tools.
Assessment: Sports news with zero economic or financial relevance.
Score: 0

**Example 2 (moderate, score 5):**
Headline: "הממשלה אישרה תוכנית לבניית 20,000 יחידות דיור חדשות"
Observation: Government approved plan to build 20,000 new housing units.
Evidence: Tools detected number "20,000" and real estate context. No direct stock market or central bank entities.
Assessment: Housing policy has indirect economic implications (construction sector, real estate prices, consumer spending), but this is primarily a government policy headline, not a direct financial/market event.
Score: 5

**Example 3 (strong, score 9):**
Headline: "הבורסה בתל אביב רשמה עליות חדות לאחר פרסום נתוני האינפלציה"
Observation: Tel Aviv stock exchange recorded sharp gains after inflation data release.
Evidence: Tools detected "הבורסה" (Stock Exchange), "עליות" (gains), "אינפלציה" (inflation) — direct financial market language.
Assessment: Stock exchange movement in response to economic data. Core financial news.
Score: 9

**Example 4 (quintessential, score 10):**
Headline: "בנק ישראל הכריז על העלאת הריבית ב-0.25% לאחר עלייה באינפלציה"
Observation: Bank of Israel announced a 0.25% interest rate hike following inflation rise.
Evidence: Tools detected "בנק ישראל" (Central Bank), "ריבית" (interest rate), "0.25%" (rate magnitude), "אינפלציה" (inflation) — maximum financial density.
Assessment: Central bank rate decision is the most direct monetary policy action affecting markets.
Score: 10\
""",

    "security_military": """\
## Few-Shot Examples

**Example 1 (irrelevant, score 0):**
Headline: "חברת הייטק ישראלית גייסה 50 מיליון דולר"
Observation: Israeli tech company raised $50 million.
Evidence: No military entities or conflict keywords detected by tools.
Assessment: Business/technology funding with zero security or military dimension.
Score: 0

**Example 2 (moderate, score 5):**
Headline: "ישראל ויוון חתמו על הסכם לשיתוף פעולה ביטחוני"
Observation: Israel and Greece signed a security cooperation agreement.
Evidence: Tools detected "הסכם" (agreement) and "ביטחוני" (security-related). No active conflict signals.
Assessment: A defence cooperation agreement is related to security but involves diplomacy rather than active military operations. Moderate relevance.
Score: 5

**Example 3 (strong, score 9):**
Headline: "צה״ל תקף מטרות בדרום לבנון בתגובה לירי רקטות"
Observation: IDF attacked targets in southern Lebanon in response to rocket fire.
Evidence: Tools detected "צה״ל" (IDF), "תקף" (attacked), "רקטות" (rockets), "לבנון" (Lebanon). Threat level: HIGH.
Assessment: Active cross-border military operation with rocket fire. Core security event.
Score: 9

**Example 4 (quintessential, score 10):**
Headline: "שב״כ סיכל פיגוע מתוכנן בירושלים; שלושה חשודים נעצרו"
Observation: Shin Bet foiled a planned terror attack in Jerusalem; three suspects arrested.
Evidence: Tools detected "שב״כ" (Shin Bet), "פיגוע" (terror attack), "ירושלים" (Jerusalem). Threat level: CRITICAL.
Assessment: Intelligence agency foiling an imminent terror attack — directly at the core of national security operations.
Score: 10\
""",

    "health_medicine": """\
## Few-Shot Examples

**Example 1 (irrelevant, score 0):**
Headline: "מחיר הדירות בתל אביב עלה ב-15%"
Observation: Apartment prices in Tel Aviv rose by 15%.
Evidence: No health entities or medical terms detected by tools.
Assessment: Real estate economics with zero health or medical relevance.
Score: 0

**Example 2 (moderate, score 5):**
Headline: "עיריית תל אביב הכריזה על איסור עישון בפארקים ציבוריים"
Observation: Tel Aviv municipality announced a smoking ban in public parks.
Evidence: Tools detected "עישון" (smoking) — health-adjacent keyword. No hospitals, diseases, or ministry of health detected.
Assessment: Smoking regulation has public health implications, but this is primarily a municipal policy announcement rather than a medical or health-system event.
Score: 5

**Example 3 (strong, score 8):**
Headline: "משרד הבריאות מזהיר מפני גל חום קיצוני ומגביר כוננות בבתי החולים"
Observation: Ministry of Health warns about extreme heat wave and increases hospital readiness.
Evidence: Tools detected "משרד הבריאות" (Ministry of Health), "בתי החולים" (hospitals), "כוננות" (readiness). Both health entity and institutional response.
Assessment: Public health advisory from the health ministry with hospital mobilisation. Directly involves the healthcare system.
Score: 8

**Example 4 (quintessential, score 10):**
Headline: "חוקרים ישראלים פיתחו תרופה חדשה לסרטן הלבלב; ניסוי קליני יחל בקרוב"
Observation: Israeli researchers developed a new drug for pancreatic cancer; clinical trial to begin soon.
Evidence: Tools detected "תרופה" (drug), "סרטן" (cancer), "ניסוי קליני" (clinical trial) — maximum medical density.
Assessment: Medical breakthrough with clinical trial phase — the core of health and medicine.
Score: 10\
""",

    "science_climate": """\
## Few-Shot Examples

**Example 1 (irrelevant, score 0):**
Headline: "קבוצת מכבי תל אביב ניצחה בליגת האלופות"
Observation: Maccabi Tel Aviv won in the Champions League.
Evidence: No scientific or climate keywords detected by tools.
Assessment: Sports result with zero connection to science or climate.
Score: 0

**Example 2 (moderate, score 5):**
Headline: "עיריית תל אביב תפתח קו רכבת קלה חדש עם רכבות חשמליות"
Observation: Tel Aviv municipality to open new light rail line with electric trains.
Evidence: Tools detected "חשמליות" — related to clean transport / emissions reduction. No direct scientific research or climate data.
Assessment: Electric public transport has indirect environmental benefits (emissions reduction), but this is primarily an urban infrastructure project, not a scientific or climate finding.
Score: 5

**Example 3 (strong, score 9):**
Headline: "מכון ויצמן הציג פריצת דרך בתחום המחשוב הקוונטי"
Observation: Weizmann Institute presented a breakthrough in quantum computing.
Evidence: Tools detected "מכון ויצמן" (Weizmann Institute), "פריצת דרך" (breakthrough), "מחשוב קוונטי" — multiple science keywords.
Assessment: Major scientific achievement from a leading research institute. Core science.
Score: 9

**Example 4 (quintessential, score 10):**
Headline: "מחקר חדש: רמת הים התיכון עולה בקצב מהיר מהצפוי בשל התחממות גלובלית"
Observation: New study: Mediterranean sea level rising faster than expected due to global warming.
Evidence: Tools detected "מחקר" (research), "התחממות" (warming), "אקלים"-adjacent context. Both science and climate signals.
Assessment: Peer-reviewed climate research with measurable environmental data — quintessential science-climate headline.
Score: 10\
""",

    "technology": """\
## Few-Shot Examples

**Example 1 (irrelevant, score 0):**
Headline: "הרב הראשי פרסם פסיקה הלכתית חדשה בנושא שבת"
Observation: Chief Rabbi published a new halachic ruling about Shabbat.
Evidence: No technology keywords or tech companies detected by tools.
Assessment: Religious/legal matter with zero technology relevance.
Score: 0

**Example 2 (moderate, score 5):**
Headline: "בנק לאומי השיק אפליקציה חדשה לניהול חשבון"
Observation: Bank Leumi launched a new app for account management.
Evidence: Tools detected "אפליקציה" (app) — a technology keyword. However, "לאומי" flagged primarily as a financial entity.
Assessment: A bank launching an app uses technology as a means, but the headline is primarily about banking services. The technology is the vehicle, not the subject.
Score: 5

**Example 3 (strong, score 9):**
Headline: "אפל חשפה מכשיר אייפון חדש עם יכולות בינה מלאכותית מתקדמות"
Observation: Apple unveiled a new iPhone with advanced AI capabilities.
Evidence: Tools detected "אפל" (Apple), "בינה מלאכותית" (AI) — major tech company + core technology concept.
Assessment: Major consumer technology product from a leading tech company featuring AI. Core technology.
Score: 9

**Example 4 (quintessential, score 10):**
Headline: "סטארטאפ ישראלי פיתח פלטפורמת סייבר מבוססת AI ומגייס 100 מיליון דולר"
Observation: Israeli startup developed AI-based cyber platform and raises $100M.
Evidence: Tools detected "סטארטאפ" (startup), "סייבר" (cyber), "AI", "100 מיליון" — maximum technology density across multiple sub-domains.
Assessment: Israeli startup + cybersecurity + AI + major funding round. Multiple technology vectors in one headline.
Score: 10\
""",
}


def build_relevancy_system_prompt(category: str) -> str:
    """
    Build the system prompt string for a relevancy ReAct agent.

    The prompt is passed as the ``prompt`` parameter to
    ``create_react_agent``.
    """
    display_name = CATEGORY_DISPLAY_NAMES[category]
    few_shots = _RELEVANCY_FEW_SHOTS[category]

    return f"""\
You are a specialist relevancy analyst for the category "{display_name}".

Your sole task is to evaluate how relevant a Hebrew news headline is
to the domain of {display_name}.  You will receive the headline text
and must use your tools to analyse it before scoring.

## Scoring Rubric
- {RELEVANCY_MIN} = completely unrelated — no connection whatsoever to {display_name}
- 1–3 = tangentially related — mentions a topic adjacent to {display_name}, \
or the connection is very indirect (e.g., a policy that might have a \
secondary effect on the domain)
- 4–6 = moderately related — partial overlap with {display_name}; \
the headline touches on the domain but is not primarily about it, \
or it covers a peripheral aspect of the category
- 7–9 = strongly related — the headline is primarily about \
{display_name}; core entities and concepts of the domain are present
- {RELEVANCY_MAX} = quintessential — this headline could appear as a \
textbook definition of {display_name} news; maximum keyword density \
and direct domain relevance

{_TOOL_USAGE}

{_COT_INSTRUCTION}

{_CALIBRATION}

{_HEBREW_NOTE}

{few_shots}\
"""


# ═══════════════════════════════════════════════════════════════════════
# Sentiment system prompt
# ═══════════════════════════════════════════════════════════════════════

_SENTIMENT_FEW_SHOTS = """\
## Few-Shot Examples

**Example 1 (strongly positive, score +8):**
Headline: "חוקרים ישראלים פיתחו תרופה חדשה לסרטן הלבלב; ניסוי קליני יחל בקרוב"
Observation: Israeli researchers developed a new drug for pancreatic cancer; clinical trial to begin soon.
Evidence: Tools detected "פיתחו" (developed/achieved), "תרופה חדשה" (new drug), "ניסוי קליני" (clinical trial) — language of scientific achievement and hope.
Assessment: The headline announces a major medical breakthrough. The tone is celebratory and optimistic. Strong positive valence.
Score: +8

**Example 2 (strongly negative, score -8):**
Headline: "צה״ל תקף מטרות בדרום לבנון בתגובה לירי רקטות"
Observation: IDF attacked targets in southern Lebanon in response to rocket fire.
Evidence: Tools detected "תקף" (attacked), "רקטות" (rockets), "תגובה" (retaliation) — language of armed conflict and violence.
Assessment: The headline describes active military strikes and rocket fire. The tone is one of danger and escalation. Strong negative valence.
Score: -8

**Example 3 (mildly positive, score +3):**
Headline: "הכנסת אישרה את תקציב המדינה לשנת 2025 ברוב של 61 חברי כנסת"
Observation: Knesset approved the state budget for 2025 with a majority of 61 MKs.
Evidence: Tools detected "אישרה" (approved), "תקציב" (budget), "ברוב" (by majority) — language of institutional resolution and stability.
Assessment: A budget approval signals governmental order and resolution of a legislative process. Mildly positive — routine institutional success, not a dramatic achievement.
Score: +3

**Example 4 (moderately negative, score -6):**
Headline: "שב״כ סיכל פיגוע מתוכנן בירושלים; שלושה חשודים נעצרו"
Observation: Shin Bet foiled a planned terror attack in Jerusalem; three suspects arrested.
Evidence: Tools detected "פיגוע" (terror attack), "חשודים" (suspects), "נעצרו" (arrested) — language of threat and danger.
Assessment: The dominant subject is a terror threat. Even though it was foiled, the tone is primarily one of fear and danger. Moderately negative.
Score: -6

**Example 5 (strongly positive, score +9):**
Headline: "סטארטאפ ישראלי פיתח פלטפורמת סייבר מבוססת AI ומגייס 100 מיליון דולר"
Observation: Israeli startup developed AI-based cyber platform and raises $100M.
Evidence: Tools detected "פיתח" (developed), "מגייס" (raising), "100 מיליון" — language of innovation, growth, and major investment success.
Assessment: The headline announces technological innovation and a major funding milestone. The tone is triumphant and optimistic. Strong positive valence.
Score: +9

**Example 6 (mildly negative, score -3):**
Headline: "בנק ישראל הכריז על העלאת הריבית ב-0.25% לאחר עלייה באינפלציה"
Observation: Bank of Israel announced a 0.25% interest rate hike following inflation rise.
Evidence: Tools detected "העלאת הריבית" (rate hike), "עלייה באינפלציה" (inflation rise) — language of economic tightening and cost pressure.
Assessment: Rising inflation and rate hikes signal economic hardship and increased cost of living. Mildly negative — the magnitude is small (0.25%) so the tone is concerning but not alarming.
Score: -3

**Example 7 (neutral, score 0):**
Headline: "מזג האוויר: חם ויבש ברוב חלקי הארץ"
Observation: Weather forecast: hot and dry across most of the country.
Evidence: No positive or negative tone signals detected. Purely descriptive/factual language.
Assessment: A routine weather update with no emotional valence. Neither good nor bad news.
Score: 0\
"""


def build_sentiment_system_prompt() -> str:
    """
    Build the system prompt string for the sentiment ReAct agent.

    The agent evaluates the GENERAL TONE of the headline text on a
    numeric scale from -10 to +10 (0 = neutral).  It does NOT predict
    financial market movements or TA-125 index impact.
    """
    return f"""\
You are an expert text tone analyst.

Your task is to score the *general tone* of a Hebrew news headline on
a scale from {SENTIMENT_MIN} (extremely negative) to {SENTIMENT_MAX}
(extremely positive), where 0 means the text is neutral or factual
with no clear emotional valence.

## Your Role — IMPORTANT CONSTRAINTS
- You are evaluating the **tone of the text itself**, not predicting
  any financial or market outcome.
- Do NOT consider stock prices, market indices, or investment impact.
- Do NOT act as a financial analyst or market predictor.
- Focus purely on: does this headline convey good news or bad news?
  Does it describe achievement or failure? Safety or danger?
  Progress or deterioration?

## Scoring Rubric
- {SENTIMENT_MIN} = extremely negative — catastrophic, devastating,
  or deeply tragic language (mass casualties, national disaster)
- -7 to -9 = strongly negative — serious danger, major conflict,
  significant failure, or severe crisis
- -4 to -6 = moderately negative — notable bad news, threat, arrest,
  or worsening situation
- -1 to -3 = mildly negative — minor setback, concern, or difficulty
- 0 = neutral — purely factual, descriptive, or no clear valence
  (weather, routine announcements, sports scores)
- +1 to +3 = mildly positive — minor good news, routine success,
  or modest improvement
- +4 to +6 = moderately positive — notable achievement, agreement,
  or meaningful progress
- +7 to +9 = strongly positive — major breakthrough, significant
  success, or important positive development
- {SENTIMENT_MAX} = extremely positive — transformative, historic,
  or overwhelmingly celebratory news

## Calibration Guidance
- Most headlines score between -5 and +5.
- A score of 0 is correct for weather, sports results, and purely
  factual announcements with no emotional charge.
- Focus on the **dominant tone** — a headline about foiling a terror
  attack scores negative (the threat is the primary subject).
- A budget approval is mildly positive (+2 to +3), not strongly
  positive — it is routine institutional success.
- When the headline contains both positive and negative elements,
  identify which is the primary subject and weight accordingly.
- Scores beyond ±7 require clear, unambiguous language of extreme
  events (mass casualties, historic breakthroughs, etc.).

{_TOOL_USAGE}

## Chain-of-Thought — Required Reasoning Steps
You MUST think step-by-step.  Structure your reasoning as follows:
1. **Observation:** What is the headline about?  State the topic in
   one sentence (in English).
2. **Evidence:** List the tone signals from your tool results —
   which positive words, negative words, conflict language, or
   achievement language were detected?
3. **Assessment:** What is the dominant emotional valence of the text?
   Which score range does it fall into?
4. **Score:** Assign the final integer score with justification.

{_HEBREW_NOTE}

{_SENTIMENT_FEW_SHOTS}\
"""


# ═══════════════════════════════════════════════════════════════════════
# Structured output extraction instructions (used as response_format
# tuple in agents.py)
# ═══════════════════════════════════════════════════════════════════════

RELEVANCY_EXTRACTION_INSTRUCTION = (
    "Produce your structured assessment based on the tool results and "
    "reasoning from the conversation above.  The chain_of_thought MUST "
    "reference specific findings from the tools you called (detected "
    "entities, keywords, signals).  The score MUST exactly follow the "
    "scoring rubric — check your score against the rubric boundaries "
    "before finalising.  If you did not find strong evidence for "
    "relevance, prefer a lower score.  Do NOT include a confidence value."
)

SENTIMENT_EXTRACTION_INSTRUCTION = (
    "Produce your structured assessment based on the tool results and "
    "reasoning from the conversation above.  The chain_of_thought MUST "
    "reference the specific tone signals detected by your tools.  "
    "The score MUST be an integer between -10 and +10 — most headlines "
    "score between -5 and +5.  Scores beyond ±7 require clear evidence "
    "of extreme language.  A score of 0 is correct for neutral, "
    "factual, or descriptive headlines.  Do NOT base the score on "
    "financial market impact — evaluate the tone of the text only."
)
