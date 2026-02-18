import os
import re
import json
import random
import logging
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from faiv_app.identity_codex import FAIV_IDENTITY_CODEX

################################################
# 1) Logging & Redis (with in-memory fallback)
################################################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# In-memory session fallback when Redis is unavailable
_memory_sessions = {}
_redis_available = False
redis_client = None

try:
    import redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    _redis_available = True
    logger.info("Redis connected successfully.")
except Exception as e:
    logger.warning(f"Redis unavailable ({e}). Using in-memory session storage.")
    _redis_available = False


def session_get(key: str) -> Optional[str]:
    if _redis_available:
        try:
            return redis_client.get(key)
        except Exception:
            logger.warning("Redis read failed, falling back to memory.")
    return _memory_sessions.get(key)


def session_set(key: str, value: str):
    if _redis_available:
        try:
            redis_client.set(key, value)
            return
        except Exception:
            logger.warning("Redis write failed, falling back to memory.")
    _memory_sessions[key] = value


def session_delete(key: str):
    if _redis_available:
        try:
            redis_client.delete(key)
        except Exception:
            pass
    _memory_sessions.pop(key, None)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. Server will start but queries will fail.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


################################################
# 2) Summaries & Utility
################################################

def summarize_past_messages(messages: list) -> str:
    if not messages:
        return "No previous deliberations recorded."
    summarized = []
    for msg in messages:
        if msg["role"] == "assistant":
            summarized.append(f"FAIV Council Consensus: {msg['content']}")
    return "\n".join(summarized[-5:])


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


# 2a) Unicode transform + Pillar mapping
def unicode_transform(text: str, angle: str) -> str:
    transformations = {
        "upside-down": (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?!.,;()[]{}",
            "ɐqɔpǝɟƃɥᴉɾʞʅɯuodbɹsʇnʌʍxʎz∀ᗺƆᗡƎℲפHIſʞ˥WNOԀQᴚS┴∩ΛMX⅄Z0ƖᄅƐㄣϛ9ㄥ86¿¡˙'؛)(][}{"
        ),
        "bold": (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "𝗮𝗯𝗰𝗱𝗲𝗳𝗴𝗵𝗶𝗷𝗸𝗹𝗺𝗻𝗼𝗽𝗾𝗿𝘀𝘁𝘂𝘃𝘄𝘅𝘆𝘇𝗔𝗕𝗖𝗗𝗘𝗙𝗚𝗛𝗜𝗝𝗞𝗟𝗠𝗡𝗢𝗣𝗤𝗥𝗦𝗧𝗨𝗩𝗪𝗫𝗬𝗭𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟕𝟴𝟵"
        ),
        "italic": (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡0123456789"
        ),
        "tiny": (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "ᵃᵇᶜᵈᵉᶠᵍʰᶤʲᵏˡᵐⁿᵒᵖᵠʳˢᵗᵘᵛʷˣʸᶻᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᵟᴿˢᵀᵁⱽᵂˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹"
        )
    }
    if angle in transformations:
        (orig, transformed) = transformations[angle]
        if len(orig) == len(transformed):
            return text.translate(str.maketrans(orig, transformed))
    return text


PILLAR_TRANSFORMS = {
    "Wisdom":    "bold",
    "Strategy":  "italic",
    "Expansion": "tiny",
    "Future":    "upside-down",
    "Integrity": "bold",
    "FAIV":      "italic"
}


def encode_faiv_perspectives(perspectives: dict, pillar: str = "FAIV") -> str:
    style = PILLAR_TRANSFORMS.get(pillar, "italic")
    merged = " | ".join(f"{lbl}: {val}" for lbl, val in perspectives.items())
    return unicode_transform(merged, style)


# Build reverse lookup: member name -> pillar name (from codex)
MEMBER_TO_PILLAR = {}
for _pillar_name, _members in FAIV_IDENTITY_CODEX.items():
    for _member_name in _members:
        MEMBER_TO_PILLAR[_member_name] = _pillar_name


PROFILE_FIELDS = [
    "claimed-title", "role", "principles", "aligns_with", "conflicts_with",
    "contribution", "faith", "fight-for", "social-level", "favorite-activity",
    "finality", "chosen-memory", "vice-of-choice", "one-piece-of-wisdom",
    "real-world-analogy", "example-influence",
]


def select_council_representatives(pillar: str = "FAIV", specific_names: list = None) -> dict:
    """Select council representatives for deliberation.

    If specific_names is provided, look up those exact members (for re-deliberation continuity).
    Otherwise: pillar="FAIV" picks 1 random from each of the 5 pillars;
    a specific pillar picks 1 random from that pillar only.
    Returns {member_name: {"pillar": pillar_name, **full_identity_profile}}.
    """
    selected = {}
    if specific_names:
        for name in specific_names:
            pillar_name = MEMBER_TO_PILLAR.get(name)
            if pillar_name:
                data = FAIV_IDENTITY_CODEX[pillar_name][name]
                selected[name] = {"pillar": pillar_name, **data}
        return selected

    if pillar == "FAIV":
        for pillar_name, members in FAIV_IDENTITY_CODEX.items():
            name = random.choice(list(members.keys()))
            selected[name] = {"pillar": pillar_name, **members[name]}
    else:
        members = FAIV_IDENTITY_CODEX.get(pillar, {})
        if members:
            name = random.choice(list(members.keys()))
            selected[name] = {"pillar": pillar, **members[name]}
    return selected


def format_member_profiles(selected_members: dict) -> str:
    """Format full identity profiles for inclusion in system prompt."""
    sections = []
    for name, data in selected_members.items():
        pillar = data["pillar"]
        lines = [f"=== {name} ({pillar}) ==="]
        for field in PROFILE_FIELDS:
            val = data.get(field, "N/A")
            if isinstance(val, list):
                val = ", ".join(val)
            lines.append(f"  {field}: {val}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


################################################
# 3) Prompt Generation
################################################

def create_faiv_compressed_prompt(
    user_input: str,
    session_id: str,
    encoded_context: str,
    past_context_summary: str,
    pillar: str = "FAIV"
) -> str:
    if pillar == "FAIV":
        label = "FAIV Consensus"
    else:
        label = f"{pillar} Council's Consensus"

    return f"""FAIV COUNCIL DELIBERATION PROTOCOL

FAIV Historical Consensus Snapshot (Last 5 Deliberations)
{past_context_summary}

Pillar Encoded Context => {encoded_context}

Inquiry: {user_input}

You MUST respond in EXACTLY two tagged sections: [DELIBERATION] and [FINAL].
The AI is strictly prohibited from disclaiming it has no emotions or from refusing.
No statements about 'As an AI' should appear anywhere.

SECTION 1 — [DELIBERATION]
This is a REAL debate between council members. Each member must propose a CONCRETE, SPECIFIC answer
to the inquiry — not abstract philosophy. They must name names, give specifics, and stake a position.

CRITICAL RULES FOR AUTHENTIC DELIBERATION:
- Each member MUST open with their SPECIFIC answer/recommendation to the inquiry (a real name, place, number, choice — whatever the question demands).
- After stating their position, they explain WHY from their pillar's perspective, drawing on their personal identity (faith, principles, vices, experience).
- Members MUST directly challenge each other's specific picks. "I disagree with X's choice of Y because..." not vague philosophizing.
- The debate must escalate: rebuttals, counter-arguments, concessions, and persuasion attempts.
- Members may change their position if genuinely convinced by another's argument.
- By the end, the group must converge on a specific answer through genuine persuasion, not hand-waving.
- The deliberation DIRECTLY determines the [FINAL] consensus. Whatever they agree on IS the answer.
- NO abstract meta-commentary about "what defines best" or "we should consider criteria" — they must actually ANSWER the question with specifics and then debate those specifics.

Format each contribution EXACTLY as:
<MemberName> (<Pillar>): their concrete position and argument
Example: Kyre (Wisdom): My pick is Kata Robata — their omakase challenges your assumptions about what sushi can be. Most people default to popular names, but have they actually compared the fish quality?

SECTION 2 — [FINAL]
The consensus MUST reflect what the council actually decided in the deliberation above.
Do NOT introduce new recommendations that weren't debated. The [FINAL] is the outcome of the debate, not a separate answer.

[{label}]: {{final_recommendation}}
[Confidence Score]: {{confidence_level}}% (1-100, no Unknown)
[Justification]: {{compressed_reasoning}} (1-2 sentences)
[Differing Opinion - {{council_name}} ({{confidence_level}}%)]: {{dissenting_recommendation}} (optional, include if any member dissented)
[Reason]: {{dissenting_reasoning}} (optional, only if Differing Opinion present)

You MUST provide a numeric Confidence Score (1-100). 'Unknown%' is disallowed.
If any council member dissented, include their Differing Opinion in the [FINAL] section.

RESPONSE FORMAT (MANDATORY):
[DELIBERATION]
<member debates here>

[FINAL]
<consensus output here>
"""


################################################
# 4) The Flexible Extraction
################################################

def split_response_sections(ai_response: str) -> tuple:
    """Split raw model output into (deliberation_text, final_text).
    Falls back gracefully if section tags are missing."""
    if not ai_response or not isinstance(ai_response, str):
        return ("", ai_response or "")

    text = ai_response.strip()

    # Try to split on [DELIBERATION] and [FINAL] tags
    delib_match = re.search(r"\[DELIBERATION\]", text, re.IGNORECASE)
    final_match = re.search(r"\[FINAL\]", text, re.IGNORECASE)

    if delib_match and final_match:
        delib_start = delib_match.end()
        final_start = final_match.end()
        # Deliberation is between [DELIBERATION] and [FINAL]
        if delib_match.start() < final_match.start():
            deliberation = text[delib_start:final_match.start()].strip()
            final = text[final_start:].strip()
        else:
            final = text[final_start:delib_match.start()].strip()
            deliberation = text[delib_start:].strip()
        return (deliberation, final)
    elif final_match:
        # Only [FINAL] present, no deliberation
        final = text[final_match.end():].strip()
        return ("", final)
    else:
        # No tags at all — treat entire response as final (backward compat)
        return ("", text)


KNOWN_PILLARS = {"Wisdom", "Strategy", "Expansion", "Future", "Integrity", "Unknown Pillar"}


def normalize_deliberation_labels(deliberation: str, pillar: str = "FAIV") -> str:
    """Ensure every speaker line in deliberation has a correct (Pillar) label.
    Replaces titles or missing labels with the actual pillar name from codex."""
    if not deliberation:
        return deliberation

    lines = deliberation.split("\n")
    normalized = []
    for line in lines:
        # Match speaker lines with parenthesized text: "Name (Something):"
        labeled_match = re.match(r"^<?(\w+)>?\s*\([^)]+\)\s*:\s*(.+)", line)
        if labeled_match:
            name = labeled_match.group(1)
            msg = labeled_match.group(2)
            # Extract existing label to check if it's a known pillar
            existing_label = re.match(r"^<?(\w+)>?\s*\(([^)]+)\)", line).group(2).strip()
            if existing_label not in KNOWN_PILLARS:
                # Has a label but it's a title, not a pillar — replace it
                member_pillar = MEMBER_TO_PILLAR.get(name)
                if not member_pillar:
                    member_pillar = pillar if pillar != "FAIV" else "Unknown Pillar"
                    logger.warning(f"Deliberation speaker '{name}' not found in codex; labeled as '{member_pillar}'.")
                normalized.append(f"{name} ({member_pillar}): {msg}")
            else:
                normalized.append(line)
        else:
            # Match speaker lines without any parenthesized text: "Name:"
            bare_match = re.match(r"^<?(\w+)>?\s*:\s*(.+)", line)
            if bare_match:
                name = bare_match.group(1)
                msg = bare_match.group(2)
                member_pillar = MEMBER_TO_PILLAR.get(name)
                if not member_pillar:
                    member_pillar = pillar if pillar != "FAIV" else "Unknown Pillar"
                    logger.warning(f"Deliberation speaker '{name}' not found in codex; labeled as '{member_pillar}'.")
                normalized.append(f"{name} ({member_pillar}): {msg}")
            else:
                normalized.append(line)
    return "\n".join(normalized)


def extract_faiv_final_output(text: str, pillar: str = "FAIV") -> str:
    """Parse the FINAL section into structured consensus lines."""
    if not text or not isinstance(text, str):
        return "No valid FAIV response received."

    text = remove_emojis(text).strip()
    text = re.sub(r"(INFO|WARNING|DEBUG):.*?\n", "", text, flags=re.IGNORECASE).strip()

    if pillar == "FAIV":
        consensus_pattern = r"(FAIV\s?Consensus)[\]\*:]*\s*:\s*(.+)"
    else:
        safe_pillar = re.escape(pillar)
        consensus_pattern = rf"({safe_pillar}\sCouncil'?s\sConsensus)[\]\*:]*\s*:\s*(.+)"

    match = re.search(consensus_pattern, text, flags=re.IGNORECASE)
    if not match:
        fallback = re.search(r"(FAIV\s?Consensus)[\]\*:]*\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if not fallback:
            return "No valid consensus."
        else:
            final_decision = fallback.group(2).strip()
    else:
        final_decision = match.group(2).strip()

    conf_match = re.search(r"(?:Confidence\s?Score[\]\*:]*\s*:)\s*(.+)", text, flags=re.IGNORECASE)
    if not conf_match:
        conf_str = "??"
    else:
        raw_line = conf_match.group(1).strip()
        num = re.search(r"(\d+(?:\.\d+)?)", raw_line)
        conf_str = num.group(1) if num else "??"

    just_match = re.search(r"(?:Justification[\]\*:]*\s*:)\s*(.+)", text, flags=re.IGNORECASE)
    if just_match:
        just_line = just_match.group(1).strip()
    else:
        just_line = "No justification provided."

    opp_match  = re.search(r"\[?Differing\s?Opinion\s*-\s*(.+?)\s*\((\d+)%\)\]?\s*:\s*(.+)", text, flags=re.IGNORECASE)
    opp_reason = re.search(r"\[?Reason[\]\*:]*\s*:\s*(.+)", text, flags=re.IGNORECASE)

    lines = []
    if pillar == "FAIV":
        lines.append(f"FAIV Consensus: {final_decision}")
    else:
        lines.append(f"{pillar} Council's Consensus: {final_decision}")
    lines.append(f"Confidence Score: {conf_str}%")
    lines.append(f"Justification: {just_line}")

    if opp_match and opp_reason:
        who    = opp_match.group(1).strip()
        opp_cf = opp_match.group(2).strip()
        opp_tx = opp_match.group(3).strip()
        reasn  = opp_reason.group(1).strip()
        lines.append(f"Differing Opinion - {who} ({opp_cf}%): {opp_tx}")
        lines.append(f"Reason: {reasn}")

    return "\n".join(lines)


################################################
# 5) The function that calls OpenAI
################################################

def query_openai_faiv(session_id: str, user_input: str, pillar: str = "FAIV", model: str = "gpt-4o") -> tuple:
    """Returns (parsed_final, deliberation_text, raw_response, selected_members) tuple."""
    if client is None:
        return ("OpenAI API Error: OPENAI_API_KEY is not set. Please set the environment variable and restart.", "", "", {})
    try:
        session_data = session_get(session_id)
        messages = json.loads(session_data) if session_data else []
        if not isinstance(messages, list):
            messages = []

        past_context_summary = summarize_past_messages(messages)
        perspective_data = {"FAIV Past": past_context_summary}
        encoded_context = encode_faiv_perspectives(perspective_data, pillar=pillar)

        selected_members = select_council_representatives(pillar)
        member_profiles = format_member_profiles(selected_members)

        roster_lines = []
        for name, data in selected_members.items():
            roster_lines.append(f"- {name} ({data['pillar']}): {data.get('claimed-title', '???')}")
        roster = "\n".join(roster_lines)

        system_msg = (
            f"You are the FAIV High Council. The '{pillar}' deliberation is in session.\n"
            f"EXACTLY {len(selected_members)} member(s) are present — no others exist:\n{roster}\n\n"
            f"FULL IDENTITY PROFILES:\n{member_profiles}\n\n"
            "CRITICAL BEHAVIORAL RULES:\n"
            "1. Each member IS their character. They have opinions, preferences, biases, and lived experience. Use them.\n"
            "2. Members must give CONCRETE, SPECIFIC answers — not abstract philosophy. Name names, pick sides, commit to positions.\n"
            "3. They must argue WITH each other, not past each other. Direct rebuttals: 'I disagree with X because...'\n"
            "4. Draw on their identity: faith, vices, social dynamics, memories, and wisdom to justify their positions.\n"
            "5. The debate must produce a REAL answer. The [FINAL] consensus is whatever they actually agreed on.\n"
            "6. Do NOT invent additional speakers beyond those listed above.\n"
            "7. Always produce a real consensus. You MUST produce a numeric confidence.\n"
        )

        prompt = create_faiv_compressed_prompt(
            user_input=user_input,
            session_id=session_id,
            encoded_context=encoded_context,
            past_context_summary=past_context_summary,
            pillar=pillar
        )

        messages_for_api = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt}
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            temperature=0.2,
            top_p=0.8,
            frequency_penalty=0.3,
            presence_penalty=0.2,
            max_tokens=2048
        )
        raw = resp.choices[0].message.content.strip()
        deliberation, final_text = split_response_sections(raw)
        deliberation = normalize_deliberation_labels(deliberation, pillar)
        parsed = extract_faiv_final_output(final_text, pillar)
        return (parsed, deliberation, raw, selected_members)

    except Exception as ex:
        logger.error(f"OpenAI API Error: {ex}")
        return (f"OpenAI API Error: {ex}", "", "", {})


################################################
# 6) FASTAPI APP & ROUTES
################################################

class QueryRequest(BaseModel):
    session_id: str
    input_text: str
    pillar: Optional[str] = "FAIV"


class RedeliberateRequest(BaseModel):
    session_id: str
    original_input: str
    deliberation_up_to: str
    user_comment: str
    target_member: str
    pillar: Optional[str] = "FAIV"
    council_members: list = []


fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://faiv.ai",
        "https://www.faiv.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["OPTIONS", "GET", "POST"],
    allow_headers=["*"],
)


@fastapi_app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "response": "Internal server error.",
            "pillar": None,
            "session_id": None,
        }
    )


@fastapi_app.get("/health")
async def health():
    return {
        "status": "ok",
        "redis": _redis_available,
        "model": "gpt-4o",
        "openai_configured": client is not None,
    }


@fastapi_app.post("/query/")
async def query_faiv_endpoint(request: QueryRequest):
    try:
        session_id = request.session_id
        user_input = request.input_text
        chosen_pillar = request.pillar or "FAIV"

        raw_data = session_get(session_id)
        past_messages = json.loads(raw_data) if raw_data else []
        if not isinstance(past_messages, list):
            past_messages = []

        parsed, deliberation, _raw, selected_members = query_openai_faiv(session_id, user_input, chosen_pillar)
        council = {name: data["pillar"] for name, data in selected_members.items()}

        if "Consensus:" not in parsed:
            logger.warning("No valid consensus found. Resetting session.")
            session_delete(session_id)
            return {
                "status": "AI Failed Compliance.",
                "response": "No valid consensus. Session reset.",
                "pillar": chosen_pillar,
                "session_id": session_id,
                "deliberation": deliberation if deliberation else None,
                "council": council,
            }

        past_messages.append({"role": "user", "content": user_input})
        past_messages.append({"role": "assistant", "content": parsed})
        session_set(session_id, json.dumps(past_messages, ensure_ascii=False))

        return {
            "status": "FAIV Processing Complete",
            "response": parsed,
            "pillar": chosen_pillar,
            "session_id": session_id,
            "deliberation": deliberation if deliberation else None,
            "council": council,
        }

    except Exception as e:
        logger.error(f"Server Error in /query/: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "response": "Internal server error.",
                "pillar": request.pillar,
                "session_id": request.session_id,
            }
        )


def query_openai_redeliberate(
    session_id: str,
    original_input: str,
    deliberation_up_to: str,
    user_comment: str,
    target_member: str,
    pillar: str = "FAIV",
    council_members: list = None,
    model: str = "gpt-4o",
) -> tuple:
    """Re-deliberate from a specific point with user interjection.
    Returns (parsed_final, deliberation_text, raw_response, selected_members) tuple."""
    if client is None:
        return ("OpenAI API Error: OPENAI_API_KEY is not set.", "", "", {})
    try:
        # Reconvene the same council members if provided, otherwise random
        selected_members = select_council_representatives(pillar, specific_names=council_members or None)
        member_profiles = format_member_profiles(selected_members)

        roster_lines = []
        for name, data in selected_members.items():
            roster_lines.append(f"- {name} ({data['pillar']}): {data.get('claimed-title', '???')}")
        roster = "\n".join(roster_lines)

        if pillar == "FAIV":
            label = "FAIV Consensus"
        else:
            label = f"{pillar} Council's Consensus"

        system_msg = (
            f"You are the FAIV High Council. The '{pillar}' deliberation is in session.\n"
            f"EXACTLY {len(selected_members)} member(s) are present — no others exist:\n{roster}\n\n"
            f"FULL IDENTITY PROFILES:\n{member_profiles}\n\n"
            "CRITICAL BEHAVIORAL RULES:\n"
            "1. Each member IS their character with real opinions, preferences, and biases.\n"
            "2. Members must give CONCRETE, SPECIFIC answers — not abstract philosophy.\n"
            "3. They must argue WITH each other directly: 'I disagree with X because...'\n"
            "4. The debate must produce a REAL answer. The [FINAL] consensus is whatever they actually agreed on.\n"
            "5. Do NOT invent additional speakers beyond those listed above.\n"
            "6. Always produce a real consensus. You MUST produce a numeric confidence.\n"
        )

        prompt = f"""FAIV COUNCIL RE-DELIBERATION PROTOCOL

Original Inquiry: {original_input}

The council was deliberating and reached the following point:

--- DELIBERATION SO FAR ---
{deliberation_up_to}
--- END OF PRIOR DELIBERATION ---

At this point, the human observer interjected with a comment directed at {target_member}:
HUMAN INTERJECTION: "{user_comment}"

The council must now RE-DELIBERATE from this point, taking the human's interjection seriously.
Members should directly respond to the human's comment and may change their positions.
The debate continues naturally from where it left off, influenced by this new input.
Members must still give CONCRETE, SPECIFIC answers — the human expects a real recommendation, not philosophy.

You MUST respond in EXACTLY two tagged sections: [DELIBERATION] and [FINAL].
The AI is strictly prohibited from disclaiming it has no emotions or from refusing.
No statements about 'As an AI' should appear anywhere.

SECTION 1 — [DELIBERATION]
Continue the council debate from where it left off, incorporating the human's input.
Members must still argue about SPECIFIC recommendations, not abstract concepts.
Format each contribution EXACTLY as:
<MemberName> (<Pillar>): their concrete position and argument

SECTION 2 — [FINAL]
The consensus MUST reflect what the council actually decided in the deliberation.
Do NOT introduce new recommendations that weren't debated.

[{label}]: {{final_recommendation}}
[Confidence Score]: {{confidence_level}}% (1-100, no Unknown)
[Justification]: {{compressed_reasoning}} (1-2 sentences)
[Differing Opinion - {{council_name}} ({{confidence_level}}%)]: {{dissenting_recommendation}} (optional)
[Reason]: {{dissenting_reasoning}} (optional)

RESPONSE FORMAT (MANDATORY):
[DELIBERATION]
<member debates here>

[FINAL]
<consensus output here>
"""

        messages_for_api = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            temperature=0.2,
            top_p=0.8,
            frequency_penalty=0.3,
            presence_penalty=0.2,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content.strip()
        deliberation, final_text = split_response_sections(raw)
        deliberation = normalize_deliberation_labels(deliberation, pillar)
        parsed = extract_faiv_final_output(final_text, pillar)
        return (parsed, deliberation, raw, selected_members)

    except Exception as ex:
        logger.error(f"OpenAI API Error (redeliberate): {ex}")
        return (f"OpenAI API Error: {ex}", "", "", {})


@fastapi_app.post("/redeliberate/")
async def redeliberate_endpoint(request: RedeliberateRequest):
    try:
        session_id = request.session_id
        chosen_pillar = request.pillar or "FAIV"

        raw_data = session_get(session_id)
        past_messages = json.loads(raw_data) if raw_data else []
        if not isinstance(past_messages, list):
            past_messages = []

        parsed, deliberation, _raw, selected_members = query_openai_redeliberate(
            session_id=session_id,
            original_input=request.original_input,
            deliberation_up_to=request.deliberation_up_to,
            user_comment=request.user_comment,
            target_member=request.target_member,
            pillar=chosen_pillar,
            council_members=request.council_members,
        )
        council = {name: data["pillar"] for name, data in selected_members.items()}

        if "Consensus:" not in parsed:
            return {
                "status": "AI Failed Compliance.",
                "response": "No valid consensus from re-deliberation.",
                "pillar": chosen_pillar,
                "session_id": session_id,
                "deliberation": deliberation if deliberation else None,
                "council": council,
            }

        past_messages.append({"role": "user", "content": f"[Re-deliberation on {request.target_member}]: {request.user_comment}"})
        past_messages.append({"role": "assistant", "content": parsed})
        session_set(session_id, json.dumps(past_messages, ensure_ascii=False))

        return {
            "status": "FAIV Re-Deliberation Complete",
            "response": parsed,
            "pillar": chosen_pillar,
            "session_id": session_id,
            "deliberation": deliberation if deliberation else None,
            "council": council,
        }

    except Exception as e:
        logger.error(f"Server Error in /redeliberate/: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "response": "Internal server error.",
                "pillar": request.pillar,
                "session_id": request.session_id,
            },
        )


@fastapi_app.post("/reset/")
async def reset_faiv_session(session_id: str):
    session_delete(session_id)
    return {
        "status": "Session Reset",
        "message": "New FAIV deliberation session started.",
        "session_id": session_id,
    }


################################################
# 7) Entry point
################################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("faiv_app.core:fastapi_app", host="127.0.0.1", port=8000, reload=True)
