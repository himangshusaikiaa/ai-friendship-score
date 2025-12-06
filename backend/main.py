from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, time
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64
import re
import math

app = FastAPI(title="AI Friendship Compatibility Score")

# --- CORS for local frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Helpers: chat parsing ----------

# Supports:
#  - 12/06/2025, 22:14 - Name: Message
#  - 12/06/25, 10:14 pm - Name: Message
#  - 12-06-2025, 22:14 – Name: Message
#  - 12.06.2025, 22:14 - Name: Message
WHATSAPP_LINE_RE = re.compile(
    r"""^
    (\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})   # date: 12/06/2025 or 12-06-25
    ,?\s+                                   # optional comma + space
    (\d{1,2}:\d{2}(?::\d{2})?)              # time: 22:14 or 22:14:05
    \s*(AM|PM|am|pm)?\s*                    # optional am/pm
    [\-\u2013\u2014]\s+                     # dash variants (-, –, —)
    ([^:]+):\s+                             # name up to colon
    (.*)$                                   # message
    """,
    re.VERBOSE,
)


SYSTEM_PATTERNS = [
    "Messages and calls are end-to-end encrypted",
    "You created group",
    "added",
    "left",
    "changed the subject",
    "changed this group's icon",
]


def is_system_message(text: str) -> bool:
    lower = text.lower()
    return any(pat.lower() in lower for pat in SYSTEM_PATTERNS)


def parse_whatsapp_chat(text: str) -> List[Dict[str, Any]]:
    messages = []
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = WHATSAPP_LINE_RE.match(line)
        if m:
            # Save previous message if exists
            if current and not is_system_message(current["message"]):
                messages.append(current)

            date_str, time_str, ampm, sender, msg = m.groups()

            # Normalize am/pm
            ampm_norm = ampm.upper() if ampm else None

            # Try to normalize timestamp
            dt = None
            # Try a few formats:
            formats = []
            if ampm_norm:
                # 12h formats
                formats = [
                    "%d/%m/%Y %I:%M %p",
                    "%d-%m-%Y %I:%M %p",
                    "%d.%m.%Y %I:%M %p",
                    "%d/%m/%y %I:%M %p",
                    "%d-%m-%y %I:%M %p",
                    "%d.%m.%y %I:%M %p",
                ]
            else:
                # 24h formats
                formats = [
                    "%d/%m/%Y %H:%M",
                    "%d-%m-%Y %H:%M",
                    "%d.%m.%Y %H:%M",
                    "%d/%m/%y %H:%M",
                    "%d-%m-%y %H:%M",
                    "%d.%m/%y %H:%M",
                ]

            dt_str_base = f"{date_str} {time_str}"
            for fmt in formats:
                try:
                    dt = datetime.strptime(dt_str_base + (f" {ampm_norm}" if ampm_norm else ""), fmt)
                    break
                except Exception:
                    continue

            current = {
                "timestamp": dt,
                "sender": sender.strip(),
                "message": msg.strip(),
            }
        else:
            # Continuation of previous message
            if current:
                current["message"] += "\n" + line

    # Final message
    if current and not is_system_message(current["message"]):
        messages.append(current)

    return messages



# --------- Metrics helpers ----------

POSITIVE_WORDS = {
    "love", "like", "happy", "amazing", "awesome", "great", "nice",
    "wonderful", "best", "cool", "cute", "fun", "funny", "perfect",
    "sweet", "good", "beautiful", "proud", "grateful", "thank you",
}

NEGATIVE_WORDS = {
    "hate", "angry", "sad", "annoyed", "upset", "bad", "worst",
    "stupid", "idiot", "dumb", "terrible", "ugly", "tired", "lonely",
    "crying", "cry", "depressed", "anxious", "anxiety",
}

TOXIC_WORDS = {
    "idiot", "stupid", "dumb", "trash", "loser", "shut up",
    "kill yourself", "kys", "bitch", "asshole",
}

LAUGH_PATTERNS = ["haha", "lmao", "lol", "rofl", "xd", "🤣", "😂", "😆", "😹"]


def count_emojis(text: str) -> int:
    # Very rough emoji count: characters with ord >= 0x1F300
    return sum(1 for ch in text if ord(ch) >= 0x1F300)


def count_laughs(text: str) -> int:
    lower = text.lower()
    return sum(lower.count(pat) for pat in LAUGH_PATTERNS)


def classify_sentiment(text: str) -> int:
    """
    Returns:
        +1 positive
         0 neutral
        -1 negative
    Very simple word-list based sentiment.
    """
    lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in lower)
    if pos > neg:
        return 1
    elif neg > pos:
        return -1
    else:
        return 0


def has_toxicity(text: str) -> bool:
    lower = text.lower()
    return any(w in lower for w in TOXIC_WORDS)


def compute_reply_times(messages):
    """
    Compute reply times when sender changes.
    Returns dict: sender -> list of seconds as float
    """
    reply_times = {}
    for i in range(1, len(messages)):
        prev = messages[i - 1]
        curr = messages[i]
        if not prev["timestamp"] or not curr["timestamp"]:
            continue
        if prev["sender"] == curr["sender"]:
            continue
        delta = (curr["timestamp"] - prev["timestamp"]).total_seconds()
        if delta < 0:
            continue
        reply_times.setdefault(curr["sender"], []).append(delta)
    return reply_times


def avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = avg(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def compute_conversation_initiations(messages):
    """
    Count who starts "sessions" separated by > 2 hours.
    """
    initiations = {}
    if not messages:
        return initiations

    initiations[messages[0]["sender"]] = initiations.get(messages[0]["sender"], 0) + 1

    for i in range(1, len(messages)):
        prev = messages[i - 1]
        curr = messages[i]
        if not prev["timestamp"] or not curr["timestamp"]:
            continue
        delta = (curr["timestamp"] - prev["timestamp"]).total_seconds()
        if delta > 2 * 60 * 60:  # 2 hours
            initiations[curr["sender"]] = initiations.get(curr["sender"], 0) + 1
    return initiations


def compute_depth_metrics(messages):
    """
    Depth based on late-night messages, message length,
    and total word count.
    """
    late_night_count = 0
    lengths = []
    total_words = 0

    for m in messages:
        msg = m["message"]
        lengths.append(len(msg))
        total_words += len(msg.split())
        ts = m["timestamp"]
        if ts:
            if ts.time() >= time(23, 0) or ts.time() <= time(4, 0):
                late_night_count += 1

    total_msgs = max(len(messages), 1)
    late_night_ratio = late_night_count / total_msgs
    length_std = std(lengths)
    # Normalize length_std by dividing by 200 chars (rough)
    length_std_norm = min(length_std / 200.0, 1.0)
    # Normalize word count by dividing by 10k words
    total_words_norm = min(total_words / 10000.0, 1.0)

    depth_score = (
        0.4 * late_night_ratio
        + 0.3 * length_std_norm
        + 0.3 * total_words_norm
    )
    depth_score = max(0.0, min(depth_score, 1.0))
    return {
        "late_night_ratio": late_night_ratio,
        "length_std_norm": length_std_norm,
        "total_words_norm": total_words_norm,
        "depth_score": depth_score,
    }


def normalize_ratio(value: float, high_is_good=True) -> float:
    value = max(0.0, min(value, 1.0))
    return value if high_is_good else 1.0 - value


def generate_insights(metrics: Dict[str, Any], person_a: str, person_b: str) -> List[str]:
    insights = []

    overall = metrics["overall_score"]
    balance = metrics["balance_score"]
    fun = metrics["fun_score"]
    positivity = metrics["positivity_score"]
    responsiveness = metrics["responsiveness_score"]
    depth = metrics["depth_score"]
    emoji_count = metrics["emoji_count"]
    laugh_count = metrics["laugh_count"]
    fast_replier = metrics["fastest_replier"]

    # Base vibe
    if overall >= 0.85:
        insights.append("This friendship is giving ✨real ones✨.")
        insights.append("Not gonna lie… this friendship is giving found-family vibes 💗")
    elif overall >= 0.7:
        insights.append("Certified bestie material detected 🤝")
    else:
        insights.append("Potential besties loading… some upgrades and you’re elite fr ⚙️")

    # Balance
    if balance > 0.8:
        insights.append("One-sided? Nah, this is balanced as hell.")
    else:
        insights.append("Lowkey a lil one-sided… someone’s carrying the convo 👀")

    # Responsiveness
    if fast_replier:
        insights.append(f"{fast_replier} replies faster than your internet speed ⚡")
    if responsiveness > 0.8:
        insights.append("You reply faster than your bestie 💀🔥")
    else:
        insights.append("Reply time is giving ‘I saw it but I’ll answer later’ energy ⏳")

    # Fun
    insights.append(f"Laugh emojis detected: {laugh_count}. Y’all are unserious fr 😂")
    if emoji_count > 0:
        ratio = emoji_count / max(metrics["total_messages"], 1)
        if ratio > 1.5:
            insights.append("Bro this chat is 70% emojis you guys are insane 😭🔥")
        else:
            insights.append("Perfect balance of text and emojis, aesthetic af ✨")

    # Positivity / toxicity
    if positivity > 0.6:
        insights.append("Vibes are mostly positive, love that for you 💅")
    elif metrics["toxicity_ratio"] > 0.05:
        insights.append("Some messages are kinda spicy 🌶️ but if it’s playful it’s fine ig")
    else:
        insights.append("Barely any negativity, we stan emotionally stable besties 🧠💖")

    # Depth
    if depth > 0.6:
        insights.append("Your midnight convo energy is unmatched 🌙🫶")
        insights.append("The loyalty score is hitting different.")
    else:
        insights.append("More late-night oversharing sessions could unlock god-tier friendship 🌌")

    # Trim to 6–10
    return insights[:10]


def find_special_moments(messages):
    """
    Find most emotional and most funny message (index + text).
    Emotional = highest |sentiment|
    Funny = highest laughs + emojis
    """
    best_emotional = (None, -1)  # (idx, score)
    best_funny = (None, -1)

    for i, m in enumerate(messages):
        msg = m["message"]
        sent = classify_sentiment(msg)
        emotional_score = abs(sent)

        funny_score = count_laughs(msg) + count_emojis(msg)

        if emotional_score > best_emotional[1]:
            best_emotional = (i, emotional_score)
        if funny_score > best_funny[1]:
            best_funny = (i, funny_score)

    emotional_text = messages[best_emotional[0]]["message"] if best_emotional[0] is not None else ""
    funny_text = messages[best_funny[0]]["message"] if best_funny[0] is not None else ""

    return emotional_text, funny_text


def compute_metrics(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not messages:
        return {}

    # Determine two main participants
    sender_counts = {}
    for m in messages:
        sender_counts[m["sender"]] = sender_counts.get(m["sender"], 0) + 1

    sorted_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_senders) < 2:
        # Single person chat or something weird
        persons = [sorted_senders[0][0], "Friend"]
    else:
        persons = [sorted_senders[0][0], sorted_senders[1][0]]

    a, b = persons[0], persons[1]
    count_a = sender_counts.get(a, 0)
    count_b = sender_counts.get(b, 0)
    total_messages = count_a + count_b

    # Balance score 0–1
    if total_messages == 0:
        balance_score = 0.5
    else:
        balance_score = 1 - abs(count_a - count_b) / total_messages

    # Reply times
    reply_times = compute_reply_times(messages)
    avg_reply_a = avg(reply_times.get(a, []))
    avg_reply_b = avg(reply_times.get(b, []))

    # Who replies faster
    fastest_replier = None
    if avg_reply_a and avg_reply_b:
        if avg_reply_a < avg_reply_b:
            fastest_replier = a
        elif avg_reply_b < avg_reply_a:
            fastest_replier = b

    # Responsiveness score
    # We want: short replies & similar speeds => high score.
    all_replies = reply_times.get(a, []) + reply_times.get(b, [])
    overall_avg_reply = avg(all_replies) if all_replies else 0

    if avg_reply_a == 0 or avg_reply_b == 0:
        similarity_factor = 0.5
    else:
        similarity_factor = 1 - abs(avg_reply_a - avg_reply_b) / max(avg_reply_a, avg_reply_b)
        similarity_factor = max(0.0, min(similarity_factor, 1.0))

    # Speed factor: <2min very good, >1h not great
    if overall_avg_reply == 0:
        speed_factor = 0.5
    else:
        speed_factor = max(0.0, min(1.0, 1 - (overall_avg_reply / 3600.0)))
    responsiveness_score = (0.6 * speed_factor) + (0.4 * similarity_factor)

    # Sentiment & toxicity & fun & emojis
    pos_msgs = 0
    neg_msgs = 0
    neutral_msgs = 0
    toxic_msgs = 0
    emoji_count = 0
    laugh_count = 0

    for m in messages:
        msg = m["message"]
        s = classify_sentiment(msg)
        if s > 0:
            pos_msgs += 1
        elif s < 0:
            neg_msgs += 1
        else:
            neutral_msgs += 1

        if has_toxicity(msg):
            toxic_msgs += 1

        emoji_count += count_emojis(msg)
        laugh_count += count_laughs(msg)

    # Positivity
    if total_messages == 0:
        positivity_delta = 0.0
    else:
        pos_ratio = pos_msgs / total_messages
        neg_ratio = neg_msgs / total_messages
        positivity_delta = pos_ratio - neg_ratio  # -1..1
    positivity_score = (positivity_delta + 1) / 2  # 0..1

    toxicity_ratio = toxic_msgs / total_messages if total_messages else 0.0

    # Fun factor: based on emoji + laugh density
    msg_based = max(total_messages, 1)
    emoji_density = emoji_count / msg_based
    laugh_density = laugh_count / msg_based

    # If you have 3+ combined per message, that's max fun
    raw_fun = emoji_density + laugh_density  # 0..+
    fun_score = min(raw_fun / 3.0, 1.0)

    depth_metrics = compute_depth_metrics(messages)

    depth_score = depth_metrics["depth_score"]

    # Final score: 25% balance, 25% emotional positivity,
    # 20% responsiveness, 20% fun, 10% depth
    overall_score = (
        0.25 * balance_score +
        0.25 * positivity_score +
        0.20 * responsiveness_score +
        0.20 * fun_score +
        0.10 * depth_score
    )
    overall_score = max(0.0, min(overall_score, 1.0))

    # Find special messages
    most_emotional, most_funny = find_special_moments(messages)

    metrics = {
        "person_a": a,
        "person_b": b,
        "message_count": {
            a: count_a,
            b: count_b,
        },
        "total_messages": total_messages,
        "balance_score": balance_score,
        "avg_reply_seconds": {
            a: avg_reply_a,
            b: avg_reply_b,
        },
        "fastest_replier": fastest_replier,
        "responsiveness_score": responsiveness_score,
        "positivity_score": positivity_score,
        "toxicity_ratio": toxicity_ratio,
        "emoji_count": emoji_count,
        "laugh_count": laugh_count,
        "emoji_density": emoji_density,
        "laugh_density": laugh_density,
        "fun_score": fun_score,
        "depth_score": depth_score,
        "depth_details": depth_metrics,
        "most_emotional_message": most_emotional,
        "most_funny_message": most_funny,
    }

    # Add final overall % score
    metrics["overall_score"] = overall_score

    # Conversation initiation counts
    initiations = compute_conversation_initiations(messages)
    metrics["conversation_initiations"] = {
        a: initiations.get(a, 0),
        b: initiations.get(b, 0),
    }

    # Add high-level question-style insights
    metrics["who_texts_more"] = a if count_a > count_b else b
    metrics["who_double_texts_more"] = metrics["who_texts_more"]  # simple approx

    metrics["late_night_ratio"] = depth_metrics["late_night_ratio"]

    return metrics


# --------- Card generator ----------

class CardRequest(BaseModel):
    person_a: str
    person_b: str
    overall_score: float  # 0..1
    balance_score: float
    fun_score: float
    positivity_score: float
    depth_score: float


def generate_share_card(req: CardRequest) -> str:
    """
    Generates 1080x1350 PNG with neon-style look and returns base64 string.
    """
    width, height = 1080, 1350
    img = Image.new("RGB", (width, height), (5, 5, 15))
    draw = ImageDraw.Draw(img, "RGBA")

    # At top of generate_share_card, after creating draw:
    def get_text_size(text, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]


    # Neon gradient aura
    for r in range(0, 900, 20):
        alpha = max(0, 180 - r // 5)
        color = (140, 0, 255, alpha) if r % 40 == 0 else (0, 255, 255, alpha)
        bbox = [
            width // 2 - r,
            height // 2 - r,
            width // 2 + r,
            height // 2 + r,
        ]
        draw.ellipse(bbox, outline=color, width=4)

    # Card rectangle
    margin = 80
    card_bbox = [margin, margin + 80, width - margin, height - margin]
    draw.rounded_rectangle(card_bbox, radius=40, outline=(255, 0, 255, 200), width=6)
    draw.rounded_rectangle(card_bbox, radius=40, outline=(0, 255, 255, 120), width=2)

    # Load fonts (fallback to default if not installed)
    try:
        title_font = ImageFont.truetype("arial.ttf", 72)
        big_font = ImageFont.truetype("arial.ttf", 160)
        label_font = ImageFont.truetype("arial.ttf", 40)
        small_font = ImageFont.truetype("arial.ttf", 34)
    except Exception:
        title_font = ImageFont.load_default()
        big_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Center helper
    def center_text(text, y, font, fill=(255, 255, 255)):
        w, h = get_text_size(text, font)
        x = (width - w) / 2
        draw.text((x, y), text, font=font, fill=fill)

    # Title
    center_text("Friendship Score", card_bbox[1] + 40, title_font, (255, 255, 255))

    # Big score
    score_pct = int(round(req.overall_score * 100))
    center_text(f"{score_pct}%", card_bbox[1] + 140, big_font, (255, 0, 255))

    # Subline
    if score_pct >= 85:
        subline = "We’re literally soul-mates 💫"
    elif score_pct >= 70:
        subline = "Elite bestie energy 🔥"
    else:
        subline = "Potential besties in progress ⚙️"
    center_text(subline, card_bbox[1] + 320, label_font, (200, 255, 255))

    # Names
    names_text = f"{req.person_a}  ×  {req.person_b}"
    center_text(names_text, card_bbox[1] + 390, label_font, (255, 255, 255))

    # Mini stats
    cx = width // 2
    y_stats = card_bbox[1] + 480

    def stat_block(label, value, x_center, y_top):
        label_w, _ = get_text_size(label, font=small_font)
        draw.text((x_center - label_w / 2, y_top), label, font=small_font, fill=(180, 180, 255))
        val = f"{int(round(value * 100))}%"
        val_w, _ = get_text_size(val, font=label_font)
        draw.text((x_center - val_w / 2, y_top + 46), val, font=label_font, fill=(255, 255, 255))

    offset = 220
    stat_block("Balance", req.balance_score, cx - offset, y_stats)
    stat_block("Positivity", req.positivity_score, cx + offset, y_stats)
    stat_block("Fun Factor", req.fun_score, cx - offset, y_stats + 160)
    stat_block("Depth", req.depth_score, cx + offset, y_stats + 160)

    # Tagline bottom
    tag = "Tag your bestie & flex this 💖"
    tw, _ = get_text_size(tag, font=small_font)
    draw.text(
        ((width - tw) / 2, card_bbox[3] - 180),
        tag,
        font=small_font,
        fill=(200, 200, 255),
    )

    # Footer
    footer = "Generated by FriendshipScore.ai"
    fw, _ = get_text_size(footer, font=small_font)
    draw.text(
        ((width - fw) / 2, card_bbox[3] - 100),
        footer,
        font=small_font,
        fill=(160, 160, 200),
    )

    # Encode base64
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


# --------- API endpoints ----------

@app.post("/analyze_chat")
async def analyze_chat(file: UploadFile = File(...)):
    """
    Accepts WhatsApp .txt export and returns metrics + scores.
    """
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        messages = parse_whatsapp_chat(text)
        metrics = compute_metrics(messages)

        if not metrics:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not parse any valid chat messages."},
            )

        insights = generate_insights(
            metrics,
            metrics["person_a"],
            metrics["person_b"],
        )

        response = {
            "persons": {
                "a": metrics["person_a"],
                "b": metrics["person_b"],
            },
            "scores": {
                "overall": round(metrics["overall_score"] * 100, 2),
                "communication_balance": round(metrics["balance_score"] * 100, 2),
                "emotional_positivity": round(metrics["positivity_score"] * 100, 2),
                "responsiveness": round(metrics["responsiveness_score"] * 100, 2),
                "fun_factor": round(metrics["fun_score"] * 100, 2),
                "depth": round(metrics["depth_score"] * 100, 2),
            },
            "raw_metrics": metrics,
            "insights": insights,
            "special_moments": {
                "most_emotional": metrics["most_emotional_message"],
                "most_funny": metrics["most_funny_message"],
            },
            "privacy_note": "Your chats stay on your device. We don’t store any conversation data.",
        }
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate_card")
async def generate_card(req: CardRequest):
    """
    Input: metrics + names
    Output: base64 PNG
    """
    try:
        b64 = generate_share_card(req)
        return {"image_base64": b64}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Run with:
# uvicorn main:app --reload
