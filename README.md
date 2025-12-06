# AI Friendship Compatibility Score 💖

Analyze WhatsApp chats between two people and get a Gen-Z flavored friendship score (0–100), bestie stats, and a neon shareable card ready for Instagram.

---

## Features

- 📂 Upload WhatsApp `.txt` chat exports
- 🤖 Auto-detect Person A & Person B
- 📊 Metrics:
  - Overall Friendship Score (0–100)
  - Communication balance
  - Emotional positivity
  - Responsiveness
  - Fun factor (emojis + laughs)
  - Conversation depth (midnight chats + message length)
- 🔍 Insights:
  - Who texts more?
  - Who replies faster?
  - Emoji & laughter density
  - Most emotional + funniest message
- 🖼 Shareable PNG card (1080×1350, IG-ready)
- 🎨 Aesthetic neon, dark UI, Gen-Z slang

> **Privacy:** Your chats stay on your device. We don’t store any conversation data.

---

## Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** HTML + Vanilla JS (single-page), neon UI
- **Image Generation:** Pillow (Python)

---

## Getting Started

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
