
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { VertexAI } from '@google-cloud/vertexai';
import { Firestore } from '@google-cloud/firestore';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 8080;
const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT;
const LOCATION = process.env.GOOGLE_CLOUD_LOCATION || 'asia-south1';
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-1.5-flash';

// Firestore client (picks up GOOGLE_APPLICATION_CREDENTIALS)
const db = new Firestore({ projectId: PROJECT_ID });

// Vertex AI client
const vertexAI = new VertexAI({
  project: PROJECT_ID,
  location: LOCATION,
});

const model = vertexAI.getGenerativeModel({ model: GEMINI_MODEL });

app.get('/api/ping', (_req, res) => {
  res.json({ ok: true, message: 'pong' });
});

app.post('/api/chat', async (req, res) => {
  try {
    const prompt = String(req.body?.prompt || '').trim();
    if (!prompt) return res.status(400).json({ error: 'prompt required' });

    const result = await model.generateContent({
      contents: [{ role: 'user', parts: [{ text: prompt }]}],
      generationConfig: {
        temperature: 0.9,
        topP: 0.95,
        maxOutputTokens: 512,
      },
      safetySettings: [
        // Add safety settings that match your youth-safe policies as needed.
      ],
    });

    const text = result?.response?.candidates?.[0]?.content?.parts?.[0]?.text || '';
    res.json({ reply: text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'chat failed', detail: String(err) });
  }
});

app.post('/api/mood', async (req, res) => {
  try {
    const { userId = 'anon', mood, score, note } = req.body || {};
    if (!mood) return res.status(400).json({ error: 'mood required' });

    const collection = process.env.FIRESTORE_MOOD_COLLECTION || 'mood_logs';
    const doc = db.collection(collection).doc();
    await doc.set({
      userId, mood, score: Number(score ?? 0), note: String(note ?? ''),
      createdAt: new Date().toISOString(),
    });

    res.json({ ok: true, id: doc.id });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'mood save failed', detail: String(err) });
  }
});

app.listen(PORT, () => {
  console.log(`Dost backend listening on http://localhost:${PORT}`);
});
