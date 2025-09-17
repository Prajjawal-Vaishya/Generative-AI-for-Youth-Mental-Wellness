# --- File 1: app.py ---
# This is your Python back-end server.
# To run this, you will need to install Flask, Flask-CORS, the Google Generative AI library,
# and the Firebase Admin SDK.
# Run these commands in your terminal:
# pip install Flask Flask-CORS google-generativeai firebase-admin
# pip install python-dotenv

import os
import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import Client as FirestoreClient
import google.auth
import sys

# --- Environment Variable Loading ---
from dotenv import load_dotenv
load_dotenv() # Loads variables from a .env file into the environment

# --- API & Firebase Setup ---
# Initialize Firebase Admin SDK using the service account key
try:
    # IMPORTANT: Ensure your 'serviceAccountKey.json' file is in the same directory as this script.
    cred = credentials.Certificate("service-account.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")
    print("Please ensure your 'serviceAccountKey.json' is in the correct path and is valid.")
    db = None

# Initialize the Gemini API client
try:
    api_key = os.getenv('GEMINI_API_KEY') 
    if api_key:
        genai.configure(api_key=api_key)
        print("Gemini API configured using environment variable.")
    else:
        print("Error: The 'GEMINI_API_KEY' environment variable is not set.")
        print("The application cannot function without it. Please set the environment variable and restart.")
        sys.exit(1)
except Exception as e:
    print(f"Error initializing Gemini API: {e}")
    sys.exit(1)

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable Cross-Origin Resource Sharing for API calls

# --- AI Models Initialization ---
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini model loaded successfully.")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    gemini_model = None

# --- Flask Routes ---
@app.route('/')
def index():
    """Route to serve the main HTML page."""
    # The browser will automatically request '/favicon.ico'.
    # You can add a favicon to your 'templates' or 'static' folder to avoid the 404 error.
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for the AI chat bot."""
    if not gemini_model:
        return jsonify({"error": "AI model not available"}), 500

    data = request.json
    user_message = data.get('message', '').strip()
    detected_emotion = data.get('emotion', 'neutral')
    user_id = data.get('userId', 'anonymous')

    if not user_message:
        return jsonify({"response": "Please speak your mind."}), 200

    try:
        # --- FIX: Updated the chat prompt for more natural conversation ---
        # The previous prompt was too restrictive and ignored the user's message content.
        # This new prompt encourages a more conversational, yet still empathetic, response.
        prompt = f"""You are an empathetic, supportive AI companion. Your goal is to listen and provide brief, validating responses. The user seems to be feeling {detected_emotion}. Respond to their message in a way that acknowledges their feelings and continues the conversation. Do not give advice or lengthy explanations. Keep your response concise and conversational.
        User: "{user_message}"
        Bot:"""

        # Generate the response using the Gemini model
        response = gemini_model.generate_content(prompt)
        bot_response = response.text

        # Log the conversation to Firestore
        if db:
            try:
                # Firestore paths should be flexible.
                # Here, we use a simple path for demonstration.
                # Your front-end should be configured to use a similar path.
                conversation_ref = db.collection('chat_logs').document(user_id).collection('conversations')
                conversation_ref.add({
                    'userMessage': user_message,
                    'botResponse': bot_response,
                    'emotion': detected_emotion,
                    'timestamp': datetime.datetime.now()
                })
                print(f"Successfully logged conversation for user {user_id}")
            except Exception as firestore_error:
                # This error indicates a problem with Firebase Admin SDK,
                # such as an invalid service account key or permissions.
                print(f"Error logging to Firestore: {firestore_error}")
                print("Conversation not saved. Please check your Firestore security rules and service account permissions.")

        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"Error calling Gemini API for chat: {e}")
        return jsonify({"error": "An error occurred with the AI service."}), 500

@app.route('/api/generate-art', methods=['POST'])
def generate_art():
    """Generates an image description based on the user's detected emotion."""
    data = request.json
    detected_emotion = data.get('emotion', 'calmness')

    if not detected_emotion:
        return jsonify({"error": "No emotion provided"}), 400

    try:
        # This endpoint returns a TEXT description, not an image itself.
        # The front-end must use this text description as input for a separate image generation service
        # or simply display it to the user.
        prompt_map = {
            "happiness": "A vibrant, abstract painting with warm yellow and orange colors, filled with swirling, joyous shapes, soft gradients, and a sense of pure energy.",
            "sadness": "A monochromatic, melancholic watercolor painting with deep blues and purples, featuring gentle, drooping forms and a sense of quiet reflection.",
            "anger": "An intense, chaotic abstract piece with sharp, jagged lines and a mix of fiery reds and deep black, conveying explosive energy and tension.",
            "calmness": "A serene landscape with soft, pastel colors, a gentle blue sky, and a flowing, tranquil river, with a sense of peaceful stillness.",
            "surprise": "A dynamic, surrealist painting with unexpected, whimsical objects floating in a brilliant, starburst-filled sky, capturing a moment of sudden wonder.",
            "fear": "A dark, atmospheric painting with shadowy, distorted figures and a looming storm cloud in the background, using deep greys and an unsettling green hue.",
            "disgust": "A complex, grotesque art piece with distorted, sickly-colored patterns and unsettling textures, with a jarring, uncomfortable composition."
        }

        art_description = prompt_map.get(detected_emotion, "An abstract image representing emotional well-being and balance.")

        return jsonify({"art_description": art_description})

    except Exception as e:
        print(f"Error generating art description: {e}")
        return jsonify({"error": "An error occurred with the art generation service."}), 500

if __name__ == '__main__':
    app.run(debug=True)
