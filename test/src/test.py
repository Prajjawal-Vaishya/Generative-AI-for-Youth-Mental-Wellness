import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# --- Configuration ---
# Load environment variables from a .env file.
# Make sure your .env file contains:
# GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"
load_dotenv()

# Your Google Cloud project details.
PROJECT_ID = "lumaveda-ai"
LOCATION = "asia-south1"
MODEL_NAME = "gemini-1.5-flash"


def test_gemini():
    """
    Initializes Vertex AI, sends a prompt to the Gemini model,
    and prints the response.
    """
    try:
        # Initialize the Vertex AI SDK.
        # The SDK will automatically use the credentials from the
        # GOOGLE_APPLICATION_CREDENTIALS environment variable.
        print("Initializing Vertex AI...")
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        # Load the specified generative model.
        print(f"Loading model: {MODEL_NAME}...")
        gemini_model = GenerativeModel(MODEL_NAME)

        # Define a test prompt to send to the model.
        prompt = "Write a short motivational quote for students facing stress."
        print(f"Sending prompt: \"{prompt}\"")

        # Generate content using the model.
        response = gemini_model.generate_content(prompt)

        # Print the model's response text.
        print("\n✅ Gemini AI Response:")
        # Access the text content from the response object.
        print(response.text)

    except Exception as e:
        # Catch and print any errors that occur.
        print(f"❌ An error occurred: {e}")
        print("\n--- Troubleshooting ---")
        print("1. Ensure you have run 'gcloud auth application-default login'.")
        print("2. Make sure the GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly in your .env file.")
        print(f"3. Check that the Vertex AI API is enabled for the project '{PROJECT_ID}'.")
        print("4. Verify the project ID and location are correct.")


if __name__ == "__main__":
    test_gemini()
