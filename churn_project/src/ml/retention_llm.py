import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_retention_message(churn_reasons):
    """
    Takes a list of reasons (from SHAP) a customer is churning,
    and calls the Gemini API to generate a personalized retention message.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    prompt = f"""
    You are an expert customer retention specialist. 
    We have identified a customer who is at high risk of churning (canceling our service).
    
    Based on our machine learning analysis, here are the top reasons this customer is likely to churn:
    {', '.join(churn_reasons)}
    
    Write a short, personalized, empathetic email to this customer. 
    Acknowledge their potential frustration (without explicitly saying "our ML model predicted you will leave")
    and offer a relevant solution or discount tailored to their specific pain points.
    
    Keep the tone helpful, professional, and warm. Limit it to 3-4 sentences.
    """
    
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found in environment. Generating a mock message.")
        return f"[MOCK MESSAGE] We noticed you might be unhappy due to: {', '.join(churn_reasons)}. Please reach out to our support team so we can find a personalized solution for you!"
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Sorry, we couldn't generate a personalized message at this time."
