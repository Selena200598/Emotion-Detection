from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the emotion classification model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Classify emotions
    results = emotion_classifier(text)
    # Assuming the model outputs scores for emotions
    # Map to our desired emotions: happy, sad, angry, surprise, fear, neutral
    emotion_map = {
        'joy': 'happy',
        'sadness': 'sad',
        'anger': 'angry',
        'surprise': 'surprise',
        'fear': 'fear',
        'disgust': 'neutral',  # Map disgust to neutral or handle differently
        'neutral': 'neutral'
    }
    
    # Get the emotion with the highest score
    top_emotion = max(results[0], key=lambda x: x['score'])
    predicted_emotion = emotion_map.get(top_emotion['label'].lower(), 'neutral')
    
    return jsonify({'emotion': predicted_emotion, 'confidence': top_emotion['score']})

if __name__ == '__main__':
    app.run(debug=True)
