"""
Emotion Detection System
Classifies text into emotions: happy, sad, angry, surprise, fear, or neutral
"""

import re
from collections import Counter
from typing import Dict, List, Tuple
import string
from flask import Flask, request, jsonify

class EmotionDetector:
    def __init__(self):
        # Emotion keywords dictionary
        self.emotion_keywords = {
            'happy': [
                'happy', 'joy', 'excited', 'great', 'awesome', 'wonderful', 'fantastic',
                'love', 'excellent', 'amazing', 'perfect', 'good', 'best', 'beautiful',
                'glad', 'delighted', 'pleased', 'cheerful', 'blessed', 'grateful',
                'wonderful', 'brilliant', 'fabulous', 'yay', 'woohoo', 'celebrate'
            ],
            'sad': [
                'sad', 'unhappy', 'depressed', 'miserable', 'disappointed', 'hurt',
                'lonely', 'down', 'cry', 'tears', 'upset', 'heartbroken', 'awful',
                'terrible', 'bad', 'worst', 'horrible', 'unfortunate', 'regret',
                'sorry', 'miss', 'grief', 'sorrow', 'despair', 'gloomy'
            ],
            'angry': [
                'angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated',
                'frustrated', 'outraged', 'pissed', 'disgusted', 'stupid', 'idiot',
                'damn', 'hell', 'fuck', 'shit', 'annoying', 'pathetic', 'ridiculous',
                'unacceptable', 'sick', 'fed up', 'infuriated', 'livid'
            ],
            'surprise': [
                'wow', 'omg', 'shocked', 'surprised', 'unexpected', 'amazing',
                'unbelievable', 'incredible', 'astonished', 'stunned', 'whoa',
                'unexpected', 'sudden', 'shock', 'startled', 'speechless',
                'mind-blowing', 'wtf', 'remarkable', 'extraordinary'
            ],
            'fear': [
                'scared', 'afraid', 'fear', 'worried', 'anxious', 'nervous',
                'terrified', 'panic', 'frightened', 'concern', 'threat', 'danger',
                'risk', 'scary', 'horror', 'dread', 'alarmed', 'paranoid',
                'uneasy', 'tense', 'stress', 'nightmare', 'phobia'
            ]
        }
        
        # Emotion intensifiers
        self.intensifiers = [
            'very', 'really', 'extremely', 'so', 'incredibly', 'absolutely',
            'totally', 'completely', 'utterly', 'super', 'quite'
        ]
        
        # Negation words
        self.negations = [
            'not', 'no', 'never', 'neither', 'nobody', 'nothing',
            'nowhere', 'none', "n't", 'hardly', 'barely'
        ]
        
        # Emotion emojis
        self.emotion_emojis = {
            'happy': ['😊', '😀', '😃', '😄', '😁', '🙂', '😍', '🥰', '😘', '❤️', '💕', '🎉', '👍', '✨'],
            'sad': ['😢', '😭', '😞', '😔', '😟', '🙁', '☹️', '💔', '😥', '😪'],
            'angry': ['😠', '😡', '🤬', '😤', '💢', '👿', '😾'],
            'surprise': ['😲', '😮', '😯', '😳', '🤯', '‼️', '⁉️'],
            'fear': ['😨', '😰', '😱', '🙀', '😧', '😦']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
        return text
    
    def detect_emojis(self, text: str) -> Dict[str, int]:
        """Count emotion-related emojis in text"""
        emoji_scores = {emotion: 0 for emotion in self.emotion_keywords.keys()}
        
        for emotion, emojis in self.emotion_emojis.items():
            for emoji in emojis:
                emoji_scores[emotion] += text.count(emoji)
        
        return emoji_scores
    
    def calculate_emotion_scores(self, text: str) -> Dict[str, float]:
        """Calculate scores for each emotion based on keywords"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
        
        # Add emoji scores
        emoji_scores = self.detect_emojis(text)
        for emotion, score in emoji_scores.items():
            emotion_scores[emotion] += score * 2.0  # Weight emojis more
        
        # Analyze words with context
        for i, word in enumerate(words):
            # Remove punctuation
            word_clean = word.strip(string.punctuation)
            
            # Check for negation in previous words
            is_negated = False
            if i > 0 and words[i-1] in self.negations:
                is_negated = True
            
            # Check for intensifiers
            intensity = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensity = 1.5
            
            # Match word with emotion keywords
            for emotion, keywords in self.emotion_keywords.items():
                if word_clean in keywords:
                    score = intensity
                    
                    # If negated, reduce score significantly
                    if is_negated:
                        score *= -0.5
                    
                    emotion_scores[emotion] += score
        
        return emotion_scores
    
    def predict_emotion(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Predict the dominant emotion in the text
        Returns: (emotion_label, confidence_scores)
        """
        if not text or not text.strip():
            return 'neutral', {}
        
        emotion_scores = self.calculate_emotion_scores(text)
        
        # If no emotions detected, return neutral
        if all(score == 0 for score in emotion_scores.values()):
            return 'neutral', emotion_scores
        
        # Find dominant emotion
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[max_emotion]
        
        # If score is too low, consider it neutral
        if max_score < 0.5:
            return 'neutral', emotion_scores
        
        return max_emotion, emotion_scores

app = Flask(__name__)

# Initialize the emotion detector
detector = EmotionDetector()

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    emotion, scores = detector.predict_emotion(text)
    
    return jsonify({'emotion': emotion, 'scores': scores})

if __name__ == '__main__':
    app.run(debug=True)
