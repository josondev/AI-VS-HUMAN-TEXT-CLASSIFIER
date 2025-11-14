# AI vs Human Text Detector - Production Ready

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class AITextDetector:
    """Production-ready AI text detector."""

    def __init__(self):
        self.max_features = 15000
        self.ngram_range = (1, 3)
        self.alpha = 0.01
        self.min_df = 2
        self.max_df = 0.9

        self.vectorizer = None
        self.classifier = None
        self.is_trained = False

    def model_exists(self, model_dir='model'):
        """Check if model exists."""
        files = ['vectorizer.joblib', 'classifier.joblib', 'config.joblib']
        return all(os.path.exists(os.path.join(model_dir, f)) for f in files)

    def train(self, filepath, test_size=0.2):
        """Train model."""
        print(f"\nLoading: {filepath}")
        data = pd.read_csv(filepath)
        X_text = data['text'].astype(str).values
        y = data['generated'].astype(int).values

        print(f"Total: {len(X_text)} | Human: {(y==0).sum()} | AI: {(y==1).sum()}")

        print(f"\nVectorizing...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )

        X = self.vectorizer.fit_transform(X_text)
        print(f"Features: {X.shape[1]}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nTraining (alpha={self.alpha})...")
        self.classifier = MultinomialNB(alpha=self.alpha, fit_prior=True)
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\n" + classification_report(y_test, y_pred, target_names=['Human', 'AI']))

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"           Pred Human  Pred AI")
        print(f"True Human    {cm[0,0]:<8}    {cm[0,1]:<8}")
        print(f"True AI       {cm[1,0]:<8}    {cm[1,1]:<8}")

        self.is_trained = True
        return accuracy

    def save_model(self, model_dir='model'):
        """Save model."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
        joblib.dump(self.classifier, os.path.join(model_dir, 'classifier.joblib'))
        joblib.dump({
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'alpha': self.alpha
        }, os.path.join(model_dir, 'config.joblib'))
        print(f"✅ Saved to {model_dir}/")

    def load_model(self, model_dir='model'):
        """Load model."""
        self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        self.classifier = joblib.load(os.path.join(model_dir, 'classifier.joblib'))
        config = joblib.load(os.path.join(model_dir, 'config.joblib'))
        self.max_features = config['max_features']
        self.ngram_range = config['ngram_range']
        self.alpha = config['alpha']
        self.is_trained = True

    def predict(self, text):
        """Predict text."""
        if not self.is_trained:
            raise ValueError("Model not loaded!")

        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]

        class_to_idx = {cls: idx for idx, cls in enumerate(self.classifier.classes_)}
        human_prob = float(probs[class_to_idx[0]])
        ai_prob = float(probs[class_to_idx[1]])

        label = "AI" if ai_prob > 0.5 else "Human"
        return label, ai_prob, human_prob

    def interactive(self):
        """Interactive CLI."""
        data_path = "balanced_ai_human_prompts.csv"
        model_dir = "model"

        while True:
            print("\n" + "="*50)
            print("AI TEXT DETECTOR")
            print("="*50)
            print(f"Model: {'✅' if self.model_exists(model_dir) else '❌'}")
            print("\n1. Train")
            print("2. Predict") 
            print("3. Exit")

            choice = input("\nSelect (1-3): ").strip()

            if choice == '1':
                print("\n" + "="*50)
                print("TRAINING")
                print("="*50)
                if os.path.exists(data_path):
                    self.train(data_path)
                    self.save_model(model_dir)
                    print("\n✅ Done!")
                else:
                    print(f"❌ File not found: {data_path}")

            elif choice == '2':
                if not self.model_exists(model_dir):
                    print("\n❌ No model! Train first (option 1)")
                    continue

                self.load_model(model_dir)

                print("\n" + "="*50)
                print("PREDICT MODE")
                print("="*50)

                while True:
                    text = input("\nText (or 'back'): ").strip()

                    if text.lower() in ['back', 'exit']:
                        break

                    if not text:
                        continue

                    label, ai_prob, human_prob = self.predict(text)
                    print(f"\n→ {label} | AI: {ai_prob:.1%} | Human: {human_prob:.1%}")

                    again = input("\nAnother? (y/n): ").strip().lower()
                    if again != 'y':
                        break

            elif choice == '3':
                print("\nBye!")
                break

            else:
                print("Invalid choice!")


if __name__ == "__main__":
    detector = AITextDetector()
    detector.interactive()
