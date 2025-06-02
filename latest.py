import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# NLP and ML libraries
from transformers import (
    AutoTokenizer, AutoModelForCausalLM
)
from sentence_transformers import SentenceTransformer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
import spacy

# Web And API

# Data processing
from sklearn.feature_extraction.text import TfidfVectorizer

# Audio processing (for integration with voice cloning)
import soundfile as sf
import librosa

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Directory Setup
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "personality_data"
MODELS_DIR = BASE_DIR / "personality_models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
DATABASE_DIR = BASE_DIR / "database"

for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR, DATABASE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "personality_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PersonalitySystem")


@dataclass
class PersonalityProfile:
    """Data structure for personality characteristics"""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    humor_style: str = "neutral"
    formality_level: float = 0.5
    emotional_expressiveness: float = 0.5
    vocabulary_complexity: float = 0.5
    response_length_preference: str = "medium"
    topics_of_interest: List[str] = None
    communication_patterns: Dict[str, float] = None

    def __post_init__(self):
        if self.topics_of_interest is None:
            self.topics_of_interest = []
        if self.communication_patterns is None:
            self.communication_patterns = {}


@dataclass
class ConversationContext:
    """Context for ongoing conversations"""
    conversation_id: str
    user_id: str
    messages: List[Dict] = None
    personality_adjustments: Dict[str, float] = None
    satisfaction_scores: List[float] = None
    topic_trends: List[str] = None
    created_at: datetime = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.personality_adjustments is None:
            self.personality_adjustments = {}
        if self.satisfaction_scores is None:
            self.satisfaction_scores = []
        if self.topic_trends is None:
            self.topic_trends = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


class DatabaseManager:
    """Manages SQLite database for personality data and interactions"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                personality_data TEXT,
                context_data TEXT
            )
        ''')

        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                satisfaction_score REAL,
                personality_match_score REAL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')

        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                message_id TEXT,
                feedback_type TEXT,
                feedback_value REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')

        # Personality profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personality_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT,
                training_data_count INTEGER,
                last_updated TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def save_conversation(self, context: ConversationContext):
        """Save conversation context to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO conversations 
            (id, user_id, created_at, last_updated, personality_data, context_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            context.conversation_id,
            context.user_id,
            context.created_at,
            context.last_updated,
            json.dumps(context.personality_adjustments),
            json.dumps(asdict(context))
        ))

        conn.commit()
        conn.close()

    def load_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Load conversation context from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT context_data FROM conversations WHERE id = ?', (conversation_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            data = json.loads(result[0])
            return ConversationContext(**data)
        return None


class PersonalityAnalyzer:
    """Analyzes text data to extract personality traits"""

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def analyze_text_corpus(self, texts: List[str]) -> PersonalityProfile:
        """Analyze a corpus of texts to extract personality traits"""
        if not texts:
            return PersonalityProfile()

        # Sentiment analysis
        sentiments = [self.sentiment_analyzer.polarity_scores(text) for text in texts]
        avg_sentiment = {
            key: np.mean([s[key] for s in sentiments])
            for key in ['pos', 'neg', 'neu', 'compound']
        }

        # Text complexity analysis
        complexities = [flesch_reading_ease(text) for text in texts if len(text) > 10]
        avg_complexity = np.mean(complexities) if complexities else 50

        # Response length analysis
        lengths = [len(text.split()) for text in texts]
        avg_length = np.mean(lengths)

        # Emotional expressiveness (exclamation marks, question marks, capitalization)
        emotion_indicators = []
        for text in texts:
            exclamations = text.count('!')
            questions = text.count('?')
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            emotion_indicators.append((exclamations + questions + caps_ratio * 10) / len(text) if text else 0)

        avg_emotion = np.mean(emotion_indicators)

        # Topic extraction
        topics = self._extract_topics(texts)

        # Communication patterns
        patterns = self._analyze_communication_patterns(texts)

        # Map to Big Five personality traits
        profile = PersonalityProfile(
            openness=min(1.0, max(0.0, (len(topics) / 10 + avg_complexity / 100))),
            conscientiousness=min(1.0, max(0.0, (1 - avg_sentiment['neg']) * 0.8)),
            extraversion=min(1.0, max(0.0, avg_emotion * 2)),
            agreeableness=min(1.0, max(0.0, avg_sentiment['pos'] * 1.2)),
            neuroticism=min(1.0, max(0.0, avg_sentiment['neg'] * 1.5)),
            formality_level=min(1.0, max(0.0, avg_complexity / 100)),
            emotional_expressiveness=min(1.0, max(0.0, avg_emotion)),
            vocabulary_complexity=min(1.0, max(0.0, avg_complexity / 100)),
            response_length_preference=self._categorize_length(avg_length),
            topics_of_interest=topics[:10],  # Top 10 topics
            communication_patterns=patterns
        )

        return profile

    def _extract_topics(self, texts: List[str]) -> List[str]:
        """Extract main topics from text corpus"""
        if not texts:
            return []

        # Combine all texts
        combined_text = ' '.join(texts)
        doc = self.nlp(combined_text)

        # Extract entities and noun phrases
        entities = [ent.text.lower() for ent in doc.ents]
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Count frequency
        topic_counts = defaultdict(int)
        for topic in entities + noun_phrases:
            if len(topic) > 2 and topic.isalpha():
                topic_counts[topic] += 1

        # Return top topics
        return sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)

    def _analyze_communication_patterns(self, texts: List[str]) -> Dict[str, float]:
        """Analyze communication patterns"""
        if not texts:
            return {}

        patterns = {
            'avg_sentence_length': np.mean([len(text.split('.')) for text in texts]),
            'question_frequency': np.mean([text.count('?') / len(text) if text else 0 for text in texts]),
            'exclamation_frequency': np.mean([text.count('!') / len(text) if text else 0 for text in texts]),
            'conjunction_usage': np.mean([len([w for w in text.split() if
                                               w.lower() in ['and', 'but', 'or', 'so', 'yet']]) / len(
                text.split()) if text.split() else 0 for text in texts]),
        }

        return patterns

    def _categorize_length(self, avg_length: float) -> str:
        """Categorize response length preference"""
        if avg_length < 10:
            return "short"
        elif avg_length < 30:
            return "medium"
        else:
            return "long"


class ReinforcementLearningAgent:
    """PPO-based reinforcement learning agent for personality adaptation"""

    def __init__(self, state_dim: int = 128, action_dim: int = 64, lr: float = 3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor-Critic networks
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5

        # Experience buffer
        self.memory = []
        self.batch_size = 64

    def _build_actor(self) -> nn.Module:
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh()  # Actions are bounded between -1 and 1
        )

    def _build_critic(self) -> nn.Module:
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from current policy"""
        state = state.to(self.device)
        action_mean = self.actor(state)
        action_std = torch.ones_like(action_mean) * 0.1  # Fixed std for simplicity

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate from critic"""
        state = state.to(self.device)
        return self.critic(state).squeeze()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })

    def update(self):
        """Update actor and critic networks using PPO"""
        if len(self.memory) < self.batch_size:
            return

        # Convert memory to tensors
        states = torch.stack([m['state'] for m in self.memory])
        actions = torch.stack([m['action'] for m in self.memory])
        rewards = torch.tensor([m['reward'] for m in self.memory], dtype=torch.float32)
        next_states = torch.stack([m['next_state'] for m in self.memory])
        dones = torch.tensor([m['done'] for m in self.memory], dtype=torch.bool)
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory])

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # Compute advantages using GAE
        values = self.get_value(states)
        next_values = self.get_value(next_states)
        next_values[dones] = 0

        advantages = torch.zeros_like(rewards)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[i] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(10):  # Multiple epochs
            # Sample mini-batch
            indices = torch.randperm(len(states))[:self.batch_size]

            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]

            # Actor update
            action_mean = self.actor(batch_states)
            action_std = torch.ones_like(action_mean) * 0.1
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().sum(dim=-1).mean()
            actor_loss = actor_loss - self.entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic update
            values = self.get_value(batch_states)
            critic_loss = F.mse_loss(values, batch_returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Clear memory
        self.memory.clear()

from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("PersonalitySystem")


class PersonalityGenerator:
    """Generates responses matching personality profile using PEFT-based fine-tuned language model"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Apply LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, peft_config).to(self.device)

        # Personality vector adapter
        self.personality_adapter = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.model.config.hidden_size),
            nn.Tanh()
        ).to(self.device)

        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_response(self, prompt: str, personality_profile: PersonalityProfile,
                          context: Optional[ConversationContext] = None, max_length: int = 150) -> str:
        personality_vector = self._encode_personality(personality_profile)
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        personality_embedding = self.personality_adapter(personality_vector)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=5,
                temperature=0.7 + personality_profile.emotional_expressiveness * 0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in response:
            response = response.replace(prompt, "").strip()
        return self._post_process_response(response, personality_profile)

    def fine_tune_on_data(self, training_texts: List[str], personality_profile: PersonalityProfile):
        logger.info("Fine-tuning model on personality data using LoRA...")

        class PersonalityDataset(Dataset):
            def __init__(self, texts, tokenizer):
                self.texts = texts
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'labels': encoding['input_ids'].squeeze(0)
                }

        dataset = PersonalityDataset(training_texts, self.tokenizer)

        training_args = TrainingArguments(
            output_dir="../peft_output",
            per_device_train_batch_size=2,
            num_train_epochs=3,
            logging_dir="../logs",
            logging_steps=1,
            save_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer
        )

        trainer.train()
        logger.info("LoRA fine-tuning completed.")

    def _encode_personality(self, profile: PersonalityProfile) -> torch.Tensor:
        features = [
            profile.openness,
            profile.conscientiousness,
            profile.extraversion,
            profile.agreeableness,
            profile.neuroticism,
            profile.formality_level,
            profile.emotional_expressiveness,
            profile.vocabulary_complexity,
            1.0 if profile.response_length_preference == "long" else 0.5 if profile.response_length_preference == "medium" else 0.0,
            len(profile.topics_of_interest) / 10.0,
            sum(profile.communication_patterns.values()) / len(profile.communication_patterns) if profile.communication_patterns else 0.5
        ]
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def _post_process_response(self, response: str, profile: PersonalityProfile) -> str:
        if profile.formality_level < 0.3:
            response = response.replace("cannot", "can't").replace("will not", "won't")
            response = response.replace("do not", "don't").replace("it is", "it's")

        if profile.emotional_expressiveness > 0.7:
            if not any(punct in response for punct in ['!', '?']):
                if np.random.random() < 0.3:
                    response += "!"

        sentences = response.split('.')
        if profile.response_length_preference == "short" and len(sentences) > 2:
            response = '.'.join(sentences[:2]) + '.'
        elif profile.response_length_preference == "long" and len(sentences) < 3:
            pass

        return response.strip()


class RewardCalculator:
    """Calculates rewards for reinforcement learning based on multiple factors"""

    def __init__(self):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def calculate_reward(self,
                         generated_response: str,
                         target_personality: PersonalityProfile,
                         user_feedback: Optional[float] = None,
                         context_relevance: float = 0.0,
                         conversation_context: Optional[ConversationContext] = None) -> float:
        """Calculate comprehensive reward score"""

        rewards = []

        # User feedback (if available) - highest weight
        if user_feedback is not None:
            rewards.append(("user_feedback", user_feedback, 0.4))

        # Personality alignment
        personality_score = self._calculate_personality_alignment(generated_response, target_personality)
        rewards.append(("personality_alignment", personality_score, 0.3))

        # Context relevance
        rewards.append(("context_relevance", context_relevance, 0.2))

        # Response quality metrics
        quality_score = self._calculate_response_quality(generated_response)
        rewards.append(("response_quality", quality_score, 0.1))

        # Calculate weighted average
        total_reward = sum(score * weight for _, score, weight in rewards)

        # Log reward components
        logger.info(f"Reward components: {dict((name, score) for name, score, _ in rewards)}")
        logger.info(f"Total reward: {total_reward}")

        return total_reward

    def _calculate_personality_alignment(self, response: str, personality: PersonalityProfile) -> float:
        """Calculate how well response aligns with personality profile"""
        scores = []

        # Sentiment alignment with personality traits
        sentiment = self.sentiment_analyzer.polarity_scores(response)

        # Extraversion vs sentiment positivity
        extraversion_score = min(1.0, abs(personality.extraversion - 0.5) * 2 * sentiment['pos'])
        scores.append(extraversion_score)

        # Emotional expressiveness vs punctuation usage
        emotion_indicators = response.count('!') + response.count('?')
        emotion_score = min(1.0,
                            personality.emotional_expressiveness * (emotion_indicators / max(1, len(response.split()))))
        scores.append(emotion_score)

        # Response length alignment
        word_count = len(response.split())
        if personality.response_length_preference == "short" and word_count <= 15:
            scores.append(1.0)
        elif personality.response_length_preference == "medium" and 15 < word_count <= 50:
            scores.append(1.0)
        elif personality.response_length_preference == "long" and word_count > 50:
            scores.append(1.0)
        else:
            scores.append(0.5)

        return np.mean(scores)

    def _calculate_response_quality(self, response: str) -> float:
        """Calculate general response quality"""
        if not response or len(response.strip()) < 3:
            return 0.0

        scores = []

        # Length appropriateness (not too short, not too long)
        word_count = len(response.split())
        if 5 <= word_count <= 100:
            scores.append(1.0)
        else:
            scores.append(max(0.0, 1.0 - abs(word_count - 50) / 50))

        # Coherence (simple heuristic based on sentence structure)
        sentences = response.split('.')
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        coherence_score = min(1.0, len(valid_sentences) / max(1, len(sentences)))
        scores.append(coherence_score)

        # Repetition penalty
        words = response.lower().split()
        unique_words = len(set(words))
        repetition_score = unique_words / max(1, len(words))
        scores.append(repetition_score)

        return np.mean(scores)


class PersonalitySystem:
    """Main system integrating all components"""

    def __init__(self, voice_cloning_module=None):
        self.db_manager = DatabaseManager(DATABASE_DIR / "personality_system.db")
        self.personality_analyzer = PersonalityAnalyzer()
        self.personality_generator = PersonalityGenerator()
        self.rl_agent = ReinforcementLearningAgent()
        self.reward_calculator = RewardCalculator()
        self.voice_cloning_module = voice_cloning_module

        # Active conversations
        self.active_conversations: Dict[str, ConversationContext] = {}

        # User profiles
        self.user_profiles: Dict[str, PersonalityProfile] = {}

        logger.info("PersonalitySystem initialized successfully")

    def _personality_summary(self, profile: PersonalityProfile) -> str:
        return (
            f"Traits:\n"
            f"- Openness: {profile.openness:.2f}\n"
            f"- Conscientiousness: {profile.conscientiousness:.2f}\n"
            f"- Extraversion: {profile.extraversion:.2f}\n"
            f"- Agreeableness: {profile.agreeableness:.2f}\n"
            f"- Neuroticism: {profile.neuroticism:.2f}\n"
            f"- Formality: {profile.formality_level:.2f}\n"
            f"- Emotional Expressiveness: {profile.emotional_expressiveness:.2f}\n"
            f"- Vocabulary Complexity: {profile.vocabulary_complexity:.2f}\n"
            f"- Preferred Length: {profile.response_length_preference}\n"
        )

    def create_personality_profile(self, user_id: str, training_texts: List[str]) -> PersonalityProfile:
        """Create personality profile from training data"""
        logger.info(f"Creating personality profile for user {user_id}")

        profile = self.personality_analyzer.analyze_text_corpus(training_texts)
        self.user_profiles[user_id] = profile

        # Save to database
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO personality_profiles 
            (user_id, profile_data, training_data_count, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (user_id, json.dumps(asdict(profile)), len(training_texts), datetime.now().isoformat()))
        conn.commit()
        conn.close()

        # Fine-tune generator on user data
        self.personality_generator.fine_tune_on_data(training_texts, profile)

        logger.info(f"Personality profile created for user {user_id}")
        return profile

    def start_conversation(self, user_id: str) -> str:
        """Start a new conversation"""
        conversation_id = str(uuid.uuid4())

        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id
        )

        self.active_conversations[conversation_id] = context
        self.db_manager.save_conversation(context)

        logger.info(f"Started conversation {conversation_id} for user {user_id}")
        return conversation_id

    def generate_response(self,
                          conversation_id: str,
                          user_message: str,
                          include_voice: bool = True) -> Dict[str, Any]:
        """Generate personality-matched response with optional voice synthesis"""

        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        context = self.active_conversations[conversation_id]
        user_id = context.user_id

        if user_id not in self.user_profiles:
            raise ValueError(f"No personality profile found for user {user_id}")

        personality_profile = self.user_profiles[user_id]

        # Add user message to context
        context.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # Generate response using RL-adapted personality
        current_state = self._encode_conversation_state(context, personality_profile)
        action, log_prob = self.rl_agent.get_action(current_state)

        # Apply RL adjustments to personality
        adjusted_personality = self._apply_rl_adjustments(personality_profile, action)

        # Generate text response
        prompt = self._build_conversation_prompt(context)
        text_response = self.personality_generator.generate_response(
            prompt, adjusted_personality, context
        )

        # Add assistant response to context
        response_entry = {
            "role": "assistant",
            "content": text_response,
            "timestamp": datetime.now().isoformat(),
            "rl_action": action.cpu().numpy().tolist(),
            "log_prob": log_prob.item()
        }
        context.messages.append(response_entry)

        # Generate voice if requested and voice cloning module is available
        audio_path = None
        if include_voice and self.voice_cloning_module:
            try:
                # Get user's reference audio (this should be set up during profile creation)
                reference_audio = self._get_user_reference_audio(user_id)
                if reference_audio:
                    audio_path = self.voice_cloning_module.clone_voice(reference_audio, text_response)
                    logger.info(f"Voice synthesis completed: {audio_path}")
            except Exception as e:
                logger.error(f"Voice synthesis failed: {e}")

        # Update context
        context.last_updated = datetime.now().isoformat()
        self.active_conversations[conversation_id] = context
        self.db_manager.save_conversation(context)

        return {
            "text_response": text_response,
            "audio_path": audio_path,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "personality_adjustments": action.cpu().numpy().tolist()
        }

    def provide_feedback(self,
                         conversation_id: str,
                         message_index: int,
                         feedback_score: float,
                         feedback_type: str = "satisfaction"):
        """Provide feedback for reinforcement learning"""

        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        context = self.active_conversations[conversation_id]

        if message_index >= len(context.messages):
            raise ValueError("Invalid message index")

        message = context.messages[message_index]
        if message["role"] != "assistant":
            raise ValueError("Can only provide feedback for assistant messages")

        # Store feedback
        context.satisfaction_scores.append(feedback_score)

        # Calculate reward
        user_id = context.user_id
        personality_profile = self.user_profiles[user_id]

        reward = self.reward_calculator.calculate_reward(
            generated_response=message["content"],
            target_personality=personality_profile,
            user_feedback=feedback_score,
            conversation_context=context
        )

        # Update RL agent
        if len(context.messages) >= 2:  # Need at least user message and assistant response
            current_state = self._encode_conversation_state(context, personality_profile)
            action = torch.tensor(message["rl_action"], dtype=torch.float32)
            log_prob = torch.tensor(message["log_prob"], dtype=torch.float32)

            # For simplicity, use current state as next state (in practice, you'd use the next conversation state)
            next_state = current_state
            done = False

            self.rl_agent.store_transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob
            )

            # Update RL agent if enough transitions collected
            if len(self.rl_agent.memory) >= self.rl_agent.batch_size:
                self.rl_agent.update()

        # Save feedback to database
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback 
            (id, conversation_id, message_id, feedback_type, feedback_value, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            conversation_id,
            str(message_index),
            feedback_type,
            feedback_score,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

        logger.info(f"Feedback provided: {feedback_score} for conversation {conversation_id}")

    def _encode_conversation_state(self, context: ConversationContext, personality: PersonalityProfile) -> torch.Tensor:
        """Encode conversation state for RL agent"""
        features = []

        # Personality features
        features.extend([
            personality.openness,
            personality.conscientiousness,
            personality.extraversion,
            personality.agreeableness,
            personality.neuroticism,
            personality.formality_level,
            personality.emotional_expressiveness,
            personality.vocabulary_complexity
        ])

        # Conversation features
        features.extend([
            len(context.messages) / 50.0,  # Normalized conversation length
            np.mean(context.satisfaction_scores) if context.satisfaction_scores else 0.5,
            len(context.topic_trends) / 10.0,  # Normalized topic diversity
        ])

        # Recent message features
        if context.messages:
            recent_message = context.messages[-1]["content"]
            features.extend([
                len(recent_message.split()) / 50.0,  # Normalized message length
                recent_message.count('!') / max(1, len(recent_message)),  # Exclamation density
                recent_message.count('?') / max(1, len(recent_message)),  # Question density
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # Pad or truncate to fixed size
        target_size = 128
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]

        return torch.tensor(features, dtype=torch.float32)

    def _apply_rl_adjustments(self, base_personality: PersonalityProfile, action: torch.Tensor) -> PersonalityProfile:
        """Apply RL action to adjust personality traits"""
        adjustments = action.cpu().numpy()

        # Scale adjustments to reasonable range
        adjustments = adjustments * 0.1  # Limit adjustments to ±0.1

        # Create adjusted personality
        adjusted = PersonalityProfile(
            openness=np.clip(base_personality.openness + adjustments[0], 0.0, 1.0),
            conscientiousness=np.clip(base_personality.conscientiousness + adjustments[1], 0.0, 1.0),
            extraversion=np.clip(base_personality.extraversion + adjustments[2], 0.0, 1.0),
            agreeableness=np.clip(base_personality.agreeableness + adjustments[3], 0.0, 1.0),
            neuroticism=np.clip(base_personality.neuroticism + adjustments[4], 0.0, 1.0),
            formality_level=np.clip(base_personality.formality_level + adjustments[5], 0.0, 1.0),
            emotional_expressiveness=np.clip(base_personality.emotional_expressiveness + adjustments[6], 0.0, 1.0),
            vocabulary_complexity=np.clip(base_personality.vocabulary_complexity + adjustments[7], 0.0, 1.0),
            response_length_preference=base_personality.response_length_preference,
            topics_of_interest=base_personality.topics_of_interest,
            communication_patterns=base_personality.communication_patterns
        )

        return adjusted

    def _build_conversation_prompt(self, context: ConversationContext) -> str:
        personality_profile = self.user_profiles.get(context.user_id)
        prompt_parts = []

        # 👉 Inject personality as natural language context
        if personality_profile:
            prompt_parts.append("The assistant has the following personality profile:\n")
            prompt_parts.append(self._personality_summary(personality_profile).strip())

        # Add recent messages
        recent_messages = context.messages[-5:] if len(context.messages) > 5 else context.messages
        for message in recent_messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                prompt_parts.append(f"Human: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")

        # Add Assistant: prompt
        if context.messages and context.messages[-1]["role"] == "user":
            prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _get_user_reference_audio(self, user_id: str) -> Optional[str]:
        """Get user's reference audio file path"""
        # This should return the path to the user's reference audio file
        # In practice, this would be stored during user registration
        reference_audio_path = DATA_DIR / f"{user_id}_reference.wav"
        return str(reference_audio_path) if reference_audio_path.exists() else None

    def load_user_profile(self, user_id: str) -> Optional[PersonalityProfile]:
        """Load user profile from database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT profile_data FROM personality_profiles WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            profile_data = json.loads(result[0])
            profile = PersonalityProfile(**profile_data)
            self.user_profiles[user_id] = profile
            return profile
        return None

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get conversation history"""
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id].messages

        # Load from database
        context = self.db_manager.load_conversation(conversation_id)
        if context:
            return context.messages
        return []

    def save_user_reference_audio(self, user_id: str, audio_file_path: str):
        """Save user's reference audio for voice cloning"""
        reference_path = DATA_DIR / f"{user_id}_reference.wav"

        # Copy and process audio file
        try:
            # Load and standardize audio
            y, sr = librosa.load(audio_file_path, sr=16000)
            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.util.normalize(y)

            # Save processed audio
            sf.write(str(reference_path), y, 16000)
            logger.info(f"Reference audio saved for user {user_id}")

        except Exception as e:
            logger.error(f"Failed to save reference audio: {e}")
            raise

