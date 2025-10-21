import pandas as pd
import os
import logging
from datetime import datetime

class UserFeedback:
    def __init__(self):
        self.log_dir = './logs'
        os.makedirs(self.log_dir, exist_ok=True)
        self.feedback_file = os.path.join(self.log_dir, 'feedback_log.csv')
        self.logger = logging.getLogger(__name__)
        if not os.path.exists(self.feedback_file):
            pd.DataFrame(columns=['user_id', 'signal_id', 'feedback', 'timestamp']).to_csv(self.feedback_file, index=False)

    def log_feedback(self, user_id, signal_id, feedback):
        """Log user feedback from Telegram."""
        try:
            feedback_data = {
                'user_id': user_id,
                'signal_id': signal_id,
                'feedback': feedback,  # e.g., 'Y' (accurate), 'N' (inaccurate)
                'timestamp': datetime.now().isoformat()
            }
            df = pd.DataFrame([feedback_data])
            df.to_csv(self.feedback_file, mode='a', header=False, index=False)
            self.logger.info(f"Logged feedback: {feedback_data}")
        except Exception as e:
            self.logger.error(f"Failed to log feedback: {e}")

    def analyze_feedback(self, signal_id):
        """Analyze feedback for a specific signal."""
        try:
            df = pd.read_csv(self.feedback_file)
            signal_feedback = df[df['signal_id'] == signal_id]
            total = len(signal_feedback)
            if total == 0:
                return {'positive_rate': 0.0, 'count': 0}
            positive = len(signal_feedback[signal_feedback['feedback'] == 'Y'])
            positive_rate = positive / total if total > 0 else 0.0
            return {'positive_rate': positive_rate, 'count': total}
        except Exception as e:
            self.logger.error(f"Failed to analyze feedback: {e}")
            return {'positive_rate': 0.0, 'count': 0}

    def get_feedback_weight(self, signal_id):
        """Calculate weight adjustment based on feedback."""
        analysis = self.analyze_feedback(signal_id)
        positive_rate = analysis['positive_rate']
        return max(0.5, min(1.0, positive_rate))  # Weight between 0.5 and 1.0