import random
import re

class MCQGenerator:
    """
    AI Agent for Automatic Multiple Choice Question (MCQ) Generation.
    Generates MCQs with difficulty levels (Easy, Medium, Hard), answers, and explanations.
    """
    def __init__(self):
        pass

    def generate_mcqs(self, full_text, num_questions=5, difficulty="Medium"):
        """
        Generate MCQs from document text.
        Difficulty levels: 'Easy', 'Medium', 'Hard'.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', full_text) if len(s.strip()) > 30]
        if not sentences:
            sentences = [full_text]

        mcqs = []
        words_vocab = list(set(re.findall(r'\b[A-Za-z]{4,15}\b', full_text)))
        if len(words_vocab) < 10:
            words_vocab += ["Information", "Document", "System", "Analysis", "Data", "Process", "Record", "Model"]

        for i in range(min(num_questions, len(sentences))):
            sent = sentences[i % len(sentences)]
            words = re.findall(r'\b[A-Za-z]{4,15}\b', sent)
            
            if not words:
                continue

            # Pick target blank word based on difficulty
            if difficulty == "Easy":
                target_word = max(words, key=len)
            elif difficulty == "Hard":
                target_word = words[0]
            else: # Medium
                target_word = sorted(words, key=len)[len(words)//2]

            question_text = sent.replace(target_word, "______")

            # Generate distractors
            distractors = [w for w in words_vocab if w.lower() != target_word.lower() and abs(len(w) - len(target_word)) <= 4]
            random.shuffle(distractors)
            distractors = distractors[:3]

            while len(distractors) < 3:
                distractors.append(f"Option_{len(distractors)+1}")

            options = distractors + [target_word]
            random.shuffle(options)

            correct_letter = chr(65 + options.index(target_word))

            mcqs.append({
                "question_num": i + 1,
                "question": f"Fill in the blank: \"{question_text}\"",
                "difficulty": difficulty,
                "options": {
                    "A": options[0],
                    "B": options[1],
                    "C": options[2],
                    "D": options[3]
                },
                "correct_answer": correct_letter,
                "correct_text": target_word,
                "explanation": f"The term '{target_word}' completes the sentence directly from the document: '{sent}'"
            })

        return mcqs
