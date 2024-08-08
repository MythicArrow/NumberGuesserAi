import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

class NumberGuessingNN(tf.keras.Model):
    def __init__(self):
        super(NumberGuessingNN, self).__init__()
        self.dense1 = Dense(10, activation='relu', input_shape=(2,))
        self.dense2 = Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

class NumberGuessingAI:
    def __init__(self, lower_bound=1, upper_bound=100):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.model = NumberGuessingNN()
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
        self.reset()

    def reset(self):
        self.low = self.lower_bound
        self.high = self.upper_bound
        self.guess_history = []

    def make_guess(self):
        if not self.guess_history:
            guess = random.randint(self.low, self.high)
        else:
            last_guess, feedback = self.guess_history[-1]
            input_tensor = np.array([[last_guess, feedback]], dtype=np.float32)
            guess = self.model.predict(input_tensor)[0][0]
            guess = max(self.low, min(self.high, int(round(guess))))
        self.current_guess = guess
        return self.current_guess

    def receive_feedback(self, feedback):
        if feedback == 'correct':
            return
        feedback_value = 1 if feedback == 'too high' else -1
        self.guess_history.append((self.current_guess, feedback_value))
        self._train()
        if feedback == 'too high':
            self.high = self.current_guess - 1
        elif feedback == 'too low':
            self.low = self.current_guess + 1

    def _train(self):
        if len(self.guess_history) < 2:
            return
        last_guess, last_feedback = self.guess_history[-2]
        current_guess, _ = self.guess_history[-1]
        input_tensor = np.array([[last_guess, last_feedback]], dtype=np.float32)
        target_tensor = np.array([[current_guess]], dtype=np.float32)
        self.model.train_on_batch(input_tensor, target_tensor)

# Main game loop
while True:
    ai = NumberGuessingAI()
    target_number = int(input("Write a number between 1 and 100"))
    print(f"Target number is: {target_number}")

    attempt = 1
    while True:
        guess = ai.make_guess()
        print(f"Attempt {attempt}: AI guesses {guess}")

        if guess == target_number:
            print("AI guessed correctly!")
            break
        elif guess > target_number:
            ai.receive_feedback('too high')
        else:
            ai.receive_feedback('too low')

        attempt += 1

    play_again = input("Would you like to play again? (yes/no): ").strip().lower()
    if play_again != 'yes':
        break

print("Thank you for playing!")

