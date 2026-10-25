# Practical 11 — Q-Learning for Tic-Tac-Toe: Performance Evaluation

import random
from collections import defaultdict

SEED = 42
random.seed(SEED)

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9

    def reset(self):
        self.board = [' '] * 9
        return tuple(self.board)

    def available_moves(self):
        return [i for i, cell in enumerate(self.board) if cell == ' ']

    def make_move(self, position, player):
        if self.board[position] != ' ':
            raise ValueError("Invalid move: cell is occupied.")
        self.board[position] = player

    def check_winner(self, player):
        return any(
            all(self.board[i] == player for i in line)
            for line in WIN_LINES
        )

    def is_draw(self):
        return (
            ' ' not in self.board
            and not self.check_winner('X')
            and not self.check_winner('O')
        )

    def show(self):
        for i in range(0, 9, 3):
            print(' | '.join(self.board[i:i+3]))
            if i < 6:
                print('--+---+--')


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.Q[(state, action)]

    def choose_action(self, state, actions, explore=True):
        # Exploration: try a random legal action.
        if explore and random.random() < self.epsilon:
            return random.choice(actions)

        # Exploitation: choose the action with the highest Q-value.
        values = [self.get_q(state, action) for action in actions]
        best_value = max(values)
        best_actions = [
            a for a, value in zip(actions, values)
            if value == best_value
        ]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_actions,
               terminal=False):
        old_value = self.get_q(state, action)

        future_value = (
            0
            if terminal or not next_actions
            else max(self.get_q(next_state, a) for a in next_actions)
        )

        target = reward + self.gamma * future_value

        self.Q[(state, action)] = (
            old_value + self.alpha * (target - old_value)
        )


def train_agent(episodes=20000, epsilon_start=1.0,
                epsilon_min=0.05, decay=0.999):
    game = TicTacToe()
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=epsilon_start
    )

    training_results = []

    for episode in range(episodes):
        state = game.reset()

        while True:
            actions = game.available_moves()
            action = agent.choose_action(state, actions, explore=True)
            game.make_move(action, 'X')

            # Agent wins
            if game.check_winner('X'):
                agent.update(
                    state, action, 1.0,
                    tuple(game.board), [],
                    terminal=True
                )
                training_results.append('win')
                break

            # Draw after agent's move
            if game.is_draw():
                agent.update(
                    state, action, 0.5,
                    tuple(game.board), [],
                    terminal=True
                )
                training_results.append('draw')
                break

            # Random opponent move
            opponent_action = random.choice(game.available_moves())
            game.make_move(opponent_action, 'O')

            # Opponent wins
            if game.check_winner('O'):
                agent.update(
                    state, action, -1.0,
                    tuple(game.board), [],
                    terminal=True
                )
                training_results.append('loss')
                break

            # Draw after opponent's move
            if game.is_draw():
                agent.update(
                    state, action, 0.5,
                    tuple(game.board), [],
                    terminal=True
                )
                training_results.append('draw')
                break

            # Continue with the state after the opponent replies
            next_state = tuple(game.board)
            next_actions = game.available_moves()

            agent.update(
                state, action, 0.0,
                next_state, next_actions
            )

            state = next_state

        # Gradually reduce exploration.
        agent.epsilon = max(
            epsilon_min,
            agent.epsilon * decay
        )

    return agent, training_results


EPISODES = 20000
agent, training_results = train_agent(episodes=EPISODES)

print("Training completed")
print("Training episodes:", EPISODES)
print("Learned state-action values:", len(agent.Q))
print("Final exploration rate (epsilon):", round(agent.epsilon, 4))


def evaluate_agent(agent, episodes=5000):
    game = TicTacToe()
    counts = {'win': 0, 'loss': 0, 'draw': 0}

    # Evaluation uses epsilon = 0, so no deliberate exploration.
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(episodes):
        game.reset()

        while True:
            state = tuple(game.board)
            action = agent.choose_action(
                state,
                game.available_moves(),
                explore=False
            )
            game.make_move(action, 'X')

            if game.check_winner('X'):
                counts['win'] += 1
                break

            if game.is_draw():
                counts['draw'] += 1
                break

            opponent_action = random.choice(game.available_moves())
            game.make_move(opponent_action, 'O')

            if game.check_winner('O'):
                counts['loss'] += 1
                break

            if game.is_draw():
                counts['draw'] += 1
                break

    agent.epsilon = old_epsilon

    rates = {
        key: value / episodes * 100
        for key, value in counts.items()
    }

    return counts, rates


TEST_EPISODES = 5000
test_counts, test_rates = evaluate_agent(agent, TEST_EPISODES)

print("\nTEST RESULTS (epsilon = 0)")
print("--------------------------")
print("Test episodes:", TEST_EPISODES)

for result in ('win', 'loss', 'draw'):
    print(
        f"{result.title():<8}: "
        f"{test_counts[result]:>5} "
        f"({test_rates[result]:.2f}%)"
    )

print("Total:", sum(test_counts.values()))
print(
    "Rate total: {:.2f}%".format(
        sum(test_rates.values())
    )
)
