import random
import tkinter as tk
from tkinter import ttk
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_weights(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])

    def set_weights(self, weights):
        start = 0
        for param in self.parameters():
            end = start + param.numel()
            param.data = weights[start:end].reshape(param.shape)
            start = end


class SnakeGame:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.distance_to_food_variation = []
        self.food = self.generate_food()
        self.score = 0
        self.steps = 50
        self.total_steps_used = 0
        self.action_list = []

    def generate_food(self):
        while True:
            food = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            if food not in self.snake:
                return food

    def get_state(self):
        head = self.snake[0]
        state = [
            int(self.direction == (1, 0)),  # Moving right
            int(self.direction == (-1, 0)),  # Moving left
            int(self.direction == (0, 1)),  # Moving down
            int(self.direction == (0, -1)),  # Moving up
            int(head[0] < self.food[0]),  # Food is to the right
            int(head[0] > self.food[0]),  # Food is to the left
            int(head[1] < self.food[1]),  # Food is below
            int(head[1] > self.food[1]),  # Food is above
            int(self.is_collision((head[0] + 1, head[1]))),  # Danger right
            int(self.is_collision((head[0] - 1, head[1]))),  # Danger left
            int(self.is_collision((head[0], head[1] + 1))),  # Danger down
            int(self.is_collision((head[0], head[1] - 1))),  # Danger up
        ]
        return torch.tensor(state, dtype=torch.float)

    def is_collision(self, pos):
        return (
            pos in self.snake[1:]
            or pos[0] < 0
            or pos[0] >= self.width
            or pos[1] < 0
            or pos[1] >= self.height
        )

    def step(self, action):
        # 0: straight, 1: right turn, 2: left turn
        if action == 1:
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:
            self.direction = (-self.direction[1], self.direction[0])

        self.action_list.append(action)

        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        )
        self.snake.insert(0, new_head)

        self.distance_to_food_variation.append(
            (
                abs(self.snake[1][0] - self.food[0])
                + abs(self.snake[1][1] - self.food[1])
            )
            - (abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1]))
        )

        if new_head == self.food:
            self.score += 1
            self.steps += 150
            self.food = self.generate_food()
        else:
            self.snake.pop()

        self.steps -= 1
        self.total_steps_used += 1
        game_over = self.is_collision(new_head) or self.steps <= 0
        return game_over, self.score


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [NeuralNetwork(12, 16, 16, 3) for _ in range(population_size)]

    def select_parents(self, fitnesses, select_best=False):
        # Select the best fitness individual
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            # If all fitnesses are zero, select randomly
            random_parent = random.choice(self.population)
            return random_parent, random_parent

        best_index = fitnesses.index(max(fitnesses))
        best_parent = self.population[best_index]

        if select_best:
            return best_parent, best_parent

        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitnesses):
            current += fitness
            if current > pick:
                random_parent = self.population[i]
                break
        else:
            # Fallback to last individual if something goes wrong
            random_parent = self.population[-1]

        return best_parent, random_parent

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(12, 16, 16, 3)
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        crossover_point = random.randint(0, len(weights1))
        new_weights = torch.cat(
            (weights1[:crossover_point], weights2[crossover_point:])
        )
        child.set_weights(new_weights)
        return child

    def mutate(self, individual):
        weights = individual.get_weights()
        mutations = torch.randn_like(weights) * self.mutation_rate
        new_weights = weights + mutations
        individual.set_weights(new_weights)

    def evolve(self, fitnesses):
        new_population = []
        print("Evolving...", max(fitnesses), sum(fitnesses) / len(fitnesses))
        parent1, parent2 = self.select_parents(fitnesses, select_best=True)
        child = self.crossover(parent1, parent2)
        self.mutate(child)
        new_population.append(child)
        for _ in range(self.population_size - 1):
            parent1, parent2 = self.select_parents(fitnesses)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        self.population = new_population


class SnakeGameAI:
    def __init__(self, master):
        self.master = master
        self.master.title("Snake Game AI")
        self.master.geometry("1300x800")

        self.population_size = 400
        self.games_drawn = 4
        self.ga = GeneticAlgorithm(self.population_size)
        self.games = [SnakeGame(20, 20) for _ in range(self.population_size)]
        self.generation = 0
        self.best_score = 0
        self.speed = 50  # Default speed (milliseconds between steps)

        # Configure the grid
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)
        self.master.grid_columnconfigure(3, weight=1)
        self.master.grid_rowconfigure(0, weight=3)  # Games row takes more space
        self.master.grid_rowconfigure(1, weight=1)  # Info row
        self.master.grid_rowconfigure(2, weight=1)  # Graph row

        # Create frames
        self.games_frame = ttk.Frame(self.master)
        self.games_frame.grid(
            row=0, column=0, columnspan=4, padx=10, pady=10, sticky="nsew"
        )

        self.info_frame = ttk.Frame(self.master)
        self.info_frame.grid(
            row=1, column=1, columnspan=2, padx=10, pady=10, sticky="nsew"
        )

        self.graph_frame = ttk.Frame(self.master)
        self.graph_frame.grid(
            row=2, column=1, columnspan=2, padx=10, pady=10, sticky="nsew"
        )

        # Set up game canvases
        self.game_frames = []
        self.step_labels = []
        for i in range(self.games_drawn):
            frame = ttk.Frame(self.games_frame)
            frame.grid(row=0, column=i, padx=5, pady=5)
            canvas = tk.Canvas(frame, width=300, height=300, bg="black")
            canvas.pack()
            step_label = ttk.Label(frame, text="Steps left: 100")
            step_label.pack()
            self.game_frames.append((frame, canvas))
            self.step_labels.append(step_label)

        # Set up info and controls
        self.generation_label = ttk.Label(self.info_frame, text="Generation: 0")
        self.generation_label.pack(side=tk.LEFT, padx=10)

        self.best_score_label = ttk.Label(self.info_frame, text="Best Score: 0")
        self.best_score_label.pack(side=tk.LEFT, padx=10)

        self.start_button = ttk.Button(
            self.info_frame, text="Start", command=self.start_games
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.speed_label = ttk.Label(self.info_frame, text="Speed:")
        self.speed_label.pack(side=tk.LEFT, padx=10)

        self.speed_slider = ttk.Scale(
            self.info_frame,
            from_=1,
            to=100,
            orient="horizontal",
            command=self.update_speed,
            length=200,
        )
        self.speed_slider.set(50)  # Set default value
        self.speed_slider.pack(side=tk.LEFT, padx=10)

        # Set up the graph
        self.fig, self.ax = plt.subplots(figsize=(3, 1.5))  # Much smaller graph
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=False, fill=tk.BOTH)

        # Initialize lists to store fitness data
        self.generations = []
        self.min_fitnesses = []
        self.avg_fitnesses = []

    def draw_game(self, canvas, game):
        canvas.delete("all")
        cell_width = 300 // game.width
        cell_height = 300 // game.height

        for segment in game.snake:
            x, y = segment
            canvas.create_rectangle(
                x * cell_width,
                y * cell_height,
                (x + 1) * cell_width,
                (y + 1) * cell_height,
                fill="green",
            )

        food_x, food_y = game.food
        canvas.create_oval(
            food_x * cell_width,
            food_y * cell_height,
            (food_x + 1) * cell_width,
            (food_y + 1) * cell_height,
            fill="red",
        )

    def start_games(self):
        self.play_games()

    def update_speed(self, value):
        self.speed = int(float(value))

    def play_games(self):
        games_active = [True] * self.population_size
        scores = [0] * self.population_size

        while any(games_active):
            only_active_games = [
                game for game, active in zip(self.games, games_active) if active
            ]
            game_with_most_steps = max(
                (game for game in only_active_games if game.steps > 0),
                key=lambda x: x.steps,
            )
            for i, (game, model) in enumerate(zip(self.games, self.ga.population)):
                if games_active[i]:
                    state = game.get_state()
                    action = torch.argmax(model(state)).item()
                    game_over, score = game.step(action)

                    if i < self.games_drawn - 1:
                        self.draw_game(self.game_frames[i][1], game)
                        self.step_labels[i].config(text=f"Steps left: {game.steps}")
                    scores[i] = score

                    if game_over:
                        games_active[i] = False

            self.draw_game(self.game_frames[-1][1], game_with_most_steps)
            self.step_labels[-1].config(
                text=f"Steps left: {game_with_most_steps.steps}"
            )

            self.master.update()
            time.sleep(self.speed / 1000)

        self.generation += 1
        self.generation_label.config(text=f"Generation: {self.generation}")

        fitnesses = []
        for game in self.games:
            fitness_score = max(
                game.score * 20
                + game.total_steps_used
                + sum(game.distance_to_food_variation) * 2,
                0,
            )
            for i in range(len(game.action_list) - 3):
                if (
                    game.action_list[i] != 0
                    and game.action_list[i + 1] == game.action_list[i]
                    and game.action_list[i + 2] == game.action_list[i]
                ):
                    fitness_score /= 2

            fitnesses.append(fitness_score)

        max_score = max(scores)
        if max_score > self.best_score:
            self.best_score = max_score
        self.best_score_label.config(text=f"Best Score: {self.best_score}")

        # Update fitness data
        self.generations.append(self.generation)
        self.min_fitnesses.append(min(fitnesses))
        self.avg_fitnesses.append(sum(fitnesses) / len(fitnesses))

        # Update the graph
        self.update_graph()

        self.ga.evolve(fitnesses)
        for game in self.games:
            game.reset()

        self.master.after(2000, self.play_games)

    def update_graph(self):
        self.ax.clear()
        self.ax.plot(self.generations, self.min_fitnesses, label="Min")
        self.ax.plot(self.generations, self.avg_fitnesses, label="Avg")
        self.ax.set_xlabel("Gen", fontsize=5)
        self.ax.set_ylabel("Fitness", fontsize=5)
        self.ax.set_title("Fitness over Generations", fontsize=5)
        self.ax.legend(fontsize=4)
        self.ax.tick_params(axis="both", which="major", labelsize=4)
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    game = SnakeGameAI(root)
    root.mainloop()
