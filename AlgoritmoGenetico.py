import matplotlib.pyplot as plt
import numpy as np
import random
from SnakeGame import SnakeGame

class NeuralNetwork:
    def __init__(self, input_size=5, hidden_size=6, output_size=4):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def predict(self, state, current_direction):
        layer1 = np.dot(state, self.weights1)
        layer1 = np.maximum(0, layer1)  # ReLU
        output = np.dot(layer1, self.weights2)

        sorted_actions = np.argsort(-output)  # Ordenar de mayor a menor
        opposite_direction = {0: 1, 1: 0, 2: 3, 3: 2}

        if sorted_actions[0] == opposite_direction[current_direction]:
            action = sorted_actions[1]
        else:
            action = sorted_actions[0]

        return action

def evaluate(nn, game, generation):
    total_score = 0
    steps = 0
    game.reset()

    prev_distance = game.get_distance_to_apple()

    while not game.game_over():
        state = game.get_state()
        action = nn.predict(state, game.direction)
        game.step(action)
        game.render(generation, total_score)
        steps += 1

        new_distance = game.get_distance_to_apple()

        if new_distance < prev_distance:
            total_score += 1
        else:
            if total_score < 300:
                total_score -= 1

        prev_distance = new_distance

        if game.ate_apple():
            total_score += 300
            prev_distance = game.get_distance_to_apple()
            steps = 0

        if game.game_over():
            total_score -= 100

        if steps > 1400:
            total_score -= 1

        if total_score < -100:
            break

    total_score += game.get_score() * 10

    return total_score

def mutate(weights, mutation_rate=0.05, mutation_strength=0.3):
    mutation_mask = np.random.rand(*weights.shape) < mutation_rate
    mutations = np.random.randn(*weights.shape) * mutation_strength
    weights += mutations * mutation_mask
    return weights

def tournament_selection(population, scores, k=5):
    selected = []
    population_size = len(population)
    for _ in range(population_size // 2):
        participants = random.sample(list(zip(population, scores)), k)
        selected.append(max(participants, key=lambda x: x[1])[0])
    return selected

def crossover(parent1, parent2):
    child = NeuralNetwork()
    child.weights1 = np.where(np.random.rand(*parent1.weights1.shape) > 0.5, parent1.weights1, parent2.weights1)
    child.weights2 = np.where(np.random.rand(*parent1.weights2.shape) > 0.5, parent1.weights2, parent2.weights2)
    return child

def genetic_algorithm(population_size=50, generations=50, mutation_rate=0.05, elitism=True):
    game = SnakeGame()
    population = [NeuralNetwork() for _ in range(population_size)]

    # Almacenar métricas por generación
    best_scores = []
    avg_scores = []
    min_scores = []

    try:
        for generation in range(generations):
            scores = [evaluate(nn, game, generation) for nn in population]

            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)

            best_scores.append(best_score)
            avg_scores.append(avg_score)
            min_scores.append(min_score)

            print(f"Generación {generation + 1}: Mejor Puntaje = {best_score}, Promedio = {avg_score:.2f}, Mínimo = {min_score}")

            selected = tournament_selection(population, scores)

            if elitism:
                elite_idx = np.argmax(scores)
                elite = population[elite_idx]
            else:
                elite = None

            new_population = []
            while len(new_population) < population_size - (1 if elitism else 0):
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1, parent2)
                child.weights1 = mutate(child.weights1, mutation_rate)
                child.weights2 = mutate(child.weights2, mutation_rate)
                new_population.append(child)

            if elitism:
                new_population.append(elite)

            population = new_population

    except KeyboardInterrupt:
        print("Ejecución interrumpida. Mostrando gráficos...")

    finally:
        game.close()

        # Graficar los resultados
        generations = range(1, len(best_scores) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_scores, label="Mejor Puntaje")
        plt.plot(generations, avg_scores, label="Puntaje Promedio")
        plt.plot(generations, min_scores, label="Puntaje Mínimo")
        plt.xlabel("Generación")
        plt.ylabel("Puntaje")
        plt.title("Evolución de Puntajes por Generación")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    genetic_algorithm()