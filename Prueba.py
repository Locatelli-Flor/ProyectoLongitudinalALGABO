import numpy as np
import random
from SnakeGame import SnakeGame

class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.rand(4, 6)
        self.weights2 = np.random.rand(6, 4)

    def predict(self, state):
        layer1 = np.dot(state, self.weights1)
        layer1 = np.maximum(0, layer1)
        output = np.dot(layer1, self.weights2)
        action = np.argmax(output)
        return action



def evaluate(nn, game, generation):
    total_score = 0
    game.reset()
    steps = 0
    while not game.game_over():
        state = game.get_state()
        action = nn.predict(state)
        print(action)
        game.step(action)
        game.render(generation, total_score)
        steps += 1

    total_score += game.get_score()

    return total_score


def genetic_algorithm():
    population_size = 100
    generations = 25
    mutation_rate = 0.3

    population = [NeuralNetwork() for _ in range(population_size)]

    for generation in range(generations):
        scores = [evaluate(nn, game, generation) for nn in population]

        selected = [population[i] for i in np.argsort(scores)[-population_size//2:]]

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = NeuralNetwork()
            
            # Cruzar pesos de los padres
            child.weights1 = (parent1.weights1 + parent2.weights1) / 2
            child.weights2 = (parent1.weights2 + parent2.weights2) / 2
            
            # Mutar los pesos del hijo
            if random.random() < mutation_rate:
                child.weights1 += np.random.randn(*child.weights1.shape) * 0.1
                child.weights2 += np.random.randn(*child.weights2.shape) * 0.1

            new_population.append(child)

        population = new_population
        print(f"Generation {generation}: Best Score = {max(scores)}")



if __name__ == "__main__":
    game = SnakeGame()
    genetic_algorithm()
    game.close()
