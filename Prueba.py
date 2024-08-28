import random
import numpy as np
from SnakeGame import SnakeGame

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(4, 4)

    def predict(self, state):
        output = np.dot(state, self.weights)
        action = np.argmax(output)
        return action

def evaluate(nn, game, generation):
    score = 0
    for _ in range(10):  # Simula varias partidas
        game.reset()
        while not game.game_over():
            state = game.get_state()
            action = nn.predict(state)
            game.step(action)
            game.render(generation, score)

    return game.get_score()  # Retorna el puntaje final

def genetic_algorithm():
    population_size = 25
    generations = 50
    mutation_rate = 0.1
    
    population = [NeuralNetwork() for _ in range(population_size)]
    
    for generation in range(generations):
        # Evaluar la población
        scores = [evaluate(nn, game, generation) for nn in population]
        
        # Selección
        selected = [population[i] for i in np.argsort(scores)[-population_size//2:]]
        
        # Cruzamiento
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = NeuralNetwork()
            child.weights = (parent1.weights + parent2.weights) / 2
            new_population.append(child)
        
        # Mutación
        for nn in new_population:
            if random.random() < mutation_rate:
                nn.weights += np.random.randn(*nn.weights.shape) * 0.1
        
        population = new_population
        print(f"Generation {generation}: Best Score = {max(scores)}")

if __name__ == "__main__":
    game = SnakeGame()
    genetic_algorithm()
    game.close()
