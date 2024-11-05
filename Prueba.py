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

        # Obtener las probabilidades de acción ordenadas por sus valores e índices
        sorted_actions = np.argsort(-output)  # Ordenar de mayor a menor

        # Definir direcciones opuestas: 0 = Izquierda, 1 = Derecha, 2 = Arriba, 3 = Abajo
        opposite_direction = {0: 1, 1: 0, 2: 3, 3: 2}

        # Si la acción más probable es la opuesta, tomar la segunda acción más probable
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

        # Obtener nueva distancia a la manzana
        new_distance = game.get_distance_to_apple()

        # Recompensa por acercarse a la manzana
        if new_distance < prev_distance:
            total_score += 1
        else:
            if total_score < 300:
                total_score -= 1

        prev_distance = new_distance

        # Recompensa por comer la manzana
        if game.ate_apple():
            total_score += 300
            prev_distance = game.get_distance_to_apple()  # Resetear la distancia con nueva comida
            steps = 0

        # Penalización adicional si choca
        if game.game_over():
            total_score -= 100

        if steps > 600 + 800:
            total_score -= 1

        if total_score < -100:
            break

    # Agregar puntaje basado en el tamaño de la serpiente
    total_score += game.get_score() * 10

    return total_score

def mutate(weights, mutation_rate=0.05, mutation_strength=0.3):
    # Aplicar mutación con una tasa y fuerza definidas
    mutation_mask = np.random.rand(*weights.shape) < mutation_rate
    mutations = np.random.randn(*weights.shape) * mutation_strength
    weights += mutations * mutation_mask
    return weights

def tournament_selection(population, scores, k=5):
    selected = []
    population_size = len(population)
    for _ in range(population_size // 2):
        # Seleccionar 'k' individuos al azar para el torneo
        participants = random.sample(list(zip(population, scores)), k)
        # Seleccionar el individuo con el mejor puntaje
        selected.append(max(participants, key=lambda x: x[1])[0])
    return selected

def crossover(parent1, parent2):
    child = NeuralNetwork()
    # Combinar los pesos de los padres usando un punto de cruce aleatorio
    child.weights1 = np.where(np.random.rand(*parent1.weights1.shape) > 0.5, parent1.weights1, parent2.weights1)
    child.weights2 = np.where(np.random.rand(*parent1.weights2.shape) > 0.5, parent1.weights2, parent2.weights2)
    return child

def genetic_algorithm(population_size=50, generations=50, mutation_rate=0.05, elitism=True):
    game = SnakeGame()
    population = [NeuralNetwork() for _ in range(population_size)]

    for generation in range(generations):
        # Evaluar el puntaje de cada individuo en la población
        scores = [evaluate(nn, game, generation) for nn in population]

        # Encontrar el mejor puntaje de la generación actual
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)
        print(f"Generación {generation + 1}: Mejor Puntaje = {best_score}, Puntaje Promedio = {avg_score:.2f}")

        # Selección de los mejores individuos
        selected = tournament_selection(population, scores)

        # Implementar elitismo: preservar el mejor individuo
        if elitism:
            elite_idx = np.argmax(scores)
            elite = population[elite_idx]
        else:
            elite = None

        # Crear una nueva población mediante cruce y mutación
        new_population = []
        while len(new_population) < population_size - (1 if elitism else 0):
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            # Mutar los pesos del hijo
            child.weights1 = mutate(child.weights1, mutation_rate)
            child.weights2 = mutate(child.weights2, mutation_rate)
            new_population.append(child)

        # Añadir el elite a la nueva población
        if elitism:
            new_population.append(elite)

        population = new_population

    # Cerrar el juego después de todas las generaciones
    game.close()

if __name__ == "__main__":
    genetic_algorithm()
