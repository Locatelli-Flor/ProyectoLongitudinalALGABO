import pygame
import random
import time

# Inicializar pygame
pygame.init()

# Definir los colores
black = (0, 0, 0)
red = (213, 50, 80)
white = (255, 255, 255)

# Definir las dimensiones de la pantalla
dis_width = 800
dis_height = 600

score = 0

class SnakeGame:
    def __init__(self):
        self.dis = pygame.display.set_mode((dis_width, dis_height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.snake_block = 10
        self.snake_speed = 10
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.get_time_since_last_apple = 0
        self.score = 0
        self.reset()

    def reset(self):
        self.x1 = dis_width / 2
        self.y1 = dis_height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.direction = None
        self.snake_List = []
        self.Length_of_snake = 1
        self.foodx = round(random.randrange(0, dis_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, dis_height - self.snake_block) / 10.0) * 10.0
        self.game_over_flag = False

    def get_state(self):
        # Normaliza las posiciones para obtener un estado del juego
        return [
            self.x1 / dis_width,
            self.y1 / dis_height,
            self.foodx / dis_width,
            self.foody / dis_height
        ]

    def game_over(self):
        return self.game_over_flag
    
    def get_time_since_last_apple(self):
        current_time = pygame.time.get_ticks()
        time_elapsed = (current_time - self.last_apple_time) / 1000
        return time_elapsed
    

    def get_game_score(self):
        time_weight = 0.1  # Ponderación para el tiempo
        time_elapsed = pygame.time.get_ticks() / 1000  # Tiempo transcurrido en segundos
        score = (self.get_score(self)) - (time_elapsed * time_weight)
        return score
    
    def get_score(self):
        return self.Length_of_snake - 1

    def step(self, action):
        # Acciones: 0 = Izquierda, 1 = Derecha, 2 = Arriba, 3 = Abajo
        if action == 0 and self.direction != 1:
            self.x1_change = -self.snake_block
            self.y1_change = 0
            self.direction = 0
        elif action == 1 and self.direction != 0:
            self.x1_change = self.snake_block
            self.y1_change = 0
            self.direction = 1
        elif action == 2 and self.direction != 3:
            self.y1_change = -self.snake_block
            self.x1_change = 0
            self.direction = 2
        elif action == 3 and self.direction != 2:
            self.y1_change = self.snake_block
            self.x1_change = 0
            self.direction = 3

        self.x1 += self.x1_change
        self.y1 += self.y1_change

        # Verificar colisiones
        if self.x1 >= dis_width or self.x1 < 0 or self.y1 >= dis_height or self.y1 < 0 or self.get_time_since_last_apple(self) > 5:
            self.score -= 10
            self.game_over_flag = True

        self.snake_List.append([self.x1, self.y1])
        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        for x in self.snake_List[:-1]:
            if x == [self.x1, self.y1]:
                self.score -= 10
                self.game_over_flag = True

        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, dis_width - self.snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, dis_height - self.snake_block) / 10.0) * 10.0
            self.Length_of_snake += 1
            self.score += 2

    def render(self, generation, score):
        self.dis.fill(black)
        pygame.draw.rect(self.dis, red, [self.foodx, self.foody, self.snake_block, self.snake_block])
        for x in self.snake_List:
            pygame.draw.rect(self.dis, white, [x[0], x[1], self.snake_block, self.snake_block])  # Cambiar la serpiente a blanca para mayor visibilidad

        # Mostrar la generación y el puntaje en pantalla
        gen_text = self.font_style.render(f"Generation: {generation}", True, white)
        score_text = self.font_style.render(f"Score: {score}", True, white)
        self.dis.blit(gen_text, [0, 0])
        self.dis.blit(score_text, [0, 30])

        pygame.display.update()

    def close(self):
        pygame.quit()
