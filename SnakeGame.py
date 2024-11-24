import pygame
import random
import math

# Inicializar pygame
pygame.init()

# Definir los colores
black = (0, 0, 0)
red = (213, 50, 80)
white = (255, 255, 255)

# Definir las dimensiones de la pantalla
dis_width = 800
dis_height = 600

class SnakeGame:
    def __init__(self):
        self.dis = pygame.display.set_mode((dis_width, dis_height))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.snake_block = 10
        self.snake_speed = 10
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.reset()
        self.focused = True  # Indica si la ventana está en foco
        self.apples_eaten = 0

    def reset(self):
        self.x1 = dis_width / 2
        self.y1 = dis_height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.direction = 0
        self.snake_List = []
        self.Length_of_snake = 1
        self.direction = 1
        self.foodx = round(random.randrange(0, dis_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, dis_height - self.snake_block) / 10.0) * 10.0
        self.game_over_flag = False
        self.apples_eaten = 0

    def get_state(self):
        head = (self.x1, self.y1)
        left, front, right = self.check_surroundings(head, self.direction, self.snake_List, dis_width)
        return [
            left,
            front,
            right,
            self.x1 - self.foodx,
            self.y1 - self.foody
        ]

    def get_distance_to_apple(self):
        # Devuelve la distancia Euclidiana desde la cabeza de la serpiente hasta la manzana
        return math.sqrt((self.x1 - self.foodx) ** 2 + (self.y1 - self.foody) ** 2)

    def check_surroundings(self, snake_head, direction, snake_body, grid_size):
        x, y = snake_head
        
        relative_directions = {
            2: [(-1, 0), (0, -1), (1, 0)],
            3: [(1, 0), (0, 1), (-1, 0)],
            0: [(0, 1), (-1, 0), (0, -1)],
            1: [(0, -1), (1, 0), (0, 1)]
        }
        
        left_move, front_move, right_move = relative_directions[direction]
        
        left = (x + left_move[0], y + left_move[1])
        front = (x + front_move[0], y + front_move[1])
        right = (x + right_move[0], y + right_move[1])
        
        def is_dangerous(pos):
            px, py = pos
            return (
                px < 0 or px >= grid_size or
                py < 0 or py >= grid_size or
                pos in snake_body
            )
        
        left_value = 1 if is_dangerous(left) else 0
        front_value = 1 if is_dangerous(front) else 0
        right_value = 1 if is_dangerous(right) else 0

        return left_value, front_value, right_value

    def get_sin_angle_to_food(self, snake_head, food_position):
        x1, y1 = snake_head
        x2, y2 = food_position
        angle = math.atan2(y2 - y1, x2 - x1)
        return math.sin(angle)

    def game_over(self):
        return self.game_over_flag

    def get_score(self):
        return self.Length_of_snake - 1

    def ate_apple(self):
        return self.x1 == self.foodx and self.y1 == self.foody

    def step(self, action):
        previous_position = self.x1, self.y1

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

        if self.x1 >= dis_width or self.x1 < 0 or self.y1 >= dis_height or self.y1 < 0:
            self.game_over_flag = True

        self.snake_List.append([self.x1, self.y1])
        if len(self.snake_List) > self.Length_of_snake:
            del self.snake_List[0]

        for x in self.snake_List[:-1]:
            if x == [self.x1, self.y1]:
                self.game_over_flag = True

        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, dis_width - self.snake_block) / 10.0) * 10.0
            self.foody = round(random.randrange(0, dis_height - self.snake_block) / 10.0) * 10.0
            self.apples_eaten += 1
            self.Length_of_snake += 1

        if (self.x1, self.y1) == previous_position:
            print("Snake stayed still! Ending game.")
            self.game_over_flag = True

    def render(self, generation, score):
        # Manejar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.ACTIVEEVENT:
                if event.gain == 0:
                    self.focused = False
                else:
                    self.focused = True

        # Si la ventana está en foco, renderizar
        if self.focused:
            self.dis.fill(black)

            # Dibujar la comida
            pygame.draw.rect(self.dis, red, [self.foodx, self.foody, self.snake_block, self.snake_block])

            # Dibujar la serpiente
            for x in self.snake_List:
                pygame.draw.rect(self.dis, white, [x[0], x[1], self.snake_block, self.snake_block])

            # Mostrar la generación y el puntaje en pantalla
            gen_text = self.font_style.render(f"Generation: {generation}", True, white)
            score_text = self.font_style.render(f"Score: {score}", True, white)
            apples_eaten = self.font_style.render(f"Apples eaten: {self.apples_eaten}", True, white)
            self.dis.blit(gen_text, [0, 0])
            self.dis.blit(score_text, [0, 30])
            self.dis.blit(apples_eaten, [0, 60])

            pygame.display.update()

        # Reducir la velocidad del juego ajustando los FPS
        self.clock.tick(120)

    def close(self):
        pygame.quit()

