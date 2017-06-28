import pygame
import numpy as np
import math
import random
import pygame
import time
import sys
import pygame.surfarray as surf
from pygame.locals import *

pygame.init()
width = 800
vertex_size = 40
height = 600
step = 20
fps = 30
clock = pygame.time.Clock()
font = pygame.font.Font('./Ubuntu-R.ttf', 15)
black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
gandalf = (200, 200, 200)
green = (0, 120, 0)

def dist(u, v):
    math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)

class Visualizer:
    def __init__(self, n, edges, circles):
        vertex = [(random.randrange(width), random.randrange(height)) for _ in range(n)]

        length = dict(map(lambda x: (x, dist(vertex[x[0]], vertex[x[1]])), edges))


        self.edges = edges
        self.length = length
        self.vertex = vertex
        self.circles = set(circles)
        self.updated = False

    def update(self, activated, actions, waits):
        if any(ev.type == pygame.QUIT for ev in pygame.event.get()):
            pygame.display.quit()
            pygame.quit()
            sys.exit()

        if not self.updated:
            self.updated = True
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Anonymous')

        points = list(map(lambda x: self.vertex[x[0]], actions))
        for f in range(step):
            self.paint(points, activated, waits)
            clock.tick(fps)
            for (i, (s, t)) in enumerate(actions):
                u = np.array(self.vertex[s])
                v = np.array(self.vertex[t])
                points[i] = tuple(map(int, u + (f/step)*(v - u)))

    def paint(self, cars=[], activated=[], waits=[]):
        activated = set(activated)
        pygame.draw.rect(self.screen, white, (0, 0, width, height))

        for u, v in self.edges:
            pygame.draw.line(self.screen, gandalf, self.vertex[u], self.vertex[v], 2)
            if (u, v) in activated:
                x = (np.array(self.vertex[u]) + np.array(self.vertex[v])) / 2.0
                pygame.draw.line(self.screen, green, x, self.vertex[v], 2)

        for i, pos in enumerate(self.vertex):
            color = green if i in self.circles else black
            pygame.draw.circle(self.screen, color, pos, vertex_size)
            text = font.render(str(i), 1, white)
            self.screen.blit(text, pos)

        for pos in cars:
            pygame.draw.rect(self.screen, red, (pos[0], pos[1], 5, 5))

        for num, pos in zip(waits, self.vertex):
            text = font.render(str(num), 1, white)
            self.screen.blit(text, (pos[0] - 10, pos[1] - 10))

        pygame.display.update()
