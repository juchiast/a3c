import math, random, time, sys, pygame
import numpy as np
import pygame.surfarray as surf
from pygame.locals import *
from pygame.draw import *

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
gandalf = (200, 200, 200)
green = (0, 120, 0)
width = 800
vertex_size = 40
height = 600
step = 20
fps = 30

pygame.init()
clock = pygame.time.Clock()
font = pygame.font.Font('./Ubuntu-R.ttf', 15)
fc = [font.render(str(i), 2, white) for i in range(1000)]

def dist(u, v):
    return math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)

def calc_len(vertex, edges):
    return dict(map(lambda x: (x, dist(vertex[x[0]], vertex[x[1]])), edges))

def get_node(vertex, pos):
    for i, x in enumerate(vertex):
        if dist(pos, x) < vertex_size:
            return i
    return None

class Visualizer:
    def __init__(self, n, edges, circles):
        vertex = [(random.randrange(width), random.randrange(height)) for _ in range(n)]

        self.edges = edges
        self.vertex = vertex
        self.circles = set(circles)
        self.updated = False
        self.selected_node = None

    def update(self, activated, actions, waits):
        points = list(map(lambda x: self.vertex[x[0]], actions))
        for f in range(step):
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()
                if ev.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    i = get_node(self.vertex, pos)
                    if i is not None:
                        self.selected_node = i
                    elif self.selected_node is not None:
                        self.vertex[self.selected_node] = pos
                        self.selected_node = None


            self.paint(points, activated, waits)
            clock.tick(fps)
            for (i, (s, t)) in enumerate(actions):
                u = np.array(self.vertex[s])
                v = np.array(self.vertex[t])
                points[i] = tuple(map(int, u + (f/step)*(v - u)))

    def paint(self, cars=[], activated=[], waits=[],):
        if not self.updated:
            self.updated = True
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Anonymous')

        activated = set(activated)
        rect(self.screen, white, (0, 0, width, height))

        for u, v in self.edges:
            line(self.screen, gandalf, self.vertex[u], self.vertex[v], 2)
            if (u, v) in activated:
                x = (np.array(self.vertex[u]) + np.array(self.vertex[v])) / 2.0
                line(self.screen, green, x, self.vertex[v], 2)

        for i, pos in enumerate(self.vertex):
            color = green if i in self.circles else black
            color = red if self.selected_node == i else color
            circle(self.screen, color, pos, vertex_size)
            self.screen.blit(fc[i], pos)

        for pos in cars:
            rect(self.screen, red, (pos[0], pos[1], 5, 5))

        for num, pos in zip(waits, self.vertex):
            self.screen.blit(fc[num], (pos[0] - 10, pos[1] - 10))

        pygame.display.update()
