#A simple runner for the neural network
import pygame
import time
import random


class Runner:
    def __init__(self):
        #Colors that we might need
        self.black = (0,0,0)
        self.white = (255, 255, 255)
        self.red   = (200,0,0)
        self.green = (0,200,0)
        self.blue  = (0,0,200)
        self.grey  = (232,232,232)
        self.player_height = 50
        self.player_width  = 50
        self.score = 0
        #Creating the Window
        self.window_height = 800
        self.window_width = 1000
        self.window = pygame.display.set_mode((self.window_width,self.window_height))
        pygame.display.set_caption("A simple Runner for Neural Net")
        self.clock = pygame.time.Clock()
    def game_start(self):
        pygame.init()
        self.game_intro()
        self.game_loop()
        pygame.quit()
        quit()
    def crash(self):
        self.display_message("You Crashed",50,self.black,self.window_width/2,self.window_height/2)
        self.display_message("Press R to restart",50,self.black,(self.window_width/2),(self.window_height/2)+50)
        
    def score(self,count):
        font = pygame.font.SysFont(None,25)
        scoreBoard = font.render("Score: "+str(count),True,self.black)
        self.window.blit(scoreBoard, (0,800))

    def draw_obstacles(self,pos_x, pos_y,radius, color):
        pygame.draw.circle(self.window,color,[pos_x,pos_y],radius)

    def draw_Player(self,pos_x, pos_y,width,height):
        pygame.draw.rect(self.window,self.blue,[pos_x,pos_y, width, height])

    def text_object(self,text,font):
        textSurface = font.render(text,True,self.black)
        return textSurface, textSurface.get_rect()
    
    def display_message(self,text,fontSize,color,x,y):
        largeText = pygame.font.Font('freesansbold.ttf',fontSize)
        TextSurf, TextRect = self.text_object(text,largeText)
        TextRect.center = (x,y)
        self.window.blit(TextSurf,TextRect)
        pygame.display.update()
        time.sleep(2)

    def game_intro(self):
        intro = True
        while intro:
            self.window.fill(self.grey)
            self.display_message("Press Space to Start the Game!!!",30,self.white,self.window_width/2,self.window_height/2)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print("SpaceBar is Pressed")
                        intro = False
                    if event.key == pygame.K_ESCAPE:
                        print("ESC is pressed")
                        pygame.quit()
                        quit()
                pygame.display.update()
                self.clock.tick(10)
        
    def game_loop(self):
        self.window.fill(self.grey) #clear Window
        pygame.draw.line(self.window,self.black,[0,self.window_height-50],[self.window_width,self.window_height-50])
        pos_x = 10
        pos_y = self.window_height - 100
        self.draw_Player(pos_x,pos_y,self.player_width, self.player_height)
        self.score  = 0
        delta_x = 0
        exitGame = False
        obstacle_x = self.window_width
        obstacle_y = random.randrange(0, self.window_height - 150)
        velocity = -4
        obstacle_radius = 20
        obstacleCount = 1
        while not exitGame:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            self.window.fill( self.grey)
            pygame.draw.line(self.window,self.black,[0,self.window_height-50],[self.window_width,self.window_height-50])
            self.draw_Player(pos_x,pos_y,self.player_width, self.player_height)
            self.draw_obstacles(obstacle_x, obstacle_y, obstacle_radius,self.black)
            obstacle_x += velocity
            pygame.display.update()
            self.clock.tick(60)


