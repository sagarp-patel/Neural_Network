#A simple runner for the neural network
import pygame
import time
import random

class Obstacle:
    def __init__(self, pos_x, pos_y, radius,velocity,color):
        self.x = pos_x
        self.y = pos_y
        self.radius = radius
        self.color = color
        self.velocity = velocity
        

class Runner:
    def __init__(self):
        #Colors that I might need
        self.black = (0,0,0)
        self.white = (255, 255, 255)
        self.red   = (200,0,0)
        self.green = (0,200,0)
        self.blue  = (0,0,200)
        self.grey  = (232,232,232)
        #Player Values
        self.player_height = 50
        self.player_width  = 50
        self.player_pos_x = 0
        self.player_pos_y = 0
        #Game Values
        self.score = 0
        self.intro = False
        self.exitGame = True
        #Creating the Window
        self.window_height = 600 #800
        self.window_width = 800 #1000
        self.window = pygame.display.set_mode((self.window_width,self.window_height))
        pygame.display.set_caption("A simple Runner for Neural Net")
        self.clock = pygame.time.Clock()
        #Creating the Obstacle
        #Obstacle Values
        self.velocity = -4
        self.obst = Obstacle(self.window_width,random.randrange(0, self.window_height - 150),20,self.velocity,self.black)

        #Starting point of game initializes pygame so that it could be used
    def game_start(self):
        pygame.init()
        self.game_intro()
        self.game_loop()
        pygame.quit()
        quit()

    #Crash Function
    def crash(self):
        self.display_message("You Crashed",50,self.black,self.window_width/2,self.window_height/2)
        self.display_message("Press R to restart",50,self.black,(self.window_width/2),(self.window_height/2)+50)

    #Keep Track of Score
    def score(self,count):
        font = pygame.font.SysFont(None,25)
        scoreBoard = font.render("Score: "+str(count),True,self.black)
        self.window.blit(scoreBoard, (0,800))

    #Draw Obstacles
    def draw_obstacle(self,pos_x, pos_y,radius, color):
        pygame.draw.circle(self.window,color,[pos_x,pos_y],radius)

    def draw_obstacles(self,obst_list,color):
        for obst in obst_list:
            self.draw_obstacle(obst.x,obst.y,obst.radius,obst.color)

    #Draw Player
    def draw_Player(self,pos_x, pos_y,width,height):
        pygame.draw.rect(self.window,self.blue,[pos_x,pos_y, width, height])

    #Creates a text object to display message
    def text_object(self,text,font):
        textSurface = font.render(text,True,self.black)
        return textSurface, textSurface.get_rect()

    #This Function will display a message at location x and y
    def display_message(self,text,fontSize,color,x,y):
        largeText = pygame.font.Font('freesansbold.ttf',fontSize)
        TextSurf, TextRect = self.text_object(text,largeText)
        TextRect.center = (x,y)
        self.window.blit(TextSurf,TextRect)
        pygame.display.update()
        time.sleep(2)
        #self.game_loop()

    #Move the Player Up
    def move_up(self):
        self.delta_y+= -10

    #Move the Player Down
    def move_down(self):
        self.delta_y+=10

    #Move Obstacles
    def move_obst(self,obst_lst,velocity):
        for obst in obst_lst:
            obst.x+=velocity
                
                    
    #Intro Screen Function
    def game_intro(self):
        self.intro = True
        while self.intro:
            self.window.fill(self.grey)
            self.display_message("Press Space to Start the Game!!!",30,self.white,self.window_width/2,self.window_height/2)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.intro = False
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                pygame.display.update()
                self.clock.tick(10)
        
    def game_loop(self):
        self.window.fill(self.grey) #clear Window
        pygame.draw.line(self.window,self.black,[0,self.window_height-50],[self.window_width,self.window_height-50])
        self.player_pos_x = 10
        self.player_pos_y = self.window_height - 100
        self.draw_Player(self.player_pos_x,self.player_pos_y,self.player_width, self.player_height)
        self.delta_y = 0
        self.score  = 0
        delta_x = 0
        self.exitGame = False
        obstacle_x = self.window_width
        obstacle_y = random.randrange(0, self.window_height - 150)
        #(self, pos_x, pos_y, radius,velocity,color)
        obst1 = Obstacle(self.window_width,random.randrange(0, self.window_height - 150),20,self.velocity,self.blue)
        obst_lst = []
        obst_lst.append(self.obst)
        #obst_lst.append(obst1)
        obstacle_radius = 20
        obstacle_count = 0
        while not self.exitGame:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                #Move the Player based on input from keyboard when a key is pressed
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if (self.player_pos_y - 5) > 0: 
                            #self.delta_y = -5
                            self.move_up()
                        else:
                            self.delta_y = 0
                    if event.key == pygame.K_DOWN:
                        if self.window_height > self.player_pos_y + self.player_height + 5:
                            #self.delta_y = 5
                            self.move_down()
                        else:
                            self.delta_y = 0
                #resetting delta when key is released
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_DOWN:
                        self.delta_y = 0
            # Making Sure the Player doesnt go off screen
            if (self.player_pos_y - 5) < 0 and self.delta_y < 0:
                self.delta_y = 0
            if self.window_height < self.player_pos_y + self.player_height + 5 and self.delta_y > 0:
                self.delta_y = 0
            #Drawing the Window
            self.window.fill(self.grey)
            self.player_pos_y+=self.delta_y
            self.draw_Player(self.player_pos_x,self.player_pos_y,self.player_width, self.player_height)
            self.draw_obstacles(obst_lst,self.black) #Keep
            self.move_obst(obst_lst,self.velocity)
            #Detecting if a collision has happened
            if self.player_pos_x + self.player_width >= self.obst.x - self.obst.radius:
                if self.obst.y - self.obst.radius <= self.player_pos_y + self.player_height and self.obst.y + self.obst.radius > self.player_pos_y:
                    self.exitGame = True
                    self.crash()
                    break
                    crashed = True
                    print("Detected Crash")
            largeText = pygame.font.Font('freesansbold.ttf',10)
            TextSurf, TextRect = self.text_object("Score: "+str(obstacle_count),largeText)
            TextRect.center = (30,10)
            self.window.blit(TextSurf,TextRect)
            pygame.display.update()
            #self.display_message("Score: "+str(obstacle_count),10,self.white,30,10)
            self.clock.tick(60)
            #Updating the Location of Obstacle
            if self.obst.x + self.obst.radius <= 0:
                self.obst.x = self.window_width
                self.obst.y = random.randrange(0, self.window_height - 150)
                obstacle_count+=1
        self.game_intro()

