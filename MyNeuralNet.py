import mnist_loader

import matplotlib
matplotlib.use('Agg')

import msvcrt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import network2
import sys


np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
np.set_printoptions(precision=2)

def train(savename):
    # here is code for training the network
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784,100, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
    net.save(savename)

class NNIMG:
    def __init__(self, fig, img, net):
        self.fig = fig
        self.img = img
        self.obj = plt.imshow(self.img, vmin=0, vmax=1)
        
        self.net = net
        self.n = 0
        self.training_data = []       
      
        self.cidpress = None
        self.cidmotion = None
        self.pressed = False

        self.lastX = None
        self.lastY = None

        self.connect()

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.obj.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidkeypress = self.obj.figure.canvas.mpl_connect(
            'key_press_event', self.on_key)

    def on_key(self,event):
        global exit_clicked
        if event.key == "e":
            exit_clicked = True
          
    def on_press(self, event):       
        self.pressed = True    
        if event.xdata and event.ydata:
            self.lastX = int(round(event.xdata))
            self.lastY = int(round(event.ydata))

            if event.button == 1: 
                col = 1
                colstr = 'yo'
            else: 
                col = 0
                colstr = 'mo'
            
            plt.plot (self.lastX, self.lastY, colstr)
            
            self.training_data.append((np.empty([2,1]),np.empty([2,1])))
            self.training_data [self.n][0][0] = float(self.lastX)/self.img.shape[1] 
            self.training_data [self.n][0][1] = float(self.lastY)/self.img.shape[0]
            self.training_data [self.n][1][0] = 0
            self.training_data [self.n][1][1] = 0
            self.training_data [self.n][1][col] = 1

            self.n += 1
            diff = 1
            cost = 1
            while cost>0.05:
                old_cost = cost
                cost,accuracy = self.net.GD(self.training_data, 1,  0.75, monitor_training_accuracy=True, monitor_training_cost=True)
                plt.title('error = %1.3f' % cost)
                diff = cost-old_cost 
            print ("final cost on training data: {}".format(cost))
            print ("final accuracy on training data: {} / {}".format(accuracy, len(self.training_data)))
            self.update_img()     

    def update_img(self):
        for ix in range(self.img.shape[1]):
            for iy in range(self.img.shape[0]):
                jx = float(ix)/self.img.shape[1]
                jy = float(iy)/self.img.shape[0]
                col = self.net.feedforward(np.array([[jx],[jy]]))[1]

                self.img[iy,ix] = col

        self.obj.set_data(self.img)        
        plt.pause(0.001)



class drawableIMG:    
    def __init__(self, fig, img, net):
        self.fig = fig
        self.img = img
        self.obj = plt.imshow(self.img, vmin=0, vmax=1)
        self.net = net
       
        self.cidpress = None
        self.cidmotion = None
        self.pressed = False

        self.lastX = None
        self.lastY = None

        self.connect()

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.obj.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.obj.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.obj.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidkeypress = self.obj.figure.canvas.mpl_connect(
            'key_press_event', self.on_key)

    def on_press(self, event):      
        self.pressed = True    
        if event.xdata and event.ydata:
            self.lastX = int(round(event.xdata))
            self.lastY = int(round(event.ydata))

            pen = np.matrix([[.05,0.25,0.05],[0.25,0.75,0.25],[.05,0.25,0.05]])
            self.img[self.lastY-pen.shape[0]//2:self.lastY+pen.shape[0]//2+1,self.lastX-pen.shape[1]//2:self.lastX+pen.shape[1]//2+1] += pen
            self.img[self.img>1] = 1

            self.obj.set_data(self.img)
            plt.pause(0.01)

    def on_motion(self, event):
        if event.xdata and event.ydata and self.pressed:
            nX = int(round(event.xdata))
            nY = int(round(event.ydata))
            if nX!=self.lastX or nY!=self.lastY:
                self.lastX = nX
                self.lastY = nY
                pen = np.matrix([[.05,0.25,0.05],[0.25,0.75,0.25],[.05,0.25,0.05]])
                #pen = np.matrix([[.25,0.5,0.25],[0.5,1,0.5],[.25,0.5,0.25]])
                self.img[self.lastY-pen.shape[0]//2:self.lastY+pen.shape[0]//2+1,self.lastX-pen.shape[1]//2:self.lastX+pen.shape[1]//2+1] += pen
                self.img[self.img>0.99] = 0.99
                self.obj.set_data(self.img)
                plt.pause(0.01)

    def on_release(self, event):
        self.pressed = False 
        self.obj.set_data(self.img)     
        plt.pause(0.001)

    def on_key(self,event):
        global exit_clicked
        if event.key == "a":
            x = self.net.feedforward(np.reshape(self.img, (784,1)))
           
            print (self.img)
            for idx,val in enumerate(x):
                print (idx, 'val = %1.3f' % val)
            x = np.argmax(x)
            
            print ('result:', x)
            result ="result: " + str(x)
            
            plt.suptitle(result)
            plt.pause(0.001)
        if event.key == "c":
            self.img = np.zeros((28,28))
            self.obj.set_data(self.img)  
            plt.suptitle('-')
            plt.pause(0.001)
        if event.key == "e":           
            exit_clicked = True
            


class testIMGs:
    
    def __init__(self, fig, img, net, test_data):
        self.fig = fig
        self.img = img
        self.obj = plt.imshow(self.img, vmin=0, vmax=1)
        self.net = net
        self.i   = 0
        self.test_data = test_data
        self.cidpress = None
        self.cidmotion = None
        self.pressed = False

        self.lastX = None
        self.lastY = None

        self.connect()

    def connect(self):
        'connect to all the events we need'
        self.cidkeypress = self.obj.figure.canvas.mpl_connect(
            'key_press_event', self.on_key)

    def on_key(self,event):
        global exit_clicked
        if event.key == " ":
            test_data_truncated = np.around(self.test_data[self.i][0],1)
            
            obj = plt.imshow(np.reshape(test_data_truncated,(28,28)), vmin=0, vmax=1)
             
            #obj.set_data(np.reshape(test_data_truncated))
            x = self.net.feedforward(test_data_truncated)
     
            print (np.reshape(test_data_truncated,(28,28)))
            for idx,val in enumerate(x):
                print (idx, 'val = %1.3f' % val)

            x = np.argmax(x)
            result ="result: " + str(x)
            print ('result:', x)
            plt.suptitle(result)
            self.i = self.i + 1
                  
        if event.key == "e":           
            exit_clicked = True


def main(): 
    
    global exit_clicked
    global keyPressed
    keyPressed = False
    exit_clicked = False
    plt.ioff()
    plt.show()
    global mode 
    mode = 3
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])

    fig = plt.plot()
    plt.switch_backend('TKAgg')
    figManager = plt.get_current_fig_manager()
    figManager.window.state('zoomed')
    if mode == 0:
        print ("Net 2, 10, 10, 2")
        img = 0.5*np.ones((140,140))
        net = network2.Network([2, 10, 10, 2], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()

        myobj = NNIMG(fig, img, net)
        plt.gca().invert_yaxis()  
        plt.draw()
    if mode == -1:
        print ("Net 2, 2, 2")
        img = 0.5*np.ones((140,140))
        net = network2.Network([2, 2, 2], cost=network2.CrossEntropyCost)
        net.large_weight_initializer()

        myobj = NNIMG(fig, img, net)
        plt.gca().invert_yaxis()  
        plt.draw()
    
    if mode == 1:       
        train ('C:/Users/ThomasReichert/source/repos/TheLIFO/NeuralNetMINT/trainednet2')

    if mode == 2:           
        img = np.zeros((28,28))
    
        net = network2.load('C:/Users/ThomasReichert/source/repos/TheLIFO/NeuralNetMINT/trainednet2')
        myobj = drawableIMG(fig, img, net)
       

    if mode == 3:
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        img = np.zeros((28,28))
        net = network2.load('C:/Users/ThomasReichert/source/repos/TheLIFO/NeuralNetMINT/trainednet2')

        myObj = testIMGs(fig, img, net, test_data)
       
    plt.ion()
    
    while not(exit_clicked):       
        plt.pause(0.001)

          
    print ("exit")
main()