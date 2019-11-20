import mnist_loader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import network2




def train(savename):
    # here is code for training the network
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784,100, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
    net.save(savename)



class drawableIMG:

    def __init__(self, fig, axs, img, net):
        self.fig = fig
        self.img = img
        self.obj = axs.imshow(self.img, vmin=0, vmax=1)
        self.net = net
       
        self.axs = axs
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
        if event.inaxes != self.axs.axes: return
        self.pressed = True    
        if event.xdata and event.ydata:
            self.lastX = int(round(event.xdata))
            self.lastY = int(round(event.ydata))

            pen = np.matrix([[.25,0.5,0.25],[0.5,1,0.5],[.25,0.5,0.25]])
            self.img[self.lastY-pen.shape[0]//2:self.lastY+pen.shape[0]//2+1,self.lastX-pen.shape[1]//2:self.lastX+pen.shape[1]//2+1] += pen
            self.img[self.img>1] = 1

            self.obj.set_data(self.img)
            self.fig.canvas.draw()
            plt.pause(0.01)

    def on_motion(self, event):
        if event.xdata and event.ydata and self.pressed:
            nX = int(round(event.xdata))
            nY = int(round(event.ydata))
            if nX!=self.lastX or nY!=self.lastY:
                self.lastX = nX
                self.lastY = nY

                pen = np.matrix([[.25,0.5,0.25],[0.5,1,0.5],[.25,0.5,0.25]])
                self.img[self.lastY-pen.shape[0]//2:self.lastY+pen.shape[0]//2+1,self.lastX-pen.shape[1]//2:self.lastX+pen.shape[1]//2+1] += pen
                self.img[self.img>1] = 1
                self.obj.set_data(self.img)
                self.fig.canvas.draw()
                plt.pause(0.01)

    def on_release(self, event):
        self.pressed = False 
        self.obj.set_data(self.img)
        self.fig.canvas.draw()
        plt.pause(0.001)

    def on_key(self,event):
        if event.key == "a":
            x = self.net.feedforward(np.reshape(self.img, (784,1)))
            x = np.argmax(x)
            print 'result:', x
            result ="result: " + str(x)
            
            plt.suptitle(result)
            self.fig.canvas.draw()
            plt.pause(0.001)
        if event.key == "c":
            self.img = np.zeros((28,28))
            self.obj.set_data(self.img)
            self.fig.canvas.draw()
            plt.pause(0.001)

class NNIMG:

    def __init__(self, fig, axs, img, net):
        self.fig = fig
        self.img = img
        self.obj = axs.imshow(self.img, vmin=0, vmax=1)

        self.net = net
        self.n = 0
        self.training_data = []
       
        self.axs = axs
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
        if event.key == "e":
            print "keypressed"
            global exit_clicked
            exit_clicked = True

    def on_press(self, event):
        if event.inaxes != self.axs.axes: return
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
            
            self.axs.plot (self.lastX, self.lastY, colstr)
            
            self.training_data.append((np.empty([2L,1L]),np.empty([2L,1L])))
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
                self.axs.set_title(cost)
                diff = cost-old_cost 
            print "final cost on training data: {}".format(cost)
            print "final accuracy on training data: {} / {}".format(accuracy, len(self.training_data))
            self.update_img()     

    def update_img(self):
        for ix in range(self.img.shape[1]):
            for iy in range(self.img.shape[0]):
                jx = float(ix)/self.img.shape[1]
                jy = float(iy)/self.img.shape[0]
                col = self.net.feedforward(np.array([[jx],[jy]]))[1]

                self.img[iy,ix] = col

        self.obj.set_data(self.img)
        self.fig.canvas.draw()
        plt.pause(0.001)


def main():  
    plt.ioff()
    plt.show()
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #train ('trainednet2')
    fig, axs = plt.subplots(2, 2)
   
    fig.canvas.draw()
    img1 = np.zeros((28,28))
    

    net1 = network2.load('trainednet2')
    
    myobj1 = drawableIMG(fig, axs[0,1], img1, net1)

    img2 = 0.5*np.ones((140,140))
    net2 = network2.Network([2, 10, 10, 2], cost=network2.CrossEntropyCost)
    net2.large_weight_initializer()

    myobj2 = NNIMG(fig, axs[1,0], img2, net2)

    plt.draw()
   # plt.pause(100.001)

        
    plt.ion()
    
    while True:       
        plt.pause(0.001)

    #obj = axs[1, 1].imshow(np.reshape(test_data[0][0],(28,28)),vmin=0, vmax=1)
    
    #for i in range(1,10):
    #    obj.set_data(np.reshape(test_data[i][0],(28,28)))
    #    x = net1.feedforward(test_data[i][0])
    #    x = np.argmax(x)
    #    result ="result: " + str(x)
    #    print 'result:', x
    #    plt.suptitle(result)
        
    #    fig.canvas.draw()
       # wait()
          
    print "exit"
main()