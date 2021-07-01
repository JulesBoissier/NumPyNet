import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import numpy as np
import pygame as pg

from neural_net import NeuralNetwork


class Board:
    def __init__(
        self,
        name: str,
        number_of_cells : int = 60
        ):
        self.name = name
        self.pixels = 500
        self.num_cells = number_of_cells
        self.num_cells = 20* self.num_cells//20  
        self.barwidth = 200
        self.cell_pixels = self.pixels / self.num_cells
        self.data = np.zeros((self.num_cells,self.num_cells))
        self.coordinates = []
        self.win = pg.display.set_mode((self.pixels+self.barwidth,self.pixels))

        #Colours
        self.black = (0,0,0)
        self.white = (225,225,225)
        self.grey = (100,100,100)
        self.darkgrey = (50,50,50)

        #Creating neural network classifier
        self.classifier = NeuralNetwork()
        self.classifier.load_model('ui_classifier')
           
    def make_button(
        self,
        name: str,
        height:int
        ) -> None:
        """
        Creates and blits image of button at a given height with text on it.

        Parameters
        ----------
        name: Text to be displayed on button, position is offset by string length to attempt to center on button.
        height: Distance from the top of the pygame window, as a ratio of total window height.
        """
        myfont = pg.font.SysFont('Arial Black', 15)
        my_button = pg.Rect(self.pixels+0.2*self.barwidth,height*self.pixels,0.6*self.barwidth,0.08*self.pixels)
        pg.draw.rect(self.win,self.darkgrey,my_button,0)
        pg.draw.rect(self.win,self.black,my_button,1)
        textsurface1 = myfont.render(name, False, self.white)
        str_length = 3.5*len(name)
        self.win.blit(textsurface1,(self.pixels+(0.57-(str_length/100))*self.barwidth,(height+0.02)*self.pixels))

    def grid(self) -> None:
        """
        Draws the entire window, including grid to draw on and buttons (clear, submit and quit).
        """
        pg.display.set_caption(self.name)
        self.win.fill(self.white)
        sidebar = pg.Rect(self.pixels,0,self.barwidth,self.pixels)
        pg.draw.rect(self.win,self.grey,sidebar,0)

        self.make_button('Clear',0.3)
        self.make_button('Submit',0.45)
        self.make_button('Quit',0.6)
        
        for i in range(self.num_cells):
            for j in range(self.num_cells):
                xcoordinate = j*self.cell_pixels
                ycoordinate = i*self.cell_pixels
                rect = pg.Rect(xcoordinate,ycoordinate,self.cell_pixels,self.cell_pixels)
                pg.draw.rect(self.win,self.grey,rect,1)
                
    def circle_update(
        self,
        x: float,
        y:float,
        r: int = 18
        ) -> None:
        """
        Selects grid cells around mouse position in a circle.

        Parameters
        ----------
        x, y: Mouse position obtained from pg.mouse.get_pos()
        r : Radius for the circle inside which cells are selected.
        """
        r = 18
        if x < self.pixels and y < self.pixels:
            for i in range(self.num_cells):
                for j in range(self.num_cells):
                    if ((x-(i*self.cell_pixels))**2) + ((y-(j*self.cell_pixels))**2) < r**2:
                        i = int(i)
                        j = int(j)
                        self.coordinates.append((i,j))
                        self.data[j][i] = 1
        self.coordinates = list(set(self.coordinates))

    def draw(self):
        """
        Fills in black rectangles on the grid which have been drawn over by cycling self.coordinates list.
        """
        for k in range(len(self.coordinates)):
            xcoordinate = self.coordinates[k][0]*self.cell_pixels
            ycoordinate = self.coordinates[k][1]*self.cell_pixels
            rect = pg.Rect(xcoordinate,ycoordinate,self.cell_pixels,self.cell_pixels)
            pg.draw.rect(self.win,self.black,rect,0)
            
    def clear(self) -> None:
        """
        Empties self.coordinates list and redraws window over previously coloured in rectangles to clear the board.
        """
        self.coordinates = []
        self.data = np.zeros((self.num_cells,self.num_cells))
        self.grid()

    def buttons(
        self,
        x : float,
        y : float
        ) -> bool:
        """
        Assigns clear, predict and quit features to mouseclicks on specific coordinates, matching previously
        drawn buttons.

        Parameters
        ----------
        x, y: Mouse position obtained from pg.mouse.get_pos()
        """
        run = True

        if self.pixels+0.2*self.barwidth < x < self.pixels+0.8*self.barwidth:
            if 0.3*self.pixels < y < 0.38*self.pixels:
                self.clear()
            elif 0.45*self.pixels < y < 0.53*self.pixels:
                self.predict()
            elif 0.6*self.pixels < y < 0.68*self.pixels:
                pg.quit()
                return not run

        return run

    def rescale(
        self,
        precision_coef : int = 10
        ) -> np.ndarray :
        """
        Rescales the data back down to a 20 by 20 array, applying greyscaling by averaging neighbouring cells in 
        the process of condensating. 
        Does so by first cropping out empty borders, stretching the array and finally averaging groups of cells 
        into singlecells.

        Parameters
        ----------
        self.data: Rescaling is directly applied to the recorded self.data array
        precision_coef: Defines the ratio for the array stretching, which increases the precision for the condensation.

        Returns
        ----------
        self.data: A 20 by 20 array with the least amount of empty borders, i.e containing the most information.
        """

        #Deletes all empty row and columns on the borders to obtain smallest information containing rectangle
        while np.all(self.data[0] == np.zeros((1,self.num_cells))):
            self.data = np.delete(self.data,0,0)    
        while np.all(self.data[-1] == np.zeros((1,self.num_cells))):
            self.data = np.delete(self.data,-1,0) 

        new_height = self.data.shape[0]

        while np.all(self.data[:,0] == np.zeros((new_height,1))):
            self.data = np.delete(self.data,0,1)
        while np.all(self.data[:,-1] == np.zeros((new_height,1))):
            self.data = np.delete(self.data,-1,1)

        #Adds rows or columns to reshape rectangle into a square
        highest_dim = np.maximum(self.data.shape[0],self.data.shape[1])
        rows_to_add = highest_dim - self.data.shape[0]
        cols_to_add = highest_dim - self.data.shape[1]
        if rows_to_add != 0:
            additive = np.zeros((rows_to_add,self.data.shape[1]))
            self.data = np.concatenate((self.data,additive),0)
        if cols_to_add != 0:
            additive = np.zeros((self.data.shape[0],cols_to_add))
            self.data = np.hstack((self.data,additive))

        #Stretches the data into a larger grid with np.repeat applied horizontally and vertically
        
        stretch_factor = precision_coef*self.num_cells/highest_dim
        self.data = np.repeat(self.data,stretch_factor,axis = 0) 
        self.data = np.repeat(self.data,stretch_factor,axis = 1) 

        #Adds missing rows/columns due to rounding of stretch_factor, so output has consistant shape
        size_dif = precision_coef*self.num_cells - self.data.shape[0] 
        if size_dif != 0:
            rows_to_add = np.zeros((size_dif,self.data.shape[1]))
            cols_to_add = np.zeros((precision_coef*self.num_cells,size_dif))
            self.data = np.concatenate((self.data,rows_to_add),0)
            self.data = np.hstack((self.data,cols_to_add))

        #Condensate back into a 20*20 grid and implements grey scaling
        condens_factor = int(precision_coef*self.num_cells/20)
        scaled_data = np.zeros((20,20))
        for i in range(20):
            for j in range(20):
                for k in range(condens_factor):
                    for l in range(condens_factor):
                        scaled_data[i][j] += self.data[condens_factor*i+k][condens_factor*j+l]/(condens_factor**2) 
        return scaled_data

    def center_of_mass(
        self,
        array
        ) -> np.ndarray:
        """
        Shifts the center of mass of the array to the center point of the array, and adds an empty border with a 
        width of four to match the 28 by 28 format of the MNIST data.

        Parameters
        ----------
        array: A 20 by 20 array the was rescaled by the self.rescale function.

        Returns
        ----------
        centered_data: A 28 by 28 array with the center of mass on the center point.
        """
        #Calculates center of mass of the 20 by 20 array
        x_com = 0
        y_com = 0
        M = np.sum(array)
        for i in range(20): 
            for j in range(20):
                y_com += (i+1)*array[i][j]/M
                x_com += (j+1)*array[i][j]/M


        #Calculates horizontal and vertical shift to center
        x_shift = np.round(x_com-10.5)
        y_shift = np.round(y_com-10.5) 
        
        #Augments matrix with empty rows and columns to 28 by 28 shape (Standard MNIST format)
        vertical_array = np.zeros((20,4))
        horizontal_array = np.zeros((4,28))
        resized_array = np.concatenate((
                                        horizontal_array,
                                        np.hstack((vertical_array,array,vertical_array)), 
                                        horizontal_array
                                        ),0)
        

        #Populates 28 by 28 matrix with data shifted to the center of mass
        centered_data = np.zeros((28,28))
        for i in range(28):
            for j in range(28):
                if resized_array[i][j] != 0:
                    k = int(i  - y_shift) 
                    l = int(j  - x_shift)
                    centered_data[k][l] = resized_array[i][j]
        return centered_data

    def preprocess(self) -> np.ndarray:
        """
        Processes the self.data array of drawn on cells to follow the processing used for the MNIST database in
        order to format the input for the classifier similarly to the training data.

        Returns
        ----------
        processed_data: A 1 by 784 array processed andready to go through a classifier.
        """
        array = self.rescale()
        centered_array = self.center_of_mass(array)
        
        #Vectorizes the array to match classifier input shape
        processed_data = np.zeros((1,784))
        
        for i in range(28):
            for j in range(28):  
                processed_data[0][i*28+j] = centered_array[i][j]  
        return processed_data

    def predict(self) -> None:
        """
        Runs the processed data through the loaded model for a neural network classifier and outputs a prediction.
        """
        if np.count_nonzero(self.data) > 0 :
            
            data = self.preprocess()
            prediction = int(self.classifier.predict(data)[0])
            probability = round(float(self.classifier.predict(data)[1]),3) * 100
            print(f'The drawn number was a {prediction} with a degree of confidence of {probability} percent.')
            self.clear()
            pg.display.update()

def run_mnist_app(name : str = 'Guesser', precision_factor : int = 60):
    """
    Runs a pygame app using the Board class on which the user can draw a digit which the classifier will try to predict.
    The classifier is a model loaded from the folder which uses the NeuralNetwork class to make a prediciton.
    An estimate of the classifier's confidence in its prediction is also printed
    """
    pg.init()
    run = True
    UI = Board(name, number_of_cells = precision_factor)
    UI.grid()
    pg.display.update()

    while run:
        px, py = pg.mouse.get_pos()
        for event in pg.event.get():
            if pg.mouse.get_pressed() == (1,0,0):
                UI.circle_update(px,py)
                UI.draw()
                run = UI.buttons(px,py)
                if run == False:
                    break
                pg.display.update()
                
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_BACKSPACE:
                    UI.clear()
                    pg.display.update()
                if event.key == pg.K_RETURN:
                    UI.predict()
                if event.key == pg.K_ESCAPE:
                    run = False
                    pg.quit()
                    break
            if event.type == pg.QUIT:
                run = False  

    pg.quit()

if __name__ == '__main__':
    
    run_mnist_app()