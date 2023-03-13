import numpy as np
from PIL import Image
import numbers
import time
import matplotlib.pyplot as plt

from .ode import rk4_system_algorithm, rk4_system_method
class ImageUtil:
    def image_from_bipolar_encode_nparray(array, scale=1):
        """
        Image from a bipolar encoded array (where +1 form black and -1 for white)

        Example:
        array = np.array(
            [[-1, +1, -1, -1],
            [+1, -1, +1, -1],
            [+1, -1, +1, -1],
            [-1, +1, -1, +1],]
        )

        display(ImageUtil.image_from_bipolar_encode_nparray(array, scale=10))
        """
        gray_array = ImageUtil.convert_bipolar2gray_encode(array)
        image = ImageUtil.image_from_gray_encode_nparray(gray_array, scale)
        return image
    
    def image_from_gray_encode_nparray(array, scale=1):
        """
        Image from gray encoded array (where 0 for black and 255 for white).

        Example:
        array = np.array(
            [[0, 200],
            [255, 0]]
        )
        image = ImageUtil.image_from_gray_encode_nparray(array, scale=20)
        display(image)
        """
        scaled_array = ImageUtil._scale_np_array(array, scale).astype(np.uint8)
        image = Image.fromarray(scaled_array, mode='L')
        return image
    
    def _scale_np_array(array, scale):
        return np.kron(array, np.ones((scale, scale)))
    
    def convert_bipolar2gray_encode(array):
        """ 
        Convert a bipolar encoded array (where +1 for black and -1 for white) into
        a gray encoded array (where 0 for black and 255 for white)
        """
        return (-255/2)*array + (255/2)

    def convert_grey2bipolar_encode(array):
        """ 
        Convert a bipolar encoded array (where +1 for black and -1 for white) into
        a gray encoded array (where 0 for black and 255 for white)
        """
        return (-2/255)*array + 1
    
    def convert_image2gray_encode(image):
        """Convert image to gray encoded np array"""
        gray_image = image.convert('L')
        return np.asarray(gray_image)

    def convert_image2bipolar_encode(image):
        """
        Convert image to bipolar encoded np array
        
        Example:
        image = Image.open("hole_fill_input.png")
        ImageUtil.convert_image2bipolar_encode(image)
        """
        gray_encode = ImageUtil.convert_image2gray_encode(image)
        return ImageUtil.convert_grey2bipolar_encode(gray_encode)
    
    def scale_image(image, scale):
        """ Scale a image using Kronecker product """
        gray_encode = ImageUtil.convert_image2gray_encode(image)
        image = ImageUtil.image_from_gray_encode_nparray(gray_encode, scale)
        return image
        
    def save_image_from_nparray(array, filename, display_scale=1):
        image = ImageUtil.image_from_bipolar_encode_nparray(array)
        display(ImageUtil.scale_image(image, display_scale))
        image.save(filename)

# implementation of CNN in python
class CNNTemplate:
    def __init__(self, A, B, I):
        self.setA(A)
        self.setB(B)
        self.setI(I)

    def setA(self, A):
        if(not isinstance(A, np.ndarray)):
            raise TypeError("'A' must be a np.ndarray")
        if(A.shape != (3,3)):
            raise TypeError("'A' must be of shape (3, 3)")
        self.A = A
    
    def setB(self, B):
        if(not isinstance(B, np.ndarray)):
            raise TypeError("'B' must be a np.ndarray")
        if(B.shape != (3,3)):
            raise TypeError("'B' must be of shape (3, 3)")
        self.B = B

    def setI(self, I):
        if(not isinstance(I, numbers.Number)):
            raise TypeError("'I' must be a number")
        self.I = I
        
    def __str__(self) -> str:
        return f"A={self.A}\nB={self.B}\nI={self.I}"
    
    def _repr_html_(self):
        return str(self)
    
class CNN:
    """
    This is python representation of Cellular Nurla Network(CNN)
    
    Attributes:
    shape
    size
    """
    def __init__(self, input, state, template):
        if(not isinstance(input, np.ndarray)):
            raise TypeError("'input' must be a np.ndarray")
        self.input = input
        self.shape = input.shape
        rows, cols = self.shape
        self.size = rows * cols
        
        self.C = 1
        self.Rx = 1

        self.set_state(state)
        self.set_template(template)
        
    def set_state(self, state):
        if(not isinstance(state, np.ndarray)):
            raise TypeError("'state' must be a np.ndarray")
        self.verify_shape_of_state(state)
        self.state = state
    
    def verify_shape_of_state(self, state):
        if(self.shape != state.shape):
            error_message = ("Shape of input and state must be same. "
                            f"{self.input.shape} != {state.shape}")
            raise TypeError(error_message)
            
    def set_template(self, template):
        # if(not isinstance(template, CNNTemplate)):
        #     raise TypeError("'template' must be a CNNTemplate")
        self.template = template
    
    def get_dynamic_at(self, i, j):
        """return dynamic value for self.state at cell (i, j)"""
        return self.get_dynamic_for_state_at(i, j, self.state)

    def get_dynamic_for_state_at(self, i, j, state):   
        """return dynamic value for state at cell (i, j)"""
        C = self.C
        Rx = self.Rx
        
        Vx = state
        Vu = self.input
        
        A = self.template.A
        B = self.template.B
        I = self.template.I

        x_sum = -(1/Rx)*Vx[i,j]
        y_sum = 0
        u_sum = 0

        for k in range(i-1,i+2):
            for l in range(j-1,j+2):
                if 0 <= k and k < Vx.shape[0]:
                    if 0 <= l and l < Vx.shape[1]:
                        Vy = (abs(Vx[k,l]+1) - abs(Vx[k,l]-1))/2
                        y_sum = y_sum + A[k-i+1, l-j+1]*Vy
                        u_sum = u_sum + B[k-i+1, l-j+1]*Vu[k,l]

        dynamic = x_sum + y_sum + u_sum + I
        return dynamic
    
    def get_dynamic_for_state(self, state):
        dynamics = np.zeros(self.shape)
        rows, cols = self.shape
        for i in range(0, rows):
            for j in range(0, cols):
                dynamics[i, j] = self.get_dynamic_for_state_at(i, j, state)
        return dynamics

class CNNSimulator:
    """Properties:
        simulator.output
        simulator.computation_time
        simulator.iteration_time_steps
        simulator.num_of_iteration
    """
    
    def __init__(self, cnn):
        self.cnn = cnn
        self.ode_method = rk4_system_method
        self.simulation_time = 2
        self.step_size = .5
        self.tol = .01
        self.max_tolerable_dynamic = .01
        self.states = np.array([cnn.state.reshape(cnn.size)])
        
        self._iteration_time_steps = np.array([0])
        self._computation_time = 0

    def get_dynamic_function_at(self, index):
        """This is adapter for ODE algorithm"""
        total_row, total_col = self.cnn.shape
        i = index // total_col
        j = index % total_col
        return lambda t, X : self.cnn.get_dynamic_for_state_at(i, j, X.reshape(self.cnn.shape))

    def simulate(self):
        """Call this function to run the simulation algorithm. Returns computation time."""

        initial_state = self.cnn.state.reshape((self.cnn.size))
        
        start_time = time.time()
        T, XX = rk4_system_algorithm(
            t0=0,
            X0=initial_state,
            tn=self.simulation_time,
            h=self.step_size,
            getFunc=self.get_dynamic_function_at,
            ode_method=self.ode_method,
            tol=self.tol,
            max_tolerable_dynamic=self.max_tolerable_dynamic)

        end_time = time.time()
        
        self.states = XX
        self._iteration_time_steps = T
        self.simulation_time = T[-1]
        self._computation_time = end_time - start_time
    
    def get_iteration_time_steps(self):
        return self._iteration_time_steps
        
    def get_num_of_iteration(self):
        return len(self.iteration_time_steps)
    
    def get_computation_time(self):
        return self._computation_time
    
    def get_final_state(self):
        """Last state of the simulation"""
        return self.states[-1].reshape(self.cnn.shape)
    
    def get_end_dynamic(self):
        return self.cnn.get_dynamic_for_state(self.end_state)

    def get_final_state_output(self):
        """Output signal of the last simulation state"""
        result = self.get_final_state()
        Vy = (abs(result+1) - abs(result-1))/2
        return Vy
    
    def get_final_state_output_image(self):
        """Convert bipolar encoded simulation output into image"""
        output = self.get_final_state_output()
        image = ImageUtil.image_from_bipolar_encode_nparray(output)
        return image
    
    def get_input(self):
        return self.cnn.input
    
    def get_input_image(self):
        return ImageUtil.image_from_bipolar_encode_nparray(self.get_input())
    
    def get_initial_state(self):
        return self.cnn.state
    
    def get_initial_state_image(self):
        return ImageUtil.image_from_bipolar_encode_nparray(self.get_initial_state())
        
        
    def plot(self):
        f, axarr = plt.subplots(1, 3)

        # axarr[0].axis('off')
        # axarr[1].axis('off')
        # axarr[2].axis('off')

        axarr[0].set_title('Input', fontsize='large', loc='center')
        axarr[1].set_title('Initial State', fontsize='large', loc='center')
        axarr[2].set_title('Output State', fontsize='large', loc='center')

        axarr[0].imshow(self.get_input_image(), cmap = plt.cm.gray)
        axarr[1].imshow(self.get_initial_state_image(), cmap = plt.cm.gray)
        axarr[2].imshow(self.get_final_state_output_image(), cmap = plt.cm.gray)
        
    def display_simulator_output(self):
        # display(ImageUtil.scale_image(self.get_final_state_output_image(), 20))
        # self.plot()
        # plt.show()
        print("computation time =", self.computation_time)
        print("simulation time =", self.simulation_time)

        # print("time steps =", self.iteration_time_steps)
        print("total iteration =", self.num_of_iteration)
        print("system max_dynamic = ", np.max(abs(self.get_end_dynamic())))
    

    
    iteration_time_steps = property(get_iteration_time_steps)
    num_of_iteration = property(get_num_of_iteration)
    end_state = property(get_final_state)
    output = property(get_final_state_output)
    computation_time = property(get_computation_time)
