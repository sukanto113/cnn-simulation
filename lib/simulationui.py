import math
import time
import numbers
import numpy as np
from PIL import Image
import ipywidgets as widgets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from ipywidgets import FloatSlider, IntSlider, Dropdown, Layout, interact

from .util import plot_XY_function_graph
from .cnn import ImageUtil, CNN, CNNTemplate, CNNSimulator
from .cnn_template import CNNTemplateLib
from .ode import rk4_system_algorithm, rk4_system_method, rkacem_system_method, ode_methods

def build_interactive_simulation_gui(simulator):
    ode_methods_name = list(ode_methods.keys())
    @widgets.interact(
        ode_method = ode_methods_name,
        time=FloatSlider(min=0, max=20, step=.1, value=1, layout=Layout(width='600px')),
        step_size=FloatSlider(min=.1, max=5, step=.1, value=1, layout=Layout(width='600px')),
        tol=".01",
        max_tol_dym=".01"
    )
    def run_hole_fill_simulator(ode_method, time, step_size,tol, max_tol_dym):

        simulator.step_size = step_size
        simulator.simulation_time = time
        simulator.max_tolerable_dynamic = float(max_tol_dym)
        simulator.tol = float(tol)
        simulator.ode_method = ode_methods[ode_method]
        simulator.simulate()
        display(ImageUtil.scale_image(simulator.get_final_state_output_image(), 5).resize((100,100)))
        simulator.display_simulator_output()
        

def build_button_simulation_gui(simulator):
    ode_methods_name = list(ode_methods.keys())
    ode_method_dropdown = widgets.Dropdown(description="ODE Method:", options=ode_methods_name, style= {'description_width': '100px'})
    time_slider = widgets.FloatSlider(description="Simulation Time:", min=0, max=20, value=2, style= {'description_width': '100px'})
    step_size_slider = widgets.FloatSlider(description="Step Size:", min=.1, max=3, style= {'description_width': '100px'})
    tolerance_field = widgets.FloatText(description="Tolerance:", value=.01, style= {'description_width': '100px'}, layout=Layout(width='200px'))
    stop_dynamic_field = widgets.FloatText(description="Stop Dynamic:", value=.01, style= {'description_width': '100px'}, layout=Layout(width='200px'))
    
    
    run_simulation_button = widgets.Button(description='Run Simulaiton')
    cnn_input_output = widgets.Output()
    cnn_initial_state_output = widgets.Output()
    cnn_final_state_output = widgets.Output()
    cnn_static_output = widgets.Output()
    
    output_file_name_field = widgets.Text(
        value='image/output.png',
        placeholder='Type something',
        description='Output File:',
    )
    
    output_file_save_button = widgets.Button(description='Save')
    
    base_ui = widgets.VBox([
        ode_method_dropdown,
        time_slider,
        step_size_slider, 
        widgets.HBox([tolerance_field, stop_dynamic_field]),
        run_simulation_button,
        widgets.HBox([cnn_input_output, cnn_initial_state_output, cnn_final_state_output]),
        cnn_static_output,
        widgets.HBox([output_file_name_field, output_file_save_button])
    ])
    
    
    with cnn_input_output:
        cnn_input_output.clear_output()
        display(ImageUtil.scale_image(simulator.get_input_image(), 5).resize((100,100)))
    with cnn_initial_state_output:
        cnn_initial_state_output.clear_output()
        display(ImageUtil.scale_image(simulator.get_initial_state_image(), 5).resize((100,100)))
            
    def setup_simulator_properties_from_ui(simulator):
        simulator.tol = float(tolerance_field.value)
        simulator.ode_method = ode_methods[ode_method_dropdown.value]
        simulator.step_size = float(step_size_slider.value)
        simulator.simulation_time = float(time_slider.value)
        simulator.max_tolerable_dynamic = float(stop_dynamic_field.value)

    def run_simulation_n(button):
        setup_simulator_properties_from_ui(simulator)
        simulator.simulate()
        with cnn_final_state_output:
            cnn_final_state_output.clear_output()
            display(ImageUtil.scale_image(simulator.get_final_state_output_image(), 5).resize((100,100)))
        with cnn_static_output:
            cnn_static_output.clear_output()
            simulator.display_simulator_output()
            
    def save_simulation_output_image(button):
        output_file_path = output_file_name_field.value
        simulator.get_final_state_output_image().save(output_file_path)
        
        
    run_simulation_button.on_click(run_simulation_n)
    output_file_save_button.on_click(save_simulation_output_image)
    display(base_ui)