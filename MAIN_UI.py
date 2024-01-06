import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

model = tf.keras.models.load_model('my_model34.h5')
scaler = joblib.load('my_scaler.pkl')

ranges = [(50,1300),(0.01,80),(0,1500),(0,180),(-600e3,600e3),(-1200e3,1200e3),
          (0,180),(0,50e3),(0,180),(-1200e3,1200e3),(0.01,50)]

default_values = [500,10,500,70,-30000,-75000,65,1000,50,-300000,5]

input_labels = ['H1_freq', 'spinning_freq', 'power_1_1_1', 'cs_beta_2', 'ss_iso_1_2', 
                'ss_ani_1_2', 'ss_beta_1_2', 'ss_ani_1_3', 'ss_beta_1_3', 'mw', 'T1e1']

root = tk.Tk()
root.title("Model Predictor")

input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

plot_frame = ttk.Frame(root, padding="10")
plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))

inputs = []
for i in range(11): 
    ttk.Label(input_frame, text=f"{input_labels[i]} ({ranges[i][0]}-{ranges[i][1]}):").grid(row=i, column=0, sticky=(tk.W))
    input_var = tk.DoubleVar()
    input_var.set(default_values[i]) 
    entry = ttk.Entry(input_frame, textvariable=input_var)
    entry.grid(row=i, column=1, sticky=(tk.W, tk.E))
    inputs.append((input_var, entry, ranges[i]))

param_var = tk.StringVar()
ttk.Label(input_frame, text="Parameter to plot:").grid(row=12, column=0, sticky=(tk.W))
ttk.Combobox(input_frame, textvariable=param_var, values=input_labels).grid(row=12, column=1, sticky=(tk.W, tk.E))

param_var2 = tk.StringVar()
ttk.Label(input_frame, text="Second parameter to plot:").grid(row=13, column=0, sticky=(tk.W))
ttk.Combobox(input_frame, textvariable=param_var2, values=input_labels).grid(row=13, column=1, sticky=(tk.W, tk.E))

plot_type_var = tk.StringVar()
ttk.Label(input_frame, text="Plot type:").grid(row=14, column=0, sticky=(tk.W))
ttk.Combobox(input_frame, textvariable=plot_type_var, values=['Normal', 'Contour']).grid(row=14, column=1, sticky=(tk.W, tk.E))

def plot_predictions():
    plot_type = plot_type_var.get()


    if not plot_type:
        messagebox.showwarning("No Plot Type Selected", "Please select a plot type (Normal or Contour).")
        return


    x = []
    for input_var, entry, range_ in inputs:
        value = input_var.get()
        if not range_[0] <= value <= range_[1]:
            messagebox.showwarning("Invalid Input", f"Please enter a value between {range_[0]} and {range_[1]} for {entry.cget('text')}.")
            return
        x.append(value)
    x = np.array(x).reshape(1, -1)


    param = input_labels.index(param_var.get())
    

    percision = 100


    plot_type = plot_type_var.get()


    if plot_type == 'Normal':
    
        param_values = np.linspace(*ranges[param], percision).reshape(-1, 1)

    
        x_values = np.repeat(x, percision, axis=0)
        x_values[:, param] = param_values[:, 0]
    elif plot_type == 'Contour':
        param2 = input_labels.index(param_var2.get())

    
        param_values = np.linspace(*ranges[param], percision)
        param_values2 = np.linspace(*ranges[param2], percision)

    
        param_values, param_values2 = np.meshgrid(param_values, param_values2)

    
        x_values = np.repeat(x, percision*percision, axis=0)
        x_values[:, param] = param_values.ravel()
        x_values[:, param2] = param_values2.ravel()


    x_values_scaled = scaler.transform(x_values)


    y_values = model.predict(x_values_scaled)


    fig = Figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))


    if plot_type == 'Normal':
        ax.plot(param_values, y_values)
        ax.set_xlabel(f'{input_labels[param]}')
        ax.set_ylabel('Output')
    elif plot_type == 'Contour':
    
        y_values = y_values.reshape(percision, percision)

    
        contour = ax.contourf(param_values, param_values2, y_values, levels=30, cmap=cm.coolwarm)

    
        ax.set_xlabel(f'{input_labels[param]}')
        ax.set_ylabel(f'{input_labels[param2]}')

    
        fig.colorbar(contour)


    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)


ttk.Button(input_frame, text="Plot", command=plot_predictions).grid(row=15, column=0, columnspan=2)

root.mainloop()

