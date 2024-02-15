import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

df = pd.read_csv("students_drahi_production_consumption_hourly.csv")[10:]
df['datetime'] = pd.to_datetime(df['datetime'])
variable_names = df.columns[1:]  # Exclude the datetime column

# Function to update the plot based on selected variables
def update_plot(selected_date, selected_variables):
    selected_data = df[df['datetime'].dt.date == selected_date]
    plt.figure(figsize=(10, 6))

    # Plot selected variables
    for variable in selected_variables:
        plt.plot(selected_data['datetime'], selected_data[variable], label=variable)

    plt.title(f'Data for {selected_date}')
    plt.xlabel('Hour')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Create a slider with dates
date_slider = widgets.SelectionSlider(
    options=pd.to_datetime(df['datetime'].dt.date.unique()),
    value=pd.to_datetime(df['datetime'].dt.date.min()),
    description='Select Date',
    continuous_update=False
)
# Create checkboxes for each variable
variable_checkboxes = [widgets.Checkbox(value=False, description=variable) for variable in variable_names]

# Create the interactive plot
interact(update_plot, selected_date=date_slider, selected_variables=variable_checkboxes)

# Show the plot
plt.show()
