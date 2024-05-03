########################################################################################################
#To get the shiny app to run or create a new one, you will need to run it in VS Code for the least amount of errors and headaches
#-- You can reference the site for more help
#https://shiny.posit.co/py/docs/install-create-run.html
#Documentaiton: 
# (I used this) Shiny Express: https://shiny.posit.co/py/api/express/
# (Original) Shiny Core: https://shiny.posit.co/py/api/core/

#--Necessary items to install:
#- You need to install the shiny extension and the python extension. 
#- pip install plotly, shinywidgets, shiny
#########################################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from shinywidgets import render_widget
from shiny import reactive, render
from shiny.express import input, output, render, ui
from shinyswatch import theme
from joblib import load
import pathlib
import asyncio



#########################################################################################################
# loads data, sets header and theme
#########################################################################################################
# sets the theme
theme.flatly()
# header and hamburger menu
ui.page_opts(title="Conditional Admits Analysis", window_title="Conditional Admits Analysis", fillable=True, id="page")

#########################################################################################################
# aggregates data
#########################################################################################################
# load and manipulate the data
conditional_admits_path = pathlib.Path(__file__).parent / "conditional_dataset_for_analysis.csv"
conditional_admits = pd.read_csv(conditional_admits_path)

######################### Statistics
stats_path= pathlib.Path(__file__).parent / "stats.csv"
stats = pd.read_csv(stats_path)

######################### Linear Regression
# loads the model into the shiny dashboard
# NOTE this will need to be changed if you train a new model

#initializes the model
# Define an async function to handle async operations
linear_regression_model_all = load(pathlib.Path(__file__).parent / "ga_linear_all_regression_model.joblib")
linear_regression_model_maj = load(pathlib.Path(__file__).parent / "ga_linear_maj_regression_model.joblib")
random_forest_model = load(pathlib.Path(__file__).parent / "ga_randomforest_model.joblib")

def make_prediction(model, input_data):
    if model is not None:
        return model.predict(input_data)
    else:
        return ui.markdown("Attempted to use an unloaded model for prediction.")

#########################  Totals
# Get only the unique items to get an accurate amount in the major
unique_values = conditional_admits.drop_duplicates(subset=None, keep='first', inplace=False)
# Group the data by major
totals_by_major = unique_values.groupby(['major_desc','math_readiness_ind','gender','ethnicity']).size().reset_index(name='total_by_major')
columns_to_clean = totals_by_major[['gender','math_readiness_ind']]
def create_better_names (full_data, partial_data):
    for index, item in partial_data.iterrows():
        for value in partial_data.columns:
            if item[value] == 'N':
                partial_data.at[index, value] = 'Not Ready'
            elif item[value] == 'Y':
                partial_data.at[index, value] = 'Ready'
            elif item[value] == 'm':
                partial_data.at[index, value] = 'Male'
            elif item[value] == 'f':
                partial_data.at[index, value] = 'Female'
            elif item[value] == 'n':
                partial_data.at[index,value] = 'Other'
            else:
                partial_data.at[index,value] = value
        # puts everything together
        for col in full_data.columns:
            if col in partial_data:
                full_data[col] = partial_data[col]
    return full_data
totals_by_major = create_better_names(totals_by_major,columns_to_clean)
def assign_school(row):
    school_of_health_science = ['Health Science', 'Exercise Science']
    school_of_arts_and_humanities = ['Visual Arts','Communication Studies','Historical Studies','Studies in the Arts']
    school_of_social_and_behavioral_sciences = ['Criminal Justice', 'Social Work']
    school_of_business = ['Business Studies', 'Computer Science']
    school_of_general_studies_and_graduate_education = ['Liberal Studies']
    school_of_natural_sciences_and_mathematics = ['Applied Physics', 'Biochemistry Molecular Biology','Biology','Chemistry',
                                                            'Environmental Studies','Marine Science','Mathematics',
                                                            'Environmental Science','Sustainability']
    if row['major_desc'] in school_of_health_science:
        return 'School of Health Science'
    elif row['major_desc'] in school_of_arts_and_humanities:
        return 'School of Arts and Humanities'
    elif row['major_desc'] in school_of_social_and_behavioral_sciences:
        return 'School of Social and Behavioral Sciences'
    elif row['major_desc'] in school_of_business:
        return 'School of Business'
    elif row['major_desc'] in school_of_natural_sciences_and_mathematics:
        return 'School of Natural Science and Mathematics'
    elif row['major_desc'] in school_of_general_studies_and_graduate_education:
        return 'School of General Studies and Graduate Education'
    else:
        return 'School not found'
filtered_columns = ['id','major_desc','final_grade','overall_gpa', 'gender']
filtered_data = conditional_admits[filtered_columns].copy()
totals_by_gpa = filtered_data.groupby(['major_desc','overall_gpa', 'gender']).size().reset_index(name='students_in_major')
totals_by_gpa['success_by_gpa'] = (totals_by_gpa['overall_gpa'] >= 3.0).astype(int)
totals_by_gpa['school'] = totals_by_gpa.apply(assign_school, axis=1)
gender = totals_by_gpa[['gender']]
totals_by_gpa = create_better_names(totals_by_gpa, gender)
unique_majors = totals_by_gpa['major_desc'].unique()
unique_majors_list = []
for item in unique_majors:
    unique_majors_list.append(item)
dfs_by_major = {}
while unique_majors_list:
    major = unique_majors_list.pop(0)
    dfs_by_major[major] = totals_by_gpa[totals_by_gpa['major_desc'] == major]
seperate_plot_df = [
    dfs_by_major['Applied Physics'],
    dfs_by_major['Biochemistry Molecular Biology'],
    dfs_by_major['Biology'],
    dfs_by_major['Business Studies'],
    dfs_by_major['Chemistry'],
    dfs_by_major['Communication Studies'],
    dfs_by_major['Computer Science'],
    dfs_by_major['Criminal Justice'],
    dfs_by_major['Environmental Science'],
    dfs_by_major['Environmental Studies'],
    dfs_by_major['Exercise Science'],
    dfs_by_major['Health Science'],
    dfs_by_major['Historical Studies'],
    dfs_by_major['Liberal Studies'],
    dfs_by_major['Marine Science'],
    dfs_by_major['Mathematics'],
    dfs_by_major['Social Work'],
    dfs_by_major['Studies in the Arts'], 
    dfs_by_major['Sustainability'],  
    dfs_by_major['Visual Arts']]
majors = [
    'Applied Physics','Biochemistry Molecular Biology','Biology','Business Studies','Chemistry',
    'Communication Studies','Computer Science','Criminal Justice','Environmental Science',
    'Environmental Studies','Exercise Science','Health Science','Historical Studies',
    'Liberal Studies','Marine Science','Mathematics','Social Work','Studies in the Arts',
    'Sustainability','Visual Arts']
overall_data = pd.concat([df.assign(Major=major) for df, major in zip(seperate_plot_df, majors)], ignore_index=True)
majors = overall_data['major_desc'].unique()

#########################################################################################################
# Code for the dashboard
#########################################################################################################
# makes the header standout on smaller screens it will turn into a hamburger menu
ui.nav_spacer()
## build Shiny elements and display graphs
with ui.sidebar(id = "sidebar"):
    with ui.panel_conditional("input.page == 'Totals by School and Major'"):
            ui.input_radio_buttons("radio_button", "Select Category", ["Math Readiness", "Student Demographics", "GPA by School"])
            with ui.panel_conditional("input.radio_button == 'Math Readiness'"):
                ui.input_select(
                    "math_readiness",
                    "Demographic:",
                    ["Gender", 
                        "Major"]
                )
            with ui.panel_conditional("input.radio_button == 'Student Demographics'"):
                ui.input_select(
                    "student_demographics",
                    "Demographic:",
                    ["Gender",
                    "Ethnicity"]
                )
            with ui.panel_conditional("input.radio_button == 'GPA by School'"):
                ui.input_select(
                    "gpa",
                    "Choose School or overall:",
                    choices={**{"overall": "Overall"}, **{major: major for major in majors}}
                )
                ui.input_checkbox("trend","Show Linear Regression Trendline", False)
# If the random forrest text is clicked then it will show the results of
    # the random forest model that was trained
    with ui.panel_conditional("input.page == 'Random Forest Classifier Predictions'"):
        # allows the user to enter text
        # id name is student_id_text
        ui.input_text(
            "student_id",
            ui.markdown("""
                        Enter Student ID <br>
                        Example: 132318
                        """)
            )
        # button to gather information
        ui.input_task_button("gather_btn", "Gather Results")
        # button to cancel items
        ui.input_task_button("cancel_btn", "Cancel")
        # clear items
        ui.input_task_button("clear_btn", "Clear")


#########################################################################################################
#This will create the plot for major and school and then it plots the linear regression model on it to make a prediction
#########################################################################################################
# Each page that will be displayed
with ui.nav_panel("Totals by School and Major"):
    with ui.panel_conditional("input.radio_button == 'Math Readiness'"):
        with ui.panel_conditional("input.math_readiness == 'Gender'"):
            # this decorator displays the graph
            @render_widget
            def mr_gen_plot():
                gen_fig = px.histogram(totals_by_major, 
                    x='total_by_major', 
                    y='gender', 
                    color='math_readiness_ind',
                    color_discrete_map={'Ready':'blue', 'Not Ready':'gold'})
                gen_fig.update_layout(
                    autosize=True,
                    title_font_size=24,
                    title='Math Readiness by Gender',
                    yaxis_title='Math Readiness',
                    xaxis_title='Student Count',
                    legend_title = 'Math Readiness',
                    showlegend=True,
                    xaxis_showgrid = False,
                    yaxis_showgrid = False,
                    plot_bgcolor='darkgrey',  
                    barmode='group',
                )
                gen_fig.update_traces(hovertemplate='Math Readiness: %{text}<br>Totals by Gender: %{y}', text=totals_by_major['math_readiness_ind'],
                                    texttemplate=' ',
                                    hoverlabel=dict(bgcolor="white",))
                gen_fig.add_annotation(text='Hover Over Bar to See Total',
                                    xref='paper', x=0.5, y=1,  
                                    yref='paper',
                                    showarrow=False,
                                    font=dict(size=16, color='white'),  
                                    align='center',
                                    xanchor='center')
                return gen_fig
        with ui.panel_conditional("input.math_readiness == 'Major'"):
            # this decorator displays the graph
            @render_widget
            def mr_maj_plot():
                maj_fig = px.histogram(totals_by_major, 
                    x='total_by_major', 
                    y='major_desc', 
                    color='math_readiness_ind',  # This will automatically assign different colors to 'Y' and 'N'
                    title='Math Readiness by Major',
                    color_discrete_map={'Ready':'blue', 'Not Ready':'gold'})
                maj_fig.update_traces(hovertemplate='Total by Major:  %{x} <br>Major: %{y}', text=totals_by_major['math_readiness_ind'],
                                    texttemplate=' ',
                                    hoverlabel=dict(bgcolor="white",))
                maj_fig.update_layout(
                    autosize=True,
                    title_font_size=24,
                    title='Math Readiness by Major',
                    yaxis_title='Major',
                    xaxis_title='Student Count',
                    legend_title = 'Math Readiness',
                    showlegend=True,# Optionally hide the legend if it's not necessary
                    xaxis_showgrid = False,
                    yaxis_showgrid = False,
                    plot_bgcolor='darkgrey',  
                    barmode='group',
                )
                maj_fig.add_annotation(text='Hover Over Bar to See Total',
                                    xref='paper', x=0.5, y=1,  
                                    yref='paper',
                                    showarrow=False,
                                    font=dict(size=16, color='white'),  
                                    align='center',
                                    xanchor='center')
                return maj_fig
    with ui.panel_conditional("input.radio_button == 'Student Demographics'"):
        with ui.panel_conditional("input.student_demographics == 'Gender'"):
            @render_widget
            def sd_maj_plot():
                sd_maj_fig = px.histogram(totals_by_major, 
                x='total_by_major', 
                y='major_desc', 
                color='gender',
                title='Total Genders Across Program',
                color_discrete_map={'Male':'blue', 'Female':'gold', 'Other':'lightblue'})
                sd_maj_fig.update_layout(
                    autosize=True,
                    title_font_size=24,
                    title='Amount of Genders in Each Major',
                    yaxis_title='Student Count',
                    xaxis_title='Student Count',
                    legend_title = 'Gender',
                    showlegend=True,
                    xaxis_showgrid = False,
                    yaxis_showgrid = False,
                    plot_bgcolor='darkgrey',  
                    barmode='group',
                )
                sd_maj_fig.update_traces(hovertemplate='Major: %{y}<br>Totals by Gender: %{x}', text=totals_by_major['gender'],
                                    texttemplate=' ',
                                    hoverlabel=dict(bgcolor="white",))
                sd_maj_fig.add_annotation(text='Hover Over Bar to See Total',
                                    xref='paper', x=0.5, y=1,  
                                    yref='paper',
                                    showarrow=False,
                                    font=dict(size=16, color='white'),  
                                    align='center',
                                    xanchor='center')
                return sd_maj_fig
        with ui.panel_conditional("input.student_demographics == 'Ethnicity'"):
            @render_widget
            def sd_eth_plot():
                sd_eth_fig = px.histogram(totals_by_major, 
                    x='total_by_major', 
                    y='major_desc', 
                    color='ethnicity',
                    title='Total Genders Across Program',
                    color_discrete_map={'Male':'blue', 'Female':'gold', 'Other':'lightblue'})
                sd_eth_fig.update_layout(
                    autosize=True,
                    title_font_size=24,
                    title='Amount of Genders in Each Major',
                    yaxis_title='Student Count',
                    xaxis_title='Student Count',
                    legend_title = 'Gender',
                    showlegend=True,
                    xaxis_showgrid = False,
                    yaxis_showgrid = False,
                    plot_bgcolor='darkgrey',  
                    barmode='group',
                )
                sd_eth_fig.update_traces(hovertemplate='Major: %{y}<br>Totals by Gender: %{x}', text=totals_by_major['gender'],
                                    texttemplate=' ',
                                    hoverlabel=dict(bgcolor="white",))
                sd_eth_fig.add_annotation(text='Hover Over Bar to See Total',
                                    xref='paper', x=0.5, y=1,  
                                    yref='paper',
                                    showarrow=False,
                                    font=dict(size=16, color='white'),  
                                    align='center',
                                    xanchor='center')
                return sd_eth_fig
    with ui.panel_conditional("input.radio_button == 'GPA by School'"):
        with ui.panel_conditional("input.gpa == 'overall'"):
            @render_widget
            def gpa_overall_plot():
                # Create the scatter plot using Plotly Express
                gpa_ov_fig = px.scatter(overall_data, x='overall_gpa', y='students_in_major', color='Major',
                                    title='GPA of Students Across Majors',
                                    labels={'overall_gpa': 'GPA', 'students_in_major': 'Amount of Students'})# Setting the size of the plot
                
                # Move the legend to the right of the plot
                gpa_ov_fig.update_layout(
                                    autosize=True,
                                    legend_title='Majors', 
                                    legend=dict(y = 1,
                                                x = 1.35,
                                                xref='paper',
                                                yref='paper',
                                                xanchor='right', 
                                                yanchor='top'
                                        ),
                                    xaxis=dict(zeroline=False),
                                    yaxis=dict(zeroline=False),
                                    plot_bgcolor='darkgrey',
                                    xaxis_showgrid = False,
                                    yaxis_showgrid = False
                                    )
                ui.panel_conditional("input.trend")
                if input.trend():
                    gpa_values = overall_data['overall_gpa'].values.reshape(-1, 1)
                    predictions = make_prediction(linear_regression_model_all, gpa_values)
                    gpa_ov_fig.add_scatter(x=overall_data['overall_gpa'], y=predictions, mode='lines', name='Linear Fit', line=dict(color='black', width=2, dash='solid'))
                return gpa_ov_fig
        with ui.panel_conditional("input.gpa != 'overall'"):
            @render_widget
            def gpa_maj_plot():
                df = totals_by_gpa[totals_by_gpa['major_desc'] == input.gpa()]
                if not df.empty:
                    gpa_maj_fig = px.scatter(df, x='overall_gpa', y='students_in_major', 
                    title=f'{input.gpa()}',
                    color='gender',
                    color_discrete_map={'Male': 'blue', 'Female': 'gold', 'Other': 'lightblue'},
                    labels={'overall_gpa': 'Overall GPA', 'students_in_major': 'Students in Major'})
                    
                    gpa_maj_fig.update_layout(
                        autosize=True,
                        title_font_size=20,
                        yaxis_title='Student Count',
                        xaxis_title='Students in Major',
                        xaxis=dict(zeroline=False),
                        yaxis=dict(zeroline=False),
                        legend_title=" ",
                        showlegend=True,
                        xaxis_showgrid=False,
                        yaxis_showgrid=False,
                        plot_bgcolor='darkgrey'
                    )
                
                ui.panel_conditional("input.trend")
                if input.trend() and len(df) >1:
                    gpa_values = df['overall_gpa'].values.reshape(-1, 1)
                    predictions = make_prediction(linear_regression_model_maj, gpa_values)
                    gpa_maj_fig.add_scatter(x=df['overall_gpa'], y=predictions, mode='lines', name='Linear Fit', line=dict(color='black', width=2, dash='solid'))
                    ui.markdown(f'')
                elif input.trend() and len(df) <= 1:
                    gpa_maj_fig.add_annotation(text='Need More Data to Make analysis',
                        xref='paper', x=0.5, y=1,  
                        yref='paper',
                        showarrow=False,
                        font=dict(size=16, color='white'),  
                        align='center',
                        xanchor='center')
                return gpa_maj_fig
            

#########################################################################################################
# builds the random forest model
#########################################################################################################
with ui.nav_panel("Random Forest Classifier Predictions"):
    # This when the Gather Results Button is clicked this will apply the
    # slow compute function
    @reactive.effect
    @reactive.event(input.gather_btn)
    def handle_click():
        slow_compute(input.student_id())
    # this will cancel the processing if the user entered in the wrong ID 
    @reactive.effect
    @reactive.event(input.cancel_btn)
    def handle_cancel():
        slow_compute.cancel()
    # this will clear the text box for the user
    @reactive.effect
    @reactive.event(input.clear_btn)
    def handle_clear(): 
        return ui.update_text("student_id", value = " ")
    @reactive.effect
    @reactive.event(input.clear_btn)
    def clear_text_area(): 
        return 

    # this will gather the information from the user when the gather results button is clicked. 
    # it will then take a pause before running showing the user that is is processing 
    # then it will return the results of the a variable
    @ui.bind_task_button(button_id="gather_btn")
    @reactive.extended_task
    async def slow_compute(student: int):
        predictions_list = []
        try:
            await asyncio.sleep(3)
            student_array = np.array([[student]])
            student_reshape =  student_array.reshape(-1, 1)
            student_prediction = make_prediction(random_forest_model(),student_reshape)
            
            # Convert the prediction to a native Python type if it's a NumPy type
            if isinstance(student_prediction[0], np.integer):
                prediction_value = int(student_prediction[0])
                predictions_list.append({
                    'student id': student,
                    'prediction': prediction_value
                                })
            else:
                # Assuming this is already in a serializable format
                prediction_value = student_prediction[0]
                predictions_list.append({
                    'student id': student,
                    'prediction': prediction_value
                                })
            return predictions_list
        except Exception as e:
            # Log the error in the console
            print(f"Error during prediction: {str(e)}")  
            prediction_value ="Error: Cannot make a prediction"
            # prints the error for the user
            predictions_list.append({
                    'student id': student,
                    'prediction': prediction_value
                                })
        return predictions_list

    # this will show the results of the slow compute on the plot
    @render.data_frame
    def show_result():
        updated_list = []
        results = slow_compute.result()
        updated_list.append(results)
        if isinstance(results, list):
            dataframe = pd.DataFrame(results)
            return render.DataGrid(
                dataframe,
                selection_mode="row"
            )

#########################################################################################################
# Build the Statistics tab
#########################################################################################################
# shows the Statistics Dataset
with ui.nav_panel("Statistics"):
    @render.data_frame
    def stats_data():
        ui.update_sidebar("sidebar", show = False)
        return render.DataGrid(
            stats, 
            selection_mode = "row",
        )


