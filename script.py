import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import time
import io
from sklearn.model_selection import train_test_split
#regression models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score
import collections
import joblib
import plotly.express as px
import plotly.graph_objects as go
# st.table(df)
df1 = 0
gr = GradientBoostingRegressor()
data = pd.read_csv("D:\Python\Major Project\health insurance.csv") 
data['sex'] = data['sex'].map({'female':0,'male':1})
data['smoker'] = data['smoker'].map({'no':0,'yes':1})
data['region'] = data['region'].map({'southwest':1,'southeast':2, 'northwest':3, 'northeast':4})

# Create a Streamlit app
# st.title("3x2 Grid Layout with Plots")
rad =st.sidebar.radio("Navigation",["Home","Visualizations", "Corelation Plot and HeatMap","Prediction Models", "Prediction", "Contribute", "About Us"])
if rad == "Home":
    # progress = st.progress(0)
    # for i in range(100):
    #     time.sleep(0.2)
    #     progress.progress(i+1) 
    # progress.empty()   

    # st.snow() 
# Define a function to generate a plot
    # def generate_plot():
    # data = np.random.randn(100)
    # fig, ax = plt.subplots()
    # ax.hist(data, bins=20)
    # return fig
#         data = """1,338	0	1	20.04	0	1	1	0
# 1,339	1	1	20.04	0	1	1	0"""
#         df = pd.read_csv(pd.compat.StringIO(data), sep='\t', header=None)

#         # Specify the rows you want to remove
#         rows_to_remove = [1338, 1339]  # Replace with the actual row indices you want to remove

#         # Remove the specified rows
#         df = df[~df.index.isin(rows_to_remove)]

    # Divide the page into a 3x2 grid
    # col1 = st.columns(1)  # Three columns in the first row
    # col4, col5, col6 = st.columns(3)  # Three columns in the second row
    # col2 = st.columns(1)
    # Add a plot to each column
    # with col1:
        st.markdown("<h2>Dataset :</h2>", True)
        if st.checkbox("Show Dataset: "):
            st.dataframe(data, height=400)
            df = pd.DataFrame(data = data)
            st.markdown("<h2>Dataset Description:</h2>", True)
            st.write(df.describe())
            # data.summary()

    
    # with col3:
        

if rad == "Visualizations":
    df = pd.DataFrame(data)
    # Select columns to plot
    # num_rows = st.slider("Select Number of Rows to Plot", min_value=1, max_value=len(df), value=len(df))

# Select columns to plot
    # st.sidebar.markdown("### Select Data Options")
    selected_columns = st.multiselect("Select columns to plot", data.columns)
    plot_type = st.selectbox("Select Plot Type", ["scatter", "bar", "line", "histogram", "box"])

    # Filter the data based on user selection
    selected_data = data[selected_columns]

    # Slider to select the number of rows to be plotted
    num_rows = st.slider("Select the number of rows to plot", 1, len(data), len(data))

    # Unique colors for columns with unique values (except for age, bmi, and charges)
    unique_colors = {}
    for column in selected_data.columns:
        if column not in ["age", "bmi", "charges"]:
            unique_values = selected_data[column].unique()
            color_palette = px.colors.qualitative.Set3 if len(unique_values) > len(px.colors.qualitative.Set3) else px.colors.qualitative.Dark24
            unique_colors[column] = {value: color_palette[i] for i, value in enumerate(unique_values)}

    # Plot the data
    fig = go.Figure()

    for column in selected_data.columns:
        if column in ["age", "bmi", "charges"]:
            # Handling specific plots for age, bmi, and charges
            if plot_type == "scatter":
                fig.add_trace(go.Scatter(x=list(range(num_rows)), y=data[column][:num_rows], mode="lines+markers", name=column))
            elif plot_type == "line":
                fig.add_trace(go.Scatter(x=list(range(num_rows)), y=data[column][:num_rows], mode="lines", name=column))
            elif plot_type == "bar":
                fig.add_trace(go.Bar(x=list(range(num_rows)), y=data[column][:num_rows], name=column))
            elif plot_type == "histogram":
                fig.add_trace(go.Histogram(x=data[column][:num_rows], name=column))
            elif plot_type == "box":
                fig.add_trace(go.Box(y=data[column][:num_rows], name=column))
            # elif plot_type == "pie":
            #     fig = px.pie(data[:num_rows], names=column, title=f"{plot_type.capitalize()} Plot")
        else:
            # Handling other columns with unique colors for unique values
            for value, color in unique_colors[column].items():
                subset = data[data[column] == value]
                if plot_type == "scatter":
                    fig.add_trace(go.Scatter(x=list(range(num_rows)), y=subset[column][:num_rows], mode="lines+markers",
                                            name=f"{column}: {value}", line=dict(color=color)))
                elif plot_type == "bar":
                    fig.add_trace(go.Bar(x=list(range(num_rows)), y=subset[column][:num_rows], name=f"{column}: {value}",
                                        marker_color=color))
                elif plot_type == "histogram":
                    fig.add_trace(go.Histogram(x=subset[column][:num_rows], name=f"{column}: {value}"))
                elif plot_type == "box":
                    fig.add_trace(go.Box(y=subset[column][:num_rows], name=f"{column}: {value}"))
                # elif plot_type == "pie":
                #     fig = px.pie(subset[:num_rows], names=column, title=f"{plot_type.capitalize()} Plot")

    fig.update_layout(title=f"Interactive {plot_type.capitalize()} Plot of Selected Columns", xaxis_title="Row Index")
    st.plotly_chart(fig)
#         st.markdown("<h2>Data Visualization:</h2>", True)
#         df = data 
#         selected_columns = st.multiselect("Select columns to plot:", df.columns, key='col')

#         # Slider for selecting the number of rows to be plotted
#         num_rows = st.slider("Number of Rows to Plot:", min_value=1, max_value=len(df), value=len(df))

#         # Select the plot type using a dropdown list
#         plot_type = st.selectbox("Select a plot type:", ['Line Plot', 'Pie Chart', 'Scatter Plot', 'Box Plot'])

#         # Plot the selected data based on the user's choices
#         # st.write(f"### Plotting selected columns for the first {num_rows} rows")

#         fig, ax = plt.subplots()

#         for col in selected_columns:
#             if plot_type == 'Line Plot':
#                 fig, ax = plt.subplots()

#                 for col in selected_columns:
#                     ax.plot(df.index[:num_rows], df[col][:num_rows], label=col)
#                 #ax.plot(df.index[:num_rows], df[col][:num_rows], label=col)
#             elif plot_type == 'Pie Chart':
#                 df = data
#                 # Select the columns to create the pie chart
#                 categorical_columns = st.multiselect("Select categorical columns for the single pie chart:", ['age', 'bmi','children', 'sex', 'smoker', 'region'])

# # Combine data from selected columns for the single pie chart
#                 pie_data = collections.Counter()
#                 for col in categorical_columns:
#                     pie_data += collections.Counter(df[col])


#                 labels = list(pie_data.keys())
#                 sizes = list(pie_data.values())

#                 fig, ax = plt.subplots()
#                 ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
#                 ax.axis('equal')


#             elif plot_type == 'Scatter Plot':

#                 selected_x = st.selectbox("Select X-axis:", df.columns)
#                 selected_y = st.multiselect("Select Y-axes:", df.columns)
#                 fig, ax = plt.subplots()
#                 for col in selected_y:
#                     ax.scatter(df[selected_x], df[col], label=col)
#                 # ax.scatter(df.index[:num_rows], df[col][:num_rows], label=col)
#             elif plot_type == 'Box Plot':
#                 sns.boxplot(x=col, data=df[:num_rows])
#                 plt.xlabel(None)
#                 plt.title(f'Box Plot for {col}')
            
#         ax.set_xlabel("Rows")
#         ax.set_ylabel("Values")
#         ax.legend()

#         st.pyplot(fig)
        # st.pyplot(generate_plot())

        # st.pyplot(generate_plot())

if rad == "Corelation Plot and HeatMap":
    # col1,col2 = st.columns(2)
    # col2 = st.columns(1)

    # Add a plot to each column
    # with col1:
        st.write("")
        df=data
        sns.set(style="ticks")
        g = sns.pairplot(df, kind="scatter", markers="o")
        # Display the plot
        st.pyplot(g)
        # fig.update_layout(
        #     width=800,  # Set the width of the plot (in pixels)
        #     height=400  # Set the height of the plot (in pixels)
        # )
    # with col2:
        corr_matrix = data.corr()
        sns.set(style="white")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)

        # Display the plot
        st.pyplot(fig)
        
if rad == "Prediction Models":
    col1, col2 = st.columns(2) 
    col3, col4, col5 = st.columns(3)
    with col1:
        
    #separating independent and dependent variable
        X = data.drop(['charges'], axis=1)
        y = data['charges']
        
        # st.write(X)
        # st.write(y)
        #train test split
        X_train, X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=45)
        #displaying X_train and y_train
        # st.markdown("<h4>X_Train: </h4>", True)
        # st.write(X_train)
        # st.markdown("<h4>y_Train: </h4>", True)
        # st.write(y_train)

        #Training Model
        st.markdown("<h4>Models used for training: </h4>", True)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        st.markdown(f"<h5>{lr}</h5>", True)
        svm = SVR()
        svm.fit(X_train, y_train)
        st.markdown(f"<h5>{svm}</h5>", True)
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        st.markdown(f"<h5>{rf}</h5>", True)
        gr = GradientBoostingRegressor()
        gr.fit(X_train, y_train)
        st.markdown(f"<h5>{gr}</h5>", True)

    with col2:
        #Prediction
        st.markdown("<h4>Prediction Comparison: </h4>", True)
        
        y_pred1 = lr.predict(X_test)
        # st.write("Predicted value of Linear Regression Model: ", y_pred1)
        y_pred2 = svm.predict(X_test)
        # st.write("Predicted value of SVR Model: ", y_pred2)
        y_pred3 = rf.predict(X_test)
        # st.write("Predicted value of RandomForestRegressor Model: ", y_pred3)
        y_pred4 = gr.predict(X_test)
        # st.write("Predicted value of GradientBoostingRegressor Model: ", y_pred4)
        # st.markdown("<h4>Comparison: </h4>", True)
        df1 = pd.DataFrame({'Actual': y_test, 'Linear Regression': y_pred1, 'SVR': y_pred2, 'Random Forest': y_pred3, 'Gradient Boosting': y_pred4})
        # st.write(df1)

        #comparing performance visually

        # st.pyplot(fig)
        plt.subplot(221) #2 rows 2 columns and 2x2
        plt.plot(df1['Actual'].iloc[0:11], label = "Actual")
        plt.plot(df1['Linear Regression'].iloc[0:11], label = "Linear Regression") 
        plt.legend()
        
        plt.subplot(222)
        plt.plot(df1['Actual'].iloc[0:11], label = "Actual")
        plt.plot(df1['SVR'].iloc[0:11], label = "SVR")
        plt.legend()

        plt.subplot(223)
        plt.plot(df1['Actual'].iloc[0:11], label = "Actual")
        plt.plot(df1['Random Forest'].iloc[0:11], label = "Random Forest")
        plt.legend()

        plt.subplot(224)
        plt.plot(df1['Actual'].iloc[0:11], label = "Actual")
        plt.plot(df1['Gradient Boosting'].iloc[0:11], label = "Gradient Boosting")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

    with col3:
        #evaluating model using r square
        score1 = metrics.r2_score(y_test,y_pred1)
        score2 = metrics.r2_score(y_test,y_pred2)
        score3 = metrics.r2_score(y_test,y_pred3)
        score4 = metrics.r2_score(y_test,y_pred4)
        st.markdown("<h4>R square values of each model: </h4>", True)
        st.write("Linear Regression: ",score1)
        st.write("SVR: ",score2)
        st.write("Random Forest: ",score3)
        st.write("Gradient Boosting: ",score4)
    with col4:
        #mean absolute error
        mae1 = metrics.mean_absolute_error(y_test, y_pred1)
        mae2 = metrics.mean_absolute_error(y_test, y_pred2)
        mae3 = metrics.mean_absolute_error(y_test, y_pred3)
        mae4 = metrics.mean_absolute_error(y_test, y_pred4)
        st.markdown("<h4>Mean Absolute Error values of each model: </h4>", True)
        st.write("Linear Regression: ",mae1)
        st.write("SVR: ",mae2)
        st.write("Random Forest: ",mae3)
        st.write("Gradient Boosting: ",mae4)

    with col5:
        mse1 = metrics.mean_squared_error(y_test, y_pred1)
        mse2 = metrics.mean_squared_error(y_test, y_pred2)
        mse3 = metrics.mean_squared_error(y_test, y_pred3)
        mse4 = metrics.mean_squared_error(y_test, y_pred4)
        st.markdown("<h4>Mean Squared Error values of each model: </h4>", True)
        st.write("Linear Regression: ",mse1)
        st.write("SVR: ",mse2)
        st.write("Random Forest: ",mse3)
        st.write("Gradient Boosting: ",mse4)
        # st.write("Linear Regression: {:.2f}".format(mse1))
        # st.write("SVR: {:.2f}".format(mse2))
        # st.write("Random Forest: {:.2f}".format(mse3))
        # st.write("Gradient Boosting: {:.2f}".format(mse4))


if rad == "Prediction":
    
    # plt.style.use('dark_background')
    data = pd.read_csv("D:\Python\Major Project\health insurance.csv") 
    data['sex'] = data['sex'].map({'female':0,'male':1})
    data['smoker'] = data['smoker'].map({'no':0,'yes':1})
    data['region'] = data['region'].map({'southwest':1,'southeast':2, 'northwest':3, 'northeast':4})
    gr = GradientBoostingRegressor()
    X = data.drop(['charges'], axis=1)
    # st.write(X)
    y = data['charges']
    gr.fit(X,y)
    
    # joblib.dump(gr,'model_train')
    # model = joblib.load('model_train')
    # model.predict()
    st.title("Health Insurance Predictor")
    age = st.number_input('Enter your age ', min_value=1)
    gender = st.radio(
    "What\'s your gender",
    ('Male', 'Female'))

    if gender == 'Male':
        gen = 1.0
    else:
        gen = 0.0
    #sns.set()
    bmi = st.number_input('BMI: ')
    children = st.number_input('Number of children ', min_value=1)
    smoker = st.radio(
    "Are you smoker?",
    ('Yes', 'No'))
    if smoker == 'Yes':
        smoke = 1.0
    else:
        smoke = 0.0
    
    region = st.radio(
    "Select your region",
    ('SouthWest', 'SouthEast', 'NorthWest', 'NorthEast'))
    if region == 'SouthWest':
        reg = 1.0
    elif region == 'SouthEast':
        reg = 2.0
    elif region == 'NorthWest':
        reg = 3.0
    else:
        reg = 4.0
    
    if st.button('PREDICT'):
        joblib.dump(gr,'model_train') #training model using joblib 
        model = joblib.load('model_train')
        result  = model.predict([[age, gen, bmi, children, smoke, reg]])
        formatted_number = "{:.2f}".format(result[0])
        st.success(f'Insurance Cost: ₹ {formatted_number}')
        # st.write('Insurance Cost: $', result[0])
    else:
        st.error('Please press Predict button')

if rad == "Contribute":
    st.markdown("<h2>Contribute Your Experience </h2>",  True)
    age = st.number_input('Enter your age: ', min_value=1)
    gender = st.radio(
    "What\'s your gender",
    ('Male', 'Female'))

    if gender == 'Male':
        gen = 'male'
    else:
        gen = 'female'
    #sns.set()
    bmi = st.number_input('BMI: ')
    children = st.number_input('Number of children ', min_value=1)
    smoker = st.radio(
    "Are you smoker?",
    ('Yes', 'No'))
    if smoker == 'Yes':
        smoke = 'yes'
    else:
        smoke = 'no'
    
    region = st.radio(
    "Select your region",
    ('SouthWest', 'SouthEast', 'NorthWest', 'NorthEast'))
    if region == 'SouthWest':
        reg = 'southwest'
    elif region == 'SouthEast':
        reg = 'southeast'
    elif region == 'NorthWest':
        reg = 'northwest'
    else:
        reg = 'northeast'
    
    charges = st.number_input('Charges($)')
    #submit button
    if st.button('Submit'):
        to_add = {'age':[age], 'sex':[gen], 'bmi':[bmi], 'children':[children], 'smoker':[smoke], 'region':[reg], 'charges ':[charges]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("D://Python//Major Project//health insurance.csv", mode='a', header = False, index = False)
        st.success("Submitted!")
        st.markdown("<h3>Thanks for sharing.</h3>",True)
    else:
        st.markdown("<h3>Can you share your experience.</h3>",True)


if rad == "About Us":

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress.progress(i+1)    

    st.snow()
    st.title("About Us")

    st.markdown(
            """
            <style>
            .container {
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin-bottom: 50px;
            }
            
            .person {
                text-align: center;
                padding: 20px;
            }
            
            .person h3 {
                margin-bottom: 10px;
                margin-top: -10px;
            }
            
            .person p {
                font-size: 18px;
                line-height: 1.5;
            }
            </style>
            """,True)


    st.markdown(
            """
            <div class="container">
                <div class="person">
                    <h3>Khushil Bhimani</h3>
                        <h5>B.E in Computer Engineering<br>
                        College Name: Vidyalankar Institute of Technology</h5>
                    <strong>Contact Information:</strong>
                    <ul>
                        <li>Email: khushilbhimani2@gmail.com.com</li>
                        <li>Phone No.:9324130035</li>
                        </ul>
                        </div>
            """,True)

    st.markdown(
            """
            <div class="container">
                <div class="person">
                    <h3>Chinmay Mhatre</h3>
                    <h5>B.E in Computer Engineering<br>
                        College Name: Vidyalankar Institute of Technology</h5>
                    <strong>Contact Information:</strong>
                    <ul>
                        <li>Email: chinmaymhatre@gmail.com</li>
                        <li>Phone No.:9324339904</li>
                    </ul>
                </div>
            """,True)

