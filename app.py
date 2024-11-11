import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Sidebar: Section for selecting the model
st.sidebar.header("Classification Model")
model_selection = st.sidebar.selectbox("Choose a model", ["K-Nearest Neighbors", "Naive Bayes", "Decision Tree"])

# Sidebar: Option for custom dataset upload
st.sidebar.header("Custom Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Sidebar: Scaling Option
scaling_option = st.sidebar.selectbox("Select Scaling Method", ["None", "Standard Scaling", "Min-Max Scaling"])

# Function to load and preprocess custom dataset
def load_custom_data(file):
    # Load dataset and try to separate target (last column) and features
    custom_df = pd.read_csv(file)
    # Assume last column is the target
    if custom_df.columns[-1] == 'species':  # Handle Iris-like format
        X = custom_df.iloc[:, :-1]
        y = custom_df['species']
    else:
        X = custom_df.iloc[:, :-1]
        y = custom_df.iloc[:, -1]  # Assuming last column is the target
    return X, y

# Load Iris dataset by default
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Prepare data (either from uploaded file or Iris dataset)
if uploaded_file is not None:
    X, y = load_custom_data(uploaded_file)
else:
    X = df[iris.feature_names]
    y = df['species']

# Sidebar: Section for input features (for live prediction)
st.sidebar.header("Input Data")
feature_values = []
for feature in iris.feature_names:
    feature_value = st.sidebar.slider(f"Enter {feature} value", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    feature_values.append(feature_value)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
if scaling_option == "Standard Scaling":
    scaler = StandardScaler()
elif scaling_option == "Min-Max Scaling":
    scaler = MinMaxScaler()
else:
    scaler = None

if scaler:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled = X_train
    X_test_scaled = X_test

# Train the selected model
if model_selection == "K-Nearest Neighbors":
    model = KNeighborsClassifier(n_neighbors=3)
elif model_selection == "Naive Bayes":
    model = GaussianNB()
else:
    model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict using the model
user_input_data = np.array(feature_values).reshape(1, -1)
if scaler:
    user_input_data_scaled = scaler.transform(user_input_data)
else:
    user_input_data_scaled = user_input_data
prediction = model.predict(user_input_data_scaled)

# Get model's accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Display the classification result
st.title("Iris Dataset Classification")

st.markdown("### Model Performance")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

st.markdown("### Model Prediction")
st.write(f"The predicted class for the input data is: **{iris.target_names[prediction][0]}**")

# Confusion Matrix
st.markdown("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
st.pyplot(fig)

# Create a 2x2 grid layout for the plots
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Scatter plot of selected features
with col1:
    st.markdown("### Scatter Plot")
    feature_x = st.sidebar.selectbox('Select feature for X-axis:', iris.feature_names)
    feature_y = st.sidebar.selectbox('Select feature for Y-axis:', iris.feature_names)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df['species'], palette='viridis', ax=ax)
    ax.set_title(f'{feature_x} vs {feature_y}', fontsize=12)
    st.pyplot(fig)

# Correlation Heatmap
with col2:
    st.markdown("### Correlation Heatmap")
    correlation_matrix = df.drop('species', axis=1).corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Pairplot
with col3:
    st.markdown("### Pairplot")
    sns.set(style="ticks")
    pairplot = sns.pairplot(df, hue='species', palette='viridis')
    st.pyplot(pairplot)

# Descriptive Statistics
with col4:
    st.markdown("### Descriptive Statistics")
    st.write(df.describe())

# Footer section
st.markdown("""
    <div style="text-align: center; font-size: 14px; color: #555;">
        <p>Created with ❤️ using Streamlit and Seaborn | Data: Iris Dataset</p>
    </div>
""", unsafe_allow_html=True)
