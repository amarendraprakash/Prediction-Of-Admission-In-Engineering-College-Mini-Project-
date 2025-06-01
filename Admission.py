import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
from tkinter import *
from tkinter import font
import os

# Load the CSV file (comma-separated)
try:
    data = pd.read_csv('college_admission_prediction.csv')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Print column names to verify
print("Column names in the CSV:", data.columns.tolist())

# Check and drop 'Serial No.' if it exists
if 'Serial No.' in data.columns:
    data = data.drop('Serial No.', axis=1)
    print("Dropped 'Serial No.' column.")
else:
    print("Column 'Serial No.' not found. Proceeding without dropping.")

# Check and drop 'Chance of Admission Getting' for features and target
if 'Chance of Admission Getting' in data.columns:
    X = data.drop('Chance of Admission Getting', axis=1)
    y = data['Chance of Admission Getting']
    print("Successfully set features and target.")
else:
    print("Column 'Chance of Admission Getting' not found in CSV. Exiting.")
    exit(1)

# Save feature names for later use in prediction
feature_names = X.columns.tolist()

# Convert target to binary (1 if > 0.8, else 0)
y = np.array([1 if value > 0.8 else 0 for value in y])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Save the model and scaler
try:
    joblib.dump(model, 'admission_model.pkl')
    joblib.dump(sc, 'scaler.pkl')
    print("Model and scaler saved successfully.")
except Exception as e:
    print(f"Error saving model/scaler: {e}")
    exit(1)

# Verify that the files exist
if not os.path.exists('admission_model.pkl') or not os.path.exists('scaler.pkl'):
    print("Model or scaler file not found after saving. Exiting.")
    exit(1)

# GUI function
def show_entry():
    print("Predict button clicked!")
    try:
        # Get input values
        p1 = float(e1.get())  # 10th Marks
        p2 = float(e2.get())  # 12th Marks
        p3 = float(e3.get())  # 12th Division
        p4 = float(e4.get())  # AIEEE Rank
        p5 = float(e5.get())  # M-1/F-0
        print(f"Input values: {p1}, {p2}, {p3}, {p4}, {p5}")

        # Load model and scaler
        try:
            model = joblib.load('admission_model.pkl')
            scaler = joblib.load('scaler.pkl')
            print("Model and scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading model/scaler: {e}")
            Label(master, text="Error loading model", font=("Arial", 12, "bold"), fg="red", bg="#f0f0f0").grid(row=8, column=0, columnspan=2, pady=10)
            master.update()  # Force GUI update
            return

        # Create a DataFrame for prediction to preserve feature names
        input_data = pd.DataFrame([[p1, p2, p3, p4, p5]], columns=feature_names)
        print(f"Input DataFrame: \n{input_data}")

        # Transform input and predict
        input_scaled = scaler.transform(input_data)
        result = model.predict(input_scaled)
        print(f"Prediction result: {result}")

        # Clear previous result if any
        for widget in master.grid_slaves(row=8, column=0):
            widget.destroy()

        # Display result
        if result[0] == 1:
            print("Displaying: High Chance of Getting Admission")  # Debug
            Label(master, text="High Chance of Getting Admission", font=("Arial", 12, "bold"), fg="green", bg="#f0f0f0").grid(row=8, column=0, columnspan=2, pady=10)
        else:
            print("Displaying: You May Get Admission")  # Debug
            Label(master, text="You May Get Admission", font=("Arial", 12, "bold"), fg="orange", bg="#f0f0f0").grid(row=8, column=0, columnspan=2, pady=10)

        # Force GUI update to ensure the label is visible
        master.update()

        # Adjust window height to ensure the label is visible
        master.update_idletasks()
        window_height = max(master.winfo_reqheight(), 350)  # Ensure enough height
        position_right = int(master.winfo_screenwidth()/2 - window_width/2)
        position_down = int(master.winfo_screenheight()/2 - window_height/2)
        master.geometry(f"{window_width}x{window_height}+{position_right}+{position_down}")

    except ValueError as ve:
        print(f"ValueError: {ve}")
        for widget in master.grid_slaves(row=8, column=0):
            widget.destroy()
        Label(master, text="Please enter valid numerical values", font=("Arial", 12, "bold"), fg="red", bg="#f0f0f0").grid(row=8, column=0, columnspan=2, pady=10)
        master.update()
    except Exception as e:
        print(f"Unexpected error in show_entry: {e}")
        for widget in master.grid_slaves(row=8, column=0):
            widget.destroy()
        Label(master, text="An error occurred", font=("Arial", 12, "bold"), fg="red", bg="#f0f0f0").grid(row=8, column=0, columnspan=2, pady=10)
        master.update()

# Set up GUI
master = Tk()
master.title("College Admission Prediction")
master.configure(bg="#f0f0f0")

# Define fonts
title_font = font.Font(family="Arial", size=16, weight="bold")
label_font = font.Font(family="Arial", size=12)

# Title label
Label(master, text="College Admission Prediction", font=title_font, bg="black", fg="white").grid(row=0, column=0, columnspan=2, pady=15, sticky="ew")

# Input labels and entries with default values for testing
Label(master, text="Enter Your 10th Marks", font=label_font, bg="#f0f0f0").grid(row=1, column=0, padx=10, pady=5, sticky="w")
Label(master, text="Enter Your 12th Marks", font=label_font, bg="#f0f0f0").grid(row=2, column=0, padx=10, pady=5, sticky="w")
Label(master, text="Enter 12th Division", font=label_font, bg="#f0f0f0").grid(row=3, column=0, padx=10, pady=5, sticky="w")
Label(master, text="Enter AIEEE Rank", font=label_font, bg="#f0f0f0").grid(row=4, column=0, padx=10, pady=5, sticky="w")
Label(master, text="Enter Gender (M=1/F=0)", font=label_font, bg="#f0f0f0").grid(row=5, column=0, padx=10, pady=5, sticky="w")

e1 = Entry(master, font=label_font, width=15)
e2 = Entry(master, font=label_font, width=15)
e3 = Entry(master, font=label_font, width=15)
e4 = Entry(master, font=label_font, width=15)
e5 = Entry(master, font=label_font, width=15)

# Set default values for testing
e1.insert(0, "95")
e2.insert(0, "92")
e3.insert(0, "2")
e4.insert(0, "100")
e5.insert(0, "1")

e1.grid(row=1, column=1, padx=10, pady=5)
e2.grid(row=2, column=1, padx=10, pady=5)
e3.grid(row=3, column=1, padx=10, pady=5)
e4.grid(row=4, column=1, padx=10, pady=5)
e5.grid(row=5, column=1, padx=10, pady=5)

# Predict button with styling
predict_button = Button(master, text="Predict", command=show_entry, font=label_font, bg="#4CAF50", fg="white", relief="raised", width=10)
predict_button.grid(row=6, column=0, columnspan=2, pady=15)
print("Button command:", predict_button.cget("command"))

# Ensure the window is centered and has a minimum size
window_width = 400
master.update_idletasks()
window_height = max(master.winfo_reqheight(), 350)  # Ensure enough height initially
position_right = int(master.winfo_screenwidth()/2 - window_width/2)
position_down = int(master.winfo_screenheight()/2 - window_height/2)
master.geometry(f"{window_width}x{window_height}+{position_right}+{position_down}")

mainloop()