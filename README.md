Handwritten Digit Recognition Application
This project is a simple graphical interface application using a Convolutional Neural Network (CNN) trained to recognize handwritten digits. Users can draw a digit, and the model attempts to predict it.
Features

User-friendly graphical interface
Real-time digit recognition
Option to train and update the model
Model improvement through user feedback

Installation

Install the required Python packages:
Copypip install -r requirements.txt


Usage
To start the application:
python main.py

Draw a digit in white on the black canvas.
Click the "Predict" button to see the model's prediction.
If the prediction is incorrect, use the "Give Feedback" button to input the correct digit.
You can retrain the model using the "Train Model" button.

Project Structure

main.py: Main application file
model.py: CNN model definition
train.py: Model training functions
gui.py: Graphical user interface



