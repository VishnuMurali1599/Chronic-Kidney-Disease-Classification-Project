from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
with open('CKD_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for predicting data
@app.route('/predict', methods=['POST'])
def predict():
    # Handle the prediction logic here
    if request.method == 'POST':
        # Retrieve form data
        age = float(request.form.get('age'))
        blood_pressure = float(request.form.get('blood_pressure'))
        specific_gravity = float(request.form.get('specific_gravity'))
        albumin = float(request.form.get('albumin'))
        sugar = float(request.form.get('sugar'))
        red_blood_cells = float(request.form.get('red_blood_cells'))
        pus_cell = float(request.form.get('pus_cell'))
        pus_cell_clumps = float(request.form.get('pus_cell_clumps'))
        bacteria = float(request.form.get('bacteria'))
        blood_glucose_random = float(request.form.get('blood_glucose_random'))
        blood_urea = float(request.form.get('blood_urea'))
        serum_creatinine = float(request.form.get('serum_creatinine'))
        sodium = float(request.form.get('sodium'))
        potassium = float(request.form.get('potassium'))
        haemoglobin = float(request.form.get('haemoglobin'))
        packed_cell_volume = float(request.form.get('packed_cell_volume'))
        white_blood_cell_count = float(request.form.get('white_blood_cell_count'))
        red_blood_cell_count = float(request.form.get('red_blood_cell_count'))
        hypertension = float(request.form.get('hypertension'))
        diabetes_mellitus = float(request.form.get('diabetes_mellitus'))
        coronary_artery_disease = float(request.form.get('coronary_artery_disease'))
        appetite = float(request.form.get('appetite'))
        peda_edema = float(request.form.get('peda_edema'))
        aanemia = float(request.form.get('aanemia'))
        
        # ... other form fields ...

        # Prepare the input data for prediction as a NumPy array
        input_data = np.array([[
            age,
            blood_pressure,
            specific_gravity,
            albumin,
            sugar,
            red_blood_cells,
            pus_cell,
            pus_cell_clumps,
            bacteria,
            blood_glucose_random,
            blood_urea,
            serum_creatinine,
            sodium,
            potassium,
            haemoglobin,
            packed_cell_volume,
            white_blood_cell_count,
            red_blood_cell_count,
            hypertension,
            diabetes_mellitus,
            coronary_artery_disease,
            appetite,
            peda_edema,
            aanemia
            # ... other input fields ...
        ]])

        # Perform prediction with the machine learning model
        prediction_result = model.predict(input_data)

        # Convert the prediction result to a user-friendly message
        result_message = "Positive" if prediction_result[0] == 1 else "Negative"

        return render_template("home.html", prediction_text=f"Chronic Kidney Disease Prediction: {result_message}")

if __name__ == "__main__":
    app.run(debug=True)
