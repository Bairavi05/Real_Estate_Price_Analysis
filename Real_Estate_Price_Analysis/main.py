from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Feature columns used in the model
feature_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
                   'basement', 'hotwaterheating', 'airconditioning', 'parking', 
                   'prefarea', 'furnishingstatus']

# Load the trained model
model = joblib.load("Model/best_lr_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.form.to_dict()

        # Convert binary categorical fields to numerical (1 for "yes", 0 for "no")
        binary_fields = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        for field in binary_fields:
            data[field] = 1 if data[field].lower() == 'yes' else 0

        # Map "furnishingstatus" to numerical values
        furnishing_mapping = {'furnished': 0, 'semi-furnished': 1, 'unfurnished': 2}
        data['furnishingstatus'] = furnishing_mapping[data.get('furnishingstatus').lower()]

        # Convert numerical fields to float
        numerical_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        for field in numerical_fields:
            data[field] = float(data[field])

        # Create a DataFrame for the input data
        input_df = pd.DataFrame([data])

        # Prepare input data for prediction
        input_data = input_df[feature_columns]

        # Make prediction
        prediction_value = model.predict(input_data)[0]

        # Return the prediction as a JSON response with a numeric value
        return jsonify({"prediction": float(prediction_value)})

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
