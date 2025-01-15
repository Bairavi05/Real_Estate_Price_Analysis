from flask import Flask, request, render_template, jsonify
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import os

app = Flask(__name__)

# Initialize Spark session with additional configurations
spark = SparkSession.builder \
    .appName("HousingPricePrediction") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
    .config("spark.hadoop.io.compression.codecs", "org.apache.hadoop.io.compress.DefaultCodec") \
    .config("spark.sql.warehouse.dir", "file:///C:/spark-warehouse") \
    .getOrCreate()

# Load the trained model
model_path = os.path.join(os.getcwd(), 'best_lr_model')  # Assuming the model is in the current directory
best_lr_model = LinearRegressionModel.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Convert categorical data to numerical
        for key in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']:
            data[key] = 1 if data[key] == 'yes' else 0
        data['furnishingstatus'] = {'furnished': 0, 'semi-furnished': 1, 'unfurnished': 2}[data['furnishingstatus']]

        # Create a PySpark DataFrame
        input_data = spark.createDataFrame([data])

        # Preprocess the data (assemble features)
        feature_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 
                           'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        assembled_data = assembler.transform(input_data)

        # Make prediction
        predictions = best_lr_model.transform(assembled_data)
        prediction_value = predictions.select("prediction").head()[0]

        return render_template('index.html', prediction=f'Predicted Price: {prediction_value}')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
