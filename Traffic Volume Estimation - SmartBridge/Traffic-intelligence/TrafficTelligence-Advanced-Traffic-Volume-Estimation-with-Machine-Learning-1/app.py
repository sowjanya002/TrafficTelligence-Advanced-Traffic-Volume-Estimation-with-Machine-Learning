from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__,template_folder='template')

# Load model and encoders dictionary
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))  # This is a dict: {'weather': encoder, 'holiday': encoder}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    weather = request.form.get('weather')
    holiday = request.form.get('holiday')

    # Create DataFrame with input data
    data = pd.DataFrame({
        'weather': [weather],
        'holiday': [holiday]
    })

    categorical_features = ['weather', 'holiday']

    # Encode each categorical column individually
    encoded_dfs = []
    for col in categorical_features:
        col_encoder = encoder[col]
        transformed = col_encoder.transform(data[col])
        # transformed is 1D array; convert to DataFrame
        encoded_df = pd.DataFrame(transformed, columns=[f"{col}_encoded"])
        encoded_dfs.append(encoded_df)

    # Concatenate all encoded columns
    encoded_features = pd.concat(encoded_dfs, axis=1).reset_index(drop=True)

    # Drop original categorical columns and add encoded ones
    data = data.drop(columns=categorical_features).reset_index(drop=True)
    data = pd.concat([data, encoded_features], axis=1)

    # Predict traffic volume
    prediction = model.predict(data.values)[0]

    # Render output page
    return render_template('output.html', result=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)

