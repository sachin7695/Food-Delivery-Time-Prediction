# 1. Library imports
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from delivery_data import Delivery_Data
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# 2. Create the app object
app = FastAPI()
MODEL = tf.keras.models.load_model('model')


class UserInput(BaseModel):
    Age_of_Delivery_Partner: int
    Rating_of_Previous_Deliveries: float
    Total_Distance: int

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
async def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
async def get_name(name: str):
    return {'Welcome To DElivery Time prediction API': f'{name}'}

@app.post('/predict/')
# def predict_delivery_time(data:Delivery_Data):
async def predict(UserInput: UserInput):
    # data = data.dict()
    # AGE = data['Age_of_Delivery_Partner']
    # RATINGS = data['Rating_of_Previous_Deliveries']
    # DISTANCE = data['Total_Distance']
    # features = np.array([[AGE, RATINGS, DISTANCE]])
   # print.predict([[AGE, RATINGS, DISTANCE]]))
    prediction = MODEL.predict(
        [[UserInput.Age_of_Delivery_Partner, UserInput.Rating_of_Previous_Deliveries, UserInput.Total_Distance]])
    return {
        'Predicted Delivery Time in Minutes =': float(prediction)
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload