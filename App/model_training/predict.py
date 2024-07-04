
import joblib
def predict(new_data):
      
      model = joblib.load("D:\\voice emotion\\mlp_classifier_new_model_with_accuracy_70.01.pkl")
      new_data = new_data.reshape(1, -1)
      predictions = model.predict(new_data)
      probabilities = model.predict_proba(new_data)
      classes = model.classes_

      return predictions, probabilities, classes