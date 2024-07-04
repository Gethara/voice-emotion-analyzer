from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from model_training.load_data import load_data
from sklearn.preprocessing import StandardScaler
import joblib
from model_training.model_summary import model_summary


x_train,x_test,y_train,y_test = load_data(test_size=0.20)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""
print((x_train.shape[0], x_test.shape[0])) ---->  (450, 151)
print(f'Features extracted: {x_train.shape[1]}') ------>  Features extracted: 180

"""


model=MLPClassifier(alpha=0.01, batch_size=250, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=1500 , random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

joblib_file = f"mlp_classifier_model_with_accuracy_{accuracy*100}.pkl"  
joblib.dump(model, joblib_file)

print(f"Model saved to {joblib_file}")
model_summary(model, y_test,y_pred)
