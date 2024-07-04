

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from model_training.load_data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  GridSearchCV
import joblib
from model_training.model_summary import model_summary


x_train,x_test,y_train,y_test = load_data(test_size=0.20)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=120)),  # Select top 120 features
    ('mlp', MLPClassifier(max_iter=300, random_state=11, verbose=True))
])

param_grid = {
    'mlp__hidden_layer_sizes': [ (300,)],
    'mlp__activation': ['tanh', 'relu'],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__alpha': [ 0.001 ,0.01],
    'mlp__learning_rate': ['constant'],
    'mlp__max_iter': [4000]
}

print((x_train.shape[0], x_test.shape[0]))
""" 
print(f'Features extracted: {x_train.shape[1]}') ------>  Features extracted: 180

"""


model = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

joblib_file = f"mlp_classifier_new_model_with_accuracy_{round(accuracy*100,2)}.pkl"  
joblib.dump(model, joblib_file)

print(f"Model saved to {joblib_file}")
model_summary(model, y_test,y_pred)
print("Best Parameters:", model.best_params_)

