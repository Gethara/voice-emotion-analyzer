import joblib


def model_summary(model):
      print("Model Summary:")
      print("===================")
      print(f"Hidden layer sizes: {model.hidden_layer_sizes}")
      print(f"Activation function: {model.activation}")
      print(f"Solver: {model.solver}")
      print(f"Alpha (regularization term): {model.alpha}")
      print(f"Number of iterations: {model.n_iter_}")
      print(f"Number of layers: {model.n_layers_}")
      print(f"Number of outputs: {model.n_outputs_}")
      print(f"Classes: {model.classes_}")
      print(f"Loss: {model.loss_}")
      print(f"Best loss: {model.best_loss_}")



model =joblib.load( "D:\\voice emotion\\mlp_classifier_new_model_with_accuracy_70.01.pkl")
best_model = model.best_estimator_
print(best_model)  
print(model.best_score_) 
print(model.cv_results_)