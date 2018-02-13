def assessSingleClassModel(classifier,X_test,threshold):

    prediction = classifier.predict(X_test)
    prediction = (prediction>threshold)

    return prediction

def assessMultiClassModel(classifier,X_test):

    # Predicting the Test set results
    prediction = classifier.predict(X_test)
    encoded_prediction = np.zeros(prediction.shape)

    for x in range(prediction.shape[0]):
        index = max(prediction[x,:])
        for y in range(prediction.shape[1]):

            if prediction[x,y] == index:
                encoded_prediction[x,y] = 1

    return encoded_prediction

def singleLabelConfusionMatrix(y_test,y_pred):

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    return cm

def multiLabelConfusionMatrix(y_test,y_pred):

     # Initializing memory for confusion matrix
     cm = np.zeros((y_test.shape[1],y_test.shape[1],))

     for x in range(y_test.shape[0]):

         row_test = list(y_test[x,:])
         row_pred = list(y_pred[x, :])

         test_index = row_test.index(max(row_test))
         pred_index = row_pred.index(max(row_pred))

         cm[test_index,pred_index] += 1

     return cm

def ROC(classifier,X_test,y_test):

    thresholds = np.arange(0.1,1,0.02)
    store = []

    for threshold in thresholds:
        prediction = assessSingleClassModel(classifier,X_test,threshold)
        cm = singleLabelConfusionMatrix(y_test,prediction)
        TPR = cm[1][1]/(cm[1][1]+cm[1][0])
        FPR = cm[0][1]/(cm[0][1]+cm[0][0])
        store.append((TPR,FPR,threshold))

    store = pd.DataFrame(store)
    store.columns = ['TPR','FPR','Threshold']

    return store

def save_classifier(model):

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


classifier_file = 'Classifiers/2/model.json'
weights_file = 'Classifiers/2/model.h5'

def load_classifier(classifier_file,weights_file):

    from keras.models import model_from_json

    json_file = open(classifier_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")

    return loaded_model