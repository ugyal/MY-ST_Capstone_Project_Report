import matplotlib.pyplot as pyplot
import pandas as pd
import sklearn
# Load libraries
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# # Data file import
# Airline_satisfaction_data = pd.read_csv("mnist_test.csv")
Airline_satisfaction_data = pd.read_csv("train.csv")

# Attribute to be predicted
predict = "Airline_satisfaction_data"

# Dataset/Column to be Predicted, X is all attributes and y is the features
#x = np.array(heart_data.drop([predict], 1)) # Will return a new data frame that doesnt have hd in it
#y = np.array(heart_data[predict])
l = preprocessing.LabelEncoder()
age = l.fit_transform(list(Airline_satisfaction_data["Age"]))#AGE OF THE PASSENGER
Gender = l.fit_transform(list(Airline_satisfaction_data["Gender"]))
id = l.fit_transform(list(Airline_satisfaction_data["id"]))
customer_type = l.fit_transform(list(Airline_satisfaction_data["Customer Type"]))
type_of_travel = l.fit_transform(list(Airline_satisfaction_data["Type of Travel"]))
class_ = l.fit_transform(list(Airline_satisfaction_data["Class"]))
flight_distance = l.fit_transform(list(Airline_satisfaction_data["Flight Distance"]))
inflight_wifi_service = l.fit_transform(list(Airline_satisfaction_data["Inflight wifi service"]))
departure_arrival_time_convenient = l.fit_transform(list(Airline_satisfaction_data["Departure/Arrival time convenient"]))
ease_of_online_booking = l.fit_transform(list(Airline_satisfaction_data["Ease of Online booking"]))
gate_location = l.fit_transform(list(Airline_satisfaction_data["Gate location"]))
food_and_drink = l.fit_transform(list(Airline_satisfaction_data["Food and drink"]))
online_boarding = l.fit_transform(list(Airline_satisfaction_data["Online boarding"]))
seat_comfort = l.fit_transform(list(Airline_satisfaction_data["Seat comfort"]))
inflight_entertainment = l.fit_transform(list(Airline_satisfaction_data["Inflight entertainment"]))
onboard_service = l.fit_transform(list(Airline_satisfaction_data["On-board service"]))
leg_room_service = l.fit_transform(list(Airline_satisfaction_data["Leg room service"]))
baggage_handling = l.fit_transform(list(Airline_satisfaction_data["Baggage handling"]))
checkin_service = l.fit_transform(list(Airline_satisfaction_data["Checkin service"]))
inflight_service = l.fit_transform(list(Airline_satisfaction_data["Inflight service"]))
cleanliness = l.fit_transform(list(Airline_satisfaction_data["Cleanliness"]))
departure_delay = l.fit_transform(list(Airline_satisfaction_data["Departure Delay in Minutes"]))
arrival_delay = l.fit_transform(list(Airline_satisfaction_data["Arrival Delay in Minutes"]))
satisfaction = l.fit_transform(list(Airline_satisfaction_data["satisfaction"]))

x = list(zip(age, Gender, class_, inflight_entertainment, departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, onboard_service, leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness, departure_delay, arrival_delay))
y = list(satisfaction)

# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that information and knows

# Check with  different Scikit-learn classification algorithms
models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []

for name, model in models:
	kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	msg += '\n'
	print(msg)

# Compare Algorithms' Performance
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Make predictions on validation/test dataset
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()

best_model = rf
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
model_accuracy = accuracy_score(y_test, y_pred)
print("Best Model Accuracy Score on Test Set:", model_accuracy)

#Model Evaluation Metric 1
print(classification_report(y_test, y_pred))

#Model Evaluation Metric 2
#Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Model Evaluation Metric 3
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

best_model = rf
best_model.fit(x_train, y_train)
rf_roc_auc = roc_auc_score(y_test,best_model.predict(x_test))
fpr,tpr,thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:,1])

plt.figure()
plt.plot(fpr,tpr,label = 'Random Forest(area = %0.2f)'% rf_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()

#Check actual/ground truth vs predicted diagnosis
for x in range(len(y_pred)):
	print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)
