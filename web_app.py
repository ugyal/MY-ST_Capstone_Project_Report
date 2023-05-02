from argparse import ArgumentParser

from flask import Flask, Blueprint, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

# from cvd_model import *
# print('predict: ', predict)
predict = 'rf'
import pandas as pd
import sklearn
# Load libraries
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#create a Flask blueprint named appweb
# decorator bind home
appweb = Blueprint('hello', __name__)

@appweb.route('/')# decorator
def home():
    return render_template("index.html")




# -------------------------------------------------------------------------------------------------  PREPARING MODEL
Airline_satisfaction_data = pd.read_csv("train.csv")

# Attribute to be predicted
predict = "Airline_satisfaction_data"

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

# specifies the evaluation metric
num_folds = 5 #sets the number of folds for cross-validation.
seed = 7 #sets the random seed for reproducibility
scoring = 'accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
# self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
model = RandomForestClassifier()
model.fit(x_train, y_train)
# -------------------------------------------------------------------------------------------------
#Flask route decorator that associates a specific URL
@appweb.route('/send', methods=['POST'])
def send(predict=predict):
    global model
    if request.method == 'POST':
        age_entry = int(request.form['age_entry'])
        gender_entry = request.form['gender_entry']
        class_entry = request.form['class_entry']
        inflight_wifi_entry = int(request.form['inflight_wifi_entry'])
        departure_arrival_entry = int(request.form['departure_arrival_entry'])
        ease_of_booking_entry = int(request.form['ease_of_booking_entry'])
        gate_location_entry = int(request.form['gate_location_entry'])
        food_drink_entry = int(request.form['food_drink_entry'])
        online_boarding_entry = int(request.form['online_boarding_entry'])
        seat_comfort_entry = int(request.form['seat_comfort_entry'])
        inflight_entertainment_entry = int(request.form['inflight_entertainment_entry'])
        onboard_service_entry = int(request.form['onboard_service_entry'])
        legroom_service_entry = int(request.form['legroom_service_entry'])
        baggage_handling_entry = int(request.form['baggage_handling_entry'])
        checkin_service_entry = int(request.form['checkin_service_entry'])
        inflight_service_entry = int(request.form['inflight_service_entry'])
        cleanliness_entry = int(request.form['cleanliness_entry'])
        departure_delay_entry = int(request.form['departure_delay_entry'])
        arrival_delay_entry = int(request.form['arrival_delay_entry'])



        if(gender_entry == "male"):
            gender_entry = 1
        else:
            gender_entry = 0


# -------------------------------------------------------------------------------------------------  USING MODEL
        # input_arr = [(age, Gender, class_, inflight_entertainment, departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, onboard_service, leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness, departure_delay, arrival_delay)] 
        # NOTE order maybe is wrong... gonna write in the way gotten'
        input_arr = [(age_entry, gender_entry, class_entry, inflight_wifi_entry, departure_arrival_entry, ease_of_booking_entry,  gate_location_entry, food_drink_entry, online_boarding_entry, seat_comfort_entry, inflight_entertainment_entry, onboard_service_entry, legroom_service_entry, baggage_handling_entry, checkin_service_entry, inflight_service_entry, cleanliness_entry, departure_delay_entry, arrival_delay_entry)]
        print('predicting')
        predicted_satisfaction_value = model.predict_proba(input_arr)
        print(predicted_satisfaction_value)
        print('prediction over')
        _str = f"Predicted satisfaction: {str(100 - predicted_satisfaction_value[0][0] * 100)}%" # NOTE don't know the reason but, by the values I enter, I guess should be 100 - whatever we find
        print(_str)
# -------------------------------------------------------------------------------------------------





# =============================================================================================================================================================================
        # # Accuracy of Model
        # model.fit(x_train, y_train) #<-- this line
        # acc = model.score(x_train, y_train)

        # predict_real = model.predict([[patient_age,patient_sex,patient_chest_pain,
        #   patient_resting_bp,patient_sereum_chol,
        #   patient_fasting_bs,patient_resting_egg,
        #   patient_max_heartrate,patient_exercise_induced_angina,
        #   patient_oldpeak,patient_slope,patient_number_vessels,
        #   patient_thalassemia]])

        # if(predict_real == [0]):
        #     predict = "The result returned with " + str(round(acc,2)*100)  + "% accuracy and you have a lower chance of getting heart disease"
        # else:
        #     predict = "The result returned with " + str(round(acc,2)*100) + "% accuracy and you have a higher chance of getting heart disease"
# ===========================================================================================================================================================================
        
        
        
        
        predict = _str




        return render_template('index.html', predict=predict)

    else:
        return render_template('index.html', predict=predict)



@appweb.route('/about')
def about():
    return render_template("about.html")



if __name__ == '__main__':

    # arg parser for the standard anaconda-project options      parse command line arguments.
    #parses the command line arguments and stores them in the args object
    #False means that if this option is not specified in the command line
    parser = ArgumentParser(prog="home",
                            description="Simple Flask Application")
    parser.add_argument('--anaconda-project-host', action='append', default=[],
                        help='Hostname to allow in requests')
    parser.add_argument('--anaconda-project-port', action='store', default=8086, type=int,
                        help='Port to listen on')
    parser.add_argument('--anaconda-project-iframe-hosts',
                        action='append',
                        help='Space-separated hosts which can embed us in an iframe per our Content-Security-Policy')
    parser.add_argument('--anaconda-project-no-browser', action='store_true',
                        default=False,
                        help='Disable opening in a browser')
    parser.add_argument('--anaconda-project-use-xheaders',
                        action='store_true',
                        default=False,
                        help='Trust X-headers from reverse proxy')
    parser.add_argument('--anaconda-project-url-prefix', action='store', default='',
                        help='Prefix in front of urls')
    parser.add_argument('--anaconda-project-address',
                        action='store',
                        #default='0.0.0.0',
                        help='IP address the application should listen on.')

    args = parser.parse_args()

    app = Flask(__name__)
    app.register_blueprint(appweb, url_prefix = args.anaconda_project_url_prefix)

    app.config['PREFERRED_URL_SCHEME'] = 'https'
    # host parameter specifies the IP address that the application should listen on
    #port parameter specifies the port number that the application should listen on.
    app.wsgi_app = ProxyFix(app.wsgi_app)#middleware is used to handle headers added by reverse proxies
    app.run(host=args.anaconda_project_address, port=args.anaconda_project_port)
