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
import tkinter as tk
# from satisfaction_model import *
# Read the test data from CSV
test_data = pd.read_csv('train.csv')

# Create the GUI
class SatisfactionPredictorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        master = self
        self.master = master
        master.title("Satisfaction Predictor")

        # Create the input fields
        tk.Label(master, text="Age:").grid(row=0, column=0)
        tk.Label(master, text="Gender:").grid(row=1, column=0)
        tk.Label(master, text="Class:").grid(row=2, column=0)
        tk.Label(master, text="Inflight wifi service:").grid(row=3, column=0)
        tk.Label(master, text="Departure/Arrival time convenient:").grid(row=4, column=0)
        tk.Label(master, text="Ease of Online booking:").grid(row=5, column=0)
        tk.Label(master, text="Gate location:").grid(row=6, column=0)
        tk.Label(master, text="Food and drink:").grid(row=7, column=0)
        tk.Label(master, text="Online boarding:").grid(row=8, column=0)
        tk.Label(master, text="Seat comfort:").grid(row=9, column=0)
        tk.Label(master, text="Inflight entertainment:").grid(row=10, column=0)
        tk.Label(master, text="On-board service:").grid(row=11, column=0)
        tk.Label(master, text="Leg room service:").grid(row=12, column=0)
        tk.Label(master, text="Baggage handling:").grid(row=13, column=0)
        tk.Label(master, text="Checkin service:").grid(row=14, column=0)
        tk.Label(master, text="Inflight service:").grid(row=15, column=0)
        tk.Label(master, text="Cleanliness:").grid(row=16, column=0)
        tk.Label(master, text="Departure Delay in Minutes:").grid(row=17, column=0)
        tk.Label(master, text="Arrival Delay in Minutes:").grid(row=18, column=0)

        # age, Gender, class_, inflight_entertainment, departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, onboard_service, leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness, departure_delay, arrival_delay
        self.age_entry = tk.Entry(master)
        self.gender_entry = tk.Entry(master)
        self.class_entry = tk.Entry(master)
        self.inflight_wifi_entry = tk.Entry(master)
        self.departure_arrival_entry = tk.Entry(master)
        self.ease_of_booking_entry = tk.Entry(master)
        self.gate_location_entry = tk.Entry(master)
        self.food_drink_entry = tk.Entry(master)
        self.online_boarding_entry = tk.Entry(master)
        self.seat_comfort_entry = tk.Entry(master)
        self.inflight_entertainment_entry = tk.Entry(master)
        self.onboard_service_entry = tk.Entry(master)
        self.legroom_service_entry = tk.Entry(master)
        self.baggage_handling_entry = tk.Entry(master)
        self.checkin_service_entry = tk.Entry(master)
        self.inflight_service_entry = tk.Entry(master)
        self.cleanliness_entry = tk.Entry(master)
        self.departure_delay_entry = tk.Entry(master)
        self.arrival_delay_entry = tk.Entry(master)
        self.age_entry.grid(row=0, column=1)
        self.gender_entry.grid(row=1, column=1)
        self.class_entry.grid(row=2, column=1)
        self.inflight_wifi_entry.grid(row=3, column=1)
        self.departure_arrival_entry.grid(row=4, column=1)
        self.ease_of_booking_entry.grid(row=5, column=1)
        self.gate_location_entry.grid(row=6, column=1)
        self.food_drink_entry.grid(row=7, column=1)
        self.online_boarding_entry.grid(row=8, column=1)
        self.seat_comfort_entry.grid(row=9, column=1)
        self.inflight_entertainment_entry.grid(row=10, column=1)
        self.onboard_service_entry.grid(row=11, column=1)
        self.legroom_service_entry.grid(row=12, column=1)
        self.baggage_handling_entry.grid(row=13, column=1)
        self.checkin_service_entry.grid(row=14, column=1)
        self.inflight_service_entry.grid(row=15, column=1)
        self.cleanliness_entry.grid(row=16, column=1)
        self.departure_delay_entry.grid(row=17, column=1)
        self.arrival_delay_entry.grid(row=18, column=1)

        self.age_entry.insert(0, '18')
        self.gender_entry.insert(0, '1')
        self.class_entry.insert(0, '1')
        self.inflight_wifi_entry.insert(0, '4')
        self.departure_arrival_entry.insert(0, '3')
        self.ease_of_booking_entry.insert(0, '3')
        self.gate_location_entry.insert(0, '2')
        self.food_drink_entry.insert(0, '4')
        self.online_boarding_entry.insert(0, '3')
        self.seat_comfort_entry.insert(0, '3')
        self.inflight_entertainment_entry.insert(0, '4')
        self.onboard_service_entry.insert(0, '4')
        self.legroom_service_entry.insert(0, '5')
        self.baggage_handling_entry.insert(0, '2')
        self.checkin_service_entry.insert(0, '1')
        self.inflight_service_entry.insert(0, '3')
        self.cleanliness_entry.insert(0, '4')
        self.departure_delay_entry.insert(0, '0')
        self.arrival_delay_entry.insert(0, '0')
        # 18, 1, 1, 4, 3, 3, 2, 4, 3, 3, 4, 4, 5, 2, 1, 3, 4, 0, 0)

        # Create the predict button

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_satisfaction)
        self.predict_button.grid(row=19, column=0, columnspan=2, pady=10)

        # Create the output field
        self.output_label = tk.Label(master, text="output prediction")
        self.output_label.grid(row=20, column=0, columnspan=2)

        # --------------------------
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
        # self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)



        # -----------------------------


    # Define the function to predict the satisfaction
    def predict_satisfaction(self):
        # # Get the input values from the GUI
        # age = l.fit_transform(list(Airline_satisfaction_data["Age"]))#AGE OF THE PASSENGER
        # Gender = l.fit_transform(list(Airline_satisfaction_data["Gender"]))
        # id = l.fit_transform(list(Airline_satisfaction_data["id"]))
        # customer_type = l.fit_transform(list(Airline_satisfaction_data["Customer Type"]))
        # type_of_travel = l.fit_transform(list(Airline_satisfaction_data["Type of Travel"]))
        # class_ = l.fit_transform(list(Airline_satisfaction_data["Class"]))
        # flight_distance = l.fit_transform(list(Airline_satisfaction_data["Flight Distance"]))
        # inflight_wifi_service = l.fit_transform(list(Airline_satisfaction_data["Inflight wifi service"]))
        # departure_arrival_time_convenient = l.fit_transform(list(Airline_satisfaction_data["Departure/Arrival time convenient"]))
        # ease_of_online_booking = l.fit_transform(list(Airline_satisfaction_data["Ease of Online booking"]))
        # gate_location = l.fit_transform(list(Airline_satisfaction_data["Gate location"]))
        # food_and_drink = l.fit_transform(list(Airline_satisfaction_data["Food and drink"]))
        # online_boarding = l.fit_transform(list(Airline_satisfaction_data["Online boarding"]))
        # seat_comfort = l.fit_transform(list(Airline_satisfaction_data["Seat comfort"]))
        # inflight_entertainment = l.fit_transform(list(Airline_satisfaction_data["Inflight entertainment"]))
        # onboard_service = l.fit_transform(list(Airline_satisfaction_data["On-board service"]))
        # leg_room_service = l.fit_transform(list(Airline_satisfaction_data["Leg room service"]))
        # baggage_handling = l.fit_transform(list(Airline_satisfaction_data["Baggage handling"]))
        # checkin_service = l.fit_transform(list(Airline_satisfaction_data["Checkin service"]))
        # inflight_service = l.fit_transform(list(Airline_satisfaction_data["Inflight service"]))
        # cleanliness = l.fit_transform(list(Airline_satisfaction_data["Cleanliness"]))
        # departure_delay = l.fit_transform(list(Airline_satisfaction_data["Departure Delay in Minutes"]))
        # arrival_delay = l.fit_transform(list(Airline_satisfaction_data["Arrival Delay in Minutes"]))
        # satisfaction = l.fit_transform(list(Airline_satisfaction_data["satisfaction"]))
# --------
        # age = int(self.age_entry.get())
        # Gender = int(self.gender_entry.get())
        # class_ = int(self.class_entry.get())
        # inflight_wifi_service = int(self.inflight_wifi_entry.get())
        # departure_arrival_time_convenient = int(self.departure_arrival_entry.get())
        # ease_of_online_booking = int(self.ease_of_booking_entry.get())
        # gate_location = int(self.gate_location_entry.get())
        # food_and_drink = int(self.food_drink_entry.get())
        # online_boarding = int(self.online_boarding_entry.get())
        # seat_comfort = int(self.seat_comfort_entry.get())
        # inflight_entertainment = int(self.inflight_entertainment_entry.get())
        # onboard_service = int(self.onboard_service_entry.get())
        # legroom_service = int(self.legroom_service_entry.get())
        # baggage_handling = int(self.baggage_handling_entry.get())
        # checkin_service = int(self.checkin_service_entry.get())
        # inflight_service = int(self.inflight_service_entry.get())
        # cleanliness = int(self.cleanliness_entry.get())
        # departure_delay = int(self.departure_delay_entry.get())
        # arrival_delay = int(self.arrival_delay_entry.get())
        age = int(self.age_entry.get())
        Gender = int(self.gender_entry.get())
        class_ = int(self.class_entry.get())
        inflight_wifi_service = int(self.inflight_wifi_entry.get())
        departure_arrival_time_convenient = int(self.departure_arrival_entry.get())
        ease_of_online_booking = int(self.ease_of_booking_entry.get())
        gate_location = int(self.gate_location_entry.get())
        food_and_drink = int(self.food_drink_entry.get())
        online_boarding = int(self.online_boarding_entry.get())
        seat_comfort = int(self.age_entry.get())
        inflight_entertainment = int(self.age_entry.get())
        onboard_service = int(self.onboard_service_entry.get())
        leg_room_service = int(self.legroom_service_entry.get())
        baggage_handling = int(self.baggage_handling_entry.get())
        checkin_service = int(self.checkin_service_entry.get())
        inflight_service = int(self.inflight_service_entry.get())
        cleanliness = int(self.cleanliness_entry.get())
        departure_delay  = int(self.departure_delay_entry.get())
        arrival_delay = int(self.arrival_delay_entry.get())

                # Create the input array for prediction
                # zip(age, Gender, class_, inflight_entertainment, departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, onboard_service, leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness, departure_delay, arrival_delay)
        input_arr = [(age, Gender, class_, inflight_entertainment, departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, onboard_service, leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness, departure_delay, arrival_delay)]
        print('predicting')
        predicted_satisfaction_value = self.model.predict_proba(input_arr)
        print(predicted_satisfaction_value)
        print('prediction over')
        _str = f"Predicted satisfaction: {str(predicted_satisfaction_value[0][0] * 100)}%"
        print(_str)
        self.output_label.config(text=_str)
        print('label updated')
        self.update_idletasks()


