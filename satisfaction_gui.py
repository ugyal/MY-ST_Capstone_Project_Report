import tkinter as tk
from satisfaction_model import *
# Read the test data from CSV
test_data = pd.read_csv('mnist_test.csv')

# Create the GUI
class SatisfactionPredictorGUI:
    def __init__(self, master):
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

        # Create the predict button

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_satisfaction)
        self.predict_button.grid(row=19, column=0, columnspan=2, pady=10)

        # Create the output field
        self.output_label = tk.Label(master, text="")
        self.output_label.grid(row=20, column=0, columnspan=2)


    # Define the function to predict the satisfaction
    def predict_satisfaction(self):
        # Get the input values from the GUI
        age = int(self.age_entry.get())
        gender = int(self.gender_entry.get())
        class_ = int(self.class_entry.get())
        inflight_wifi_service = int(self.inflight_wifi_entry.get())
        departure_arrival_time_convenient = int(self.departure_arrival_entry.get())
        ease_of_online_booking = int(self.ease_of_booking_entry.get())
        gate_location = int(self.gate_location_entry.get())
        food_and_drink = int(self.food_drink_entry.get())
        online_boarding = int(self.online_boarding_entry.get())
        seat_comfort = int(self.seat_comfort_entry.get())
        inflight_entertainment = int(self.inflight_entertainment_entry.get())
        onboard_service = int(self.onboard_service_entry.get())
        legroom_service = int(self.legroom_service_entry.get())
        baggage_handling = int(self.baggage_handling_entry.get())
        checkin_service = int(self.checkin_service_entry.get())
        inflight_service = int(self.inflight_service_entry.get())
        cleanliness = int(self.cleanliness_entry.get())
        departure_delay = int(self.departure_delay_entry.get())
        arrival_delay = int(self.arrival_delay_entry.get())

                # Create the input array for prediction
        input_arr = [[age, gender, class_, inflight_wifi_service, departure_arrival_time_convenient,
                    ease_of_online_booking, food_and_beverages, gate_location, inflight_entertainment,
                    onboard_service, leg_room_service, baggage_handling, checkin_service, inflight_service,
                    cleanliness, departure_delay_in_minutes, arrival_delay_in_minutes]]
        predicted_satisfaction = predict_satisfaction(input_arr)
        self.output_label.config(text=f"Predicted satisfaction: {predicted_satisfaction[0]}")

