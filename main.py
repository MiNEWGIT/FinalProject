import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from pymongo import MongoClient
import cv2
import numpy as np
from scipy.signal import find_peaks
import asyncio
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from general_users import fetch_data, group_users, analyze_heart_rates
from toga import ScrollContainer

# Import algorithms from different files
from OCSVM import detect_anomalies_ocsvm
from IsolationForests import detect_anomalies
from arima_model2 import arima_forecast_for_user
from moving_average import moving_average_forecast, plot_moving_avg_forecast

client = MongoClient("mongodb://localhost:27017/")
db = client['HRMonitoring']  # Ensure the database name is correct
users_collection = db['Users']

class HeartRateApp(toga.App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db = MongoClient("mongodb://localhost:27017/")['HRMonitoring']
        self.users_collection = self.db['Users']
        self.heart_rate_collection = self.db['heart_rate']
        self.check_user_task = None
        self.image_view = None  # Add this line inside the __init__ method
        
    def startup(self):
        self.main_window = toga.MainWindow(title=self.formal_name, size=(800, 600))  # Adjust size as needed
    
        # Main layout
        self.main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))
    
        # Welcome message and HR information
        self.welcome_label = toga.Label(
            '       Welcome to the Heart Rate Monitoring App!\n\n',
            style=Pack(padding=(0, 6), font_size=24, font_weight='normal')
        )
    
        self.info_label = toga.Label(
            'Authors:\n'
            'Ali Mohsen && Mohammad Khamaisy\n\n'
            'Importance of Heart Rate:\n\n'
            'Heart rate is a key indicator of cardiovascular health, reflecting how well the\n'
            'heart pumps blood and delivers oxygen. Monitoring it provides insights into physical\n'
            'fitness, stress levels, and potential health issues. An abnormal heart rate can signal\n'
            'conditions like cardiovascular disease or hormonal imbalances, allowing for early\n'
            'intervention. For athletes, tracking heart rate helps optimize training and recovery,\n'
            'highlighting its importance in maintaining overall health and preventing chronic conditions.\n\n\n',
            style=Pack(padding=(0, 10))
        )
    
        # User name input
        self.user_name_label = toga.Label('Enter your name:', style=Pack(padding=(0, 5)))
        self.user_name_input = toga.TextInput(on_change=self.on_user_name_change, style=Pack(flex=1))
        self.user_name_box = toga.Box(children=[self.user_name_label, self.user_name_input], style=Pack(direction=ROW, padding=(0, 5)))
    
        # Measure Heart Rate button
        self.measure_heart_rate_button = toga.Button('Measure Heart Rate', on_press=self.measure_heart_rate, style=Pack(padding=10))
        self.measure_heart_rate_button.enabled = False
        
        self.personal_analysis_button = toga.Button('Personal Analysis', on_press=self.show_personal_analysis, style=Pack(padding=10))
        self.personal_analysis_button.enabled = False
    
        self.main_box.add(self.welcome_label)
        self.main_box.add(self.info_label)
        self.main_box.add(self.user_name_box)
        self.main_box.add(self.measure_heart_rate_button)
        self.main_box.add(self.personal_analysis_button)
    
        self.heart_rate_label = toga.Label('Heart Rate: Not measured yet', style=Pack(padding=10))
        self.main_box.add(self.heart_rate_label)
    
        # Add the Show HR Analysis button
        self.show_analysis_button = toga.Button('Show HR Analysis', on_press=self.show_hr_analysis, style=Pack(padding=10))
        self.main_box.add(self.show_analysis_button)
        
        # Scroll Container
        self.scroll_container = ScrollContainer(content=self.main_box, style=Pack(flex=1))
    

        # Set ScrollContainer as the content of the main window
        self.main_window.content = self.scroll_container
    
        
        # Show the main window
        self.main_window.show()


            
    def show_hr_analysis(self, widget):

        # Fetch and process data
        users = fetch_data()
        groups = group_users(users)
        analysis = analyze_heart_rates(groups)
        
        # Remove any previous content from the main window
        if self.image_view is not None:
            self.main_box.remove(self.image_view)
        
        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))  # Three side-by-side graphs
        
        # Plot heart rates by age on the first axis
        ax1.clear()
        for age, rates in analysis['age'].items():
            ax1.scatter([age] * len(rates), rates, alpha=0.7, label=f'Age {age}')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Heart Rate')
        ax1.set_title('Heart Rate vs Age')
        ax1.legend()
        ax1.grid(True)
        
        # Plot heart rates by gender on the second axis
        ax2.clear()
        for gender, rates in analysis['gender'].items():
            ax2.scatter([gender] * len(rates), rates, alpha=0.7, label=f'Gender {gender}')
        ax2.set_xlabel('Gender')
        ax2.set_ylabel('Heart Rate')
        ax2.set_title('Heart Rate vs Gender')
        ax2.legend()
        ax2.grid(True)
        
        # Plot heart rates by heart problems status on the third axis
        ax3.clear()
        for heart_problems_status, rates in analysis['heart_problems'].items():
            ax3.scatter([heart_problems_status] * len(rates), rates, alpha=0.7, label=f'Heart Problems: {heart_problems_status}')
        ax3.set_xlabel('Heart Problems Status')
        ax3.set_ylabel('Heart Rate')
        ax3.set_title('Heart Rate vs Heart Problems')
        ax3.legend()
        ax3.grid(True)
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot to an image file
        plot_image_path = 'hr_analysis_plot.png'
        fig.savefig(plot_image_path, format='png')
        plt.close(fig)
        
        # Load the image into an ImageView
        image = toga.Image(path=plot_image_path)
        self.image_view = toga.ImageView(image, style=Pack(padding=12))
        
        # Add the new ImageView to the main box
        self.main_box.add(self.image_view)
        
        # Refresh the main window
        self.main_window.content = self.scroll_container  # Change to set the scroll container as the main content


    async def show_personal_analysis(self, widget):
        user_name = self.user_name_input.value.strip()
    
        if not user_name:
            await self.main_window.info_dialog('Error', 'Please enter your name to proceed with personal analysis.')
            return
        
        user = self.read_user_by_name(user_name)
        
        if not user:
            await self.main_window.info_dialog('Error', 'You need to be in the database to perform this action.')
            return
        
        # Fetch personal heart rate data
        heart_rate_data = {key: value for key, value in user.items() if key.startswith('HR at')}
        
        if not heart_rate_data:
            await self.main_window.info_dialog('No Data', 'No heart rate data available for personal analysis.')
            return
        
        # Extract time and heart rate values for plotting
        times = [datetime.strptime(key, 'HR at %H:%M (%A)') for key in heart_rate_data.keys()]
        heart_rates = [float(value) for value in heart_rate_data.values()]
        
       # Calculate the moving average forecast
        moving_avg_value = moving_average_forecast(heart_rates, window_size=3)

        # Prepare forecast as a list (one value for simplicity)
        forecast = [moving_avg_value] * (len(times) - len(heart_rates))

        # Plot the forecast
        plot_moving_avg_forecast(times, heart_rates, forecast, len(heart_rates) - 1)
        
        # Detect anomalies using Isolation Forest
        detect_anomalies(user['_id'])
        
        # Detect anomalies using OC-SVM
        detect_anomalies_ocsvm(user['_id'])
        
        # Load and display the generated plots
        plot_files = [ 'moving_avg_plot.png']
        for plot_file in plot_files:
            # Remove previous content
            if self.image_view is not None:
                self.main_box.remove(self.image_view)
            
            # Load new image
            image = toga.Image(path=plot_file)
            self.image_view = toga.ImageView(image, style=Pack(padding=12))
            self.main_box.add(self.image_view)
        
        # Refresh the main window
        self.main_window.content = self.scroll_container



    async def on_user_name_change(self, widget):
        if self.check_user_task and not self.check_user_task.done():
            self.check_user_task.cancel()  # Cancel the previous task if a new change is detected

        self.check_user_task = asyncio.ensure_future(self.check_user_existence())

    async def check_user_existence(self):
        await asyncio.sleep(1.0)  # Delay for 1 second

        user_name = self.user_name_input.value.strip()

        if not user_name:
            self.measure_heart_rate_button.enabled = False
            self.personal_analysis_button.enabled = False
            return
        
        user = self.read_user_by_name(user_name)

        if user:
            await self.handle_existing_user(user)
            self.measure_heart_rate_button.enabled = True
            self.personal_analysis_button.enabled = True
        else:
            await self.open_create_user_window(user_name)

    def read_user_by_name(self, name):
        return self.users_collection.find_one({"Name": name})

    async def open_create_user_window(self, user_name):
        self.create_user_window = toga.Window(title='Create New User', size=(300, 400))
        self.create_user_box = toga.Box(style=Pack(direction=COLUMN, padding=10))
    
        # User input fields (same as before)
        self.age_label = toga.Label('Age:', style=Pack(padding=(0, 5)))
        self.age_input = toga.TextInput(style=Pack(flex=1))
    
        self.gender_label = toga.Label('Gender (male/female):', style=Pack(padding=(0, 5)))
        self.gender_input = toga.TextInput(style=Pack(flex=1))
    
        self.smoking_status_label = toga.Label('Smoking Status (yes/no):', style=Pack(padding=(0, 5)))
        self.smoking_status_input = toga.TextInput(style=Pack(flex=1))
    
        self.heart_problems_label = toga.Label('Heart Problems (yes/no):', style=Pack(padding=(0, 5)))
        self.heart_problems_input = toga.TextInput(style=Pack(flex=1))
    
        self.smartwatch_label = toga.Label('Smart Watch (yes/no):', style=Pack(padding=(0, 5)))
        self.smartwatch_input = toga.TextInput(style=Pack(flex=1))
    
        self.activity_level_label = toga.Label('Activity Level (1-5):', style=Pack(padding=(0, 5)))
        self.activity_level_input = toga.TextInput(style=Pack(flex=1))
    
        # Terms and Conditions
        self.terms_label = toga.Label(
            'Terms and Conditions:\n'
            '1. This app is not a replacement for a doctor appointment.\n'
            '2. If you have heart problems/disease, we recommend visiting a doctor and not relying solely on the app.\n'
            '3. If you have downloaded the smartwatch app, your HR will be measured automatically.\n'
            '4. I agree that the app will use, measure, and analyze my heart rate.\n'
            '5. I agree that the app will store and use my information.\n',
            style=Pack(padding=(0, 5))
        )
        
        self.terms_checkbox = toga.Switch('I agree to the terms and conditions', style=Pack(padding=(0, 5)))
    
        # Submit button
        self.submit_button = toga.Button('Submit', on_press=self.submit_new_user, style=Pack(padding=10))
    
        # Add elements to the box
        self.create_user_box.add(self.age_label)
        self.create_user_box.add(self.age_input)
        self.create_user_box.add(self.gender_label)
        self.create_user_box.add(self.gender_input)
        self.create_user_box.add(self.smoking_status_label)
        self.create_user_box.add(self.smoking_status_input)
        self.create_user_box.add(self.heart_problems_label)
        self.create_user_box.add(self.heart_problems_input)
        self.create_user_box.add(self.smartwatch_label)
        self.create_user_box.add(self.smartwatch_input)
        self.create_user_box.add(self.activity_level_label)
        self.create_user_box.add(self.activity_level_input)
        self.create_user_box.add(self.terms_label)
        self.create_user_box.add(self.terms_checkbox)
        self.create_user_box.add(self.submit_button)
    
        self.create_user_window.content = self.create_user_box
        self.create_user_window.show()


    async def submit_new_user(self, widget):
        name = self.user_name_input.value.strip()
        age = self.age_input.value.strip()
        gender = self.gender_input.value.strip().lower()
        smoking_status = self.smoking_status_input.value.strip().lower() == 'yes'
        heart_problems = self.heart_problems_input.value.strip().lower() == 'yes'
        smartwatch = self.smartwatch_input.value.strip().lower() == 'yes'
        activity_level = self.activity_level_input.value.strip()
    
        # Validate input data
        try:
            age = int(age)
            activity_level = int(activity_level)
            
            if activity_level < 1 or activity_level > 5:
                raise ValueError("Activity level must be between 1 and 5.")
            if gender not in ["male", "female"]:
                raise ValueError("Gender must be 'male' or 'female'.")
            if age <= 0:
                raise ValueError("Age must be a positive integer.")
        except ValueError as e:
            await self.create_user_window.error_dialog('Invalid Input', str(e))
            return
    
        # Create heart rate data
        heart_rate_data = {}
        start_time = datetime.strptime("00:00", "%H:%M")
        
        for _ in range(48):  # 48 half-hour intervals in a day
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
                #hour = start_time.hour
                
                # Determine heart rate range based on age and gender
                if 6 <= age <= 14:
                    initial_hr = random.randint(70, 110)
                elif 15 <= age <= 60:
                    if gender == "male":
                        initial_hr = random.randint(60, 100)
                    else:
                        initial_hr = random.randint(70, 110)
                elif age > 60:
                    initial_hr = random.randint(50, 90)
                else:
                    initial_hr = random.randint(60, 100)  # Default if age doesn't fit in any category
                
                # Adjust heart rate based on smoking or heart problems
                if smoking_status or heart_problems:
                    initial_hr = random.randint(80, 120)
                    
                # Adjust heart rate based on activity level
                if activity_level == 3 or activity_level == 4:
                    initial_hr += 5
                elif activity_level == 5:
                    initial_hr += 10
    
                hr_field = f"HR at {start_time.strftime('%H:%M')} ({day})"
                heart_rate_data[hr_field] = initial_hr
            
            start_time += timedelta(minutes=30)
    
        # Create user data dictionary
        user_data = {
            'Name': name,
            'Age': age,
            'Gender': gender,
            'Smoking': smoking_status,
            'Heart Problems': heart_problems,
            'Smart Watch': smartwatch,
            'Activity Level (1-5)': activity_level,
            **heart_rate_data
        }
    
        # Add the user to the database
        try:
            result = self.users_collection.insert_one(user_data)
            if result.inserted_id:
                await self.create_user_window.info_dialog('Success', 'New user created successfully!')
                self.create_user_window.close()
                self.measure_heart_rate_button.enabled = True
                self.personal_analysis_button.enabled = True
            else:
                await self.create_user_window.error_dialog('Error', 'Failed to create new user.')
        except Exception as e:
            await self.create_user_window.error_dialog('Error', f'An error occurred while creating the user: {e}')
    


    async def handle_existing_user(self, user):
        await self.main_window.info_dialog('Welcome Back!', f'Welcome back, {user["Name"]}!')

    async def measure_heart_rate(self, widget=None):
        """
        This function will measure the heart rate, either manually or automatically.
        """
        # Perform any ARIMA forecasting or preprocessing if needed
        arima_forecast_for_user("Ali", 10)
    
        cap = cv2.VideoCapture(0)  # Open the camera (0 is the default camera)
    
        if not cap.isOpened():
            await self.main_window.info_dialog('Error', 'Could not open camera.')
            return
    
        if widget is not None:  # If called via button press, show dialog
            await self.main_window.info_dialog('Measure Heart Rate', 'Place your finger on the camera and press OK to continue.')
    
        times = []
        signal = []
        bpm = None
        start_time = datetime.now()
    
        try:
            while (datetime.now() - start_time).total_seconds() < 10:  # Measure for 10 seconds
                ret, frame = cap.read()
                if not ret:
                    await self.main_window.info_dialog('Error', 'Failed to capture image.')
                    break
    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray)
                times.append(datetime.now())
                signal.append(mean_intensity)
    
                cv2.imshow('Frame', frame)
    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
            # Calculate BPM
            if len(times) > 1:
                times = np.array(times)
                signal = np.array(signal)
                fps = len(times) / (times[-1] - times[0]).total_seconds()
                peaks, _ = find_peaks(signal, height=np.mean(signal), distance=int(fps * 0.6))
                bpm = len(peaks) / (times[-1] - times[0]).total_seconds() * 60
                self.heart_rate_label.text = f"Heart Rate: {bpm:.2f} BPM"
    
                # Update the database with the heart rate
                user_name = self.user_name_input.value.strip()
                await self.check_for_discrepancy(user_name, bpm)
                await self.update_heart_rate_in_db(user_name, bpm)
    
        except Exception as e:
            await self.main_window.error_dialog('Error', f'An error occurred while measuring heart rate: {e}')
    
        finally:
            cap.release()
            cv2.destroyAllWindows()
       
    async def update_heart_rate_in_db(self, user_name, heart_rate):
        user = self.read_user_by_name(user_name)
        
        if user:
            # Get current day of the week and time in 30-minute intervals
            current_time = datetime.now()
            day_of_week = current_time.strftime('%A')  # e.g., "Monday"
            
            # Round the time to the nearest 30-minute interval
            hour = current_time.hour
            minute = current_time.minute
        
            if minute < 30:
                time_slot = f"HR at {hour:02}:00 ({day_of_week})"
            else:
                time_slot = f"HR at {hour:02}:30 ({day_of_week})"
        
            # Prepare the update query
            update_query = {time_slot: heart_rate}
            
            try:
                # Update the user's heart rate at the specific time slot
                result = self.users_collection.update_one(
                    {"Name": user_name},
                    {"$set": update_query}
                )
                
                if result.modified_count > 0:
                    await self.main_window.info_dialog('Success', f'Heart rate of {heart_rate:.2f} BPM updated at {time_slot} for {user_name}.')
                else:
                    await self.main_window.error_dialog('Update Failed', f'Failed to update heart rate for {user_name}. No changes made.')
                    
            except Exception as e:
                await self.main_window.error_dialog('Error', f'Failed to update heart rate: {e}')
        else:
            await self.main_window.error_dialog('User Not Found', f'User {user_name} does not exist.')

    async def check_for_discrepancy(self, user_name, measured_bpm):
        user = self.read_user_by_name(user_name)
        
        if user:
            # Get current day of the week and time in 30-minute intervals
            current_time = datetime.now()
            day_of_week = current_time.strftime('%A')  # e.g., "Monday"
            
            # Round the time to the nearest 30-minute interval
            hour = current_time.hour
            minute = current_time.minute
            
            if minute < 30:
                time_slot = f"HR at {hour:02}:00 ({day_of_week})"
            else:
                time_slot = f"HR at {hour:02}:30 ({day_of_week})"
            
            stored_heart_rate = user.get(time_slot)
            
            if stored_heart_rate is not None and abs(stored_heart_rate - measured_bpm) > 0:
                # Determine if the heart rate is high or low
                if measured_bpm < stored_heart_rate:
                    heart_rate_status = 'High'
                else:
                    heart_rate_status = 'Low'
                    
                reason = await self.ask_for_reason(heart_rate_status)
                if reason:
                    await self.update_discrepancy_in_db(user_name, time_slot, measured_bpm, reason, heart_rate_status)
            else:
                # Handle the case where there is no significant discrepancy
                pass
        else:
            await self.main_window.error_dialog('User Not Found', f'User {user_name} does not exist.')
    
    async def ask_for_reason(self, heart_rate_status):
        self.reason_event = asyncio.Event()
        
        # Define reason options based on heart rate status
        if heart_rate_status == 'High':
            reason_options = ['Exercise', 'Stress', 'Illness', 'Other']
        else:  # heart_rate_status == 'Low'
            reason_options = ['Sleeping', 'Meditating', 'Other']
        
        # Create a new window for selecting the reason
        self.reason_window = toga.Window(title='Heart Rate Abnormality', size=(300, 200))
        self.reason_box = toga.Box(style=Pack(direction=COLUMN, padding=10))
        
        # Label to prompt user for reason
        self.reason_label = toga.Label(f'Please select the reason for the abnormal heart rate ({heart_rate_status}):', style=Pack(padding=(0, 5)))
        self.reason_box.add(self.reason_label)
        
        # Store the selected reason
        self.selected_reason = None
        
        # Create buttons for each reason option
        for reason in reason_options:
            button = toga.Button(reason, on_press=self.on_reason_button_press, style=Pack(padding=5))
            self.reason_box.add(button)
        
        # Submit button
        self.submit_button = toga.Button('Submit', on_press=self.on_submit_button_press, style=Pack(padding=5))
        self.reason_box.add(self.submit_button)
        
        self.reason_window.content = self.reason_box
        self.reason_window.show()
        
        # Wait for the user to select a reason and submit
        await self.reason_event.wait()
        self.reason_window.close()
        
        return self.selected_reason
    
    def on_reason_button_press(self, widget):
        # Update selected reason when a reason button is pressed
        self.selected_reason = widget.text
    
    def on_submit_button_press(self, widget):
        # Response messages for different reasons
        response_messages = {
            'Exercise': 'Have fun!',
            'Stress': 'Try to calm down.',
            'Illness': 'I hope you feel better.',
            'Sleeping': 'Sleep well.',
            'Meditating': 'Keep it up!',
            'Other': 'Thank you for sharing.'
        }
        
        # Get the response message based on the selected reason
        response_message = response_messages.get(self.selected_reason, 'Thank you for sharing.')
        
        # Show the response message
        self.main_window.info_dialog('Heart Rate Response', response_message)
        
        # Signal that the reason selection is complete
        self.reason_event.set()
        


def main():
    return HeartRateApp('Heart Rate Monitoring App', 'org.example.heart_rate_app')



if __name__ == '__main__':
    main().main_loop()