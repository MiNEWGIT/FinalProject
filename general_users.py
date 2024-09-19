import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import defaultdict

def fetch_data():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')  # Adjust the connection string as needed
    db = client['HRMonitoring']  # Replace with your database name
    users_collection = db['Users']  # Replace with your collection name

    # Fetch all users
    users = list(users_collection.find({}))
    print(f"Fetched {len(users)} users.")  # Debug: number of users fetched
    return users

def group_users(users):
    groups = defaultdict(lambda: defaultdict(list))
    for user in users:
        age = user.get('Age')
        gender = user.get('Gender')
        heart_problems = user.get('Heart Problems')
        heart_rates = [value for key, value in user.items() if key.startswith('HR at') and isinstance(value, (int, float))]

        # Debug: Print user details and heart rates
        print(f"User: {user.get('Name')}, Age: {age}, Gender: {gender}, Heart Problems: {heart_problems}")
        print(f"Heart Rates: {heart_rates}")

        if heart_rates:  # Ensure there are heart rates to process
            for rate in heart_rates:
                if age is not None:
                    groups['age'][age].append(rate)
                if gender is not None:
                    groups['gender'][gender].append(rate)
                if heart_problems is not None:
                    groups['heart_problems'][heart_problems].append(rate)
    
    # Debug: Print grouped data
    for category, subgroups in groups.items():
        for key, rates in subgroups.items():
            print(f"Group - Category: {category}, Key: {key}, Heart Rates Count: {len(rates)}")

    return groups

def analyze_heart_rates(groups):
    analysis = {
        'age': defaultdict(list),
        'gender': defaultdict(list),
        'heart_problems': defaultdict(list)
    }
    
    for category, subgroups in groups.items():
        for key, rates in subgroups.items():
            analysis[category][key].extend(rates)
    
    # Debug: Print analysis data
    for category, data in analysis.items():
        for key, rates in data.items():
            print(f"Analysis - Category: {category}, Key: {key}, Heart Rates Count: {len(rates)}")

    return analysis


def plot_general_analysis(analysis):
    # Plot heart rates by age
    plt.figure(figsize=(12, 6))
    for age, rates in analysis['age'].items():
        plt.scatter([age] * len(rates), rates, alpha=0.7, label=f'Age {age}')
    plt.xlabel('Age')
    plt.ylabel('Heart Rate')
    plt.title('Heart Rate vs Age')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('age_analysis.png')  # Save the plot as an image file
    plt.close()

    # Plot heart rates by gender
    gender_map = {'male': 1, 'female': 2, 'other': 3}
    plt.figure(figsize=(12, 6))
    for gender, rates in analysis['gender'].items():
        if gender in gender_map:
            plt.scatter([gender_map[gender]] * len(rates), rates, alpha=0.7, label=f'Gender {gender}')
        else:
            plt.scatter([4] * len(rates), rates, alpha=0.7, label=f'Gender Unknown')  # Handle unexpected values
    plt.xlabel('Gender')
    plt.ylabel('Heart Rate')
    plt.title('Heart Rate vs Gender')
    plt.xticks(ticks=[1, 2, 3, 4], labels=['Male', 'Female', 'Other', 'Unknown'])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot heart rates by heart problems
    heart_problems_map = {True: 1, False: 0}
    plt.figure(figsize=(12, 6))
    for heart_problems, rates in analysis['heart_problems'].items():
        if heart_problems in heart_problems_map:
            plt.scatter([heart_problems_map[heart_problems]] * len(rates), rates, alpha=0.7, label=f'Heart Problems {heart_problems}')
        else:
            plt.scatter([2] * len(rates), rates, alpha=0.7, label='Heart Problems Unknown')  # Handle unexpected values
    plt.xlabel('Heart Problems')
    plt.ylabel('Heart Rate')
    plt.title('Heart Rate vs Heart Problems')
    plt.xticks(ticks=[0, 1, 2], labels=['No', 'Yes', 'Unknown'])
    plt.legend()
    plt.grid(True)
    plt.show()
