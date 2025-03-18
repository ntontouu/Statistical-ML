import datetime
import socket
import mysql.connector

# --- Σύνδεση στη MySQL ---
def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  # Αν χρειάζεται αλλαγή, προσαρμόστε
            password="",  # Αν υπάρχει κωδικός, προσθέστε τον εδώ
            database="swim_events_db"  # Προσαρμόστε το όνομα της βάσης δεδομένων
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Σφάλμα σύνδεσης στη MySQL: {err}")
        return None

connection = connect_to_mysql()
if not connection:
    exit()

# --- Ανάκτηση δεδομένων από MySQL ---
cursor = connection.cursor()
cursor.execute("SELECT event, gender, time FROM swimming")
dataset = cursor.fetchall()
cursor.close()
connection.close()

if not dataset:
    print("⚠️ Δεν βρέθηκαν δεδομένα στη βάση!")
    exit()

# --- Συνάρτηση για μετατροπή χρόνου σε δευτερόλεπτα ---
def time_to_seconds(time_value):
    if isinstance(time_value, datetime.timedelta):
        return time_value.total_seconds()
    try:
        t = datetime.datetime.strptime(time_value, "%H:%M:%S.%f")
    except ValueError:
        t = datetime.datetime.strptime(time_value, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000

# --- Επεξεργασία δεδομένων ---
events = list(set(row[0] for row in dataset))
genders = {"male": 1, "female": 0}  # Διορθώθηκε ώστε να ταιριάζει με πεζά γράμματα

converted_data = []
for row in dataset:
    event_encoded = events.index(row[0])
    gender_encoded = genders[row[1].lower()]  # Μετατροπή σε πεζά πριν την αναζήτηση στο λεξικό
    time_seconds = time_to_seconds(row[2])
    converted_data.append((event_encoded, time_seconds, gender_encoded))

# --- Κανονικοποίηση χρόνων ---
min_time = min(row[1] for row in converted_data)
max_time = max(row[1] for row in converted_data)

for i in range(len(converted_data)):
    event_encoded, time_seconds, gender_encoded = converted_data[i]
    time_scaled = (time_seconds - min_time) / (max_time - min_time)
    converted_data[i] = (event_encoded, time_scaled, gender_encoded)

# --- Διαχωρισμός σε train και test ---
split_index = int(0.7 * len(converted_data))
train_data = converted_data[:split_index]
test_data = converted_data[split_index:]

X_train = [row[:2] for row in train_data]
y_train = [row[2] for row in train_data]

X_test = [row[:2] for row in test_data]
y_test = [row[2] for row in test_data]

# --- Εκπαίδευση του Naïve Bayes χωρίς scikit-learn ---
def gaussian_nb(X_train, y_train, X_test):
    import math
    from collections import defaultdict

    class_stats = defaultdict(list)
    for i in range(len(X_train)):
        class_stats[y_train[i]].append(X_train[i])

    means = {}
    stds = {}
    for c, values in class_stats.items():
        means[c] = [sum(feature) / len(feature) for feature in zip(*values)]
        stds[c] = [math.sqrt(sum((x - mean) ** 2 for x in feature) / len(feature)) for mean, feature in zip(means[c], zip(*values))]

    def predict(x):
        probabilities = {}
        for c in means:
            probabilities[c] = 1
            for i in range(len(x)):
                mean = means[c][i]
                std = stds[c][i]
                exponent = math.exp(-((x[i] - mean) ** 2 / (2 * std ** 2))) if std > 0 else 1
                probabilities[c] *= (1 / (math.sqrt(2 * math.pi) * std)) * exponent if std > 0 else 1
        return max(probabilities, key=probabilities.get)

    return [predict(x) for x in X_test]

# --- Πρόβλεψη και Αξιολόγηση ---
y_pred = gaussian_nb(X_train, y_train, X_test)

accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / len(y_test)
print(f"\n🔹 Ακρίβεια μοντέλου: {accuracy * 100:.2f}%")

# --- Διαδραστική πρόβλεψη ---
while True:
    try:
        print("\nΔιαθέσιμα αγωνίσματα:", events)
        event_input = input("Εισάγετε το αγώνισμα (ή 'q' για έξοδο): ")
        if event_input == 'q':
            break
        if event_input not in events:
            print(" Μη έγκυρο αγώνισμα.")
            continue

        time_input = input("Εισάγετε τον χρόνο (hh:mm:ss.fff): ")
        if time_input.count(':') == 2 and '.' not in time_input:
            time_input = time_input + '.000'

        time_seconds = time_to_seconds(time_input)
        if not min_time <= time_seconds <= max_time:
            print("Ο χρόνος είναι εκτός ορίων των δεδομένων.")
            continue

        event_encoded = events.index(event_input)
        time_scaled = (time_seconds - min_time) / (max_time - min_time)
        X_input = [event_encoded, time_scaled]

        predicted_gender = gaussian_nb(X_train, y_train, [X_input])[0]
        gender_map = {1: "Male", 0: "Female"}
        print(f"🔹 Προβλεπόμενο φύλο: {gender_map[predicted_gender]}")

    except ValueError:
        print(" Σφάλμα εισόδου. Δοκιμάστε ξανά.")