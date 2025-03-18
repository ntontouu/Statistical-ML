import mysql.connector
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import datetime

# Σύνδεση στη βάση δεδομένων
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  
        password="",  
        database="swim_events_db"  
    )
    print("Σύνδεση στη βάση δεδομένων επιτυχής!")
except mysql.connector.Error as err:
    print(f"Σφάλμα σύνδεσης: {err}")
    exit()

# Διαβάζουμε τα δεδομένα από τον πίνακα swimming
query = "SELECT event, gender, time FROM swimming"
df = pd.read_sql(query, conn)

# Κλείνουμε τη σύνδεση
conn.close()

# Αν δεν υπάρχουν δεδομένα
if df.empty:
    print("Ο πίνακας 'swimming' είναι άδειος! Εισάγετε δεδομένα και δοκιμάστε ξανά.")
    exit()

# Μετατροπή του χρόνου σε δευτερόλεπτα
def time_to_seconds(time_value):
    if isinstance(time_value, pd.Timedelta):
        time_str = str(time_value)
    else:
        time_str = time_value
        
    time_str = time_str.split()[-1]  # Αφαιρούμε την ημέρα αν υπάρχει
    try:
        t = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000
    except ValueError:
        t = datetime.datetime.strptime(time_str, "%H:%M:%S")
        return t.hour * 3600 + t.minute * 60 + t.second

# Διαγραφή των μη έγκυρων γραμμών (αν υπάρχει σφάλμα μετατροπής)
df = df.dropna()

# Εκτύπωση των στηλών του DataFrame για να δούμε αν υπάρχει η στήλη 'time_seconds'
print("Στήλες του DataFrame:", df.columns)

# Μετατροπή του event σε αριθμούς
event_encoder = LabelEncoder()
df['event_encoded'] = event_encoder.fit_transform(df['event'])

# Μετατροπή του gender σε αριθμούς
gender_encoder = LabelEncoder()
df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])  # 0: Female, 1: Male

# Μετατροπή του χρόνου σε δευτερόλεπτα
df['time_seconds'] = df['time'].apply(time_to_seconds)

# Εκτύπωση των πρώτων γραμμών για να δούμε αν η μετατροπή έγινε σωστά
print("Πρώτες γραμμές με time_seconds:")
print(df[['time', 'time_seconds']].head())

# Εφαρμογή της κανονικοποίησης στον χρόνο
scaler = StandardScaler()
df['time_scaled'] = scaler.fit_transform(df[['time_seconds']])

# Εκτύπωση των πρώτων γραμμών για να δούμε τη στήλη 'time_scaled'
print("Πρώτες γραμμές μετά την κανονικοποίηση:")
print(df[['time_seconds', 'time_scaled']].head())

# Χαρακτηριστικά (X) και Ετικέτες (y)
X = df[['event_encoded', 'time_scaled']]
y = df['gender_encoded']

# Διαχωρισμός σε train-test (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Εκπαίδευση του Naïve Bayes μοντέλου
model = GaussianNB()
model.fit(X_train, y_train)

# Πρόβλεψη
y_pred = model.predict(X_test)

# Ακρίβεια & Αναφορά
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔹 Ακρίβεια μοντέλου: {accuracy * 100:.2f}%")
print("\n🔹 Αναφορά ταξινόμησης:\n", classification_report(y_test, y_pred, target_names=gender_encoder.classes_))

# Διαδραστική πρόβλεψη
while True:
    try:
        print("\nΔιαθέσιμα αγωνίσματα:", event_encoder.classes_)
        event_input = input("Εισάγετε το αγώνισμα (ή 'q' για έξοδο): ")
        if event_input == 'q':
            break

        if event_input not in event_encoder.classes_:
            print(" Μη έγκυρο αγώνισμα.")
            continue

        # Εισαγωγή του χρόνου με σωστή μορφή
        time_input = input("Εισάγετε τον χρόνο (hh:mm:ss.fff): ")

        if time_input.count(':') == 2 and '.' not in time_input:
            time_input = time_input + '.000'

        # Μετατροπή του χρόνου σε δευτερόλεπτα
        time_seconds = time_to_seconds(time_input)
        if time_seconds is None:
            print(" Σφάλμα εισόδου. Δοκιμάστε ξανά.")
            continue

        # Μετατροπή των εισόδων
        event_encoded = event_encoder.transform([event_input])[0]

        # Σιγουρευόμαστε ότι τα δεδομένα περιλαμβάνουν τις σωστές στήλες
        X_input = [[event_encoded, time_seconds]]

        # Πρόβλεψη φύλου
        predicted_gender = model.predict(X_input)
        print(f"🔹 Προβλεπόμενο φύλο: {gender_encoder.inverse_transform(predicted_gender)[0]}")

    except ValueError:
        print(" Σφάλμα εισόδου. Δοκιμάστε ξανά.")