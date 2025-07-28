'''-----------------------------------------------
THE AI MODEL IS TRAINED FROM aimodel.py AND DUMPED 
INTO THE FILES modelo_sueno.pki AND scaler.pki SO
FIRST aimodel.py MUST BE EXECUTED
--------------------------------------------------
'''

import tkinter as tk
from tkinter import messagebox  
import joblib                  
import numpy as np             

class SleepPredictorApp:
    def __init__(self, root):
        self.root = root
        root.title("Sleep Quality Predictor")

        self.entries = {}

        fields = [
            ("Sleep Duration (hrs, 1-12)", "sleepDuration"),
            ("Sleep Quality (1-5)", "sleepQuality"),
            ("Caffeine Intake (mg, 0-1000)", "caffeineIntake"),
            ("Screen Time (mins, 1-720)", "screenTime"),
            ("Physical Activity (mins, 0-300)", "physicalActivity"),
            ("Stress Level (1-10)", "stressLevel"),
            ("Alcohol (0 = no, 1 = yes)", "alcohol"),
            ("Nap Duration (mins)", "napDuration"),
            ("Mood (1-5)", "mood")
        ]

        for i, (label_text, key) in enumerate(fields):
            label = tk.Label(root, text=label_text)
            label.grid(row=i, column=0, sticky="e", padx=5, pady=5)

            entry = tk.Entry(root)
            entry.grid(row=i, column=1, padx=5, pady=5)

            self.entries[key] = entry

        predict_btn = tk.Button(root, text="Predict", command=self.predict)
        predict_btn.grid(row=len(fields), column=0, columnspan=2, pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.grid(row=len(fields)+1, column=0, columnspan=2)

        self.modelo = joblib.load('modelo_sueno.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def predict(self):
        try:
            datos = []
            for key in ["sleepDuration","sleepQuality","caffeineIntake","screenTime",
                        "physicalActivity","stressLevel","alcohol","napDuration","mood"]:
                val = self.entries[key].get()
                if val == "":
                    raise ValueError(f"Campo {key} está vacío.")
                if key in ["sleepDuration", "sleepQuality", "stressLevel", "alcohol", "mood"]:
                    val = int(val)
                else:
                    val = float(val)
                datos.append(val)

            datos_np = np.array(datos).reshape(1, -1)
            datos_escalados = self.scaler.transform(datos_np)
            pred = self.modelo.predict(datos_escalados)[0]
            mensaje = "Good sleep 😴" if pred == 1 else "Bad sleep 😕"
            self.result_label.config(text=f"Prediction result: {mensaje}")
        except Exception as e:
            messagebox.showerror("Error", f"Make sure all inputs are correctly assigned!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SleepPredictorApp(root)
    root.mainloop()
