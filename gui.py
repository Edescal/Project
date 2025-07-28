'''
THE AI MODEL IS TRAINED FROM aimodel.py AND DUMPED 
INTO THE FILES modelo_sueno.pki AND scaler.pki SO
FIRST aimodel.py MUST BE EXECUTED
'''

import tkinter as tk
from tkinter import messagebox  # Para mostrar mensajes emergentes (errores)
import joblib                  # Para cargar modelos y objetos serializados
import numpy as np             # Para manejo de arrays numéricos

class SleepPredictorApp:
    def __init__(self, root):
        self.root = root
        root.title("Sleep Quality Predictor")  # Título de la ventana

        # Diccionario para guardar las entradas de texto del formulario
        self.entries = {}

        # Lista de campos con etiquetas y nombres clave para acceso posterior
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

        # Crear etiquetas y cajas de texto para cada campo
        for i, (label_text, key) in enumerate(fields):
            # Crear y posicionar la etiqueta a la izquierda
            label = tk.Label(root, text=label_text)
            label.grid(row=i, column=0, sticky="e", padx=5, pady=5)

            # Crear y posicionar la caja de entrada a la derecha de la etiqueta
            entry = tk.Entry(root)
            entry.grid(row=i, column=1, padx=5, pady=5)

            # Guardar el widget Entry para acceder a su contenido más tarde
            self.entries[key] = entry

        # Botón para ejecutar la predicción al hacer clic
        predict_btn = tk.Button(root, text="Predict", command=self.predict)
        predict_btn.grid(row=len(fields), column=0, columnspan=2, pady=10)

        # Etiqueta para mostrar el resultado de la predicción debajo del botón
        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.grid(row=len(fields)+1, column=0, columnspan=2)

        # Cargar el modelo y el scaler solo una vez al iniciar la aplicación
        self.modelo = joblib.load('modelo_sueno.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def predict(self):
        try:
            datos = []
            # Recorrer las claves para leer valores desde los Entry widgets
            for key in ["sleepDuration","sleepQuality","caffeineIntake","screenTime",
                        "physicalActivity","stressLevel","alcohol","napDuration","mood"]:
                val = self.entries[key].get()  # Obtener texto ingresado

                # Validar que no esté vacío
                if val == "":
                    raise ValueError(f"Campo {key} está vacío.")

                # Convertir a entero o float según el campo
                if key in ["sleepDuration", "sleepQuality", "stressLevel", "alcohol", "mood"]:
                    val = int(val)
                else:
                    val = float(val)

                datos.append(val)  # Agregar a la lista de datos

            # Convertir lista a array numpy y darle forma para el scaler
            datos_np = np.array(datos).reshape(1, -1)

            # Escalar los datos de entrada con el scaler previamente entrenado
            datos_escalados = self.scaler.transform(datos_np)

            # Obtener la predicción del modelo
            pred = self.modelo.predict(datos_escalados)[0]

            # Mostrar mensaje según la predicción
            mensaje = "Good sleep 😴" if pred == 1 else "Bad sleep 😕"
            self.result_label.config(text=f"Prediction result: {mensaje}")

        except Exception as e:
            # En caso de error, mostrar ventana emergente con el mensaje
            messagebox.showerror("Error", f"Make sure all inputs are correctly assigned!")


if __name__ == "__main__":
    root = tk.Tk()                # Crear ventana principal
    app = SleepPredictorApp(root) # Inicializar nuestra aplicación con la ventana
    root.mainloop()               # Ejecutar el loop principal de la GUI
