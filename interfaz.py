# ====================================================================
# INTERFAZ DE CLASIFICACIÓN COVID-19 - RADIOGRAFÍAS
# ====================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import threading
import time

# ====================================================================
# CONTROLADOR DE PREDICCIÓN
# ====================================================================

class CovidClassifierController:
    """Controlador para manejar la lógica de clasificación"""
    
    def __init__(self):
        self.model = None
        self.class_names = ['COVID', 'Normal', 'Viral Pneumonia']
        self.model_loaded = False
        self.models_dir = Path("covid_classifier/models")
        
    def load_model(self):
        """Cargar el modelo entrenado"""
        try:
            # Posibles ubicaciones del modelo
            model_paths = [
                self.models_dir / 'final_covid_model.keras',
                self.models_dir / 'best_mobilenetv2_finetune.keras',
                self.models_dir / 'best_mobilenetv2.keras',
                Path('final_covid_model.keras'),
                Path('best_mobilenetv2.keras')
            ]
            
            for model_path in model_paths:
                if model_path.exists():
                    print(f"Cargando modelo desde: {model_path}")
                    self.model = keras.models.load_model(str(model_path))
                    self.model_loaded = True
                    return True, f"Modelo cargado exitosamente desde {model_path.name}"
            
            return False, "No se encontró ningún modelo entrenado"
            
        except Exception as e:
            return False, f"Error al cargar modelo: {str(e)}"
    
    def preprocess_image(self, image_path):
        """Preprocesar imagen para predicción"""
        try:
            # Cargar imagen
            image = cv2.imread(str(image_path))
            if image is None:
                return None, "No se pudo cargar la imagen"
            
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionar a 150x150 (tamaño esperado por el modelo)
            image = cv2.resize(image, (150, 150))
            
            # Normalizar píxeles (0-1)
            image = image.astype(np.float32) / 255.0
            
            # Agregar dimensión de batch
            image = np.expand_dims(image, axis=0)
            
            return image, "Imagen procesada correctamente"
            
        except Exception as e:
            return None, f"Error al procesar imagen: {str(e)}"
    
    def predict_image(self, image_path):
        """Realizar predicción sobre una imagen"""
        if not self.model_loaded:
            return None, "Modelo no cargado"
        
        try:
            # Preprocesar imagen
            processed_image, message = self.preprocess_image(image_path)
            if processed_image is None:
                return None, message
            
            # Realizar predicción
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Obtener todas las probabilidades
            probabilities = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
            
            result = {
                'predicted_class': self.class_names[predicted_class_idx],
                'confidence': float(confidence),
                'probabilities': probabilities
            }
            
            return result, "Predicción realizada exitosamente"
            
        except Exception as e:
            return None, f"Error en predicción: {str(e)}"

# ====================================================================
# INTERFAZ GRÁFICA
# ====================================================================

class CovidClassifierGUI:
    """Interfaz gráfica para clasificación de COVID-19"""
    
    def __init__(self):
        self.controller = CovidClassifierController()
        self.current_image_path = None
        self.setup_gui()
        
    def setup_gui(self):
        """Configurar la interfaz gráfica"""
        # Ventana principal (más compacta)
        self.root = tk.Tk()
        self.root.title("Clasificador COVID-19 - Radiografías")
        self.root.geometry("700x650")
        self.root.configure(bg='#f0f0f0')
        
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(
            main_frame, 
            text="🔬 Clasificador COVID-19", 
            font=('Arial', 18, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Frame para botones de control
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        control_frame.columnconfigure(1, weight=1)
        
        # Botón cargar modelo
        self.load_model_btn = ttk.Button(
            control_frame,
            text="📦 Cargar Modelo",
            command=self.load_model_async
        )
        self.load_model_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Estado del modelo
        self.model_status_label = ttk.Label(
            control_frame,
            text="⚠️ Modelo no cargado",
            foreground='red'
        )
        self.model_status_label.grid(row=0, column=1, sticky=tk.W)
        
        # Botón seleccionar imagen
        self.select_image_btn = ttk.Button(
            control_frame,
            text="🖼️ Seleccionar Imagen",
            command=self.select_image,
            state='disabled'
        )
        self.select_image_btn.grid(row=0, column=2, padx=(10, 0))
        
        # Frame para imagen
        image_frame = ttk.LabelFrame(main_frame, text="Imagen Seleccionada", padding="10")
        image_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Canvas para mostrar imagen (más pequeño)
        self.image_canvas = tk.Canvas(
            image_frame,
            width=250,
            height=250,
            bg='white',
            relief='sunken',
            borderwidth=2
        )
        self.image_canvas.grid(row=0, column=0, pady=5)
        
        # Botón clasificar
        self.classify_btn = ttk.Button(
            image_frame,
            text="🎯 Clasificar Imagen",
            command=self.classify_image_async,
            state='disabled'
        )
        self.classify_btn.grid(row=1, column=0, pady=5)
        
        # Frame para resultados con scroll
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Canvas con scrollbar para resultados
        results_canvas = tk.Canvas(results_frame, height=200)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_canvas.yview)
        scrollable_results = ttk.Frame(results_canvas)
        
        scrollable_results.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        )
        
        results_canvas.create_window((0, 0), window=scrollable_results, anchor="nw")
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        results_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        scrollable_results.columnconfigure(0, weight=1)
        
        # Resultado principal
        self.result_label = ttk.Label(
            scrollable_results,
            text="Seleccione una imagen para clasificar",
            font=('Arial', 11),
            anchor='center'
        )
        self.result_label.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Frame para probabilidades
        prob_frame = ttk.Frame(scrollable_results)
        prob_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        prob_frame.columnconfigure(1, weight=1)
        
        # Etiquetas de probabilidades
        self.prob_labels = {}
        self.prob_bars = {}
        
        classes = ['COVID', 'Normal', 'Viral Pneumonia']
        colors = ['#ff6b6b', '#4ecdc4', '#45aaf2']
        
        for i, (class_name, color) in enumerate(zip(classes, colors)):
            # Etiqueta de clase
            class_label = ttk.Label(prob_frame, text=f"{class_name}:", font=('Arial', 9))
            class_label.grid(row=i, column=0, sticky=tk.W, padx=(0, 8), pady=1)
            
            # Barra de progreso (más pequeña)
            progress_bar = ttk.Progressbar(
                prob_frame,
                length=150,
                mode='determinate'
            )
            progress_bar.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=1)
            self.prob_bars[class_name] = progress_bar
            
            # Etiqueta de porcentaje
            percent_label = ttk.Label(prob_frame, text="0.0%", font=('Arial', 9))
            percent_label.grid(row=i, column=2, sticky=tk.W, padx=(8, 0), pady=1)
            self.prob_labels[class_name] = percent_label
        
        # Hacer scroll con rueda del mouse
        def _on_mousewheel(event):
            results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        results_canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows
        results_canvas.bind("<Button-4>", lambda e: results_canvas.yview_scroll(-1, "units"))  # Linux
        results_canvas.bind("<Button-5>", lambda e: results_canvas.yview_scroll(1, "units"))  # Linux
        
        # Barra de progreso para carga
        self.progress_bar = ttk.Progressbar(
            main_frame,
            mode='indeterminate'
        )
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Configurar redimensionamiento
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(3, weight=1)
    
    def load_model_async(self):
        """Cargar modelo de forma asíncrona"""
        def load_model_thread():
            self.progress_bar.start()
            self.load_model_btn.configure(state='disabled')
            
            success, message = self.controller.load_model()
            
            self.progress_bar.stop()
            
            if success:
                self.model_status_label.configure(
                    text="✅ Modelo cargado correctamente",
                    foreground='green'
                )
                self.select_image_btn.configure(state='normal')
                messagebox.showinfo("Éxito", message)
            else:
                self.model_status_label.configure(
                    text="❌ Error al cargar modelo",
                    foreground='red'
                )
                messagebox.showerror("Error", message)
            
            self.load_model_btn.configure(state='normal')
        
        threading.Thread(target=load_model_thread, daemon=True).start()
    
    def select_image(self):
        """Seleccionar imagen para clasificar"""
        file_types = [
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("Todos los archivos", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de radiografía",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.classify_btn.configure(state='normal')
            self.reset_results()
    
    def display_image(self, image_path):
        """Mostrar imagen en el canvas"""
        try:
            # Cargar y redimensionar imagen (más pequeña para el canvas)
            image = Image.open(image_path)
            image.thumbnail((230, 230), Image.Resampling.LANCZOS)
            
            # Convertir para tkinter
            self.photo = ImageTk.PhotoImage(image)
            
            # Limpiar canvas y mostrar imagen
            self.image_canvas.delete("all")
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Centrar imagen
            x = (canvas_width - image.width) // 2
            y = (canvas_height - image.height) // 2
            
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
    
    def classify_image_async(self):
        """Clasificar imagen de forma asíncrona"""
        def classify_thread():
            self.progress_bar.start()
            self.classify_btn.configure(state='disabled')
            
            result, message = self.controller.predict_image(self.current_image_path)
            
            self.progress_bar.stop()
            
            if result:
                self.display_results(result)
            else:
                messagebox.showerror("Error", message)
            
            self.classify_btn.configure(state='normal')
        
        threading.Thread(target=classify_thread, daemon=True).start()
    
    def display_results(self, result):
        """Mostrar resultados de clasificación"""
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        # Resultado principal
        confidence_percent = confidence * 100
        
        if confidence >= 0.7:
            confidence_text = "Alta"
            confidence_color = "green"
        elif confidence >= 0.5:
            confidence_text = "Media"
            confidence_color = "orange"
        else:
            confidence_text = "Baja"
            confidence_color = "red"
        
        result_text = f"🎯 Predicción: {predicted_class}\n📊 Confianza: {confidence_percent:.1f}% ({confidence_text})"
        self.result_label.configure(text=result_text)
        
        # Actualizar barras de probabilidad
        for class_name, probability in probabilities.items():
            prob_percent = probability * 100
            self.prob_bars[class_name]['value'] = prob_percent
            self.prob_labels[class_name].configure(text=f"{prob_percent:.1f}%")
    
    def reset_results(self):
        """Resetear resultados"""
        self.result_label.configure(text="Presione 'Clasificar Imagen' para obtener resultados")
        for class_name in self.prob_bars:
            self.prob_bars[class_name]['value'] = 0
            self.prob_labels[class_name].configure(text="0.0%")
    
    def run(self):
        """Ejecutar la aplicación"""
        self.root.mainloop()

# ====================================================================
# FUNCIÓN PRINCIPAL
# ====================================================================

def main():
    """Función principal"""
    print("🚀 Iniciando Clasificador COVID-19")
    
    try:
        app = CovidClassifierGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error al iniciar aplicación: {e}")
        messagebox.showerror("Error Fatal", f"No se pudo iniciar la aplicación:\n{str(e)}")

if __name__ == "__main__":
    main()