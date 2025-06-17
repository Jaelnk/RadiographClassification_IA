"# RadiographClassification_IA" 
# Entrenamiento de Modelo COVID-19 para Radiografías
## Resumen Ejecutivo para Exposición - developed by: Jhael Nicolalde

---

## 🎯 **Objetivo del Proyecto**
Crear un modelo de inteligencia artificial que pueda clasificar radiografías en 3 categorías:
- **COVID-19** 🦠
- **NEUMONÍA** 🫁  
- **NORMAL** ✅

---
![image](https://github.com/user-attachments/assets/70ebb30b-a1df-4ea0-bb1a-ba907aac58f3)
![image](https://github.com/user-attachments/assets/4bf25cc5-b443-4664-8c20-8cc9e392c356)
![image](https://github.com/user-attachments/assets/f6bf6c2b-0cc2-4ac3-abdc-f82057778fa1)


## 🛠️ **Herramientas Principales Utilizadas**

### **1. Framework de Deep Learning**
- **TensorFlow/Keras** - Para construir y entrenar el modelo
- **Versión optimizada** para CPU y GPU

### **2. Arquitectura del Modelo**
- **MobileNetV2** - Red neuronal preentrenada
- **Transfer Learning** - Aprovecha conocimiento previo
- **Fine-tuning** - Ajuste fino para nuestro problema específico

### **3. Herramientas de Datos**
- **Pandas** - Manejo de datos tabulares
- **NumPy** - Operaciones matemáticas
- **Generadores personalizados** - Carga eficiente de imágenes

### **4. Optimización y Monitoreo**
- **Callbacks inteligentes** - Control automático del entrenamiento
- **Monitor de recursos** - CPU y memoria en tiempo real
- **Early Stopping** - Evita sobreentrenamiento

---

## 📊 **Proceso de Entrenamiento (Simplificado)**

### **Fase 1: Preparación**
```
Datos → Aumento → Generadores → Listo para entrenar
```

### **Fase 2: Entrenamiento Inicial (25 épocas)**
- Capas base **CONGELADAS**
- Solo entrenar clasificador final
- Aprendizaje rápido y estable

### **Fase 3: Fine-tuning (15 épocas)**  
- Descongelar últimas 20 capas de MobileNetV2
- Learning rate **reducido**
- Ajuste fino y preciso

---

## ⚙️ **Configuraciones Clave**

| Parámetro | Valor | Propósito |
|-----------|--------|-----------|
| **Batch Size** | 64 | Procesamiento eficiente |
| **Learning Rate** | 0.001 → 0.0001 | Control de aprendizaje |
| **Input Size** | 150x150x3 | Imágenes RGB optimizadas |
| **Dropout** | 0.3 | Prevenir sobreentrenamiento |
| **Early Stopping** | 8 épocas paciencia | Detención inteligente |

---

## 🚀 **Optimizaciones Implementadas**

### **Hardware**
- ✅ **Soporte GPU/CPU** automático
- ✅ **Paralelización** de datos
- ✅ **Gestión eficiente** de memoria

### **Software**
- ✅ **Pesos de clase balanceados** (dataset desbalanceado)
- ✅ **Callbacks inteligentes** (guardar mejor modelo)
- ✅ **Monitoreo en tiempo real** de recursos

### **Datos**
- ✅ **Generador personalizado** para archivos .npy
- ✅ **Carga bajo demanda** (no todo en memoria)
- ✅ **Shuffle automático** por época

---

## 📈 **Métricas Monitoreadas**

Durante el entrenamiento se trackean:
- **Accuracy** (precisión general)
- **Loss** (función de pérdida)
- **Precision** (verdaderos positivos)
- **Recall** (sensibilidad)
- **AUC** (área bajo la curva ROC)

---

## 🎯 **Resultados del Entrenamiento**

### **Tiempo Total:** ~40-60 minutos
### **Épocas Completadas:** 25 + 15 = 40 épocas máximo
### **Archivo Final:** `final_covid_model.keras`

### **Outputs Generados:**
- 📊 **Gráficos** de pérdida y precisión
- 📈 **Métricas** de cada época
- 💾 **Modelo guardado** automáticamente
- 📝 **Logs detallados** del proceso

---

## 💡 **Puntos Clave para la Exposición**

### **¿Por qué MobileNetV2?**
- 🚀 **Eficiente** - Menos parámetros, más rápido
- 🎯 **Efectivo** - Preentrenado en ImageNet
- 📱 **Versátil** - Diseñado para dispositivos móviles

### **¿Por qué Transfer Learning?**
- ⏱️ **Ahorra tiempo** - No entrenar desde cero
- 📊 **Mejores resultados** - Aprovecha conocimiento previo
- 💻 **Menos recursos** - Requiere menos datos y cómputo

### **¿Por qué Fine-tuning?**
- 🎯 **Especialización** - Adapta al problema específico
- ⚖️ **Balance** - Entre generalización y especialización
- 📈 **Mejora incremental** - Optimiza resultados finales

---

## 🔧 **Aspectos Técnicos Destacables**

1. **Automatización completa** - El proceso se ejecuta sin intervención manual
2. **Manejo robusto de errores** - Continúa funcionando ante problemas
3. **Optimización de recursos** - Usa eficientemente CPU/GPU disponible
4. **Reproducibilidad** - Semillas fijas para resultados consistentes
5. **Escalabilidad** - Fácil ajustar parámetros según recursos

---

## 📝 **Resumen en 3 Puntos**

1. **HERRAMIENTAS:** TensorFlow + MobileNetV2 + Transfer Learning
2. **PROCESO:** Entrenamiento inicial + Fine-tuning automatizado  
3. **RESULTADO:** Modelo optimizado para clasificar radiografías COVID-19

---

*Este entrenamiento representa un enfoque moderno y eficiente para resolver problemas de clasificación médica usando inteligencia artificial.*
