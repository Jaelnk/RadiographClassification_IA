# ====================================================================
# FASE 5: EVALUACIÓN Y MÉTRICAS - COVID-19 RADIOGRAFÍAS (CORREGIDA)
# ====================================================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
import time

# ====================================================================
# CONFIGURACIÓN
# ====================================================================

# Directorios
BASE_DIR = Path("covid_classifier")
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
EVALUATION_DIR = RESULTS_DIR / "evaluation"

# Crear directorio de evaluación
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de clases
CLASS_NAMES = ['COVID', 'Normal', 'Viral Pneumonia']
NUM_CLASSES = len(CLASS_NAMES)

print("🔍 FASE 5: EVALUACIÓN Y MÉTRICAS DEL MODELO")
print("=" * 50)

# ====================================================================
# PASO 1: CARGAR MODELO Y DATOS
# ====================================================================

def load_trained_model():
    """Cargar el modelo entrenado"""
    print("📦 Cargando modelo entrenado...")
    
    # Intentar cargar el modelo final
    model_paths = [
        MODELS_DIR / 'final_covid_model.keras',
        MODELS_DIR / 'best_mobilenetv2_finetune.keras',
        MODELS_DIR / 'best_mobilenetv2.keras'
    ]
    
    model = None
    model_path_used = None
    
    for model_path in model_paths:
        if model_path.exists():
            try:
                model = keras.models.load_model(str(model_path))
                model_path_used = model_path
                print(f"✅ Modelo cargado desde: {model_path}")
                break
            except Exception as e:
                print(f"⚠️ Error cargando {model_path}: {e}")
                continue
    
    if model is None:
        raise FileNotFoundError("❌ No se pudo cargar ningún modelo entrenado")
    
    return model, model_path_used

def find_test_directory():
    """Buscar el directorio de datos de prueba"""
    print("🔍 Buscando directorio de datos de prueba...")
    
    # Posibles ubicaciones del directorio de prueba
    possible_test_dirs = [
        BASE_DIR / "data" / "augmented" / "test",
        BASE_DIR / "data" / "test",
        BASE_DIR / "test",
        Path("data") / "test",
        Path("test"),
        Path("COVID-19_Radiography_Dataset") / "test",
        BASE_DIR / "COVID-19_Radiography_Dataset" / "test"
    ]
    
    # Buscar directorios que contengan las subcarpetas de clases
    for test_dir in possible_test_dirs:
        if test_dir.exists():
            print(f"📂 Verificando: {test_dir}")
            # Verificar si contiene subdirectorios
            subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
            if subdirs:
                print(f"   Subdirectorios encontrados: {[d.name for d in subdirs]}")
                # Contar imágenes en cada subdirectorio
                total_images = 0
                for subdir in subdirs:
                    image_files = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg"))
                    total_images += len(image_files)
                    print(f"   {subdir.name}: {len(image_files)} imágenes")
                
                if total_images > 0:
                    print(f"✅ Directorio válido encontrado: {test_dir}")
                    return test_dir
    
    # Si no se encuentra, buscar en el directorio actual
    print("🔍 Buscando en estructura alternativa...")
    current_dir = Path(".")
    for item in current_dir.rglob("*"):
        if item.is_dir() and ("test" in item.name.lower() or "val" in item.name.lower()):
            subdirs = [d for d in item.iterdir() if d.is_dir()]
            if len(subdirs) >= 2:  # Al menos 2 clases
                total_images = 0
                for subdir in subdirs:
                    image_files = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg"))
                    total_images += len(image_files)
                
                if total_images > 0:
                    print(f"✅ Directorio alternativo encontrado: {item}")
                    return item
    
    return None

def create_test_generator():
    """Crear generador para datos de prueba"""
    print("📊 Configurando generador de datos de prueba...")
    
    # Buscar directorio de datos de prueba
    test_dir = find_test_directory()
    
    if test_dir is None:
        # Si no hay directorio de prueba, usar directorio de validación
        print("⚠️ No se encontró directorio de prueba, buscando validación...")
        possible_val_dirs = [
            BASE_DIR / "data" / "augmented" / "validation",
            BASE_DIR / "data" / "validation", 
            BASE_DIR / "validation",
            Path("data") / "validation",
            Path("validation")
        ]
        
        for val_dir in possible_val_dirs:
            if val_dir.exists():
                subdirs = [d for d in val_dir.iterdir() if d.is_dir()]
                if subdirs:
                    total_images = sum(len(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))) for d in subdirs)
                    if total_images > 0:
                        test_dir = val_dir
                        print(f"✅ Usando directorio de validación: {test_dir}")
                        break
    
    if test_dir is None:
        raise FileNotFoundError("❌ No se encontró ningún directorio con datos de prueba o validación")
    
    print(f"📂 Directorio de datos: {test_dir}")
    
    # Listar subdirectorios y mapear clases
    subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
    print(f"📋 Clases encontradas: {[d.name for d in subdirs]}")
    
    # Generador sin augmentación para evaluación
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    test_generator = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,  # Importante: no mezclar para métricas precisas
        seed=42
    )
    
    if test_generator.samples == 0:
        raise ValueError(f"❌ No se encontraron imágenes en {test_dir}")
    
    print(f"✅ Generador creado: {test_generator.samples} imágenes de prueba")
    print(f"📋 Mapeo de clases: {test_generator.class_indices}")
    
    # Actualizar nombres de clases basado en lo que se encontró
    found_classes = list(test_generator.class_indices.keys())
    global CLASS_NAMES
    if len(found_classes) == len(CLASS_NAMES):
        # Mapear clases encontradas a nombres estándar
        class_mapping = {}
        for found_class in found_classes:
            if 'covid' in found_class.lower():
                class_mapping[found_class] = 'COVID'
            elif 'normal' in found_class.lower():
                class_mapping[found_class] = 'Normal'
            elif 'pneumonia' in found_class.lower() or 'viral' in found_class.lower():
                class_mapping[found_class] = 'Viral Pneumonia'
            else:
                class_mapping[found_class] = found_class
        
        # Actualizar CLASS_NAMES en el orden correcto
        CLASS_NAMES = [class_mapping.get(found_classes[i], found_classes[i]) 
                      for i in range(len(found_classes))]
        print(f"📋 Clases mapeadas: {CLASS_NAMES}")
    
    return test_generator

# ====================================================================
# PASO 2: PREDICCIONES Y MÉTRICAS BÁSICAS
# ====================================================================

def evaluate_model(model, test_generator):
    """Evaluar modelo y obtener métricas básicas"""
    print("🎯 Evaluando modelo...")
    
    if test_generator.samples == 0:
        raise ValueError("❌ El generador no tiene imágenes para evaluar")
    
    # Realizar predicciones
    print("🔄 Realizando predicciones...")
    try:
        predictions = model.predict(test_generator, verbose=1)
    except Exception as e:
        print(f"❌ Error durante predicción: {e}")
        # Intentar con un batch más pequeño
        print("🔄 Intentando con batch size menor...")
        test_generator.batch_size = 16
        predictions = model.predict(test_generator, verbose=1)
    
    # Obtener etiquetas verdaderas
    true_labels = test_generator.classes
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Verificar que tenemos datos
    if len(true_labels) == 0:
        raise ValueError("❌ No se obtuvieron etiquetas verdaderas")
    
    # Métricas básicas
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None, zero_division=0
    )
    
    # Métricas promedio
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )
    
    print(f"✅ Evaluación completada")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"📊 Precision promedio: {precision_avg:.4f}")
    print(f"📊 Recall promedio: {recall_avg:.4f}")
    print(f"📊 F1-Score promedio: {f1_avg:.4f}")
    
    return {
        'predictions': predictions,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support,
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg
    }

# ====================================================================
# PASO 3: MATRIZ DE CONFUSIÓN
# ====================================================================

def plot_confusion_matrix(true_labels, predicted_labels, save_path):
    """Crear y guardar matriz de confusión"""
    print("📊 Creando matriz de confusión...")
    
    # Calcular matriz de confusión
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Cantidad de Imágenes'})
    
    plt.title('Matriz de Confusión - Clasificador COVID-19', fontsize=16, pad=20)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Etiqueta Real', fontsize=12)
    
    # Agregar estadísticas
    accuracy = accuracy_score(true_labels, predicted_labels)
    plt.text(0.02, 0.98, f'Accuracy: {accuracy:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matriz de confusión guardada en: {save_path}")
    
    return cm

# ====================================================================
# PASO 4: REPORTE DE CLASIFICACIÓN
# ====================================================================

def generate_classification_report(true_labels, predicted_labels, save_path):
    """Generar reporte detallado de clasificación"""
    print("📋 Generando reporte de clasificación...")
    
    # Reporte de sklearn
    report = classification_report(
        true_labels, predicted_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    # Convertir a DataFrame para mejor visualización
    report_df = pd.DataFrame(report).transpose()
    
    # Guardar como CSV
    report_df.to_csv(save_path.with_suffix('.csv'))
    
    # Crear reporte visual
    plt.figure(figsize=(12, 8))
    
    # Métricas por clase
    metrics = ['precision', 'recall', 'f1-score']
    classes = CLASS_NAMES
    
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        plt.bar(x + i*width, values, width, label=metric.title(), alpha=0.8)
    
    plt.xlabel('Clases')
    plt.ylabel('Puntuación')
    plt.title('Métricas por Clase - Clasificador COVID-19')
    plt.xticks(x + width, classes)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        for j, v in enumerate(values):
            plt.text(j + i*width, v + 0.01, f'{v:.3f}', 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Reporte guardado en: {save_path}")
    
    return report, report_df

# ====================================================================
# PASO 5: CURVAS ROC
# ====================================================================

def plot_roc_curves(true_labels, predictions, save_path):
    """Crear curvas ROC para clasificación multiclase"""
    print("📈 Creando curvas ROC...")
    
    # Verificar número de clases
    n_classes = len(np.unique(true_labels))
    
    # Binarizar las etiquetas para ROC multiclase
    true_labels_bin = label_binarize(true_labels, classes=list(range(n_classes)))
    
    # Si solo hay 2 clases, label_binarize devuelve un array 1D
    if n_classes == 2:
        true_labels_bin = np.column_stack([1 - true_labels_bin, true_labels_bin])
    
    plt.figure(figsize=(12, 8))
    
    # Colores para cada clase
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Calcular ROC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(min(n_classes, len(CLASS_NAMES))):
        if i < predictions.shape[1] and i < true_labels_bin.shape[1]:
            fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            color = colors[i % len(colors)]
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Línea diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC - Clasificador COVID-19')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Curvas ROC guardadas en: {save_path}")
    
    return roc_auc

# ====================================================================
# PASO 6: ANÁLISIS DE ERRORES
# ====================================================================

def analyze_prediction_errors(test_generator, true_labels, predicted_labels, predictions, save_path):
    """Analizar errores de predicción"""
    print("🔍 Analizando errores de predicción...")
    
    # Encontrar predicciones incorrectas
    incorrect_indices = np.where(true_labels != predicted_labels)[0]
    
    if len(incorrect_indices) == 0:
        print("🎉 ¡No hay errores de predicción!")
        # Crear archivo vacío para consistencia
        pd.DataFrame().to_csv(save_path, index=False)
        return pd.DataFrame()
    
    # Crear DataFrame con errores
    errors_data = []
    filenames = test_generator.filenames if hasattr(test_generator, 'filenames') else []
    
    for idx in incorrect_indices:
        true_class = CLASS_NAMES[true_labels[idx]] if true_labels[idx] < len(CLASS_NAMES) else f"Class_{true_labels[idx]}"
        pred_class = CLASS_NAMES[predicted_labels[idx]] if predicted_labels[idx] < len(CLASS_NAMES) else f"Class_{predicted_labels[idx]}"
        confidence = np.max(predictions[idx])
        filename = filenames[idx] if idx < len(filenames) else f"image_{idx}"
        
        errors_data.append({
            'filename': filename,
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': confidence,
            'true_label_idx': true_labels[idx],
            'pred_label_idx': predicted_labels[idx]
        })
    
    errors_df = pd.DataFrame(errors_data)
    
    # Guardar análisis de errores
    errors_df.to_csv(save_path, index=False)
    
    # Estadísticas de errores
    print(f"❌ Total de errores: {len(incorrect_indices)} de {len(true_labels)}")
    print(f"📊 Tasa de error: {len(incorrect_indices)/len(true_labels)*100:.2f}%")
    
    # Errores por clase
    if len(errors_df) > 0:
        error_by_class = errors_df.groupby(['true_class', 'predicted_class']).size().reset_index(name='count')
        print("\n📋 Errores más comunes:")
        for _, row in error_by_class.sort_values('count', ascending=False).head(5).iterrows():
            print(f"   {row['true_class']} → {row['predicted_class']}: {row['count']} casos")
    
    print(f"✅ Análisis de errores guardado en: {save_path}")
    
    return errors_df

# ====================================================================
# PASO 7: RESUMEN FINAL
# ====================================================================

def generate_final_report(evaluation_results, roc_auc, save_path):
    """Generar reporte final completo"""
    print("📄 Generando reporte final...")
    
    # Recopilar todas las métricas
    final_report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_performance': {
            'overall_accuracy': float(evaluation_results['accuracy']),
            'average_precision': float(evaluation_results['precision_avg']),
            'average_recall': float(evaluation_results['recall_avg']),
            'average_f1_score': float(evaluation_results['f1_avg'])
        },
        'per_class_metrics': {},
        'roc_auc_scores': {str(k): float(v) for k, v in roc_auc.items()},
        'total_test_samples': len(evaluation_results['true_labels']),
        'class_distribution': {},
        'class_names': CLASS_NAMES
    }
    
    # Métricas por clase
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(evaluation_results['precision']):
            final_report['per_class_metrics'][class_name] = {
                'precision': float(evaluation_results['precision'][i]),
                'recall': float(evaluation_results['recall'][i]),
                'f1_score': float(evaluation_results['f1_score'][i]),
                'support': int(evaluation_results['support'][i])
            }
            
            # Distribución de clases
            class_count = np.sum(evaluation_results['true_labels'] == i)
            final_report['class_distribution'][class_name] = int(class_count)
    
    # Guardar reporte
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    # Mostrar resumen en consola
    print("\n" + "="*60)
    print("🎯 RESUMEN FINAL DE EVALUACIÓN")
    print("="*60)
    print(f"📊 Accuracy General: {final_report['model_performance']['overall_accuracy']:.4f}")
    print(f"📊 Precision Promedio: {final_report['model_performance']['average_precision']:.4f}")
    print(f"📊 Recall Promedio: {final_report['model_performance']['average_recall']:.4f}")
    print(f"📊 F1-Score Promedio: {final_report['model_performance']['average_f1_score']:.4f}")
    
    print(f"\n📋 Métricas por Clase:")
    for class_name in CLASS_NAMES:
        if class_name in final_report['per_class_metrics']:
            metrics = final_report['per_class_metrics'][class_name]
            print(f"   {class_name}:")
            print(f"     Precision: {metrics['precision']:.4f}")
            print(f"     Recall: {metrics['recall']:.4f}")
            print(f"     F1-Score: {metrics['f1_score']:.4f}")
            print(f"     Muestras: {metrics['support']}")
    
    print(f"\n📈 AUC Scores:")
    for class_idx, auc_score in roc_auc.items():
        if class_idx < len(CLASS_NAMES):
            print(f"   {CLASS_NAMES[class_idx]}: {auc_score:.4f}")
    
    print(f"\n✅ Reporte completo guardado en: {save_path}")
    
    return final_report

# ====================================================================
# FUNCIÓN PRINCIPAL
# ====================================================================

def evaluate_covid_model():
    """Función principal para evaluar el modelo COVID-19"""
    print("🚀 INICIANDO EVALUACIÓN COMPLETA DEL MODELO")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Cargar modelo
        model, model_path = load_trained_model()
        
        # 2. Preparar datos de prueba
        test_generator = create_test_generator()
        
        # 3. Evaluar modelo
        evaluation_results = evaluate_model(model, test_generator)
        
        # 4. Matriz de confusión
        cm_path = EVALUATION_DIR / "confusion_matrix.png"
        confusion_matrix_result = plot_confusion_matrix(
            evaluation_results['true_labels'],
            evaluation_results['predicted_labels'],
            cm_path
        )
        
        # 5. Reporte de clasificación
        report_path = EVALUATION_DIR / "classification_report"
        classification_report_result, report_df = generate_classification_report(
            evaluation_results['true_labels'],
            evaluation_results['predicted_labels'],
            report_path
        )
        
        # 6. Curvas ROC
        roc_path = EVALUATION_DIR / "roc_curves.png"
        roc_auc_scores = plot_roc_curves(
            evaluation_results['true_labels'],
            evaluation_results['predictions'],
            roc_path
        )
        
        # 7. Análisis de errores
        errors_path = EVALUATION_DIR / "prediction_errors.csv"
        errors_analysis = analyze_prediction_errors(
            test_generator,
            evaluation_results['true_labels'],
            evaluation_results['predicted_labels'],
            evaluation_results['predictions'],
            errors_path
        )
        
        # 8. Reporte final
        final_report_path = EVALUATION_DIR / "final_evaluation_report.json"
        final_report = generate_final_report(
            evaluation_results,
            roc_auc_scores,
            final_report_path
        )
        
        # Tiempo total
        total_time = time.time() - start_time
        
        print(f"\n🎉 EVALUACIÓN COMPLETADA EXITOSAMENTE")
        print(f"⏱️ Tiempo total: {total_time:.1f} segundos")
        print(f"📁 Resultados guardados en: {EVALUATION_DIR}")
        print(f"🎯 Modelo evaluado: {model_path.name}")
        
        return {
            'evaluation_results': evaluation_results,
            'final_report': final_report,
            'model_path': model_path,
            'evaluation_time': total_time
        }
        
    except Exception as e:
        print(f"❌ Error durante evaluación: {e}")
        import traceback
        traceback.print_exc()
        raise

# ====================================================================
# EJECUCIÓN PRINCIPAL
# ====================================================================

if __name__ == "__main__":
    print("============================================================")
    print("🔍 EVALUACIÓN Y MÉTRICAS - CLASIFICADOR COVID-19")
    print("============================================================")
    
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Ejecutar evaluación
    try:
        evaluation_results = evaluate_covid_model()
        
        print(f"\n✅ EVALUACIÓN FINALIZADA")
        print(f"📊 Todos los archivos de evaluación están disponibles en:")
        print(f"   📁 {EVALUATION_DIR}")
        print(f"\n📋 Archivos generados:")
        print(f"   🖼️ confusion_matrix.png - Matriz de confusión")
        print(f"   📊 classification_report.png/csv - Reporte de clasificación")
        print(f"   📈 roc_curves.png - Curvas ROC")
        print(f"   ❌ prediction_errors.csv - Análisis de errores")
        print(f"   📄 final_evaluation_report.json - Reporte completo")
        
    except Exception as e:
        print(f"\n❌ ERROR EN EVALUACIÓN:")
        print(f"Error: {e}")
        print(f"\n🔧 POSIBLES SOLUCIONES:")
        print(f"1. Verificar que existe un modelo entrenado en covid_classifier/models/")
        print(f"2. Verificar estructura de datos:")
        print(f"   - Debe existir un directorio con subdirectorios por clase")
        print(f"   - Cada subdirectorio debe contener imágenes (.png, .jpg, .jpeg)")
        print(f"3. Estructuras posibles:")
        print(f"   - covid_classifier/data/test/")
        print(f"   - covid_classifier/data/validation/")
        print(f"   - test/ o validation/ en directorio actual")
        print(f"4. Verificar que las dependencias están instaladas correctamente")
        
        # Mostrar estructura actual para debug
        print(f"\n🔍 ESTRUCTURA ACTUAL:")
        base_path = Path(".")
        for item in base_path.rglob("*"):
            if item.is_dir() and any(keyword in item.name.lower() for keyword in ['test', 'val', 'data']):
                print(f"   📂 {item}")
                if item.exists():
                    subdirs = [d for d in item.iterdir() if d.is_dir()]
                    if subdirs:
                        print(f"      Subdirectorios: {[d.name for d in subdirs[:5]]}")