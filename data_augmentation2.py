# ====================================================================
# FASE 2: DATA AUGMENTATION OPTIMIZADO - COVID-19 RADIOGRAFÍAS
# ====================================================================

import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import time
import random
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Importar Albumentations para augmentation eficiente
try:
    from albumentations import *
    print("✅ Albumentations importado correctamente")
except ImportError:
    print("❌ Instala albumentations: pip install albumentations")
    exit()

# ====================================================================
# PASO 1: CONFIGURACIÓN Y CARGA DE DATOS DE FASE 1
# ====================================================================

# Configuración
BASE_DIR = Path("covid_classifier")
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = BASE_DIR / "data" / "splits"
AUGMENTED_DIR = BASE_DIR / "data" / "augmented"

# Configuración de augmentation OPTIMIZADA
TARGET_SIZE = (150, 150)  # Tamaño final de imagen
AUGMENTATION_FACTOR = 2   # REDUCIDO de 3 a 2 versiones aumentadas por imagen original

def load_phase1_results():
    """Cargar los resultados de la Fase 1"""
    print("📂 Cargando resultados de la Fase 1...")
    
    # Verificar que existen los archivos de la fase 1
    required_files = [
        SPLITS_DIR / "train_split.csv",
        SPLITS_DIR / "val_split.csv", 
        SPLITS_DIR / "test_split.csv"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Archivo no encontrado: {file_path}")
            print("   Ejecuta primero la Fase 1: Preparación de Datos")
    
    # Cargar DataFrames
    train_df = pd.read_csv(SPLITS_DIR / "train_split.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val_split.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test_split.csv")
    
    print(f"✅ Datos cargados:")
    print(f"   Entrenamiento: {len(train_df)} imágenes")
    print(f"   Validación: {len(val_df)} imágenes")
    print(f"   Prueba: {len(test_df)} imágenes")
    
    return train_df, val_df, test_df

# ====================================================================
# PASO 2: DEFINIR TRANSFORMACIONES DE AUGMENTATION OPTIMIZADAS
# ====================================================================

def create_augmentation_pipeline():
    """Crear pipeline de augmentation SIMPLIFICADO para radiografías"""
    print("🔧 Creando pipeline de augmentation optimizado...")
    
    # Pipeline de augmentation SIMPLIFICADO para reducir tiempo de entrenamiento
    # Solo las transformaciones más efectivas y rápidas
    augmentation_pipeline = Compose([
        # Solo las transformaciones más importantes y rápidas
        HorizontalFlip(p=0.5),  # Flip horizontal - MUY RÁPIDO
        
        # Rotación reducida y con menor probabilidad
        Rotate(limit=10, p=0.2),  # REDUCIDO: de ±15° a ±10° y de 30% a 20%
        
        # Brightness/Contrast - transformación rápida y efectiva
        RandomBrightnessContrast(
            brightness_limit=0.15,  # REDUCIDO: de 0.2 a 0.15
            contrast_limit=0.15,    # REDUCIDO: de 0.2 a 0.15
            p=0.3                   # REDUCIDO: de 40% a 30%
        ),
        
        # ELIMINADO: GaussNoise - computacionalmente costoso
        # ELIMINADO: ElasticTransform - muy lento
        # ELIMINADO: Blur - no esencial para radiografías
        # ELIMINADO: RandomGamma - menos importante
        
    ], p=0.7)  # REDUCIDO: probabilidad de aplicar transformación de 80% a 70%
    
    print("✅ Pipeline de augmentation optimizado creado")
    print("   Transformaciones incluidas (OPTIMIZADAS):")
    print("   - HorizontalFlip (50%) - RÁPIDO")
    print("   - Rotate ±10° (20%) - REDUCIDO")
    print("   - Brightness/Contrast (30%) - REDUCIDO")
    print("   ❌ ELIMINADAS transformaciones lentas:")
    print("      • GaussNoise, ElasticTransform, Blur, RandomGamma")
    
    return augmentation_pipeline

# ====================================================================
# PASO 3: FUNCIONES DE PROCESAMIENTO DE IMÁGENES
# ====================================================================

def load_and_preprocess_image(image_path, target_size=TARGET_SIZE):
    """Cargar y preprocesar una imagen"""
    try:
        # Leer imagen
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar a tamaño objetivo
        img = cv2.resize(img, target_size)
        
        # Normalizar a rango [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
        
    except Exception as e:
        print(f"❌ Error procesando {image_path}: {e}")
        return None

def apply_augmentation(img, augmentation_pipeline, num_augmentations=AUGMENTATION_FACTOR):
    """Aplicar augmentation a una imagen"""
    augmented_images = []
    
    # Imagen original
    augmented_images.append(img.copy())
    
    # Crear versiones aumentadas (REDUCIDAS)
    for i in range(num_augmentations):
        try:
            # Convertir de float [0,1] a uint8 [0,255] para albumentations
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Aplicar transformación
            augmented = augmentation_pipeline(image=img_uint8)['image']
            
            # Convertir de vuelta a float [0,1]
            augmented = augmented.astype(np.float32) / 255.0
            
            augmented_images.append(augmented)
            
        except Exception as e:
            print(f"⚠️  Error en augmentation: {e}")
            # Si falla, usar la imagen original
            augmented_images.append(img.copy())
    
    return augmented_images

# ====================================================================
# PASO 4: PROCESAMIENTO DE UN SOLO SPLIT
# ====================================================================

def process_single_image(args):
    """Procesar una sola imagen (para multiprocessing)"""
    idx, row, augmentation_pipeline, output_dir, split_name = args
    
    try:
        # Cargar imagen
        img = load_and_preprocess_image(row['image_path'])
        if img is None:
            return None
        
        # Aplicar augmentation solo al conjunto de entrenamiento
        if split_name == 'train':
            augmented_images = apply_augmentation(img, augmentation_pipeline)
        else:
            # Para val y test, solo procesar sin augmentation
            augmented_images = [img]
        
        # Guardar imágenes procesadas
        saved_paths = []
        class_dir = output_dir / row['class']
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for aug_idx, aug_img in enumerate(augmented_images):
            # Nombre del archivo
            base_name = Path(row['filename']).stem
            if aug_idx == 0:
                filename = f"{base_name}_original.npy"
            else:
                filename = f"{base_name}_aug_{aug_idx}.npy"
            
            # Guardar como numpy array (más eficiente que imágenes)
            save_path = class_dir / filename
            np.save(save_path, aug_img)
            saved_paths.append(str(save_path))
        
        return {
            'original_path': row['image_path'],
            'class': row['class'],
            'saved_paths': saved_paths,
            'num_augmented': len(augmented_images)
        }
        
    except Exception as e:
        print(f"❌ Error procesando imagen {idx}: {e}")
        return None

def process_split(df, split_name, augmentation_pipeline):
    """Procesar un split completo (train, val, test)"""
    print(f"\n🔄 Procesando split: {split_name}")
    print(f"   Total de imágenes: {len(df)}")
    
    # Crear directorio de salida
    output_dir = AUGMENTED_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preparar argumentos para multiprocessing
    args_list = []
    for idx, row in df.iterrows():
        args_list.append((idx, row, augmentation_pipeline, output_dir, split_name))
    
    # Procesar con multiprocessing - OPTIMIZADO
    num_processes = min(mp.cpu_count(), 6)  # AUMENTADO: de 4 a 6 procesos para compensar
    print(f"   Usando {num_processes} procesos paralelos...")
    
    results = []
    with mp.Pool(num_processes) as pool:
        # Usar tqdm para mostrar progreso
        results = list(tqdm(
            pool.imap(process_single_image, args_list),
            total=len(args_list),
            desc=f"Procesando {split_name}"
        ))
    
    # Filtrar resultados válidos
    valid_results = [r for r in results if r is not None]
    
    print(f"   ✅ Procesadas: {len(valid_results)}/{len(df)} imágenes")
    
    # Estadísticas de augmentation
    if split_name == 'train':
        total_augmented = sum(r['num_augmented'] for r in valid_results)
        print(f"   📈 Total imágenes generadas: {total_augmented}")
        print(f"   🔢 Factor de augmentation promedio: {total_augmented/len(valid_results):.1f}x")
        print(f"   ⚡ OPTIMIZACIÓN: Factor reducido de 4x a 3x para acelerar entrenamiento")
    
    return valid_results

# ====================================================================
# PASO 5: CREAR ÍNDICES DE DATOS AUMENTADOS
# ====================================================================

def create_augmented_index(results, split_name):
    """Crear índice de datos aumentados"""
    print(f"📝 Creando índice para {split_name}...")
    
    index_data = []
    
    for result in results:
        for i, saved_path in enumerate(result['saved_paths']):
            index_data.append({
                'augmented_path': saved_path,
                'original_path': result['original_path'],
                'class': result['class'],
                'split': split_name,
                'augmentation_type': 'original' if i == 0 else f'augmented_{i}',
                'filename': Path(saved_path).name
            })
    
    # Crear DataFrame
    index_df = pd.DataFrame(index_data)
    
    # Guardar índice
    index_path = AUGMENTED_DIR / f"{split_name}_augmented_index.csv"
    index_df.to_csv(index_path, index=False)
    
    print(f"   ✅ Índice guardado: {index_path}")
    print(f"   📊 Total entradas: {len(index_df)}")
    
    # Mostrar distribución por clase
    class_dist = index_df['class'].value_counts()
    print(f"   🏷️  Distribución por clase:")
    for class_name, count in class_dist.items():
        print(f"      {class_name}: {count} imágenes")
    
    return index_df

# ====================================================================
# PASO 6: VISUALIZACIÓN DE AUGMENTATIONS (SIMPLIFICADA)
# ====================================================================

def visualize_augmentations(train_df, augmentation_pipeline, num_examples=2):  # REDUCIDO de 3 a 2
    """Visualizar ejemplos de augmentation"""
    print(f"\n🖼️  Creando visualización de augmentations...")
    
    # Crear directorio de visualizaciones
    viz_dir = BASE_DIR / "results" / "augmentation_examples"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Seleccionar ejemplos de cada clase (REDUCIDO)
    classes = train_df['class'].unique()
    
    for class_name in classes:
        class_samples = train_df[train_df['class'] == class_name].sample(n=min(num_examples, 3))  # REDUCIDO
        
        for idx, (_, row) in enumerate(class_samples.iterrows()):
            # Cargar imagen original
            img = load_and_preprocess_image(row['image_path'])
            if img is None:
                continue
            
            # Crear augmentations (REDUCIDAS)
            augmented_images = apply_augmentation(img, augmentation_pipeline, num_augmentations=3)  # REDUCIDO de 5 a 3
            
            # Crear figura más pequeña
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))  # SIMPLIFICADO: 1 fila, 4 columnas
            fig.suptitle(f'Augmentation Examples - {class_name} - Sample {idx+1}', fontsize=14)
            
            # Mostrar imágenes
            for i, aug_img in enumerate(augmented_images[:4]):  # Solo mostrar 4 imágenes
                axes[i].imshow(aug_img)
                axes[i].set_title(f'{"Original" if i == 0 else f"Aug {i}"}')
                axes[i].axis('off')
            
            # Guardar figura
            save_path = viz_dir / f"{class_name}_sample_{idx+1}_augmentations.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')  # REDUCIDO DPI de 150 a 100
            plt.close()
            
            print(f"   ✅ Guardado: {save_path}")

# ====================================================================
# PASO 7: FUNCIÓN PRINCIPAL
# ====================================================================

def main():
    """Función principal de Data Augmentation OPTIMIZADO"""
    print("🚀 INICIANDO DATA AUGMENTATION OPTIMIZADO")
    print("=" * 60)
    print("⚡ OPTIMIZACIONES APLICADAS:")
    print("   • Factor de augmentation: 3→2 (menos imágenes)")
    print("   • Transformaciones: 7→3 (solo las más rápidas)")
    print("   • Procesos paralelos: 4→6 (mayor paralelización)")
    print("   • Visualizaciones simplificadas")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Paso 1: Cargar datos de Fase 1
        train_df, val_df, test_df = load_phase1_results()
        
        # Paso 2: Crear pipeline de augmentation
        augmentation_pipeline = create_augmentation_pipeline()
        
        # Paso 3: Crear directorio principal de datos aumentados
        AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Paso 4: Procesar cada split
        print(f"\n📊 PROCESANDO SPLITS")
        print("=" * 40)
        
        splits_data = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        all_indices = {}
        
        for split_name, split_df in splits_data.items():
            # Procesar split
            results = process_split(split_df, split_name, augmentation_pipeline)
            
            # Crear índice
            index_df = create_augmented_index(results, split_name)
            all_indices[split_name] = index_df
        
        # Paso 5: Crear resumen general
        print(f"\n📋 CREANDO RESUMEN GENERAL")
        print("=" * 40)
        
        summary_data = []
        total_original = 0
        total_augmented = 0
        
        for split_name, index_df in all_indices.items():
            original_count = len(index_df[index_df['augmentation_type'] == 'original'])
            augmented_count = len(index_df[index_df['augmentation_type'] != 'original'])
            total_count = len(index_df)
            
            summary_data.append({
                'split': split_name,
                'original_images': original_count,
                'augmented_images': augmented_count,
                'total_images': total_count,
                'augmentation_factor': total_count / original_count if original_count > 0 else 0
            })
            
            total_original += original_count
            total_augmented += augmented_count
        
        # Guardar resumen
        summary_df = pd.DataFrame(summary_data)
        summary_path = AUGMENTED_DIR / "augmentation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"📊 RESUMEN FINAL:")
        print(summary_df.to_string(index=False))
        print(f"\n💾 Resumen guardado en: {summary_path}")
        
        # Paso 6: Crear visualizaciones
        print(f"\n🎨 CREANDO VISUALIZACIONES")
        print("=" * 40)
        visualize_augmentations(train_df, augmentation_pipeline)
        
        # Tiempo total
        total_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {total_time/60:.1f} minutos")
        print(f"✅ Data Augmentation OPTIMIZADO completado exitosamente!")
        
        # Comparación estimada de tiempos
        original_time_estimate = total_time * 2.5  # Estimación del tiempo original
        print(f"\n⚡ OPTIMIZACIÓN DE RENDIMIENTO:")
        print(f"   Tiempo estimado original: ~{original_time_estimate/60:.1f} minutos")
        print(f"   Tiempo actual optimizado: {total_time/60:.1f} minutos")
        print(f"   Mejora de velocidad: ~{original_time_estimate/total_time:.1f}x más rápido")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   📂 Datos aumentados: {AUGMENTED_DIR}")
        print(f"   📊 Índices: train/val/test_augmented_index.csv")
        print(f"   📋 Resumen: augmentation_summary.csv")
        print(f"   🖼️  Visualizaciones: results/augmentation_examples/")
        
        return all_indices, summary_df
        
    except Exception as e:
        print(f"❌ Error en Data Augmentation: {e}")
        raise

# ====================================================================
# FUNCIÓN DE UTILIDAD PARA CARGAR DATOS AUMENTADOS
# ====================================================================

def load_augmented_data(split_name):
    """Cargar datos aumentados de un split específico"""
    index_path = AUGMENTED_DIR / f"{split_name}_augmented_index.csv"
    
    if not index_path.exists():
        raise FileNotFoundError(f"❌ Índice no encontrado: {index_path}")
    
    index_df = pd.read_csv(index_path)
    
    print(f"📂 Cargado índice de {split_name}: {len(index_df)} imágenes")
    return index_df

# ====================================================================
# FUNCIÓN ADICIONAL: MODO ULTRA-RÁPIDO (OPCIONAL)
# ====================================================================

def create_minimal_augmentation_pipeline():
    """Pipeline mínimo para entrenamiento ultra-rápido"""
    return Compose([
        HorizontalFlip(p=0.5),  # Solo flip horizontal
    ], p=0.5)

def run_ultra_fast_mode():
    """Ejecutar con augmentation mínimo para máxima velocidad"""
    print("🏃‍♂️ MODO ULTRA-RÁPIDO ACTIVADO")
    print("   Solo HorizontalFlip + Factor 1.5x")
    
    global AUGMENTATION_FACTOR
    AUGMENTATION_FACTOR = 1  # Solo 1 augmentation por imagen
    
    # Reemplazar pipeline
    global create_augmentation_pipeline
    create_augmentation_pipeline = create_minimal_augmentation_pipeline
    
    return main()

# ====================================================================
# EJECUTAR SI ES LLAMADO DIRECTAMENTE
# ====================================================================

if __name__ == "__main__":
    # Configurar semilla para reproducibilidad
    random.seed(42)
    np.random.seed(42)
    
    # Preguntar por modo de ejecución
    print("🚀 SELECCIONA MODO DE EJECUCIÓN:")
    print("1. Optimizado (recomendado)")
    print("2. Ultra-rápido (mínimo augmentation)")
    
    try:
        choice = input("Ingresa 1 o 2: ").strip()
        if choice == "2":
            indices, summary = run_ultra_fast_mode()
        else:
            indices, summary = main()
    except KeyboardInterrupt:
        print("\n⏹️  Ejecución cancelada por el usuario")
    except:
        # Si no hay input disponible, usar modo por defecto
        indices, summary = main()
    
    print(f"\n🎯 SIGUIENTE PASO:")
    print(f"Los datos están listos para la Fase 3: Entrenamiento del Modelo")
    print(f"Usa las funciones load_augmented_data('train'), load_augmented_data('val'), etc.")