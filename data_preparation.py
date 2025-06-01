# ====================================================================
# FASE 1: PREPARACIÓN DE DATOS - COVID-19 RADIOGRAFÍAS
# ====================================================================

import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time

# ====================================================================
# PASO 1: CONFIGURACIÓN INICIAL
# ====================================================================

# Configuración de rutas
BASE_DIR = Path("covid_classifier")
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = BASE_DIR / "data" / "splits"

# Crear estructura de carpetas
def create_project_structure():
    """Crear la estructura de carpetas del proyecto"""
    print("📁 Creando estructura de carpetas...")
    
    folders = [
        BASE_DIR / "data" / "raw",
        BASE_DIR / "data" / "processed", 
        BASE_DIR / "data" / "splits" / "train",
        BASE_DIR / "data" / "splits" / "val",
        BASE_DIR / "data" / "splits" / "test",
        BASE_DIR / "src",
        BASE_DIR / "results",
        BASE_DIR / "notebooks"
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Creada: {folder}")
    
    print("✅ Estructura de carpetas creada exitosamente!\n")

# ====================================================================
# PASO 2: EXPLORACIÓN Y CATALOGACIÓN DEL DATASET
# ====================================================================

def explore_dataset():
    """Explorar el dataset y crear un catálogo de todas las imágenes"""
    print("🔍 Explorando el dataset...")
    
    # Asumiendo que el dataset está organizado así:
    # COVID-19_Radiography_Dataset/
    # ├── COVID/
    # ├── Normal/
    # └── Viral Pneumonia/
    
    dataset_info = []
    
    # Mapeo de nombres de carpetas a etiquetas
    class_mapping = {
        'COVID': 'COVID-19',
        'Normal': 'NORMAL', 
        'Viral Pneumonia': 'PNEUMONIA'
    }
    
    total_images = 0
    
    for folder_name, class_label in class_mapping.items():
        folder_path = RAW_DATA_DIR / folder_name
        
        if folder_path.exists():
            # Buscar archivos de imagen
            image_extensions = ['*.png', '*.jpg', '*.jpeg']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(folder_path.glob(ext)))
            
            print(f"📊 Clase {class_label}: {len(image_files)} imágenes")
            
            # Agregar información de cada imagen
            for img_path in image_files:
                dataset_info.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'class': class_label,
                    'class_folder': folder_name,
                    'file_size_mb': img_path.stat().st_size / (1024*1024)
                })
                total_images += 1
        else:
            print(f"⚠️  Carpeta no encontrada: {folder_path}")
    
    print(f"\n📈 Total de imágenes encontradas: {total_images}")
    
    # Crear DataFrame
    df = pd.DataFrame(dataset_info)
    
    # Guardar catálogo
    catalog_path = BASE_DIR / "data" / "dataset_catalog.csv"
    df.to_csv(catalog_path, index=False)
    print(f"💾 Catálogo guardado en: {catalog_path}")
    
    return df

# ====================================================================
# PASO 3: ANÁLISIS EXPLORATORIO BÁSICO
# ====================================================================

def analyze_dataset(df):
    """Realizar análisis exploratorio básico del dataset"""
    print("\n📊 ANÁLISIS EXPLORATORIO DEL DATASET")
    print("=" * 50)
    
    # Distribución de clases
    class_counts = df['class'].value_counts()
    print("\n🏷️  Distribución de clases:")
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {class_name}: {count} imágenes ({percentage:.1f}%)")
    
    # Estadísticas de tamaño de archivos
    print(f"\n📏 Tamaño de archivos:")
    print(f"   Promedio: {df['file_size_mb'].mean():.2f} MB")
    print(f"   Mínimo: {df['file_size_mb'].min():.2f} MB")
    print(f"   Máximo: {df['file_size_mb'].max():.2f} MB")
    
    # Verificar balance del dataset
    min_class = class_counts.min()
    max_class = class_counts.max()
    balance_ratio = min_class / max_class
    
    print(f"\n⚖️  Balance del dataset:")
    print(f"   Ratio balance: {balance_ratio:.2f}")
    if balance_ratio < 0.7:
        print("   ⚠️  Dataset desbalanceado - considerar técnicas de balanceo")
    else:
        print("   ✅ Dataset relativamente balanceado")
    
    return class_counts

# ====================================================================
# PASO 4: VERIFICACIÓN DE CALIDAD DE IMÁGENES
# ====================================================================

def check_image_quality(df, sample_size=100):
    """Verificar la calidad y características de una muestra de imágenes"""
    print(f"\n🔍 Verificando calidad de imágenes (muestra de {sample_size})...")
    
    # Tomar muestra aleatoria
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    valid_images = 0
    invalid_images = []
    resolutions = []
    
    for idx, row in sample_df.iterrows():
        try:
            # Intentar leer la imagen
            img = cv2.imread(row['image_path'])
            
            if img is not None:
                height, width = img.shape[:2]
                resolutions.append((width, height))
                valid_images += 1
            else:
                invalid_images.append(row['image_path'])
                
        except Exception as e:
            invalid_images.append(row['image_path'])
            print(f"❌ Error leyendo {row['filename']}: {e}")
    
    print(f"✅ Imágenes válidas: {valid_images}/{len(sample_df)}")
    
    if invalid_images:
        print(f"❌ Imágenes inválidas: {len(invalid_images)}")
        for invalid_img in invalid_images[:5]:  # Mostrar solo las primeras 5
            print(f"   - {invalid_img}")
    
    # Análisis de resoluciones
    if resolutions:
        widths, heights = zip(*resolutions)
        print(f"\n📐 Resoluciones encontradas:")
        print(f"   Ancho promedio: {np.mean(widths):.0f}px")
        print(f"   Alto promedio: {np.mean(heights):.0f}px")
        print(f"   Resolución más común: {Counter(resolutions).most_common(1)[0]}")
    
    return valid_images, invalid_images

# ====================================================================
# PASO 5: DIVISIÓN EN CONJUNTOS (TRAIN/VAL/TEST)
# ====================================================================

def create_data_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Dividir el dataset en conjuntos de entrenamiento, validación y prueba"""
    print(f"\n📂 Creando división de datos...")
    print(f"   Entrenamiento: {train_ratio*100:.0f}%")
    print(f"   Validación: {val_ratio*100:.0f}%") 
    print(f"   Prueba: {test_ratio*100:.0f}%")
    
    # Verificar que las proporciones sumen 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Las proporciones deben sumar 1.0"
    
    splits = {}
    
    # Dividir por clase para mantener distribución
    for class_name in df['class'].unique():
        class_data = df[df['class'] == class_name].copy()
        
        # Primera división: train vs (val + test)
        train_data, temp_data = train_test_split(
            class_data, 
            test_size=(val_ratio + test_ratio),
            random_state=42,
            shuffle=True
        )
        
        # Segunda división: val vs test
        val_data, test_data = train_test_split(
            temp_data,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42,
            shuffle=True
        )
        
        # Agregar etiqueta de split
        train_data['split'] = 'train'
        val_data['split'] = 'val' 
        test_data['split'] = 'test'
        
        # Combinar
        if class_name not in splits:
            splits[class_name] = []
        
        splits[class_name].extend([train_data, val_data, test_data])
        
        print(f"   {class_name}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Combinar todas las clases
    train_df = pd.concat([splits[cls][0] for cls in splits.keys()], ignore_index=True)
    val_df = pd.concat([splits[cls][1] for cls in splits.keys()], ignore_index=True)  
    test_df = pd.concat([splits[cls][2] for cls in splits.keys()], ignore_index=True)
    
    # Mezclar los conjuntos
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Resumen final:")
    print(f"   Entrenamiento: {len(train_df)} imágenes")
    print(f"   Validación: {len(val_df)} imágenes")
    print(f"   Prueba: {len(test_df)} imágenes")
    print(f"   Total: {len(train_df) + len(val_df) + len(test_df)} imágenes")
    
    return train_df, val_df, test_df

# ====================================================================
# PASO 6: GUARDAR INFORMACIÓN DE SPLITS
# ====================================================================

def save_splits_info(train_df, val_df, test_df):
    """Guardar información de los splits en archivos CSV"""
    print("\n💾 Guardando información de splits...")
    
    # Guardar cada split
    splits_info = {
        'train': train_df,
        'val': val_df, 
        'test': test_df
    }
    
    for split_name, split_df in splits_info.items():
        # Guardar CSV con información del split
        csv_path = SPLITS_DIR / f"{split_name}_split.csv"
        split_df.to_csv(csv_path, index=False)
        print(f"   ✅ {split_name}: {csv_path}")
        
        # Mostrar distribución de clases en este split
        class_dist = split_df['class'].value_counts()
        print(f"      Distribución: {dict(class_dist)}")
    
    # Crear archivo resumen
    summary_data = []
    for split_name, split_df in splits_info.items():
        for class_name in split_df['class'].unique():
            count = len(split_df[split_df['class'] == class_name])
            summary_data.append({
                'split': split_name,
                'class': class_name, 
                'count': count
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = BASE_DIR / "data" / "splits_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   📊 Resumen: {summary_path}")

# ====================================================================
# PASO 7: FUNCIÓN PRINCIPAL 
# ====================================================================

def main():
    """Función principal que ejecuta toda la preparación de datos"""
    print("🚀 INICIANDO PREPARACIÓN DE DATOS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Paso 1: Crear estructura
    create_project_structure()
    
    # Paso 2: Explorar dataset
    df = explore_dataset()
    
    if df.empty:
        print("❌ No se encontraron imágenes. Verifica la ruta del dataset.")
        return
    
    # Paso 3: Análisis exploratorio
    class_counts = analyze_dataset(df)
    
    # Paso 4: Verificar calidad
    valid_count, invalid_images = check_image_quality(df)
    
    # Filtrar imágenes inválidas si las hay
    if invalid_images:
        print(f"\n🧹 Filtrando {len(invalid_images)} imágenes inválidas...")
        df = df[~df['image_path'].isin(invalid_images)].reset_index(drop=True)
    
    # Paso 5: Crear splits
    train_df, val_df, test_df = create_data_splits(df)
    
    # Paso 6: Guardar información
    save_splits_info(train_df, val_df, test_df)
    
    # Tiempo total
    total_time = time.time() - start_time
    print(f"\n⏱️  Tiempo total: {total_time:.1f} segundos")
    print(f"✅ Preparación de datos completada exitosamente!")
    
    return train_df, val_df, test_df

# ====================================================================
# EJECUTAR SI ES LLAMADO DIRECTAMENTE
# ====================================================================

if __name__ == "__main__":
    # Ejecutar preparación
    train_df, val_df, test_df = main()
    
    # Opcional: Mostrar estadísticas finales
    print(f"\n📈 ESTADÍSTICAS FINALES:")
    print(f"Dataset listo para entrenamiento!")
    print(f"Total de imágenes procesadas: {len(train_df) + len(val_df) + len(test_df)}")