# ====================================================================
# FASE 3: ARQUITECTURA DEL MODELO - COVID-19 RADIOGRAFÍAS
# ====================================================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import time
import json

# Configurar TensorFlow para usar CPU eficientemente
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

print(f"✅ TensorFlow versión: {tf.__version__}")
print(f"🔧 Dispositivos disponibles: {tf.config.list_physical_devices()}")

# ====================================================================
# PASO 1: CONFIGURACIÓN Y PARÁMETROS
# ====================================================================

# Directorios del proyecto
BASE_DIR = Path("covid_classifier")
AUGMENTED_DIR = BASE_DIR / "data" / "augmented"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Crear directorios necesarios
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros del modelo
MODEL_CONFIG = {
    'input_shape': (150, 150, 3),
    'num_classes': 3,
    'batch_size': 32,
    'learning_rate': 0.001,
    'dropout_rate': 0.3,
    'l2_regularization': 0.01
}

# Mapeo de clases
CLASS_NAMES = ['COVID-19', 'NORMAL', 'PNEUMONIA']
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INT_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

print(f"🏷️  Clases del modelo: {CLASS_NAMES}")
print(f"📊 Configuración del modelo: {MODEL_CONFIG}")

# ====================================================================
# PASO 2: GENERADOR DE DATOS PERSONALIZADO
# ====================================================================

class COVID19DataGenerator(tf.keras.utils.Sequence):
    """Generador de datos personalizado para cargar imágenes .npy"""
    
    def __init__(self, dataframe, batch_size=32, shuffle=True, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.dataframe))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Número de batches por época"""
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        """Generar un batch de datos"""
        # Índices del batch
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.dataframe))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Inicializar arrays
        batch_x = np.zeros((len(batch_indices), *MODEL_CONFIG['input_shape']), dtype=np.float32)
        batch_y = np.zeros((len(batch_indices), MODEL_CONFIG['num_classes']), dtype=np.float32)
        
        # Cargar imágenes del batch
        for i, idx in enumerate(batch_indices):
            row = self.dataframe.iloc[idx]
            
            # Cargar imagen .npy
            try:
                img = np.load(row['augmented_path'])
                
                # Asegurar que la imagen tenga el shape correcto
                if img.shape != MODEL_CONFIG['input_shape']:
                    # Si es grayscale, convertir a RGB
                    if len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=-1)
                    elif img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                
                batch_x[i] = img
                
                # One-hot encoding de la clase
                class_idx = CLASS_TO_INT[row['class']]
                batch_y[i, class_idx] = 1.0
                
            except Exception as e:
                print(f"⚠️  Error cargando {row['augmented_path']}: {e}")
                # Usar imagen en negro si falla
                batch_x[i] = np.zeros(MODEL_CONFIG['input_shape'])
                batch_y[i, 0] = 1.0  # Clase por defecto
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Barajar al final de cada época"""
        if self.shuffle:
            np.random.shuffle(self.indices)

# ====================================================================
# PASO 3: CONSTRUCCIÓN DE LA ARQUITECTURA MOBILENETV2
# ====================================================================

def create_mobilenetv2_model():
    """Crear modelo basado en MobileNetV2 para clasificación COVID-19"""
    print("🏗️  Construyendo arquitectura MobileNetV2...")
    
    # Cargar MobileNetV2 preentrenado (sin la capa final)
    base_model = MobileNetV2(
        input_shape=MODEL_CONFIG['input_shape'],
        alpha=1.0,  # Factor de escalado de anchura
        include_top=False,  # No incluir clasificador final
        weights='imagenet',  # Pesos preentrenados
        pooling='avg'  # Global Average Pooling
    )
    
    # Congelar las primeras capas del modelo base
    base_model.trainable = False
    print(f"   📊 Capas congeladas: {len(base_model.layers)}")
    
    # Construir el modelo completo
    inputs = keras.Input(shape=MODEL_CONFIG['input_shape'])
    
    # Preprocesamiento (normalización para ImageNet)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Modelo base MobileNetV2
    x = base_model(x, training=False)
    
    # Capas de clasificación personalizadas
    x = layers.Dropout(MODEL_CONFIG['dropout_rate'])(x)
    
    # Capa densa con regularización L2
    x = layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(MODEL_CONFIG['l2_regularization'])
    )(x)
    x = layers.Dropout(MODEL_CONFIG['dropout_rate'])(x)
    
    # Capa de salida
    outputs = layers.Dense(
        MODEL_CONFIG['num_classes'], 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Crear modelo
    model = Model(inputs, outputs)
    
    print("✅ Arquitectura MobileNetV2 creada")
    print(f"   📏 Parámetros totales: {model.count_params():,}")
    
    # Calcular parámetros no entrenables de forma compatible
    try:
        non_trainable_params = sum([np.prod(v.shape) for v in base_model.non_trainable_variables])
        print(f"   🔒 Parámetros no entrenable: {non_trainable_params:,}")
    except:
        print(f"   🔒 Parámetros no entrenable: {len(base_model.non_trainable_variables)} variables")
    
    return model, base_model

# ====================================================================
# PASO 4: CONFIGURACIÓN DE ENTRENAMIENTO
# ====================================================================

def compile_model(model):
    """Compilar el modelo con optimizador y métricas"""
    print("⚙️  Compilando modelo...")
    
    # Optimizador Adam con learning rate específico
    optimizer = Adam(learning_rate=MODEL_CONFIG['learning_rate'])
    
    # Compilar modelo
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',  # Para clasificación multiclase
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("✅ Modelo compilado con:")
    print(f"   🎯 Loss: categorical_crossentropy")
    print(f"   📊 Métricas: accuracy, precision, recall, AUC")
    print(f"   🔧 Optimizador: Adam (lr={MODEL_CONFIG['learning_rate']})")
    
    return model

# ====================================================================
# PASO 5: CALLBACKS PARA ENTRENAMIENTO
# ====================================================================

def create_callbacks():
    """Crear callbacks para el entrenamiento"""
    print("📋 Configurando callbacks...")
    
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reducir learning rate cuando se estanque
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Guardar mejor modelo
        ModelCheckpoint(
            filepath=str(MODELS_DIR / 'best_mobilenetv2_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    print("✅ Callbacks configurados:")
    print("   ⏹️  EarlyStopping (paciencia: 10)")
    print("   📉 ReduceLROnPlateau (paciencia: 5)")
    print("   💾 ModelCheckpoint (mejor val_accuracy)")
    
    return callbacks

# ====================================================================
# PASO 6: CARGA DE DATOS Y PREPARACIÓN
# ====================================================================

def load_and_prepare_data():
    """Cargar y preparar los datos aumentados"""
    print("📂 Cargando datos aumentados...")
    
    # Cargar índices de cada split
    train_df = pd.read_csv(AUGMENTED_DIR / "train_augmented_index.csv")
    val_df = pd.read_csv(AUGMENTED_DIR / "val_augmented_index.csv")
    test_df = pd.read_csv(AUGMENTED_DIR / "test_augmented_index.csv")
    
    print(f"✅ Datos cargados:")
    print(f"   🚂 Entrenamiento: {len(train_df)} imágenes")
    print(f"   ✅ Validación: {len(val_df)} imágenes")
    print(f"   🧪 Prueba: {len(test_df)} imágenes")
    
    # Mostrar distribución de clases
    print(f"\n📊 Distribución de clases (entrenamiento):")
    train_class_dist = train_df['class'].value_counts()
    for class_name, count in train_class_dist.items():
        percentage = (count / len(train_df)) * 100
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    return train_df, val_df, test_df

# ====================================================================
# PASO 7: CÁLCULO DE PESOS DE CLASE
# ====================================================================

def calculate_class_weights(train_df):
    """Calcular pesos de clase para balancear el dataset"""
    print("⚖️  Calculando pesos de clase...")
    
    # Obtener clases y sus frecuencias
    class_counts = train_df['class'].value_counts()
    
    # Calcular pesos usando sklearn
    classes = [CLASS_TO_INT[class_name] for class_name in class_counts.index]
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.array(classes),
        y=[CLASS_TO_INT[class_name] for class_name in train_df['class']]
    )
    
    # Crear diccionario de pesos
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    
    print("✅ Pesos de clase calculados:")
    for class_idx, weight in class_weights.items():
        class_name = INT_TO_CLASS[class_idx]
        print(f"   {class_name}: {weight:.3f}")
    
    return class_weights

# ====================================================================
# PASO 8: FUNCIÓN PRINCIPAL DE CREACIÓN DEL MODELO
# ====================================================================

def create_covid_model():
    """Función principal para crear el modelo completo"""
    print("🚀 CREANDO MODELO COVID-19")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Paso 1: Cargar datos
        train_df, val_df, test_df = load_and_prepare_data()
        
        # Paso 2: Crear generadores de datos
        print(f"\n🔄 Creando generadores de datos...")
        train_generator = COVID19DataGenerator(
            train_df, 
            batch_size=MODEL_CONFIG['batch_size'], 
            shuffle=True
        )
        val_generator = COVID19DataGenerator(
            val_df, 
            batch_size=MODEL_CONFIG['batch_size'], 
            shuffle=False
        )
        test_generator = COVID19DataGenerator(
            test_df, 
            batch_size=MODEL_CONFIG['batch_size'], 
            shuffle=False
        )
        
        print(f"✅ Generadores creados:")
        print(f"   🚂 Train batches: {len(train_generator)}")
        print(f"   ✅ Val batches: {len(val_generator)}")
        print(f"   🧪 Test batches: {len(test_generator)}")
        
        # Paso 3: Crear arquitectura
        model, base_model = create_mobilenetv2_model()
        
        # Paso 4: Compilar modelo
        model = compile_model(model)
        
        # Paso 5: Crear callbacks
        callbacks = create_callbacks()
        
        # Paso 6: Calcular pesos de clase
        class_weights = calculate_class_weights(train_df)
        
        # Paso 7: Mostrar resumen del modelo
        print(f"\n📋 RESUMEN DEL MODELO:")
        print("=" * 30)
        model.summary()
        
        # Paso 8: Guardar configuración
        config_data = {
            'model_config': MODEL_CONFIG,
            'class_names': CLASS_NAMES,
            'class_to_int': CLASS_TO_INT,
            'class_weights': class_weights,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'model_params': int(model.count_params())
        }
        
        config_path = MODELS_DIR / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Configuración guardada en: {config_path}")
        
        # Tiempo de creación
        creation_time = time.time() - start_time
        print(f"\n⏱️  Tiempo de creación: {creation_time:.2f} segundos")
        print(f"✅ Modelo listo para entrenamiento!")
        
        return {
            'model': model,
            'base_model': base_model,
            'train_generator': train_generator,
            'val_generator': val_generator,
            'test_generator': test_generator,
            'callbacks': callbacks,
            'class_weights': class_weights,
            'config': config_data
        }
        
    except Exception as e:
        print(f"❌ Error creando modelo: {e}")
        raise

# ====================================================================
# PASO 9: FUNCIÓN DE FINE-TUNING
# ====================================================================

def setup_fine_tuning(model, base_model):
    """Configurar el modelo para fine-tuning"""
    print("🔧 Configurando fine-tuning...")
    
    # Descongelar las últimas capas del modelo base
    base_model.trainable = True
    
    # Congelar todas las capas excepto las últimas 20
    fine_tune_at = len(base_model.layers) - 20
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=Adam(learning_rate=MODEL_CONFIG['learning_rate'] / 10),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print(f"✅ Fine-tuning configurado:")
    print(f"   🔓 Capas entrenables: {fine_tune_at} - {len(base_model.layers)}")
    print(f"   📉 Learning rate reducido: {MODEL_CONFIG['learning_rate'] / 10}")
    
    return model

# ====================================================================
# FUNCIÓN DE UTILIDAD PARA VISUALIZAR ARQUITECTURA
# ====================================================================

def visualize_model_architecture(model):
    """Crear visualización de la arquitectura del modelo"""
    print("🎨 Creando visualización de la arquitectura...")
    
    try:
        # Crear directorio de visualizaciones
        viz_dir = RESULTS_DIR / "model_architecture"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar plot del modelo
        plot_path = viz_dir / "mobilenetv2_architecture.png"
        tf.keras.utils.plot_model(
            model,
            to_file=str(plot_path),
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150
        )
        
        print(f"✅ Arquitectura guardada en: {plot_path}")
        
    except Exception as e:
        print(f"⚠️  No se pudo crear visualización: {e}")

# ====================================================================
# EJECUTAR SI ES LLAMADO DIRECTAMENTE
# ====================================================================

if __name__ == "__main__":
    # Configurar semillas para reproducibilidad
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Crear modelo
    model_components = create_covid_model()
    
    # Visualizar arquitectura
    visualize_model_architecture(model_components['model'])
    
    print(f"\n🎯 SIGUIENTE PASO:")
    print(f"El modelo está listo para la Fase 4: Entrenamiento Optimizado")
    print(f"Usa model_components para acceder a todos los componentes")
    
    # Ejemplo de uso para la siguiente fase
    print(f"\n💡 EJEMPLO DE USO:")
    print(f"model = model_components['model']")
    print(f"train_gen = model_components['train_generator']")
    print(f"val_gen = model_components['val_generator']")
    print(f"callbacks = model_components['callbacks']")
    print(f"class_weights = model_components['class_weights']")