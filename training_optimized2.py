# ====================================================================
# FASE 4: ENTRENAMIENTO OPTIMIZADO GPU - COVID-19 RADIOGRAFÍAS (CORREGIDO)
# ====================================================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
import psutil
import threading
import gc
from contextlib import contextmanager

# Importar la Fase 3
from arq_model import create_covid_model, setup_fine_tuning

# ====================================================================
# PASO 1: CONFIGURACIÓN OPTIMIZADA DE GPU Y MEMORIA
# ====================================================================

def setup_gpu_optimization():
    """Configurar GPU y optimizaciones de memoria"""
    print("🚀 CONFIGURANDO OPTIMIZACIONES GPU Y MEMORIA")
    print("=" * 50)
    
    # Detectar GPUs disponibles
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"🎯 GPUs detectadas: {len(gpus)}")
    
    if gpus:
        try:
            # Configurar crecimiento dinámico de memoria GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Configurar GPU principal
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Habilitar XLA (Accelerated Linear Algebra)
            tf.config.optimizer.set_jit(True)
            
            # Configurar Mixed Precision para GPUs compatibles
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"✅ GPU configurada: {gpus[0].name}")
            print(f"✅ Mixed Precision habilitado: {policy.name}")
            print(f"✅ XLA optimización habilitada")
            
        except RuntimeError as e:
            print(f"⚠️  Error configurando GPU: {e}")
            return False
    else:
        print("⚠️  No se detectaron GPUs, usando CPU optimizado")
        # Optimizaciones para CPU
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Usar todos los cores
        tf.config.threading.set_inter_op_parallelism_threads(0)
        
        # Habilitar optimizaciones de CPU
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        
        return False
    
    return True

# ====================================================================
# PASO 2: GENERADOR DE DATOS OPTIMIZADO CON CACHE
# ====================================================================

class OptimizedCOVID19DataGenerator(tf.keras.utils.Sequence):
    """Generador optimizado con cache y prefetch"""
    
    def __init__(self, dataframe, batch_size=32, shuffle=True, cache_size=1000):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataframe))
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        """Generar batch optimizado con cache"""
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.dataframe))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Pre-allocar arrays con dtype específico
        batch_x = np.empty((len(batch_indices), 150, 150, 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_indices), 3), dtype=np.float32)
        
        # Cargar imágenes con cache
        for i, idx in enumerate(batch_indices):
            row = self.dataframe.iloc[idx]
            img_path = row['augmented_path']
            
            # Verificar cache
            if img_path in self.cache:
                img = self.cache[img_path]
                self.cache_hits += 1
            else:
                try:
                    img = np.load(img_path).astype(np.float32)
                    
                    # Verificar y corregir shape
                    if img.shape != (150, 150, 3):
                        if len(img.shape) == 2:
                            img = np.stack([img] * 3, axis=-1)
                        elif img.shape[-1] == 1:
                            img = np.repeat(img, 3, axis=-1)
                    
                    # Agregar al cache si hay espacio
                    if len(self.cache) < self.cache_size:
                        self.cache[img_path] = img
                    
                    self.cache_misses += 1
                    
                except Exception as e:
                    print(f"⚠️  Error cargando {img_path}: {e}")
                    img = np.zeros((150, 150, 3), dtype=np.float32)
            
            batch_x[i] = img
            
            # One-hot encoding optimizado
            class_name = row['class']
            if class_name == 'COVID-19':
                batch_y[i, 0] = 1.0
            elif class_name == 'NORMAL':
                batch_y[i, 1] = 1.0
            else:  # PNEUMONIA
                batch_y[i, 2] = 1.0
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_size': len(self.cache),
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate
        }

# ====================================================================
# PASO 3: CONFIGURACIÓN OPTIMIZADA DE ENTRENAMIENTO
# ====================================================================

# Configuración optimizada
OPTIMIZED_TRAINING_CONFIG = {
    'initial_epochs': 25,        # Reducido para pruebas rápidas
    'fine_tune_epochs': 15,      # Reducido para eficiencia
    'patience': 8,               # Paciencia reducida
    'monitor_metric': 'val_accuracy',
    'save_best_only': True,
    'verbose': 1,
    'batch_size': 64,            # Batch size más grande para GPU
    'cache_size': 2000,          # Cache más grande
    'prefetch_buffer': 2,        # Buffer de prefetch
    'num_parallel_calls': tf.data.AUTOTUNE
}

print("🚀 FASE 4 OPTIMIZADA: ENTRENAMIENTO CON GPU")
print("=" * 50)
print(f"📊 Configuración optimizada: {OPTIMIZED_TRAINING_CONFIG}")

# ====================================================================
# PASO 4: MONITOR DE RECURSOS OPTIMIZADO
# ====================================================================

class OptimizedResourceMonitor:
    """Monitor optimizado de recursos del sistema"""
    
    def __init__(self, monitoring_interval=15):  # Intervalo reducido
        self.monitoring = False
        self.monitoring_interval = monitoring_interval
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'gpu_memory_used': [],
            'gpu_utilization': [],
            'timestamps': []
        }
        self.thread = None
    
    def start_monitoring(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("📊 Monitor de recursos optimizado iniciado")
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        print("📊 Monitor de recursos detenido")
    
    def _monitor_loop(self):
        while self.monitoring:
            try:
                # Estadísticas básicas
                self.stats['cpu_percent'].append(psutil.cpu_percent(interval=1))
                self.stats['memory_percent'].append(psutil.virtual_memory().percent)
                self.stats['timestamps'].append(datetime.now())
                
                # Estadísticas GPU si está disponible
                try:
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    if gpus:
                        # Usar nvidia-ml-py si está disponible
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            
                            # Memoria GPU
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_mem_used = (mem_info.used / mem_info.total) * 100
                            self.stats['gpu_memory_used'].append(gpu_mem_used)
                            
                            # Utilización GPU
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            self.stats['gpu_utilization'].append(util.gpu)
                            
                        except ImportError:
                            self.stats['gpu_memory_used'].append(0)
                            self.stats['gpu_utilization'].append(0)
                    else:
                        self.stats['gpu_memory_used'].append(0)
                        self.stats['gpu_utilization'].append(0)
                except:
                    self.stats['gpu_memory_used'].append(0)
                    self.stats['gpu_utilization'].append(0)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"⚠️  Error en monitoreo: {e}")
                break

# ====================================================================
# PASO 5: CALLBACKS OPTIMIZADOS
# ====================================================================

def create_optimized_callbacks():
    """Crear callbacks optimizados para GPU"""
    print("📋 Configurando callbacks optimizados...")
    
    # Directorios
    BASE_DIR = Path("covid_classifier")
    MODELS_DIR = BASE_DIR / "models"
    TRAINING_DIR = BASE_DIR / "results" / "training"
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Early Stopping optimizado
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Cambiar a accuracy para convergencia más rápida
            patience=OPTIMIZED_TRAINING_CONFIG['patience'],
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        
        # Reducir learning rate optimizado
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # Reducción más agresiva
            patience=4,  # Paciencia reducida
            min_lr=1e-8,
            verbose=1,
            cooldown=2
        ),
        
        # ModelCheckpoint optimizado
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / 'best_mobilenetv2_optimized.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        ),
        
        # Guardar solo weights para checkpoint rápido
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / 'weights_checkpoint.weights.h5'),
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            save_freq='epoch',
            verbose=0
        ),
        
        # TensorBoard optimizado
        keras.callbacks.TensorBoard(
            log_dir=str(TRAINING_DIR / 'tensorboard_logs'),
            histogram_freq=0,  # Desactivar histogramas para velocidad
            write_graph=False,  # No escribir grafo
            write_images=False,  # No escribir imágenes
            update_freq='epoch',
            profile_batch=0  # Desactivar profiling
        )
    ]
    
    print("✅ Callbacks optimizados configurados")
    return callbacks

# ====================================================================
# PASO 6: GESTIÓN DE MEMORIA Y LIMPIEZA
# ====================================================================

@contextmanager
def memory_cleanup():
    """Context manager para limpieza automática de memoria"""
    try:
        yield
    finally:
        # Limpieza de memoria
        gc.collect()
        if tf.config.experimental.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()

# ====================================================================
# PASO 7: FUNCIÓN DE ENTRENAMIENTO OPTIMIZADA (CORREGIDA)
# ====================================================================

def train_optimized_initial_phase(model, train_gen, val_gen, callbacks, class_weights):
    """Entrenar fase inicial optimizada - VERSIÓN CORREGIDA"""
    print(f"\n🎯 FASE INICIAL OPTIMIZADA")
    print("=" * 40)
    print(f"📊 Épocas: {OPTIMIZED_TRAINING_CONFIG['initial_epochs']}")
    print(f"🔒 Capas base: CONGELADAS")
    print(f"🚀 Batch size: {OPTIMIZED_TRAINING_CONFIG['batch_size']}")
    
    with memory_cleanup():
        start_time = time.time()
        
        # Entrenar con configuración optimizada - PARÁMETROS CORREGIDOS
        history_initial = model.fit(
            train_gen,
            epochs=OPTIMIZED_TRAINING_CONFIG['initial_epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=OPTIMIZED_TRAINING_CONFIG['verbose']
            # Removidos parámetros incompatibles: workers, use_multiprocessing, max_queue_size
        )
        
        training_time = time.time() - start_time
        print(f"⏱️  Tiempo fase inicial: {training_time/60:.1f} minutos")
        
        return history_initial

def train_optimized_fine_tuning_phase(model, base_model, train_gen, val_gen, callbacks, class_weights):
    """Entrenar fase de fine-tuning optimizada - VERSIÓN CORREGIDA"""
    print(f"\n🔧 FASE FINE-TUNING OPTIMIZADA")
    print("=" * 40)
    print(f"📊 Épocas: {OPTIMIZED_TRAINING_CONFIG['fine_tune_epochs']}")
    print(f"🔓 Capas base: PARCIALMENTE DESCONGELADAS")
    
    # Configurar fine-tuning con learning rate más conservador
    model = setup_fine_tuning(model, base_model)
    
    with memory_cleanup():
        start_time = time.time()
        
        # Entrenar con configuración optimizada - PARÁMETROS CORREGIDOS
        history_finetune = model.fit(
            train_gen,
            epochs=OPTIMIZED_TRAINING_CONFIG['fine_tune_epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=OPTIMIZED_TRAINING_CONFIG['verbose']
            # Removidos parámetros incompatibles: workers, use_multiprocessing, max_queue_size
        )
        
        training_time = time.time() - start_time
        print(f"⏱️  Tiempo fine-tuning: {training_time/60:.1f} minutos")
        
        return history_finetune

# ====================================================================
# PASO 8: FUNCIÓN PRINCIPAL OPTIMIZADA
# ====================================================================

def train_covid_model_optimized():
    """Función principal optimizada para entrenar el modelo COVID-19"""
    print("🚀 INICIANDO ENTRENAMIENTO OPTIMIZADO")
    print("=" * 50)
    
    total_start_time = time.time()
    
    # Configurar GPU y optimizaciones
    gpu_available = setup_gpu_optimization()
    
    # Inicializar monitor de recursos optimizado
    resource_monitor = OptimizedResourceMonitor()
    resource_monitor.start_monitoring()
    
    try:
        # Crear modelo y componentes con configuración optimizada
        print("🔧 Creando modelo y componentes optimizados...")
        
        with memory_cleanup():
            model_components = create_covid_model()
            
            # Reemplazar generadores con versión optimizada
            train_df = pd.read_csv(Path("covid_classifier/data/augmented/train_augmented_index.csv"))
            val_df = pd.read_csv(Path("covid_classifier/data/augmented/val_augmented_index.csv"))
            
            optimized_train_gen = OptimizedCOVID19DataGenerator(
                train_df,
                batch_size=OPTIMIZED_TRAINING_CONFIG['batch_size'],
                shuffle=True,
                cache_size=OPTIMIZED_TRAINING_CONFIG['cache_size']
            )
            
            optimized_val_gen = OptimizedCOVID19DataGenerator(
                val_df,
                batch_size=OPTIMIZED_TRAINING_CONFIG['batch_size'],
                shuffle=False,
                cache_size=OPTIMIZED_TRAINING_CONFIG['cache_size']
            )
            
            model = model_components['model']
            base_model = model_components['base_model']
            class_weights = model_components['class_weights']
            
            # Callbacks optimizados
            callbacks = create_optimized_callbacks()
        
        print("✅ Componentes optimizados listos")
        print(f"🎯 GPU disponible: {gpu_available}")
        print(f"📊 Batch size: {OPTIMIZED_TRAINING_CONFIG['batch_size']}")
        print(f"💾 Cache size: {OPTIMIZED_TRAINING_CONFIG['cache_size']}")
        
        # Entrenamiento inicial optimizado
        print(f"\n📚 FASE INICIAL OPTIMIZADA")
        history_initial = train_optimized_initial_phase(
            model, optimized_train_gen, optimized_val_gen, callbacks, class_weights
        )
        
        # Fine-tuning optimizado
        history_finetune = None
        if OPTIMIZED_TRAINING_CONFIG['fine_tune_epochs'] > 0:
            print(f"\n🔧 FASE FINE-TUNING OPTIMIZADA")
            
            # Cargar mejor modelo
            best_model_path = Path("covid_classifier/models/best_mobilenetv2_optimized.h5")
            if best_model_path.exists():
                with memory_cleanup():
                    model = keras.models.load_model(str(best_model_path))
                print(f"📂 Mejor modelo cargado")
            
            history_finetune = train_optimized_fine_tuning_phase(
                model, base_model, optimized_train_gen, optimized_val_gen, callbacks, class_weights
            )
        
        # Guardar modelo final optimizado
        final_model_path = Path("covid_classifier/models/final_covid_model_optimized.h5")
        model.save(str(final_model_path))
        print(f"💾 Modelo final optimizado guardado")
        
        # Estadísticas finales
        total_time = time.time() - total_start_time
        resource_monitor.stop_monitoring()
        
        # Estadísticas de cache
        train_cache_stats = optimized_train_gen.get_cache_stats()
        val_cache_stats = optimized_val_gen.get_cache_stats()
        
        final_stats = {
            'total_training_time_minutes': total_time / 60,
            'gpu_available': gpu_available,
            'batch_size': OPTIMIZED_TRAINING_CONFIG['batch_size'],
            'cache_stats': {
                'train': train_cache_stats,
                'val': val_cache_stats
            },
            'best_val_accuracy': max(history_initial.history['val_accuracy']),
            'final_val_accuracy': history_initial.history['val_accuracy'][-1],
            'training_completed': datetime.now().isoformat()
        }
        
        if history_finetune:
            final_stats['best_val_accuracy_finetune'] = max(history_finetune.history['val_accuracy'])
            final_stats['final_val_accuracy'] = history_finetune.history['val_accuracy'][-1]
        
        # Guardar estadísticas
        stats_path = Path("covid_classifier/results/training/optimized_training_stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"\n🎉 ENTRENAMIENTO OPTIMIZADO COMPLETADO")
        print("=" * 50)
        print(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
        print(f"🎯 Mejor val_accuracy: {final_stats['best_val_accuracy']:.4f}")
        print(f"🚀 GPU utilizada: {gpu_available}")
        print(f"📊 Cache hit rate (train): {train_cache_stats['hit_rate']:.3f}")
        print(f"📊 Cache hit rate (val): {val_cache_stats['hit_rate']:.3f}")
        
        return {
            'model': model,
            'history_initial': history_initial,
            'history_finetune': history_finetune,
            'final_stats': final_stats
        }
        
    except Exception as e:
        resource_monitor.stop_monitoring()
        print(f"❌ Error durante entrenamiento optimizado: {e}")
        raise
    finally:
        # Limpieza final
        with memory_cleanup():
            pass

# ====================================================================
# FUNCIÓN DE BENCHMARK
# ====================================================================

def benchmark_training_speed():
    """Benchmark para comparar velocidad de entrenamiento"""
    print("🏁 EJECUTANDO BENCHMARK DE VELOCIDAD")
    print("=" * 50)
    
    # Simular algunas épocas para medir velocidad
    OPTIMIZED_TRAINING_CONFIG['initial_epochs'] = 3
    OPTIMIZED_TRAINING_CONFIG['fine_tune_epochs'] = 2
    
    start_time = time.time()
    
    try:
        results = train_covid_model_optimized()
        total_time = time.time() - start_time
        
        print(f"\n📊 RESULTADOS DEL BENCHMARK:")
        print(f"⏱️  Tiempo total: {total_time/60:.2f} minutos")
        print(f"⚡ Tiempo por época: {total_time/5:.1f} segundos")
        print(f"🎯 Accuracy alcanzada: {results['final_stats']['final_val_accuracy']:.4f}")
        
        return total_time
        
    except Exception as e:
        print(f"❌ Error en benchmark: {e}")
        return None

# ====================================================================
# FUNCIONES ADICIONALES PARA COMPATIBILIDAD
# ====================================================================

def get_tensorflow_version_info():
    """Obtener información de la versión de TensorFlow"""
    print(f"🔧 TensorFlow versión: {tf.__version__}")
    print(f"🔧 Keras versión: {keras.__version__}")
    
    # Verificar si hay funciones obsoletas
    try:
        # Verificar si model.fit acepta workers
        import inspect
        fit_signature = inspect.signature(tf.keras.Model.fit)
        has_workers = 'workers' in fit_signature.parameters
        print(f"🔧 Soporte para 'workers': {has_workers}")
        
        return {
            'tf_version': tf.__version__,
            'keras_version': keras.__version__,
            'supports_workers': has_workers
        }
    except:
        return {
            'tf_version': tf.__version__,
            'keras_version': keras.__version__,
            'supports_workers': False
        }

# ====================================================================
# EJECUTAR SI ES LLAMADO DIRECTAMENTE  
# ====================================================================

if __name__ == "__main__":
    # Configurar semillas para reproducibilidad
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Mostrar información de la versión
    version_info = get_tensorflow_version_info()
    print(f"📦 Información del sistema:")
    for key, value in version_info.items():
        print(f"   {key}: {value}")
    
    # Verificar recursos del sistema
    print(f"💻 CPU cores: {psutil.cpu_count()}")
    print(f"🧠 RAM total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Mostrar GPUs disponibles
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"🎯 GPUs disponibles: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    else:
        print("⚠️  No se detectaron GPUs")
    
    # Ejecutar entrenamiento optimizado
    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO OPTIMIZADO COMPLETO")
    print("="*60)
    
    training_results = train_covid_model_optimized()
    
    print(f"\n🎯 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print(f"El modelo optimizado está listo para la Fase 5: Evaluación y Métricas")
    print(f"Modelo final: covid_classifier/models/final_covid_model_optimized.h5")