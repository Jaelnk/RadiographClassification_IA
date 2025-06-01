# ====================================================================
# FASE 4: ENTRENAMIENTO OPTIMIZADO - COVID-19 RADIOGRAFÍAS (CORREGIDO)
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

# Importar la Fase 3
from arq_model import create_covid_model, setup_fine_tuning

# ====================================================================
# PASO 1: CONFIGURACIÓN DEL ENTRENAMIENTO
# ====================================================================

# Directorios
BASE_DIR = Path("covid_classifier")
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
TRAINING_DIR = RESULTS_DIR / "training"

# Crear directorios
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de entrenamiento optimizada
TRAINING_CONFIG = {
    'initial_epochs': 25,        # Épocas para entrenamiento inicial
    'fine_tune_epochs': 15,      # Épocas para fine-tuning
    'patience': 8,               # Paciencia para early stopping (reducida)
    'monitor_metric': 'val_accuracy',  # Métrica a monitorear
    'save_best_only': True,      # Solo guardar el mejor modelo
    'verbose': 1,                # Nivel de verbose
    'batch_size': 64,           # Batch size optimizado
    'cache_size': 2000,         # Tamaño de caché
    'prefetch_buffer': 2,       # Buffer de prefetch
    'num_parallel_calls': -1    # Llamadas paralelas
}

print("🚀 FASE 4 OPTIMIZADA: ENTRENAMIENTO CON GPU")
print("=" * 50)
print(f"📊 Configuración optimizada: {TRAINING_CONFIG}")

# ====================================================================
# PASO 2: VERIFICACIÓN DE COMPATIBILIDAD
# ====================================================================

def check_system_compatibility():
    """Verificar compatibilidad del sistema"""
    print("🔧 Verificando compatibilidad del sistema...")
    
    # Información de TensorFlow
    tf_version = tf.__version__
    keras_version = keras.__version__
    
    print(f"🔧 TensorFlow versión: {tf_version}")
    print(f"🔧 Keras versión: {keras_version}")
    
    # Verificar soporte para workers
    supports_workers = hasattr(tf.data, 'AUTOTUNE')
    print(f"🔧 Soporte para 'workers': {supports_workers}")
    
    # Información del sistema
    system_info = {
        'tf_version': tf_version,
        'keras_version': keras_version,
        'supports_workers': supports_workers
    }
    
    print(f"📦 Información del sistema:")
    for key, value in system_info.items():
        print(f"   {key}: {value}")
    
    return system_info

# ====================================================================
# PASO 3: MONITOR DE RECURSOS OPTIMIZADO
# ====================================================================

class ResourceMonitor:
    """Monitor de recursos del sistema durante entrenamiento"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'timestamps': []
        }
        self.thread = None
    
    def start_monitoring(self):
        """Iniciar monitoreo de recursos"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True  # Hacer el thread daemon
        self.thread.start()
        print("📊 Monitor de recursos optimizado iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo de recursos"""
        self.monitoring = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)  # Timeout de 1 segundo
        print("📊 Monitor de recursos detenido")
    
    def _monitor_loop(self):
        """Loop de monitoreo en segundo plano"""
        while self.monitoring:
            try:
                self.stats['cpu_percent'].append(psutil.cpu_percent(interval=1))
                self.stats['memory_percent'].append(psutil.virtual_memory().percent)
                self.stats['timestamps'].append(datetime.now())
                time.sleep(30)  # Monitorear cada 30 segundos
            except Exception as e:
                print(f"⚠️ Error en monitoreo: {e}")
                break
    
    def get_average_stats(self):
        """Obtener estadísticas promedio"""
        if not self.stats['cpu_percent']:
            return {'avg_cpu': 0, 'avg_memory': 0, 'max_cpu': 0, 'max_memory': 0}
        
        return {
            'avg_cpu': np.mean(self.stats['cpu_percent']),
            'avg_memory': np.mean(self.stats['memory_percent']),
            'max_cpu': np.max(self.stats['cpu_percent']),
            'max_memory': np.max(self.stats['memory_percent'])
        }
    
    def save_stats(self, filepath):
        """Guardar estadísticas de recursos"""
        if self.stats['timestamps']:
            stats_df = pd.DataFrame({
                'timestamp': self.stats['timestamps'],
                'cpu_percent': self.stats['cpu_percent'],
                'memory_percent': self.stats['memory_percent']
            })
            stats_df.to_csv(filepath, index=False)
            print(f"📈 Estadísticas guardadas en: {filepath}")

# ====================================================================
# PASO 4: CALLBACK PERSONALIZADO OPTIMIZADO
# ====================================================================

class TrainingLogger(keras.callbacks.Callback):
    """Callback personalizado para logging detallado"""
    
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_logs = []
        self.training_start_time = None
        self.epoch_start_time = None
        
    def on_train_begin(self, logs=None):
        """Al inicio del entrenamiento"""
        self.training_start_time = time.time()
        print(f"🚀 Iniciando entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def on_epoch_begin(self, epoch, logs=None):
        """Al inicio de cada época"""
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        """Al final de cada época"""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            
            # Registrar métricas de la época
            epoch_info = {
                'epoch': epoch + 1,
                'timestamp': datetime.now().isoformat(),
                'epoch_time': epoch_time,
                **(logs or {})
            }
            self.epoch_logs.append(epoch_info)
            
            # Mostrar progreso mejorado
            print(f"📊 Época {epoch + 1}: "
                  f"loss={logs.get('loss', 0):.4f}, "
                  f"acc={logs.get('accuracy', 0):.4f}, "
                  f"val_loss={logs.get('val_loss', 0):.4f}, "
                  f"val_acc={logs.get('val_accuracy', 0):.4f} "
                  f"({epoch_time:.1f}s)")
    
    def on_train_end(self, logs=None):
        """Al final del entrenamiento"""
        if self.training_start_time is not None:
            total_time = time.time() - self.training_start_time
            print(f"✅ Entrenamiento completado en {total_time/60:.1f} minutos")
            self.save_logs()
    
    def save_logs(self):
        """Guardar logs de entrenamiento"""
        if self.epoch_logs:
            logs_df = pd.DataFrame(self.epoch_logs)
            logs_path = self.log_dir / "training_logs.csv"
            logs_df.to_csv(logs_path, index=False)
            print(f"📝 Logs guardados en: {logs_path}")

# ====================================================================
# PASO 5: CONFIGURACIÓN DE OPTIMIZACIONES GPU/CPU
# ====================================================================

def setup_gpu_optimizations():
    """Configurar optimizaciones para GPU/CPU"""
    print("🚀 CONFIGURANDO OPTIMIZACIONES GPU Y MEMORIA")
    print("=" * 50)
    
    # Configurar GPUs si están disponibles
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"🎯 GPUs detectadas: {len(gpus)}")
    
    if gpus:
        try:
            # Configurar crecimiento de memoria
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Configuración GPU optimizada")
            return True
        except RuntimeError as e:
            print(f"⚠️ Error configurando GPU: {e}")
            return False
    else:
        print("⚠️ No se detectaron GPUs, usando CPU optimizado")
        # Configurar CPU
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Usar todos los cores
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Usar todos los cores
        return False

# ====================================================================
# PASO 6: FUNCIÓN PRINCIPAL DE ENTRENAMIENTO OPTIMIZADA
# ====================================================================

def train_covid_model_optimized():
    """Función principal optimizada para entrenar el modelo COVID-19"""
    print("🚀 INICIANDO ENTRENAMIENTO OPTIMIZADO COMPLETO")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # Verificar compatibilidad del sistema
    system_info = check_system_compatibility()
    
    # Configurar optimizaciones
    gpu_available = setup_gpu_optimizations()
    
    # Inicializar monitor de recursos
    resource_monitor = ResourceMonitor()
    resource_monitor.start_monitoring()
    
    try:
        print("🔧 Creando modelo y componentes optimizados...")
        
        # Crear modelo y componentes
        model_components = create_covid_model()
        
        model = model_components['model']
        base_model = model_components['base_model']
        train_gen = model_components['train_generator']
        val_gen = model_components['val_generator']
        callbacks = model_components['callbacks']
        class_weights = model_components['class_weights']
        
        # Configurar callbacks optimizados
        print("📋 Configurando callbacks optimizados...")
        
        # Agregar logger personalizado
        training_logger = TrainingLogger(TRAINING_DIR)
        callbacks.append(training_logger)
        
        # Configurar early stopping más agresivo
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=TRAINING_CONFIG['patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        # Actualizar callbacks
        callbacks = [cb for cb in callbacks if not isinstance(cb, keras.callbacks.EarlyStopping)]
        callbacks.append(early_stopping)
        
        print("✅ Callbacks optimizados configurados")
        print("✅ Componentes optimizados listos")
        
        # Información del entrenamiento
        print(f"🎯 GPU disponible: {gpu_available}")
        print(f"📊 Batch size: {TRAINING_CONFIG['batch_size']}")
        print(f"💾 Cache size: {TRAINING_CONFIG['cache_size']}")
        
        # FASE INICIAL OPTIMIZADA
        print(f"\n📚 FASE INICIAL OPTIMIZADA")
        print("=" * 40)
        print(f"🎯 FASE INICIAL OPTIMIZADA")
        print("=" * 40)
        print(f"📊 Épocas: {TRAINING_CONFIG['initial_epochs']}")
        print(f"🔒 Capas base: CONGELADAS")
        print(f"🚀 Batch size: {TRAINING_CONFIG['batch_size']}")
        
        initial_start_time = time.time()
        
        # Entrenar fase inicial
        history_initial = model.fit(
            train_gen,
            epochs=TRAINING_CONFIG['initial_epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=TRAINING_CONFIG['verbose']
        )
        
        initial_time = time.time() - initial_start_time
        print(f"⏱️ Tiempo fase inicial: {initial_time/60:.1f} minutos")
        
        # FASE FINE-TUNING OPTIMIZADA
        history_finetune = None
        if TRAINING_CONFIG['fine_tune_epochs'] > 0:
            print(f"\n🔧 FASE FINE-TUNING OPTIMIZADA")
            print("=" * 40)
            
            try:
                # En lugar de cargar el modelo desde archivo, usar el modelo actual
                # que ya tiene los mejores pesos restaurados por EarlyStopping
                print("🔧 Configurando fine-tuning...")
                
                # Configurar fine-tuning
                model = setup_fine_tuning(model, base_model)
                
                # Actualizar callbacks para fine-tuning
                # Cambiar el path del checkpoint para fine-tuning
                best_model_path = MODELS_DIR / 'best_mobilenetv2_finetune.keras'  # Usar .keras
                
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    filepath=str(best_model_path),
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
                
                # Actualizar callbacks
                callbacks_finetune = [cb for cb in callbacks if not isinstance(cb, keras.callbacks.ModelCheckpoint)]
                callbacks_finetune.append(checkpoint_callback)
                
                finetune_start_time = time.time()
                
                # Entrenar fine-tuning
                history_finetune = model.fit(
                    train_gen,
                    epochs=TRAINING_CONFIG['fine_tune_epochs'],
                    validation_data=val_gen,
                    callbacks=callbacks_finetune,
                    class_weight=class_weights,
                    verbose=TRAINING_CONFIG['verbose']
                )
                
                finetune_time = time.time() - finetune_start_time
                print(f"⏱️ Tiempo fine-tuning: {finetune_time/60:.1f} minutos")
                
            except Exception as e:
                print(f"⚠️ Error en fine-tuning: {e}")
                print("🔄 Continuando sin fine-tuning...")
                history_finetune = None
        
        # Guardar modelo final usando formato Keras nativo
        final_model_path = MODELS_DIR / 'final_covid_model.keras'
        model.save(str(final_model_path))
        print(f"💾 Modelo final guardado en: {final_model_path}")
        
        # Crear visualizaciones
        plot_training_history_optimized(history_initial, history_finetune)
        
        # Estadísticas finales
        total_time = time.time() - total_start_time
        resource_monitor.stop_monitoring()
        
        # Guardar estadísticas de recursos
        resource_stats_path = TRAINING_DIR / "resource_usage.csv"
        resource_monitor.save_stats(resource_stats_path)
        
        # Resumen final
        resource_stats = resource_monitor.get_average_stats()
        
        # Calcular mejor accuracy
        best_val_acc_initial = max(history_initial.history['val_accuracy'])
        best_val_acc_final = best_val_acc_initial
        
        if history_finetune:
            best_val_acc_finetune = max(history_finetune.history['val_accuracy'])
            best_val_acc_final = max(best_val_acc_initial, best_val_acc_finetune)
        
        final_stats = {
            'total_training_time_minutes': total_time / 60,
            'initial_epochs_completed': len(history_initial.history['loss']),
            'finetune_epochs_completed': len(history_finetune.history['loss']) if history_finetune else 0,
            'best_val_accuracy_initial': best_val_acc_initial,
            'best_val_accuracy_final': best_val_acc_final,
            'final_val_accuracy': history_finetune.history['val_accuracy'][-1] if history_finetune else history_initial.history['val_accuracy'][-1],
            'resource_usage': resource_stats,
            'system_info': system_info,
            'gpu_available': gpu_available,
            'training_completed': datetime.now().isoformat()
        }
        
        # Guardar estadísticas finales
        stats_path = TRAINING_DIR / "final_training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"\n🎉 ENTRENAMIENTO OPTIMIZADO COMPLETADO")
        print("=" * 60)
        print(f"⏱️ Tiempo total: {total_time/60:.1f} minutos")
        print(f"🎯 Mejor val_accuracy inicial: {best_val_acc_initial:.4f}")
        print(f"🎯 Mejor val_accuracy final: {best_val_acc_final:.4f}")
        print(f"💻 Uso promedio CPU: {resource_stats['avg_cpu']:.1f}%")
        print(f"🧠 Uso promedio RAM: {resource_stats['avg_memory']:.1f}%")
        print(f"📊 Estadísticas en: {stats_path}")
        
        return {
            'model': model,
            'history_initial': history_initial,
            'history_finetune': history_finetune,
            'final_stats': final_stats
        }
        
    except Exception as e:
        resource_monitor.stop_monitoring()
        print(f"❌ Error durante entrenamiento optimizado: {e}")
        import traceback
        traceback.print_exc()
        raise

# ====================================================================
# PASO 7: VISUALIZACIÓN OPTIMIZADA
# ====================================================================

def plot_training_history_optimized(history_initial, history_finetune=None):
    """Crear gráficos optimizados del historial de entrenamiento"""
    print("📈 Creando gráficos optimizados de entrenamiento...")
    
    try:
        # Combinar historiales si hay fine-tuning
        if history_finetune:
            metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
            combined_history = {}
            
            for metric in metrics:
                if metric in history_initial.history and metric in history_finetune.history:
                    combined_history[metric] = (
                        history_initial.history[metric] + 
                        history_finetune.history[metric]
                    )
                elif metric in history_initial.history:
                    combined_history[metric] = history_initial.history[metric]
            
            separation_point = len(history_initial.history['loss'])
        else:
            combined_history = history_initial.history
            separation_point = None
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Historial de Entrenamiento COVID-19 MobileNetV2 (Optimizado)', fontsize=16)
        
        # Gráfico 1: Loss
        if 'loss' in combined_history and 'val_loss' in combined_history:
            axes[0, 0].plot(combined_history['loss'], label='Training Loss', color='blue', linewidth=2)
            axes[0, 0].plot(combined_history['val_loss'], label='Validation Loss', color='red', linewidth=2)
            if separation_point:
                axes[0, 0].axvline(x=separation_point, color='green', linestyle='--', alpha=0.7, label='Fine-tuning')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Época')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Accuracy
        if 'accuracy' in combined_history and 'val_accuracy' in combined_history:
            axes[0, 1].plot(combined_history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
            axes[0, 1].plot(combined_history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
            if separation_point:
                axes[0, 1].axvline(x=separation_point, color='green', linestyle='--', alpha=0.7, label='Fine-tuning')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Época')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Métricas adicionales
        if 'precision' in combined_history and 'recall' in combined_history:
            axes[1, 0].plot(combined_history['precision'], label='Precision', color='purple', linewidth=2)
            axes[1, 0].plot(combined_history['recall'], label='Recall', color='brown', linewidth=2)
            axes[1, 0].set_title('Precision & Recall')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Métrica')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Precision & Recall\nno disponibles', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Precision & Recall')
        
        # Gráfico 4: Progreso de validación
        if 'val_accuracy' in combined_history:
            axes[1, 1].plot(combined_history['val_accuracy'], label='Val Accuracy', color='red', linewidth=2, marker='o', markersize=3)
            axes[1, 1].set_title('Validation Accuracy Progress')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Validation Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Agregar línea del mejor resultado
            best_val_acc = max(combined_history['val_accuracy'])
            axes[1, 1].axhline(y=best_val_acc, color='green', linestyle=':', alpha=0.7, 
                              label=f'Best: {best_val_acc:.4f}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_path = TRAINING_DIR / "training_history_optimized.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráficos optimizados guardados en: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ Error creando gráficos: {e}")

# ====================================================================
# FUNCIÓN PRINCIPAL
# ====================================================================

if __name__ == "__main__":
    print("============================================================")
    print("🚀 INICIANDO ENTRENAMIENTO OPTIMIZADO COMPLETO")
    print("============================================================")
    
    # Configurar semillas para reproducibilidad
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Verificar recursos disponibles
    print(f"💻 CPU cores: {psutil.cpu_count()}")
    print(f"🧠 RAM total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"⚠️ No se detectaron GPUs" if not tf.config.experimental.list_physical_devices('GPU') else f"✅ GPUs disponibles: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    
    # Ejecutar entrenamiento optimizado
    try:
        training_results = train_covid_model_optimized()
        
        print(f"\n🎯 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"🎯 SIGUIENTE PASO:")
        print(f"El modelo está entrenado y listo para la Fase 5: Evaluación y Métricas")
        print(f"Modelo final guardado en: covid_classifier/models/final_covid_model.keras")
        
    except Exception as e:
        print(f"\n❌ ERROR DURANTE ENTRENAMIENTO:")
        print(f"Error: {e}")
        print(f"\n🔧 Posibles soluciones:")
        print(f"1. Verificar que arq_model.py está disponible")
        print(f"2. Verificar que los datos están en la ubicación correcta")
        print(f"3. Reducir batch_size si hay problemas de memoria")
        print(f"4. Verificar dependencias de TensorFlow")