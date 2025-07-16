import time
import psutil

class SimpleMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Inicia monitoramento"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().percent
        
    def stop(self):
        """Para monitoramento e exibe métricas"""
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        execution_time = end_time - self.start_time
        memory_usage = end_memory
        
        print("\n" + "="*50)
        print("📊 MÉTRICAS DE MONITORAMENTO")
        print("="*50)
        print(f"⏱️ Tempo de execução: {execution_time:.2f}s")
        print(f"💾 Uso de memória: {memory_usage:.1f}%")
        print(f"🖥️ Uso de CPU: {cpu_usage:.1f}%")
        
        # Alertas simples
        if execution_time > 30:
            print("🔴 ALERTA: Tempo de execução alto!")
        if memory_usage > 80:
            print("🔴 ALERTA: Uso de memória crítico!")
        if cpu_usage > 80:
            print("🔴 ALERTA: Uso de CPU crítico!")
            
        print("="*50)