import time
import psutil


class SimpleMonitor:
    """Classe simples para monitoramento de desempenho em scripts Python.

    Realiza medições do tempo de execução, uso de memória e uso de CPU, exibindo
    um resumo ao final com possíveis alertas em caso de consumo elevado.
    """

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Inicia o monitoramento de recursos.

        Registra o tempo inicial e o percentual de uso de memória no momento da chamada.
        Deve ser chamado antes do trecho de código a ser monitorado.
        """

        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().percent
        
    def stop(self):
        """Finaliza o monitoramento e exibe as métricas coletadas.

        Calcula o tempo total de execução, coleta o uso atual de memória e CPU,
        e imprime um relatório resumido no terminal. Também emite alertas se
        algum dos recursos estiver acima de limites críticos (80% ou 30s).
        """
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
