import psutil
import os
import signal

current_pid = os.getpid()

# Trova tutti i processi figli del processo attuale (es. l'IDE)
parent = psutil.Process(current_pid)
for child in parent.children(recursive=True):
    print(f"Terminando il processo PID: {child.pid}, Nome: {child.name()}")
    try:
        child.terminate()  # Termina il processo
    except psutil.NoSuchProcess:
        continue

# Assicurati che i processi siano terminati
gone, still_alive = psutil.wait_procs(parent.children(), timeout=3)
for p in still_alive:
    print(f"Forzando la terminazione del processo PID: {p.pid}, Nome: {p.name()}")
    p.kill()  # Forza la terminazione se non è stato possibile terminarlo
