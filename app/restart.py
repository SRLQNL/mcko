import logging
import os
import subprocess

_log = logging.getLogger("mcko.restart")

# Корень проекта — на два уровня выше этого файла (app/restart.py → app/ → mcko/)
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def restart_app() -> None:
    """Запускает перезапуск приложения:
    1. Отдельный shell-процесс (новая сессия, не зависит от текущего)
    2. Ждёт 1с, чтобы текущий процесс успел завершиться
    3. Вызывает kill.sh — убивает все процессы MCKO (включая нас)
    4. Запускает run.sh — запускает watchdog и main.py заново
    """
    kill_sh = os.path.join(_PROJECT_DIR, "kill.sh")
    run_sh = os.path.join(_PROJECT_DIR, "run.sh")
    _log.info("Restart initiated: kill=%s, run=%s", kill_sh, run_sh)

    cmd = f"sleep 1 && bash '{kill_sh}' && sleep 0.5 && bash '{run_sh}'"
    subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # setsid — процесс не зависит от нашего PID
    )
    _log.info("Restart subprocess launched (detached). Waiting to be killed by kill.sh.")


def terminate_app() -> None:
    """Останавливает все процессы MCKO через kill.sh."""
    kill_sh = os.path.join(_PROJECT_DIR, "kill.sh")
    _log.info("Terminate initiated: kill=%s", kill_sh)
    subprocess.Popen(
        ["bash", "-c", f"bash '{kill_sh}'"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _log.info("Terminate subprocess launched (detached). Waiting to be killed by kill.sh.")
