import sys

def timed_input(prompt: str, timeout: int = 10, default: str = "n") -> str:
    """
    Ожидает ввод от пользователя timeout секунд. Если ввода нет — возвращает default.
    Работает на Unix благодаря select, и имеет fallback на поток для других платформ.
    """
    try:
        # try select (работает в Linux/Unix)
        import select
        sys.stdout.write(prompt)
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            line = sys.stdin.readline().strip().lower()
            return line if line else default
        else:
            print()  # перенос строки после тайм-аута
            print(f"⏲ Таймаут {timeout}s — выбран вариант '{default}'")
            return default
    except Exception:
        # fallback: поток + очередь (подходит и для Windows)
        import threading, queue
        q = queue.Queue()

        def _reader():
            try:
                q.put(input(prompt))
            except Exception:
                q.put("")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            answer = q.get(timeout=timeout).strip().lower()
            return answer if answer else default
        except queue.Empty:
            print()
            print(f"⏲ Таймаут {timeout}s — выбран вариант '{default}'")
            return default
