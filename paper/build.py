import os
import sys
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class LatexEventHandler(FileSystemEventHandler):
    LATEX_FLAGS = "-interaction nonstopmode -halt-on-error -file-line-error"
    FILETYPE_INPUT = [".tex"]

    def on_any_event(self, event):
        for ext in self.FILETYPE_INPUT:
            if event.src_path.endswith(ext):
                self.compile(event)

    def compile(self, event):
        print("=== LATEX ===")
        os.system(f"pdflatex {self.LATEX_FLAGS} {event.src_path}")
        os.system(f"md5 {event.src_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."

    observer = Observer()
    observer.schedule(LatexEventHandler(), path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
