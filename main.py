from pathlib import Path
from utils import


if __name__ == "__main__":
    # Получить список всех файлов
    folder_path = Path(r"C:\Users\dev\Desktop\Video")
    files = list(folder_path.iterdir())
    for file in files:
        print(str(file))
