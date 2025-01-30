import os
import subprocess


def install_gigaam():
    try:
        # 0. Установка ffmpeg
        # Используя apt
        subprocess.run(["sudo", "apt", "update"])
        subprocess.run(["sudo", "apt", "install", "ffmpeg"])

        # 1. Клонирование репозитория
        repo_url = "https://github.com/salute-developers/GigaAM.git"
        clone_command = ["git", "clone", repo_url]
        subprocess.run(clone_command, check=True)

        # 2. Переход в директорию GigaAM
        os.chdir("GigaAM")

        # 3. Установка пакета в режиме разработки
        install_command = ["pip", "install", "-e", "."]
        subprocess.run(install_command, check=True)

        print("GigaAM успешно установлен!")

    except subprocess.CalledProcessError as e:
        print(f"Произошла ошибка при выполнении команды: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == '__main__':
    install_gigaam()