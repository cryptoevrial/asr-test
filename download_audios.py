from utils import HuggingFaceAPI


if __name__ == '__main__':
    api = HuggingFaceAPI()
    api.download_wav_files()