import json
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

from utils import GigaAM, Whisper


def worker(gpu_id, input_queue, output_queue):
    asr_model = GigaAM(model_name='ctc', gpu_id=gpu_id)
    asr_model.load_model(model_name=asr_model.model_name)
    while True:
        file = input_queue.get()
        if file == 'STOP':
            break
        result = asr_model.transcribe(file)
        output_queue.put(
            {
                file: result
            }
        )

def process(num_gpu: int):

    input_queue = Queue()
    output_queue = Queue()

    files = []
    '''
    Указать папку с файлами!!!
    '''
    for file in os.listdir(Path.cwd()):
        if file.endswith(('.wav',)):
            files.append(file)

    processes = []
    for gpu_id in range(num_gpu):
        p = Process(
            target=worker,
            args=(gpu_id, input_queue, output_queue)
        )
        p.start()
        processes.append(p)

    for file in files:
        input_queue.put(file)

    for _ in range(num_gpu):
        input_queue.put('STOP')

    results = []
    for _ in range(len(files)):
        results.append(output_queue.get())

    for p in processes:
        p.join()

    return results

if __name__ == '__main__':
    start_time = time.time()
    results = process(1)
    end_time = time.time()
    result_time = end_time-start_time
    '''
    Указать имя файла для каждой модели
    '''
    with open('data.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(result_time)




