import ray
import whisper

from ray.utils.queue import Queue


@ray.remote(num_gpus=1)
class WhisperWorker:
	def __init__(self, worker_id: int, model_size: str = 'large'):
		self.worker_id = worker_id
		self.model = whisper.load_model(model_size).to('cuda')
		print(f"Worker {worker_id} initialized & {model_size} model loaded")

	def transcribe(self, audio_path):
		transcribe_options = {
			"task": "transcribe",
			"fp16": True,
			"verbose": None,
			"language": "ru"
		}	
        print(f"Worker {self.worker_id} processing {audio_path}")
		result = self.model.transcribe(str(audio_path), **transcribe_options)

		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()

		return result

		async def process_queue(self, input_queue: Queue, output_queue: Queue):
	        while True:
	            try:
	                # Получаем задачу из очереди с таймаутом
	                task = await input_queue.get_async(timeout=5)
	                
	                # Обрабатываем задачу
	                result = self.transcribe(task)
	                
	                # Кладем результат в выходную очередь
	                await output_queue.put_async((task, result))
	                
	            except ray.util.queue.Empty:
	                print(f"Worker {self.worker_id}: Queue is empty, stopping")
	                break
	            except Exception as e:
	                print(f"Worker {self.worker_id}: Error processing task: {e}")
	                await output_queue.put_async((task, f"Error: {str(e)}"))


# Основной код для работы с очередями:
@ray.remote
def process_audio_files(audio_files, num_workers=4):
    # Создаем входную и выходную очереди
    input_queue = Queue()
    output_queue = Queue()
    
    # Создаем воркеров
    workers = [WhisperWorker.remote(i) for i in range(num_workers)]
    
    # Заполняем входную очередь
    for audio_file in audio_files:
        input_queue.put(audio_file)
    
    # Запускаем обработку очереди всеми воркерами
    worker_tasks = [
        worker.process_queue.remote(input_queue, output_queue) 
        for worker in workers
    ]
    
    # Собираем результаты по мере их поступления
    results = {}
    tasks_remaining = len(audio_files)
    
    while tasks_remaining > 0:
        try:
            audio_file, result = output_queue.get(timeout=10)
            results[audio_file] = result
            tasks_remaining -= 1
            print(f"Processed {audio_file}, {tasks_remaining} tasks remaining")
        except ray.util.queue.Empty:
            print("Timeout waiting for results")
            break
    
    return results


def print_cluster_info():
	print(ray.nodes())
	print('*'*50)
	print(ray.cluster_resources())

'''
if __name__ == '__main__':
	audio_files = []
	

	future = process_audio_files.remote(audio_files)
	results = ray.get(future)
    with open('data.json', 'w') as f:	
    	json.dump(results, f, ensure_ascii=False, indent=2)
'''



