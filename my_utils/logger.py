import os
import json

class PSNRLogger:
    def __init__(self, save_path, data_path):
        self.results = {}
        self.results_time = {}
        self.metadata = {'path': data_path}
        self.file_name = data_path
        self.path = save_path
        self.push(0,0)

    
    def push(self, psnr, epoch):
        self.results[str(epoch)] = psnr
        
    def push_time(self, forward, backward, per_epoch_whole, epoch):
        if str(epoch) not in self.results_time:
            self.results_time[str(epoch)] = {}
        
        self.results_time[str(epoch)].update({
            'forward': forward,
            'backward': backward,
            'per_epoch_whole': per_epoch_whole
        })

    def push_infer_time(self, inference, epoch):
        if str(epoch) not in self.results_time:
            self.results_time[str(epoch)] = {}

        self.results_time[str(epoch)].update({
            'inference_time': inference
        })
            
    def save_results(self):
        full_path = f"{self.path}/{self.file_name}.json"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            json.dump({ **self.metadata,'result': self.results,'result_time': self.results_time,}, f, indent=4)

    def load_results(self):
        full_path = f"{self.path}/{self.file_name}.json"
        # 파일이 존재하는지 확인
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                data = json.load(f)
                self.results = data['result']
                self.results_time = data['result_time']
                self.metadata = {key: data[key] for key in data if key != ('result' or 'result_time')}
        else:
            # 파일이 존재하지 않으면 새 파일을 생성하고 초기 데이터 구조를 저장
            os.makedirs(os.path.dirname(full_path), exist_ok=True)  # 필요한 디렉토리 생성
            with open(full_path, 'w') as f:
                json.dump({ **self.metadata,'result': self.results,'result_time': self.results_time,}, f, indent=4)
            print(f"No existing file found. Created a new file at {full_path}")


    def set_result_path(self, path):
        self.file_name = path
        
    def set_metadata(self, key, value):
        self.metadata[key] = value