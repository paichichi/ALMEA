import subprocess
class TensorBoardManager:
    def __init__(self, logger_path, port=6006):
        self.logger_path = logger_path
        self.port = port
        self.process = None

    def start(self):
        command = f"tensorboard --logdir={self.logger_path} --port={self.port}"
        self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self):
        if self.process:
            self.process.terminate()

    def get_url(self):
        return f"http://localhost:{self.port}/"