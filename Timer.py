import threading


class Timer:
    def __init__(self):
        pass

    def post_timer(self):
        threading.Timer(10, self.post_timer).start()