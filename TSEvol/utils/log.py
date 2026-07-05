"""
Describe: logging utilities for the TSEvol pipeline
"""
import time, os, sys, csv
class FileDealer:
    def __init__(self, saved_path):
        self.saved_path = saved_path
        self.timestamp = self.create_father_dir()
        self.father_dir = os.path.join(saved_path, self.timestamp)
        self.file_initilizer = True
        self.raw_data_file = os.path.join(self.father_dir, "raw_data.csv")                      # raw inter-agent conversation log (CSV)
        self.conversation_data_file = os.path.join(self.father_dir, "conversation_data.csv")    # curated multi-turn conversation log (CSV)


    def create_father_dir(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.saved_path, timestamp)
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            print(f"Wrong For Father Dir: {e}")
            return None
        return timestamp

    def data_writer(self, new_data, file_type):
        with open(file_type, mode="a", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data)




