import os
import time


free_memery = False
kill_all = False

if __name__ == '__main__':
    if free_memery:
        excute_str = 'sudo fuser -v /dev/nvidia*'
        out_list = list(set(os.popen(excute_str).readlines()))
        print(out_list)
    if kill_all:
        excute_str = 'nvidia-smi'
        out_list = os.popen(excute_str).readlines()
        for oo in out_list:
            if oo.find('python') != -1:
                proc_list = oo.split()
                pid = proc_list[2].strip()
                kill_str = 'kill -9' + ' '.join(pid)
                print(kill_str)
                time.sleep(0.3)
                os.system(kill_str)
