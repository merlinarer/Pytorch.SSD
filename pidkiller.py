import os
import time


free_memery = False
free_memery_all = True
kill_all = False

if __name__ == '__main__':
    if free_memery:
        excute_str = 'sudo fuser -v /dev/nvidia*'
        out_list = list(set(os.popen(excute_str).readlines()))
        print(out_list)  # for safe manual kill command
    if free_memery_all:
        pid = list(set(os.popen('sudo fuser -v /dev/nvidia*').read().split()))
        kill_cmd = 'sudo kill -9 ' + ' '.join(pid)
        print(kill_cmd)
        os.popen(kill_cmd)
    if kill_all:
        excute_str = 'nvidia-smi'
        out_list = os.popen(excute_str).readlines()
        for oo in out_list:
            if oo.find('python') != -1:
                proc_list = oo.split()
                pid = proc_list[2].strip()
                kill_str = 'sudo kill -9 ' + pid
                print(kill_str)
                time.sleep(0.3)
                os.system(kill_str)
