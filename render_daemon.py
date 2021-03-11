import schedule
import time
import os
from pathlib import Path


running_policy_list = []


def render(self):
    global running_policy_list
    policy_list = os.listdir(str(Path.home())+'/policies')
    new_policies = set(self.running_policy_list)-set(policy_list)
    running_policy_list = policy_list

    for policy in new_policies:
        os.system('python3 /home/justin_terry/rl_scratch/render.py '+str(Path.home())+'/'+policy)


schedule.every(60).seconds.do(render())

while True:
    schedule.run_pending()
    time.sleep(1)
