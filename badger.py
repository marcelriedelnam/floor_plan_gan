import os
QUEUE_FILE = 'queue.txt'
def peek_queue():
    with open(QUEUE_FILE, 'r') as f:
        return f.readline()
def pop_queue():
    with open(QUEUE_FILE, 'r') as f:
        lines = f.readlines()
    with open(QUEUE_FILE, 'w') as f:
        f.writelines(lines[1:])
while True:
    command = peek_queue()
    if not command:
        break
    command = command.strip()
    print(f"[🦡 {os.getcwd()}]$ {command}")
    if len(command) > 0 and command[0] != "#" :
        os.system(command)
    pop_queue()
print("🦡☁️💤")