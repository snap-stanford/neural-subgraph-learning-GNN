import sys
out_mem = sys.argv[1]

with open(out_mem, mode='r') as f:
    out_mem = f.read()
    out_mem = out_mem.split('\n')

    out_mem = [int(x) for x in out_mem if x]
    max_mem = max(out_mem) / (1024**2)

    print("Max memory: %.3fMB" %max_mem)
