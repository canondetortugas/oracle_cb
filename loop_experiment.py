import os

#cmd = "python Semibandits.py --T 31278 --dataset mslr30k --L 1 --alg lin  --param 0.1 --I {}"

# cmd = "python Semibandits.py --T 31278 --dataset yahoo --alg lin  --param 0.01 --I {}"
cmd = "python Semibandits.py --T 31278 --dataset yahoo --alg rucb --learning_alg gb5fast  --param 0.01 --I {}"

# cmd = "python Semibandits.py --T 31278 --dataset mslr30k --L 1 --alg eps --learning_alg gb5  --param 0.005 --I {}"

# cmd = "python Semibandits.py --T 8000 --dataset mslr30k --L 1 --alg lin  --param 0.05 --I {}"

# cmd1 = "python Semibandits.py --T 30000 --dataset mslr30k --L 1 --alg rucb --learning_alg gb5  --param 0.01 --I {}"
# cmd2 = "python Semibandits.py --T 30000 --dataset mslr30k --L 1 --alg rucb --learning_alg gb5_fast  --param 0.01 --I {}"

# cmd = "python .\Semibandits.py --dataset mslr30k --T 30000 --alg lin --param 0.05 --I {}"
# cmd = "python .\Semibandits.py --dataset mslr30k --T 30000 --alg mini --learning_alg gb5 --param 0.01 --I {}"

if __name__=='__main__':
    for idx in range(10):
        os.system(cmd.format(idx))

#     for idx in range(10):
#         os.system(cmd1.format(idx))

# if __name__=='__main__':

#     for idx in range(10):
#         os.system.('mv rucb_gb5_fast_0.01000_validation_{}.out rucb_gb5fast_0.01000_validation_{}.out'.format(idx, idx)
