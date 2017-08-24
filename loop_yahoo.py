import os


cmds = ["python Semibandits.py --T 31278 --dataset yahoo --alg eps --learning_alg gb5  --param 0.01 --I {}",
        "python Semibandits.py --T 31278 --dataset yahoo --alg lin  --param 0.01 --I {}",
        "python Semibandits.py --T 31278 --dataset yahoo --alg rucb --learning_alg gb5fast  --param 0.01 --I {}",
        "python Semibandits.py --T 31278 --dataset yahoo --alg mini --learning_alg gb5  --param 0.01 --I {}"
]


if __name__=='__main__':
    for cmd in cmds:
        for idx in range(8):
            os.system(cmd.format(idx))

