if __name__ == '__main__':
    import subprocess
    n_cores = 6
    procs = list()
    for core in range(n_cores):
        procs.append(subprocess.Popen('wandb agent hauska/HTM/f8nl99vh'.split()))

    for p in procs:
        p.wait()