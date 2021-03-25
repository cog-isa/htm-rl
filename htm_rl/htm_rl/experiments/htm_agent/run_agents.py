if __name__ == '__main__':
    import subprocess
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_processes", type=int, default=1)
    parser.add_argument("-c", "--command", type=str, default='')

    args = parser.parse_args()
    procs = list()
    n_processes = args.n_processes
    command = args.command
    tokens = command.split()
    for core in range(n_processes):
        procs.append(subprocess.Popen(tokens))

    for p in procs:
        p.wait()