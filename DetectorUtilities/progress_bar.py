def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_last="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_last)
    # Print new line on completion
    if iteration == total:
        print()
