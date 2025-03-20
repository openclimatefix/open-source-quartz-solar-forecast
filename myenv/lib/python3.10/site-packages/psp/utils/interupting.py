import contextlib
import sys


@contextlib.contextmanager
def continue_on_interupt(prompt: bool = False):
    try:
        yield
    except KeyboardInterrupt:
        if prompt:
            answer = None
            while answer not in ["Y", "n"]:
                answer = input("Interupted, continue? [Y/n] ")
            if answer == "n":
                print("Exiting without saving")
                sys.exit()
