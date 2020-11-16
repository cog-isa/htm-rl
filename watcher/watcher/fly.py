from utils.drawer import Drawer

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="the name of the datafile")
    parser.add_argument("--size", help="width,height")
    args = parser.parse_args()
    if args.size is None:
        width, height = 1280, 720
    else:
        width, height = args.size.split(',')
    drawer = Drawer('data/'+args.name, [int(width), int(height)])

    while not drawer.window.should_close():
        drawer.update()
        # the main application loop
        while not drawer.window.should_close() and not drawer.window.next and not drawer.window.previous:
            drawer.process()
        if drawer.window.next and drawer.current + 2 < len(drawer.data_base.keys()): drawer.current = drawer.current + 1
        if drawer.window.previous and drawer.current > 0: drawer.current = drawer.current - 1
        drawer.window.next = False
        drawer.window.previous = False

    drawer.window.terminate()

