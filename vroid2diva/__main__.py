import sys


def run() -> int:
    args = sys.argv[1:]
    # A single .vrm argument (e.g. a model dropped onto the launcher) opens the GUI with it.
    if len(args) == 1 and args[0].lower().endswith(".vrm"):
        from .gui import main
        return main(vrm_path=args[0])
    if args:
        from .cli import main
        return main()
    from .gui import main
    return main()


if __name__ == "__main__":
    sys.exit(run())
