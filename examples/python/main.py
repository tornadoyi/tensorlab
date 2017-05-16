import argparse

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["object-detection", "playgame", ], help="choice command")

    object_detect_arg = parser.add_argument_group('object-detection')
    object_detect_arg.add_argument('--data-path', type=str, default="../data/training.xml")

    playgame_arg = parser.add_argument_group('playgame')
    playgame_arg.add_argument('--game', type=str, default='CartPole-v0')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse()

    if args.command == "object-detection":
        import examples.object_detection.object_detection as object_detection
        object_detection.main(args.data_path)

    elif args.command == "playgame":
        import examples.rl.a3c.main as main
        main.main(args)