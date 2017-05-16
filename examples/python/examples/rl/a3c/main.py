from a3c import A3C


def main(args):

    game = args.game
    a3c = A3C(game)

    a3c.train(log_per_second=1)