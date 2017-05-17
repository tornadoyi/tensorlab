from a3c import A3C


def main(args):

    game = args.game
    a3c = A3C(game, net_type='ff')

    a3c.train(log_per_second=1, save_statistic_per_second=10)