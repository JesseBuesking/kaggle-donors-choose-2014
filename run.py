

import argparse
from datetime import datetime
from genericpath import exists
import os
import sys
from kaggle_donors_choose_2014.model_runner import ModelRunner, models


import subprocess
git_hash = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'])[:10]


def save_stats(filename, stats):
    with open(filename, 'wt') as f:
        for line in stats:
            f.write('{}\n'.format(line))


def save_predictions(filename, predictions):
    with open(filename, 'wt') as f:
        f.write('projectid,is_exciting\n')
        predictions = list(predictions.itertuples())
        for prediction in predictions:
            f.write("%s,%s\n" % (
                prediction[1],
                prediction[2]
            ))


# http://stackoverflow.com/a/3637103
class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stdout.write(os.linesep)
        sys.stdout.write('ERROR: {0}{1}{1}'.format(message, os.linesep))
        self.print_help()
        sys.exit(2)


def parse_args():
    p = DefaultHelpParser(
        description='Model runner for KDD 2014 kaggle competition.')

    p.add_argument(
        '--model',
        required=True,
        dest='model_name',
        action='store',
        choices=models.keys(),
        help='the name of the model to run'
    )

    p.add_argument(
        '--submit',
        dest='submit',
        action='store_true',
        help='toggle on to run for a submission'
    )

    args = p.parse_args()
    return args


def last_few(pscore):
    last_n = []
    lastfile = 'output/last-100.txt'
    if exists(lastfile):
        with open(lastfile, 'r') as f:
            for line in f:
                dt, ghash, name, score = [i.strip() for i in line.split('\t')]
                last_n.append([dt, ghash, name, float(score)])
    last_n.append([
        datetime.utcnow().strftime('%Y%m%d%H%M%S'),
        git_hash,
        model_name,
        pscore
    ])
    last_n = last_n[-100:]
    with open(lastfile, 'w') as f:
        for row in last_n:
            f.write('{}\n'.format('\t'.join([str(i) for i in row])))
        f.flush()
    print('')
    print('last 10 scores:')
    for line in last_n[-10:]:
        print('  {} {:>15} {:>25} {:>20}'.format(
            line[0],
            line[1],
            line[2],
            '{:.5f}'.format(line[3])
        ))


if __name__ == '__main__':
    args = parse_args()

    mr = ModelRunner()

    model_name = args.model_name
    if model_name not in models:
        raise Exception('\n`{}` is not a valid model\n\nmodels:\n  {}'.format(
            model_name,
            '\n  '.join(models.keys())
        ))
    mr.init(model_name)

    # train
    if not args.submit:
        predictions = mr.train_model()
        sname = 'output/{}-{}-{}.stats.txt'.format(
            datetime.utcnow().strftime('%Y%m%d%H%M%S'),
            model_name,
            git_hash
        )
        stats = mr.stats(predictions)
        print('')
        for line in stats:
            print(line)
        save_stats(sname, stats)

        pscore = mr.auc_roc_score(predictions)

    # real run for submission
    else:
        predictions = mr.test_model()

        stats = mr.stats(predictions)
        print('')
        for line in stats:
            print(line)
        save_stats(predictions, stats)

        # save to file
        oname = 'output/{}-{}-{}.csv'.format(
            datetime.utcnow().strftime('%Y%m%d%H%M%S'),
            model_name,
            git_hash
        )
        save_predictions(oname, predictions)

        pscore = mr.auc_roc_score(predictions)

    last_few(pscore)
